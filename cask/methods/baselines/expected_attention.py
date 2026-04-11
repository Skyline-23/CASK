from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch

from ..pruning_utils import build_rotary, determine_rope_style, rotate_half


class ExpectedAttention:
    def __init__(
        self,
        budget: int = 128,
        window_size: int = 8,
        n_future_positions: int = 512,
        n_sink: int = 4,
        use_covariance: bool = True,
        use_vnorm: bool = True,
        epsilon: float = 0.0,
        protect_prefill: bool = False,
        model_path: str | None = None,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = int(budget)
        self.window_size = int(window_size)
        self.n_future_positions = int(n_future_positions)
        self.n_sink = int(n_sink)
        self.use_covariance = bool(use_covariance)
        self.use_vnorm = bool(use_vnorm)
        self.epsilon = float(epsilon)
        self.protect_prefill = bool(protect_prefill)
        self.prefill_length = 0
        self.model_path = Path(model_path) if model_path else Path(".")
        self._rotary = None
        self._rope_style: Optional[str] = None

    @property
    def requires_prerope_queries(self) -> bool:
        return True

    def attach_prefill_length(self, prefill_length: int) -> None:
        self.prefill_length = int(prefill_length)

    def _lazy_rotary(self, module, device: torch.device, dtype: torch.dtype):
        if self._rotary is None:
            self._rotary = build_rotary(
                device,
                self.model_path,
                dtype,
                config=module.config,
            )
            self._rope_style = determine_rope_style(module.config)
        return self._rotary

    def _average_rope_transform(
        self,
        module,
        start_position: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rotary = self._lazy_rotary(module, device, dtype)
        head_dim = module.head_dim
        position_ids = torch.arange(
            int(start_position),
            int(start_position) + self.n_future_positions,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        probe = torch.zeros(1, self.n_future_positions, head_dim, device=device, dtype=dtype)
        cos, sin = rotary(probe, position_ids)
        cos = cos[0]
        sin = sin[0]
        basis = torch.eye(head_dim, device=device, dtype=dtype).unsqueeze(0).expand(self.n_future_positions, -1, -1)
        rotated_basis = basis * cos.unsqueeze(1) + rotate_half(basis, style=self._rope_style or "half") * sin.unsqueeze(1)
        return rotated_basis.mean(dim=0)

    def _get_query_statistics(
        self,
        cached_queries_prerope: torch.Tensor,
        module,
        start_position: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        queries = cached_queries_prerope
        if queries.size(2) > self.n_sink:
            queries = queries[:, :, self.n_sink :, :]

        mu = queries.mean(dim=2)
        cov = None
        if self.use_covariance:
            centered = queries - mu.unsqueeze(2)
            cov = torch.einsum("bnsi,bnsj->bnij", centered, centered) / max(1, queries.shape[2])

        avg_rope = self._average_rope_transform(module, start_position, mu.device, mu.dtype)
        mu = torch.matmul(mu, avg_rope)
        if cov is not None:
            cov = torch.matmul(avg_rope.transpose(-1, -2), torch.matmul(cov, avg_rope))
        return mu, cov

    def _compute_scores(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cached_queries_prerope: torch.Tensor,
        module,
    ) -> torch.Tensor:
        mean_query, cov_query = self._get_query_statistics(
            cached_queries_prerope=cached_queries_prerope,
            module=module,
            start_position=key_states.shape[2],
        )

        bsz, num_kv_heads, seq_len, head_dim = key_states.shape
        num_attention_heads = int(module.config.num_attention_heads)
        num_key_value_groups = max(1, num_attention_heads // num_kv_heads)

        keys = key_states.repeat_interleave(num_key_value_groups, dim=1)
        value_norm = value_states.norm(dim=-1).repeat_interleave(num_key_value_groups, dim=1)

        scores = (mean_query.unsqueeze(2) * keys).sum(dim=-1) / math.sqrt(head_dim)
        if cov_query is not None:
            scores = scores + torch.einsum("bhld,bhde,bhle->bhl", keys, cov_query, keys) / head_dim / 2.0

        scores = torch.softmax(scores, dim=-1)
        if self.use_vnorm:
            scores = (scores + self.epsilon) * value_norm
        scores = scores.view(bsz, num_kv_heads, num_key_value_groups, seq_len).mean(dim=2)
        return scores

    def _compress_with_scores(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        scores: torch.Tensor,
    ):
        head_dim = key_states.shape[-1]
        kv_cache_len = key_states.shape[-2]
        if kv_cache_len <= self.budget:
            return key_states, value_states

        protected_prefix = self.n_sink
        if self.protect_prefill and self.prefill_length > 0:
            protected_prefix = max(protected_prefix, self.prefill_length)
        protected_prefix = min(protected_prefix, kv_cache_len)

        if protected_prefix >= self.budget:
            return (
                key_states[:, :, : self.budget, :],
                value_states[:, :, : self.budget, :],
            )

        window = min(self.window_size, max(0, kv_cache_len - protected_prefix))
        candidate_end = max(protected_prefix, kv_cache_len - window)
        keep_count = self.budget - protected_prefix - window

        k_prefix = key_states[:, :, :protected_prefix, :]
        v_prefix = value_states[:, :, :protected_prefix, :]
        k_window = key_states[:, :, candidate_end:, :]
        v_window = value_states[:, :, candidate_end:, :]

        if keep_count <= 0 or candidate_end <= protected_prefix:
            return (
                torch.cat([k_prefix, k_window], dim=2),
                torch.cat([v_prefix, v_window], dim=2),
            )

        candidate_scores = scores[:, :, protected_prefix:candidate_end]
        candidate_keys = key_states[:, :, protected_prefix:candidate_end, :]
        candidate_values = value_states[:, :, protected_prefix:candidate_end, :]

        if candidate_scores.shape[-1] <= keep_count:
            k_selected = candidate_keys
            v_selected = candidate_values
        else:
            indices = candidate_scores.topk(keep_count, dim=-1).indices
            gather = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_selected = candidate_keys.gather(dim=2, index=gather)
            v_selected = candidate_values.gather(dim=2, index=gather)

        return (
            torch.cat([k_prefix, k_selected, k_window], dim=2),
            torch.cat([v_prefix, v_selected, v_window], dim=2),
        )

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        **kwargs,
    ):
        if key_states.shape[-2] <= self.budget:
            return key_states, value_states

        cached_queries_prerope = kwargs.get("cached_queries_prerope")
        module = kwargs.get("module")
        if cached_queries_prerope is None or module is None:
            raise ValueError("ExpectedAttention requires cached_queries_prerope and module inputs")

        scores = self._compute_scores(
            key_states=key_states,
            value_states=value_states,
            cached_queries_prerope=cached_queries_prerope,
            module=module,
        )
        return self._compress_with_scores(key_states, value_states, scores)
