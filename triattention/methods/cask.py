"""CASK v2 two-stage compressor for HuggingFace decode-time runs.

Stage 1 applies TriAttention eviction to prompt/prefix tokens when prompt-heavy
regimes would otherwise exhaust the budget. Stage 2 reuses TriAttention scoring
to protect a stable decode core subset and consolidates the remaining scratch
tokens with local pre-RoPE key merges.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .pruning_utils import invert_rope, rotate_half, verify_rotary_alignment
from .triattention import (
    TriAttention,
    TriAttentionConfig,
    _cache_to_legacy_tuple,
    _legacy_tuple_to_cache,
)


@dataclass
class CASKConfig(TriAttentionConfig):
    """Configuration for phase-2 CASK."""

    prefix_coverage_ratio: float = 0.0625
    recent_window_size: int = 128
    protected_core_ratio: float = 0.5
    min_protected_core_tokens: int = 1
    core_selection_mode: str = "vote"
    merge_operator: str = "keepkv"
    merge_local_window: int = 32
    merge_similarity_threshold: float = 0.985
    value_projection_threshold: float | None = None
    representative_mode: str = "score_max_source"
    promotion_score_ratio: float | None = None
    merge_score_mass_ratio_threshold: float | None = None
    use_phase_markers: bool = True
    phase_marker_token_ids: Optional[Tuple[Tuple[int, ...], ...]] = None


@dataclass
class _ScratchDescriptor:
    source_indices: List[int]
    representative_position: int
    representative_source_index: int
    representative_score: float
    score_mass: float
    count: int
    span: Tuple[int, int]
    phase_id: int
    protected_core: bool = False
    support_weight: float = 1.0
    rep_vector: Optional[torch.Tensor] = None
    value_vector: Optional[torch.Tensor] = None


class CASK(TriAttention):
    """Two-stage prefix eviction + decode scratch merging."""

    def __init__(self, config: CASKConfig) -> None:
        super().__init__(config)
        self.recent_window_size = int(config.recent_window_size)
        self.prefix_coverage_ratio = float(config.prefix_coverage_ratio)
        self.protected_core_ratio = float(config.protected_core_ratio)
        self.min_protected_core_tokens = int(config.min_protected_core_tokens)
        self.core_selection_mode = str(config.core_selection_mode).strip().lower()
        self.merge_operator = str(config.merge_operator).strip().lower()
        self.merge_local_window = int(config.merge_local_window)
        self.merge_similarity_threshold = float(config.merge_similarity_threshold)
        self.value_projection_threshold = (
            None if config.value_projection_threshold is None else float(config.value_projection_threshold)
        )
        self.representative_mode = str(config.representative_mode).strip().lower()
        self.promotion_score_ratio = (
            None if config.promotion_score_ratio is None else float(config.promotion_score_ratio)
        )
        self.merge_score_mass_ratio_threshold = (
            None
            if config.merge_score_mass_ratio_threshold is None
            else float(config.merge_score_mass_ratio_threshold)
        )
        self.use_phase_markers = bool(config.use_phase_markers)
        if not (0.0 <= self.protected_core_ratio <= 1.0):
            raise ValueError("protected_core_ratio must be in [0, 1]")
        if not (0.0 <= self.prefix_coverage_ratio <= 1.0):
            raise ValueError("prefix_coverage_ratio must be in [0, 1]")
        if self.min_protected_core_tokens < 0:
            raise ValueError("min_protected_core_tokens must be >= 0")
        if self.core_selection_mode not in {"vote", "score"}:
            raise ValueError("core_selection_mode must be one of {'vote', 'score'}")
        if self.merge_operator not in {"keepkv", "mean"}:
            raise ValueError("merge_operator must be one of {'keepkv', 'mean'}")
        if self.representative_mode not in {"weighted_latest", "score_max_source"}:
            raise ValueError("representative_mode must be one of {'weighted_latest', 'score_max_source'}")
        if self.promotion_score_ratio is not None and self.promotion_score_ratio < 0.0:
            raise ValueError("promotion_score_ratio must be >= 0 when provided")
        if self.merge_score_mass_ratio_threshold is not None and self.merge_score_mass_ratio_threshold <= 0.0:
            raise ValueError("merge_score_mass_ratio_threshold must be > 0 when provided")
        if self.recent_window_size < 0:
            raise ValueError("recent_window_size must be >= 0")
        if self.merge_local_window < 0:
            raise ValueError("merge_local_window must be >= 0")
        if not (-1.0 <= self.merge_similarity_threshold <= 1.0):
            raise ValueError("merge_similarity_threshold must be in [-1, 1]")
        self.merge_counts: List[int] = []
        self.merge_supports: List[float] = []
        self.merge_spans: List[Tuple[int, int]] = []
        self.phase_ids: List[int] = []
        self.current_phase_id = 0
        self.decode_token_history: List[int] = []
        self.output_projection_slices: Dict[Tuple[int, int], torch.Tensor] = {}
        self.phase_marker_token_ids: Tuple[Tuple[int, ...], ...] = ()
        self.max_phase_marker_length = 0
        self.runtime_stats: Dict[str, object] = {}
        self._last_selection_dump_payload: Dict[str, object] | None = None
        self._set_phase_markers(config.phase_marker_token_ids)
        self._reset_runtime_stats()

    def reset_compression_state(self) -> None:
        super().reset_compression_state()
        self.merge_counts = []
        self.merge_supports = []
        self.merge_spans = []
        self.phase_ids = []
        self.current_phase_id = 0
        self.decode_token_history = []
        self._last_selection_dump_payload = None
        self._reset_runtime_stats()

    def _reset_runtime_stats(self) -> None:
        self.runtime_stats = {
            "prefix_coverage_ratio": self.prefix_coverage_ratio,
            "core_selection_mode": self.core_selection_mode,
            "merge_operator": self.merge_operator,
            "prefix_stage_mode": "triattention_eviction",
            "prefix_compression_events": 0,
            "total_prefix_evicted_tokens": 0,
            "last_prefix_event": None,
            "compression_events": 0,
            "total_protected_core_tokens": 0,
            "total_scratch_descriptors": 0,
            "total_scratch_source_tokens": 0,
            "total_scratch_merged_groups": 0,
            "total_scratch_saved_tokens": 0,
            "guard_triggered": False,
            "guard_reason_counts": {},
            "last_guard": None,
            "last_event": None,
        }

    def _record_compression_event(self, **event: int) -> None:
        if not self.runtime_stats:
            self._reset_runtime_stats()
        self.runtime_stats["compression_events"] = int(self.runtime_stats["compression_events"]) + 1
        self.runtime_stats["total_protected_core_tokens"] = (
            int(self.runtime_stats["total_protected_core_tokens"]) + int(event["protected_core_tokens"])
        )
        self.runtime_stats["total_scratch_descriptors"] = (
            int(self.runtime_stats["total_scratch_descriptors"]) + int(event["scratch_descriptors"])
        )
        self.runtime_stats["total_scratch_source_tokens"] = (
            int(self.runtime_stats["total_scratch_source_tokens"]) + int(event["scratch_source_tokens"])
        )
        self.runtime_stats["total_scratch_merged_groups"] = (
            int(self.runtime_stats["total_scratch_merged_groups"]) + int(event["scratch_merged_groups"])
        )
        self.runtime_stats["total_scratch_saved_tokens"] = (
            int(self.runtime_stats["total_scratch_saved_tokens"]) + int(event["scratch_saved_tokens"])
        )
        self.runtime_stats["last_event"] = {key: int(value) for key, value in event.items()}

    def _record_guard_event(self, reason: str, **event: int) -> None:
        if not self.runtime_stats:
            self._reset_runtime_stats()
        counts = dict(self.runtime_stats.get("guard_reason_counts") or {})
        counts[reason] = int(counts.get(reason, 0)) + 1
        payload: Dict[str, object] = {"reason": reason}
        payload.update({key: int(value) for key, value in event.items()})
        self.runtime_stats["guard_triggered"] = True
        self.runtime_stats["guard_reason_counts"] = counts
        self.runtime_stats["last_guard"] = payload

    def _record_prefix_event(self, **event: int) -> None:
        if not self.runtime_stats:
            self._reset_runtime_stats()
        self.runtime_stats["prefix_compression_events"] = int(self.runtime_stats["prefix_compression_events"]) + 1
        self.runtime_stats["total_prefix_evicted_tokens"] = (
            int(self.runtime_stats["total_prefix_evicted_tokens"]) + int(event["evicted_prefix_tokens"])
        )
        self.runtime_stats["last_prefix_event"] = {key: int(value) for key, value in event.items()}

    def _guard_keep_indices(
        self,
        *,
        seq_len: int,
        prefix_length: int,
        candidate_end: int,
        recent_start: int,
        reserve_candidate_slots: int = 0,
    ) -> List[int]:
        keep: List[int] = []
        if prefix_length > 0:
            keep.extend(range(prefix_length))
        reserve_candidate_slots = max(0, min(int(reserve_candidate_slots), max(0, candidate_end - prefix_length)))
        if reserve_candidate_slots > 0:
            keep.extend(range(candidate_end - reserve_candidate_slots, candidate_end))
        if recent_start < seq_len:
            keep.extend(range(recent_start, seq_len))
        if len(keep) < self.budget:
            missing = self.budget - len(keep)
            extra_start = max(prefix_length, candidate_end - reserve_candidate_slots - missing)
            extra = list(range(extra_start, candidate_end - reserve_candidate_slots))
            keep.extend(extra[-missing:])
        deduped = list(dict.fromkeys(keep))
        if len(deduped) > self.budget:
            deduped = deduped[-self.budget :]
        return deduped

    def _target_prefix_budget(
        self,
        *,
        seq_len: int,
        prefix_length: int,
    ) -> int:
        decode_tokens = max(0, seq_len - prefix_length)
        if decode_tokens <= 0:
            return min(prefix_length, self.budget)
        recent_target = min(self.recent_window_size, decode_tokens)
        active_scratch_slots = 2 if decode_tokens > recent_target else 0
        decode_budget_target = min(self.budget, recent_target + active_scratch_slots)
        return max(0, min(prefix_length, self.budget - decode_budget_target))

    def _select_prefix_keep_indices(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        *,
        prefix_length: int,
        prefix_budget: int,
    ) -> List[int]:
        if prefix_budget <= 0:
            return []
        if prefix_length <= prefix_budget:
            return list(range(prefix_length))

        prefix_head_matrix, _ = self._compute_candidate_score_bundle(
            pkv_tuple,
            start_index=0,
            count=prefix_length,
        )
        if prefix_head_matrix.numel() == 0:
            return list(range(prefix_budget))

        scored_matrix = prefix_head_matrix
        if self.generator is not None and scored_matrix.numel() > 0:
            noise = torch.rand(
                scored_matrix.shape,
                device=scored_matrix.device,
                generator=self.generator,
            ) * 1e-6
            scored_matrix = scored_matrix + noise
        combined_scores = scored_matrix.max(dim=0).values
        coverage_budget = min(
            prefix_budget // 2,
            max(0, int(round(prefix_budget * self.prefix_coverage_ratio))),
        )
        if coverage_budget <= 0:
            keep_relative = self._select_union_based(scored_matrix, combined_scores, prefix_budget)
            return [int(idx) for idx in keep_relative.detach().to("cpu").tolist()]

        anchor_positions = torch.linspace(
            0,
            prefix_length - 1,
            steps=coverage_budget,
            device=scored_matrix.device,
        )
        anchor_indices = sorted(
            {
                int(round(float(pos.item())))
                for pos in anchor_positions
            }
        )
        if not anchor_indices:
            keep_relative = self._select_union_based(scored_matrix, combined_scores, prefix_budget)
            return [int(idx) for idx in keep_relative.detach().to("cpu").tolist()]

        selected = set(anchor_indices)
        remaining_budget = max(0, prefix_budget - len(selected))
        if remaining_budget > 0:
            available_mask = torch.ones(prefix_length, device=scored_matrix.device, dtype=torch.bool)
            available_mask[torch.tensor(anchor_indices, device=scored_matrix.device, dtype=torch.long)] = False
            available_count = int(available_mask.sum().item())
            if available_count > 0:
                available_matrix = scored_matrix[:, available_mask]
                available_scores = combined_scores[available_mask]
                take = min(remaining_budget, available_count)
                keep_relative = self._select_union_based(available_matrix, available_scores, take)
                available_indices = torch.arange(prefix_length, device=scored_matrix.device)[available_mask]
                selected.update(int(idx) for idx in available_indices[keep_relative].detach().to("cpu").tolist())

        if len(selected) < prefix_budget:
            score_order = torch.argsort(combined_scores, descending=True)
            for idx in score_order.detach().to("cpu").tolist():
                selected.add(int(idx))
                if len(selected) >= prefix_budget:
                    break

        return sorted(selected)[:prefix_budget]

    def _apply_prefix_eviction_stage(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        *,
        prefix_length: int,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], int, bool]:
        if prefix_length <= 0 or not self.config.count_prompt_tokens:
            return pkv_tuple, prefix_length, False

        seq_len = pkv_tuple[0][0].shape[2]
        target_prefix_budget = self._target_prefix_budget(seq_len=seq_len, prefix_length=prefix_length)
        if target_prefix_budget >= prefix_length:
            return pkv_tuple, prefix_length, False

        prefix_keep = self._select_prefix_keep_indices(
            pkv_tuple,
            prefix_length=prefix_length,
            prefix_budget=target_prefix_budget,
        )
        keep = prefix_keep + list(range(prefix_length, seq_len))
        pkv_tuple = self._gather_existing_slots(pkv_tuple, keep)
        self.prefix_length = len(prefix_keep)
        self._record_prefix_event(
            original_prefix_tokens=prefix_length,
            kept_prefix_tokens=self.prefix_length,
            evicted_prefix_tokens=max(0, prefix_length - self.prefix_length),
            target_prefix_budget=target_prefix_budget,
            decode_tokens=max(0, seq_len - prefix_length),
        )
        return pkv_tuple, self.prefix_length, True

    def get_runtime_summary(self) -> Dict[str, object]:
        summary = dict(self.runtime_stats)
        summary["current_cache_tokens"] = int(len(self.cache_positions))
        summary["current_prefix_tokens"] = int(self.prefix_length)
        summary["current_total_cardinality"] = int(sum(self.merge_counts)) if self.merge_counts else 0
        return summary

    def _set_phase_markers(
        self,
        marker_token_ids: Optional[Tuple[Tuple[int, ...], ...]],
    ) -> None:
        if not self.use_phase_markers or not marker_token_ids:
            self.phase_marker_token_ids = ()
            self.max_phase_marker_length = 0
            return
        normalized = []
        seen = set()
        for marker in marker_token_ids:
            token_tuple = tuple(int(token) for token in marker if token is not None)
            if not token_tuple or token_tuple in seen:
                continue
            seen.add(token_tuple)
            normalized.append(token_tuple)
        normalized.sort(key=len)
        self.phase_marker_token_ids = tuple(normalized)
        self.max_phase_marker_length = max((len(marker) for marker in normalized), default=0)

    def attach_runtime_assets(
        self,
        model,
        *,
        phase_marker_token_ids: Optional[Tuple[Tuple[int, ...], ...]] = None,
    ) -> None:
        self._set_phase_markers(phase_marker_token_ids)
        self.output_projection_slices = {}
        if self.value_projection_threshold is None:
            return
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            return
        for layer_idx, attn_head in self.sampled_heads:
            layers = getattr(model.model, "layers", None)
            if layers is None or layer_idx >= len(layers):
                continue
            self_attn = getattr(layers[layer_idx], "self_attn", None)
            o_proj = getattr(self_attn, "o_proj", None)
            weight = getattr(o_proj, "weight", None)
            if weight is None or getattr(weight, "is_meta", False):
                continue
            start = attn_head * self.head_dim
            end = start + self.head_dim
            if end > weight.shape[1]:
                continue
            self.output_projection_slices[(layer_idx, attn_head)] = (
                weight[:, start:end].detach().to(device=self.config.device, dtype=torch.float32).contiguous()
            )

    def append_decode_tokens(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
    ) -> None:
        for token_id, position in zip(token_ids, positions):
            self.cache_positions.append(int(position))
            self.merge_counts.append(1)
            self.merge_supports.append(1.0)
            self.merge_spans.append((int(position), int(position)))
            phase_id = self.current_phase_id if self.use_phase_markers else 0
            self.phase_ids.append(phase_id)
            if not self.use_phase_markers or not self.phase_marker_token_ids:
                continue
            self.decode_token_history.append(int(token_id))
            if self.max_phase_marker_length > 0 and len(self.decode_token_history) > self.max_phase_marker_length:
                self.decode_token_history = self.decode_token_history[-self.max_phase_marker_length :]
            longest_match = 0
            history = tuple(self.decode_token_history)
            for marker in self.phase_marker_token_ids:
                if len(marker) > len(history):
                    continue
                if history[-len(marker) :] == marker and len(marker) > longest_match:
                    longest_match = len(marker)
            if longest_match > 0:
                self.current_phase_id += 1

    def compress_pkv(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        *,
        prefix_length: int = 0,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        if not pkv_tuple:
            return pkv_tuple

        batch_size = pkv_tuple[0][0].shape[0]
        if batch_size != 1:
            raise ValueError("CASK currently requires batch_size == 1")

        seq_len = pkv_tuple[0][0].shape[2]
        if seq_len <= self.budget:
            return pkv_tuple

        if self.allow_prefill_compression:
            prefix_length = 0
        prefix_length = min(prefix_length, seq_len)

        # Stage 1: shrink the pinned prefix with TriAttention eviction when it
        # crowds out the decode-stage merge budget.
        pkv_tuple, prefix_length, _ = self._apply_prefix_eviction_stage(
            pkv_tuple,
            prefix_length=prefix_length,
        )
        seq_len = pkv_tuple[0][0].shape[2]

        if prefix_length >= self.budget:
            keep = list(range(max(0, seq_len - self.budget), seq_len))
            self._record_guard_event(
                "prefix_budget_exhausted",
                seq_len=seq_len,
                prefix_length=prefix_length,
                budget=self.budget,
            )
            return self._gather_existing_slots(pkv_tuple, keep)

        recent = min(self.recent_window_size, max(0, seq_len - prefix_length))
        decode_budget = max(0, self.budget - prefix_length)
        if decode_budget <= 0:
            keep = list(range(max(0, seq_len - self.budget), seq_len))
            self._record_guard_event(
                "prefix_budget_exhausted",
                seq_len=seq_len,
                prefix_length=prefix_length,
                budget=self.budget,
            )
            return self._gather_existing_slots(pkv_tuple, keep)

        # Reserve at least two scratch slots whenever older decode tokens exist
        # so the decode-stage merge planner stays active in v2.
        if seq_len - prefix_length > decode_budget and recent >= decode_budget:
            recent = max(0, decode_budget - 2)

        candidate_start = prefix_length
        candidate_end = max(candidate_start, seq_len - recent)
        candidate_count = max(0, candidate_end - candidate_start)
        available_slots = max(0, decode_budget - recent)

        if candidate_count <= available_slots:
            keep = list(range(seq_len))
            return self._gather_existing_slots(pkv_tuple, keep)

        if available_slots <= 0:
            keep = self._guard_keep_indices(
                seq_len=seq_len,
                prefix_length=prefix_length,
                candidate_end=candidate_end,
                recent_start=candidate_end,
                reserve_candidate_slots=0,
            )
            self._record_guard_event(
                "merge_inactive",
                seq_len=seq_len,
                prefix_length=prefix_length,
                candidate_tokens=candidate_count,
                recent_tokens=recent,
                available_slots=available_slots,
            )
            return self._gather_existing_slots(pkv_tuple, keep)

        if available_slots <= 1:
            keep = self._guard_keep_indices(
                seq_len=seq_len,
                prefix_length=prefix_length,
                candidate_end=candidate_end,
                recent_start=candidate_end,
                reserve_candidate_slots=available_slots,
            )
            self._record_guard_event(
                "merge_inactive",
                seq_len=seq_len,
                prefix_length=prefix_length,
                candidate_tokens=candidate_count,
                recent_tokens=recent,
                available_slots=available_slots,
            )
            return self._gather_existing_slots(pkv_tuple, keep)

        candidate_indices = list(range(candidate_start, candidate_end))
        candidate_head_matrix, candidate_scores = self._compute_candidate_score_bundle(
            pkv_tuple,
            start_index=candidate_start,
            count=candidate_count,
        )
        candidate_supports = self._compute_candidate_supports(candidate_scores)
        for offset, idx in enumerate(candidate_indices):
            self.merge_supports[idx] = float(candidate_supports[offset].item())
        self._last_selection_dump_payload = None
        candidate_descriptors = self._build_candidate_plan(
            pkv_tuple=pkv_tuple,
            candidate_indices=candidate_indices,
            candidate_head_matrix=candidate_head_matrix,
            candidate_scores=candidate_scores,
            available_slots=available_slots,
        )

        final_descriptors: List[_ScratchDescriptor] = []
        final_descriptors.extend(self._single_descriptor(idx) for idx in range(prefix_length))
        final_descriptors.extend(candidate_descriptors)
        final_descriptors.extend(self._single_descriptor(idx) for idx in range(candidate_end, seq_len))

        if len(final_descriptors) > self.budget:
            raise RuntimeError(
                f"CASK produced {len(final_descriptors)} descriptors for budget {self.budget}"
            )

        protected_core_tokens = sum(1 for descriptor in candidate_descriptors if descriptor.protected_core)
        scratch_descriptors = [descriptor for descriptor in candidate_descriptors if not descriptor.protected_core]
        scratch_source_tokens = sum(len(descriptor.source_indices) for descriptor in scratch_descriptors)
        scratch_merged_groups = sum(1 for descriptor in scratch_descriptors if len(descriptor.source_indices) > 1)
        scratch_saved_tokens = scratch_source_tokens - len(scratch_descriptors)
        self._record_compression_event(
            candidate_tokens=candidate_count,
            protected_core_tokens=protected_core_tokens,
            scratch_descriptors=len(scratch_descriptors),
            scratch_source_tokens=scratch_source_tokens,
            scratch_merged_groups=scratch_merged_groups,
            scratch_saved_tokens=scratch_saved_tokens,
            recent_tokens=recent,
            available_slots=available_slots,
        )
        self._dump_scratch_selection_event(
            candidate_indices=candidate_indices,
            candidate_head_matrix=candidate_head_matrix,
            candidate_scores=candidate_scores,
            candidate_supports=candidate_supports,
            kv_cache_len=seq_len,
            prefix_length=prefix_length,
            recent_tokens=recent,
            available_slots=available_slots,
            protected_core_tokens=protected_core_tokens,
            scratch_descriptors=scratch_descriptors,
            scratch_source_tokens=scratch_source_tokens,
            scratch_saved_tokens=scratch_saved_tokens,
        )
        return self._materialize_descriptors(pkv_tuple, final_descriptors)

    def _compute_candidate_score_bundle(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        *,
        start_index: int,
        count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            empty_scores = torch.empty(0, device=self.config.device, dtype=torch.float32)
            empty_matrix = torch.empty((0, 0), device=self.config.device, dtype=torch.float32)
            return empty_matrix, empty_scores

        dummy_positions = torch.arange(count, device=self.config.device, dtype=torch.long)
        all_head_scores: List[torch.Tensor] = []
        for layer_idx, (key_states, _) in enumerate(pkv_tuple):
            layer_scores = self._compute_layer_head_scores(
                key_states,
                dummy_positions,
                layer_idx,
                start_index=start_index,
            )
            if layer_scores is not None:
                all_head_scores.append(layer_scores)

        if not all_head_scores:
            empty_matrix = torch.empty((0, count), device=self.config.device, dtype=torch.float32)
            return empty_matrix, torch.zeros(count, device=self.config.device, dtype=torch.float32)

        head_matrix = torch.cat(all_head_scores, dim=0).to(dtype=torch.float32)
        if self.normalize_scores and head_matrix.numel() > 0:
            mean = head_matrix.mean(dim=1, keepdim=True)
            std = head_matrix.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            head_matrix = (head_matrix - mean) / std
        return head_matrix, head_matrix.max(dim=0).values

    def _build_candidate_plan(
        self,
        *,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        candidate_indices: List[int],
        candidate_head_matrix: torch.Tensor,
        candidate_scores: torch.Tensor,
        available_slots: int,
    ) -> List[_ScratchDescriptor]:
        candidate_count = len(candidate_indices)
        if candidate_count <= available_slots:
            return [self._single_descriptor(idx) for idx in candidate_indices]

        score_map = {
            int(idx): float(candidate_scores[offset].item())
            for offset, idx in enumerate(candidate_indices)
        }
        initial_core = self._initial_core_budget(candidate_count, available_slots)
        attempt_modes = [(True, True)]
        if self.promotion_score_ratio is not None or self.merge_score_mass_ratio_threshold is not None:
            attempt_modes.append((False, False))
        for allow_promotion, allow_mass_gate in attempt_modes:
            for core_count in range(initial_core, -1, -1):
                descriptors = self._try_candidate_plan(
                    pkv_tuple=pkv_tuple,
                    candidate_indices=candidate_indices,
                    candidate_head_matrix=candidate_head_matrix,
                    candidate_scores=candidate_scores,
                    available_slots=available_slots,
                    core_count=core_count,
                    score_map=score_map,
                    allow_promotion=allow_promotion,
                    allow_mass_gate=allow_mass_gate,
                )
                if descriptors is not None:
                    return descriptors

        raise RuntimeError("Unable to build a feasible CASK candidate plan")

    def _compute_candidate_supports(self, candidate_scores: torch.Tensor) -> torch.Tensor:
        if candidate_scores.numel() == 0:
            return candidate_scores
        return F.softplus(candidate_scores.to(dtype=torch.float32)) + 1e-4

    def _score_slot_reference_mass(
        self,
        candidate_scores: torch.Tensor,
        available_slots: int,
    ) -> float:
        if candidate_scores.numel() == 0 or available_slots <= 0:
            return 0.0
        topk = torch.topk(
            candidate_scores,
            k=min(int(available_slots), int(candidate_scores.numel())),
            largest=True,
        ).values
        if topk.numel() == 0:
            return 0.0
        return float(topk.mean().item())

    def _initial_core_budget(self, candidate_count: int, available_slots: int) -> int:
        if available_slots <= 0:
            return 0
        proposed = int(round(self.protected_core_ratio * available_slots))
        proposed = max(0, min(proposed, available_slots, candidate_count))
        minimum_core = min(self.min_protected_core_tokens, candidate_count, max(0, available_slots - 1))
        proposed = max(proposed, minimum_core)
        if candidate_count > proposed and proposed == available_slots:
            proposed = max(minimum_core, available_slots - 1)
        return proposed

    def _apply_dynamic_core_promotion(
        self,
        *,
        candidate_indices: Sequence[int],
        core_indices: Sequence[int],
        score_map: Dict[int, float],
        available_slots: int,
        enabled: bool,
    ) -> List[int]:
        if not enabled or self.promotion_score_ratio is None or not core_indices:
            return sorted(int(idx) for idx in core_indices)
        core_scores = [float(score_map.get(int(idx), 0.0)) for idx in core_indices]
        if not core_scores:
            return sorted(int(idx) for idx in core_indices)
        threshold = min(core_scores) * float(self.promotion_score_ratio)
        promoted = {int(idx) for idx in core_indices}
        eligible = [
            int(idx)
            for idx in candidate_indices
            if int(idx) not in promoted and float(score_map.get(int(idx), 0.0)) >= threshold
        ]
        eligible.sort(key=lambda idx: (float(score_map.get(idx, 0.0)), idx), reverse=True)
        promotion_budget = max(0, int(available_slots) - len(promoted) - 1)
        if promotion_budget > 0:
            promoted.update(eligible[:promotion_budget])
        return sorted(promoted)

    def _try_candidate_plan(
        self,
        *,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        candidate_indices: List[int],
        candidate_head_matrix: torch.Tensor,
        candidate_scores: torch.Tensor,
        available_slots: int,
        core_count: int,
        score_map: Dict[int, float],
        allow_promotion: bool,
        allow_mass_gate: bool,
    ) -> Optional[List[_ScratchDescriptor]]:
        core_indices: List[int] = []
        core_set: set[int] = set()
        base_core_indices: List[int] = []
        if core_count > 0:
            base_core_indices = self._select_protected_core_indices(
                candidate_indices=candidate_indices,
                candidate_head_matrix=candidate_head_matrix,
                candidate_scores=candidate_scores,
                core_count=core_count,
            )
            core_indices = self._apply_dynamic_core_promotion(
                candidate_indices=candidate_indices,
                core_indices=base_core_indices,
                score_map=score_map,
                available_slots=available_slots,
                enabled=allow_promotion,
            )
            core_set = set(core_indices)

        scratch_indices = [idx for idx in candidate_indices if idx not in core_set]
        scratch_slots = available_slots - len(core_indices)
        if scratch_slots < 0:
            return None
        slot_reference_mass = self._score_slot_reference_mass(candidate_scores, available_slots)
        score_mass_gate_relaxed = not allow_mass_gate
        if len(scratch_indices) <= scratch_slots:
            descriptors = [
                self._single_descriptor(
                    idx,
                    protected_core=(idx in core_set),
                    representative_score=score_map.get(idx),
                    score_mass=score_map.get(idx),
                )
                for idx in candidate_indices
            ]
            self._last_selection_dump_payload = {
                "plan_strategy": "normal_no_merge",
                "heuristic_attempt": "strict" if (allow_promotion or allow_mass_gate) else "fallback_relaxed",
                "core_count": int(len(core_indices)),
                "core_indices": [int(idx) for idx in core_indices],
                "promoted_core_indices": [int(idx) for idx in core_indices if idx not in set(base_core_indices)],
                "scratch_slots": int(scratch_slots),
                "phase_boundary_relaxed": False,
                "score_mass_gate_relaxed": bool(score_mass_gate_relaxed),
                "merge_trace": [],
                "candidate_descriptors": [self._descriptor_to_dump(item) for item in descriptors],
            }
            return descriptors

        rep_map, value_map = self._build_representation_maps(pkv_tuple, scratch_indices)
        phase_constrained = bool(self.use_phase_markers)
        phase_boundary_relaxed = False
        merge_trace: List[Dict[str, object]] = []
        items, runs = self._partition_candidate_runs(
            candidate_indices=candidate_indices,
            core_set=core_set,
            rep_map=rep_map,
            value_map=value_map,
            score_map=score_map,
            respect_phase_boundaries=phase_constrained,
        )
        current_scratch_descriptors = sum(len(run) for run in runs)
        while current_scratch_descriptors > scratch_slots:
            pair = self._choose_best_merge_pair(
                runs,
                slot_reference_mass=slot_reference_mass,
                allow_score_mass_gate=not score_mass_gate_relaxed,
            )
            if pair is None:
                if phase_constrained:
                    phase_constrained = False
                    phase_boundary_relaxed = True
                    items, runs = self._partition_candidate_runs(
                        candidate_indices=candidate_indices,
                        core_set=core_set,
                        rep_map=rep_map,
                        value_map=value_map,
                        score_map=score_map,
                        respect_phase_boundaries=False,
                    )
                    current_scratch_descriptors = sum(len(run) for run in runs)
                    continue
                if not score_mass_gate_relaxed and self.merge_score_mass_ratio_threshold is not None:
                    score_mass_gate_relaxed = True
                    continue
                return None
            run_idx, pair_idx = pair
            run = runs[run_idx]
            left = run[pair_idx]
            right = run[pair_idx + 1]
            merge_trace.append(
                self._merge_trace_entry(
                    left=left,
                    right=right,
                    phase_boundaries_active=phase_constrained,
                )
            )
            merged = self._merge_descriptors(left, right)
            run[pair_idx : pair_idx + 2] = [merged]
            current_scratch_descriptors -= 1

        output: List[_ScratchDescriptor] = []
        for kind, payload in items:
            if kind == "core":
                output.append(
                    self._single_descriptor(
                        int(payload),
                        protected_core=True,
                        representative_score=score_map.get(int(payload)),
                        score_mass=score_map.get(int(payload)),
                    )
                )
                continue
            output.extend(runs[int(payload)])
        self._last_selection_dump_payload = {
            "plan_strategy": "normal_merge",
            "heuristic_attempt": "strict" if (allow_promotion or allow_mass_gate) else "fallback_relaxed",
            "core_count": int(len(core_indices)),
            "core_indices": [int(idx) for idx in core_indices],
            "promoted_core_indices": [int(idx) for idx in core_indices if idx not in set(base_core_indices)],
            "scratch_slots": int(scratch_slots),
            "phase_boundary_relaxed": bool(phase_boundary_relaxed),
            "score_mass_gate_relaxed": bool(score_mass_gate_relaxed),
            "merge_trace": merge_trace,
            "candidate_descriptors": [self._descriptor_to_dump(item) for item in output],
        }
        return output

    def _partition_candidate_runs(
        self,
        *,
        candidate_indices: Sequence[int],
        core_set: set[int],
        rep_map: Dict[int, torch.Tensor],
        value_map: Dict[int, torch.Tensor],
        score_map: Dict[int, float],
        respect_phase_boundaries: bool,
    ) -> Tuple[List[Tuple[str, object]], List[List[_ScratchDescriptor]]]:
        items: List[Tuple[str, object]] = []
        runs: List[List[_ScratchDescriptor]] = []
        current_run: List[int] = []
        current_phase_id: Optional[int] = None

        for idx in candidate_indices:
            if idx in core_set:
                if current_run:
                    runs.append(self._build_run_descriptors(current_run, rep_map, value_map, score_map))
                    items.append(("run", len(runs) - 1))
                    current_run = []
                    current_phase_id = None
                items.append(("core", idx))
                continue
            idx_phase = self.phase_ids[idx] if idx < len(self.phase_ids) else 0
            if (
                respect_phase_boundaries
                and current_run
                and current_phase_id is not None
                and idx_phase != current_phase_id
            ):
                runs.append(self._build_run_descriptors(current_run, rep_map, value_map, score_map))
                items.append(("run", len(runs) - 1))
                current_run = []
            current_run.append(idx)
            current_phase_id = idx_phase

        if current_run:
            runs.append(self._build_run_descriptors(current_run, rep_map, value_map, score_map))
            items.append(("run", len(runs) - 1))
        return items, runs

    def _select_protected_core_indices(
        self,
        *,
        candidate_indices: Sequence[int],
        candidate_head_matrix: torch.Tensor,
        candidate_scores: torch.Tensor,
        core_count: int,
    ) -> List[int]:
        if core_count <= 0:
            return []
        candidate_count = len(candidate_indices)
        if candidate_count == 0:
            return []

        if self.core_selection_mode == "score" or candidate_head_matrix.numel() == 0:
            topk = torch.topk(candidate_scores, k=core_count, largest=True).indices.tolist()
            return sorted(candidate_indices[offset] for offset in topk)

        per_head_quota = min(core_count, candidate_count)
        vote_counts = torch.zeros(candidate_count, device=candidate_scores.device, dtype=torch.int32)
        for head_scores in candidate_head_matrix:
            head_k = min(per_head_quota, head_scores.numel())
            if head_k <= 0:
                continue
            top_idx = torch.topk(head_scores, k=head_k, largest=True).indices
            vote_counts.scatter_add_(
                0,
                top_idx,
                torch.ones_like(top_idx, dtype=vote_counts.dtype),
            )

        ranked_offsets = list(range(candidate_count))
        ranked_offsets.sort(
            key=lambda offset: (
                int(vote_counts[offset].item()),
                float(candidate_scores[offset].item()),
                -int(candidate_indices[offset]),
            ),
            reverse=True,
        )
        return sorted(candidate_indices[offset] for offset in ranked_offsets[:core_count])

    def _build_representation_maps(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        indices: Sequence[int],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        if not indices:
            return {}, {}

        index_tensor = torch.tensor(indices, device=self.config.device, dtype=torch.long)
        positions = torch.tensor(
            [self.cache_positions[idx] for idx in indices],
            device=self.config.device,
            dtype=torch.long,
        )
        cos, sin = self._rotary_terms(positions, dtype=torch.float32)
        accum = torch.zeros(
            len(indices),
            self.head_dim,
            device=self.config.device,
            dtype=torch.float32,
        )
        value_accum: Optional[torch.Tensor] = None
        if self.value_projection_threshold is not None and self.output_projection_slices:
            first_projection = next(iter(self.output_projection_slices.values()))
            value_accum = torch.zeros(
                len(indices),
                first_projection.shape[0],
                device=self.config.device,
                dtype=torch.float32,
            )
        used_heads = 0
        used_value_heads = 0

        for layer_idx, attn_head in self.sampled_heads:
            key_states, value_states = pkv_tuple[layer_idx]
            kv_head = attn_head
            if self.num_key_value_heads and self.num_attention_heads:
                kv_head = min(key_states.shape[1] - 1, attn_head // max(1, self.num_key_value_groups))
            k_rot = key_states[0, kv_head].index_select(0, index_tensor).to(dtype=torch.float32)
            k_prerope = invert_rope(
                k_rot,
                cos,
                sin,
                self.attention_scale,
                style=self.rope_style,
            )
            accum += F.normalize(k_prerope, dim=-1, eps=1e-6)
            used_heads += 1
            proj_slice = self.output_projection_slices.get((layer_idx, attn_head))
            if value_accum is not None and proj_slice is not None:
                value_head = value_states[0, kv_head].index_select(0, index_tensor).to(dtype=torch.float32)
                value_accum += F.linear(value_head, proj_slice)
                used_value_heads += 1

        if used_heads == 0:
            key_map = {idx: accum[offset] for offset, idx in enumerate(indices)}
        else:
            accum = accum / float(used_heads)
            key_map = {idx: accum[offset] for offset, idx in enumerate(indices)}

        value_map: Dict[int, torch.Tensor] = {}
        if value_accum is not None:
            if used_value_heads > 0:
                value_accum = value_accum / float(used_value_heads)
            value_map = {idx: value_accum[offset] for offset, idx in enumerate(indices)}
        return key_map, value_map

    def _build_run_descriptors(
        self,
        run_indices: Sequence[int],
        rep_map: Dict[int, torch.Tensor],
        value_map: Dict[int, torch.Tensor],
        score_map: Dict[int, float],
    ) -> List[_ScratchDescriptor]:
        return [
            _ScratchDescriptor(
                source_indices=[idx],
                representative_position=self.cache_positions[idx],
                representative_source_index=idx,
                representative_score=float(score_map.get(idx, self.merge_supports[idx])),
                score_mass=float(score_map.get(idx, 0.0)),
                count=self.merge_counts[idx],
                span=self.merge_spans[idx],
                phase_id=self.phase_ids[idx] if idx < len(self.phase_ids) else 0,
                protected_core=False,
                support_weight=float(self.merge_supports[idx]),
                rep_vector=rep_map.get(idx),
                value_vector=value_map.get(idx),
            )
            for idx in run_indices
        ]

    def _choose_best_merge_pair(
        self,
        runs: Sequence[Sequence[_ScratchDescriptor]],
        *,
        slot_reference_mass: float,
        allow_score_mass_gate: bool,
    ) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[int, float]] = None

        for run_idx, run in enumerate(runs):
            if len(run) < 2:
                continue
            for pair_idx in range(len(run) - 1):
                left = run[pair_idx]
                right = run[pair_idx + 1]
                if not self._projection_gate_ok(left, right):
                    continue
                if (
                    allow_score_mass_gate
                    and not self._score_mass_gate_ok(left, right, slot_reference_mass=slot_reference_mass)
                ):
                    continue
                gap = max(0, right.span[0] - left.span[1])
                local_ok = gap <= self.merge_local_window
                similarity = self._descriptor_similarity(left, right)
                tier = 2 if local_ok and similarity >= self.merge_similarity_threshold else 1 if local_ok else 0
                key = (tier, float(similarity))
                if best_key is None or key > best_key:
                    best_key = key
                    best = (run_idx, pair_idx)
        return best

    def _descriptor_similarity(
        self,
        left: _ScratchDescriptor,
        right: _ScratchDescriptor,
    ) -> float:
        if left.rep_vector is None or right.rep_vector is None:
            return -1.0
        lhs = F.normalize(left.rep_vector.unsqueeze(0), dim=-1, eps=1e-6)
        rhs = F.normalize(right.rep_vector.unsqueeze(0), dim=-1, eps=1e-6)
        return float((lhs * rhs).sum().item())

    def _projection_gate_ok(
        self,
        left: _ScratchDescriptor,
        right: _ScratchDescriptor,
    ) -> bool:
        if self.value_projection_threshold is None:
            return True
        if left.value_vector is None or right.value_vector is None:
            return True
        distance = torch.linalg.vector_norm(left.value_vector - right.value_vector, ord=2).item()
        return distance <= self.value_projection_threshold

    def _score_mass_gate_ok(
        self,
        left: _ScratchDescriptor,
        right: _ScratchDescriptor,
        *,
        slot_reference_mass: float,
    ) -> bool:
        if self.merge_score_mass_ratio_threshold is None:
            return True
        if slot_reference_mass <= 0.0:
            return True
        combined = float(left.score_mass) + float(right.score_mass)
        return combined <= float(self.merge_score_mass_ratio_threshold) * float(slot_reference_mass)

    def _projection_distance(
        self,
        left: _ScratchDescriptor,
        right: _ScratchDescriptor,
    ) -> float | None:
        if left.value_vector is None or right.value_vector is None:
            return None
        return float(torch.linalg.vector_norm(left.value_vector - right.value_vector, ord=2).item())

    def _descriptor_to_dump(self, descriptor: _ScratchDescriptor) -> Dict[str, object]:
        return {
            "source_indices": [int(idx) for idx in descriptor.source_indices],
            "representative_position": int(descriptor.representative_position),
            "representative_source_index": int(descriptor.representative_source_index),
            "representative_score": float(descriptor.representative_score),
            "score_mass": float(descriptor.score_mass),
            "count": int(descriptor.count),
            "span": [int(descriptor.span[0]), int(descriptor.span[1])],
            "phase_id": int(descriptor.phase_id),
            "protected_core": bool(descriptor.protected_core),
            "support_weight": float(descriptor.support_weight),
        }

    def _merge_trace_entry(
        self,
        *,
        left: _ScratchDescriptor,
        right: _ScratchDescriptor,
        phase_boundaries_active: bool,
    ) -> Dict[str, object]:
        gap = max(0, right.span[0] - left.span[1])
        similarity = self._descriptor_similarity(left, right)
        projection_distance = self._projection_distance(left, right)
        combined_score_mass = float(left.score_mass) + float(right.score_mass)
        local_ok = gap <= self.merge_local_window
        tier = 2 if local_ok and similarity >= self.merge_similarity_threshold else 1 if local_ok else 0
        return {
            "left": self._descriptor_to_dump(left),
            "right": self._descriptor_to_dump(right),
            "gap": int(gap),
            "similarity": float(similarity),
            "projection_distance": projection_distance,
            "combined_score_mass": combined_score_mass,
            "local_ok": bool(local_ok),
            "tier": int(tier),
            "phase_boundaries_active": bool(phase_boundaries_active),
        }

    def _dump_scratch_selection_event(
        self,
        *,
        candidate_indices: Sequence[int],
        candidate_head_matrix: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_supports: torch.Tensor,
        kv_cache_len: int,
        prefix_length: int,
        recent_tokens: int,
        available_slots: int,
        protected_core_tokens: int,
        scratch_descriptors: Sequence[_ScratchDescriptor],
        scratch_source_tokens: int,
        scratch_saved_tokens: int,
    ) -> None:
        if self.score_dump_dir is None:
            return
        if self.score_dump_max_events is not None and self._score_dump_event_index >= int(self.score_dump_max_events):
            return

        payload = self._last_selection_dump_payload or {}
        context = dict(self._score_dump_context)
        shard_id = int(context.get("shard_id", -1))
        run_id = int(context.get("run_id", -1))
        sample_idx = int(context.get("sample_idx", -1))
        event_index = self._score_dump_event_index
        file_name = (
            f"shard{shard_id:02d}_run{run_id:03d}_sample{sample_idx:05d}_"
            f"event{event_index:04d}.pt"
        )
        dump_payload: Dict[str, Any] = {
            "metadata": {
                "dump_kind": "cask_selection",
                "event_index": event_index,
                "absolute_position": int(self.absolute_position),
                "budget": int(self.budget),
                "kv_cache_len": int(kv_cache_len),
                "prefix_length": int(prefix_length),
                "candidate_count": int(len(candidate_indices)),
                "recent_tokens": int(recent_tokens),
                "available_slots": int(available_slots),
                "protected_core_ratio": float(self.protected_core_ratio),
                "core_selection_mode": self.core_selection_mode,
                "merge_operator": self.merge_operator,
                "merge_local_window": int(self.merge_local_window),
                "merge_similarity_threshold": float(self.merge_similarity_threshold),
                "value_projection_threshold": self.value_projection_threshold,
                "representative_mode": self.representative_mode,
                "promotion_score_ratio": self.promotion_score_ratio,
                "merge_score_mass_ratio_threshold": self.merge_score_mass_ratio_threshold,
                "min_protected_core_tokens": int(self.min_protected_core_tokens),
                "protected_core_tokens": int(protected_core_tokens),
                "scratch_descriptor_count": int(len(scratch_descriptors)),
                "scratch_source_tokens": int(scratch_source_tokens),
                "scratch_saved_tokens": int(scratch_saved_tokens),
                "plan_strategy": str(payload.get("plan_strategy", "unknown")),
                "heuristic_attempt": str(payload.get("heuristic_attempt", "unknown")),
                "phase_boundary_relaxed": bool(payload.get("phase_boundary_relaxed", False)),
                "score_mass_gate_relaxed": bool(payload.get("score_mass_gate_relaxed", False)),
                "core_count": int(payload.get("core_count", protected_core_tokens)),
                "promoted_core_indices": list(payload.get("promoted_core_indices", [])),
                "scratch_slots": int(payload.get("scratch_slots", max(0, available_slots - protected_core_tokens))),
                **context,
            },
            "tensors": {
                "candidate_indices": torch.tensor(candidate_indices, dtype=torch.long),
                "candidate_scores": candidate_scores.detach().to("cpu"),
                "candidate_supports": candidate_supports.detach().to("cpu"),
                "candidate_head_matrix": candidate_head_matrix.detach().to("cpu"),
                "protected_core_indices": torch.tensor(
                    payload.get("core_indices", []),
                    dtype=torch.long,
                ),
            },
            "scratch_descriptors": [self._descriptor_to_dump(item) for item in scratch_descriptors],
            "candidate_descriptors": list(payload.get("candidate_descriptors", [])),
            "merge_trace": list(payload.get("merge_trace", [])),
        }
        torch.save(dump_payload, self.score_dump_dir / file_name)
        self._score_dump_event_index += 1

    def _merge_descriptors(
        self,
        left: _ScratchDescriptor,
        right: _ScratchDescriptor,
    ) -> _ScratchDescriptor:
        total = max(1, left.count + right.count)
        left_mass = float(left.count) * max(float(left.support_weight), 1e-4)
        right_mass = float(right.count) * max(float(right.support_weight), 1e-4)
        total_mass = max(left_mass + right_mass, 1e-4)
        left_is_anchor = (
            left.representative_score > right.representative_score
            or (
                left.representative_score == right.representative_score
                and left.representative_position >= right.representative_position
            )
        )
        anchor = left if left_is_anchor else right
        if left.rep_vector is None:
            merged_vec = right.rep_vector
        elif right.rep_vector is None:
            merged_vec = left.rep_vector
        else:
            if self.representative_mode == "score_max_source":
                merged_vec = anchor.rep_vector
            elif self.merge_operator == "keepkv":
                merged_vec = (left.rep_vector * left_mass + right.rep_vector * right_mass) / total_mass
            else:
                merged_vec = (
                    left.rep_vector * float(left.count) + right.rep_vector * float(right.count)
                ) / float(total)
        if left.value_vector is None:
            merged_value = right.value_vector
        elif right.value_vector is None:
            merged_value = left.value_vector
        else:
            if self.merge_operator == "keepkv":
                merged_value = (left.value_vector * left_mass + right.value_vector * right_mass) / total_mass
            else:
                merged_value = (
                    left.value_vector * float(left.count) + right.value_vector * float(right.count)
                ) / float(total)

        return _ScratchDescriptor(
            source_indices=left.source_indices + right.source_indices,
            representative_position=(
                anchor.representative_position
                if self.representative_mode == "score_max_source"
                else max(left.representative_position, right.representative_position)
            ),
            representative_source_index=anchor.representative_source_index,
            representative_score=float(anchor.representative_score),
            score_mass=float(left.score_mass + right.score_mass),
            count=left.count + right.count,
            span=(min(left.span[0], right.span[0]), max(left.span[1], right.span[1])),
            phase_id=left.phase_id,
            protected_core=False,
            support_weight=(left_mass + right_mass) / float(total),
            rep_vector=merged_vec,
            value_vector=merged_value,
        )

    def _single_descriptor(
        self,
        index: int,
        *,
        protected_core: bool = False,
        representative_score: float | None = None,
        score_mass: float | None = None,
    ) -> _ScratchDescriptor:
        return _ScratchDescriptor(
            source_indices=[index],
            representative_position=self.cache_positions[index],
            representative_source_index=index,
            representative_score=float(
                self.merge_supports[index] if representative_score is None else representative_score
            ),
            score_mass=float(
                self.merge_supports[index] if score_mass is None else score_mass
            ),
            count=self.merge_counts[index],
            span=self.merge_spans[index],
            phase_id=self.phase_ids[index] if index < len(self.phase_ids) else 0,
            protected_core=protected_core,
            support_weight=float(self.merge_supports[index]),
            rep_vector=None,
            value_vector=None,
        )

    def _gather_existing_slots(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        keep_indices: Sequence[int],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        keep_tensor = torch.tensor(keep_indices, device=self.config.device, dtype=torch.long)
        new_pkv = []
        for key_states, value_states in pkv_tuple:
            k_new = key_states.index_select(2, keep_tensor)
            v_new = value_states.index_select(2, keep_tensor)
            new_pkv.append((k_new.contiguous(), v_new.contiguous()))
        self.cache_positions = [self.cache_positions[idx] for idx in keep_indices]
        self.merge_counts = [self.merge_counts[idx] for idx in keep_indices]
        self.merge_supports = [self.merge_supports[idx] for idx in keep_indices]
        self.merge_spans = [self.merge_spans[idx] for idx in keep_indices]
        self.phase_ids = [self.phase_ids[idx] for idx in keep_indices]
        self.prefix_length = min(self.prefix_length, len(self.cache_positions))
        return tuple(new_pkv)

    def _materialize_descriptors(
        self,
        pkv_tuple: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        descriptors: Sequence[_ScratchDescriptor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        new_pkv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for key_states, value_states in pkv_tuple:
            layer_keys: List[torch.Tensor] = []
            layer_values: List[torch.Tensor] = []
            for descriptor in descriptors:
                k_desc, v_desc = self._materialize_descriptor_for_layer(
                    key_states,
                    value_states,
                    descriptor,
                )
                layer_keys.append(k_desc)
                layer_values.append(v_desc)
            new_layer_keys = torch.cat(layer_keys, dim=2)
            new_layer_values = torch.cat(layer_values, dim=2)
            new_pkv.append((new_layer_keys.contiguous(), new_layer_values.contiguous()))

        self.cache_positions = [descriptor.representative_position for descriptor in descriptors]
        self.merge_counts = [descriptor.count for descriptor in descriptors]
        self.merge_supports = [float(descriptor.support_weight) for descriptor in descriptors]
        self.merge_spans = [descriptor.span for descriptor in descriptors]
        self.phase_ids = [descriptor.phase_id for descriptor in descriptors]
        self.prefix_length = min(self.prefix_length, len(self.cache_positions))
        return tuple(new_pkv)

    def _materialize_descriptor_for_layer(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        descriptor: _ScratchDescriptor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_indices = descriptor.source_indices
        if len(source_indices) == 1:
            source_idx = int(source_indices[0])
            return (
                key_states[:, :, source_idx : source_idx + 1, :].contiguous(),
                value_states[:, :, source_idx : source_idx + 1, :].contiguous(),
            )

        source_tensor = torch.tensor(source_indices, device=key_states.device, dtype=torch.long)
        source_positions = torch.tensor(
            [self.cache_positions[idx] for idx in source_indices],
            device=key_states.device,
            dtype=torch.long,
        )
        source_weights = torch.tensor(
            [
                float(self.merge_counts[idx]) * max(float(self.merge_supports[idx]), 1e-4)
                if self.merge_operator == "keepkv"
                else float(self.merge_counts[idx])
                for idx in source_indices
            ],
            device=key_states.device,
            dtype=torch.float32,
        )
        source_weights = source_weights / source_weights.sum().clamp_min(1.0)

        values = value_states[0].index_select(1, source_tensor).to(dtype=torch.float32)
        if self.merge_operator == "keepkv":
            merged_value = (values * source_weights.view(1, -1, 1)).sum(dim=1).to(dtype=value_states.dtype)
        else:
            merged_value = values.sum(dim=1).to(dtype=value_states.dtype)

        if self.representative_mode == "score_max_source":
            rep_idx = int(descriptor.representative_source_index)
            merged_rot = key_states[:, :, rep_idx : rep_idx + 1, :].contiguous()
            return (
                merged_rot,
                merged_value.unsqueeze(0).unsqueeze(2),
            )

        keys = key_states[0].index_select(1, source_tensor).to(dtype=torch.float32)
        cos, sin = self._rotary_terms(source_positions, dtype=torch.float32, device=key_states.device)
        keys_prerope = invert_rope(
            keys,
            cos.unsqueeze(0),
            sin.unsqueeze(0),
            self.attention_scale,
            style=self.rope_style,
        )
        merged_prerope = (keys_prerope * source_weights.view(1, -1, 1)).sum(dim=1)

        rep_pos = torch.tensor([descriptor.representative_position], device=key_states.device, dtype=torch.long)
        rep_cos, rep_sin = self._rotary_terms(rep_pos, dtype=torch.float32, device=key_states.device)
        merged_rot = self._apply_rope(
            merged_prerope,
            rep_cos.squeeze(0),
            rep_sin.squeeze(0),
        ).to(dtype=key_states.dtype)

        return (
            merged_rot.unsqueeze(0).unsqueeze(2),
            merged_value.unsqueeze(0).unsqueeze(2),
        )

    def _rotary_terms(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_device = device or self.config.device
        position_ids = positions.to(device=target_device, dtype=torch.long).unsqueeze(0)
        probe = torch.zeros(
            1,
            position_ids.shape[1],
            self.head_dim,
            device=target_device,
            dtype=self.config.dtype,
        )
        if self.rotary.inv_freq.device != target_device:
            self.rotary.to(device=target_device)
        cos, sin = self.rotary(probe, position_ids)
        return cos[0].to(dtype=dtype), sin[0].to(dtype=dtype)

    def _apply_rope(
        self,
        base: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return base * cos + rotate_half(base, style=self.rope_style) * sin


def apply_cask_patch(
    model,
    *,
    stats_path: Path,
    model_path: Path,
    kv_budget: int,
    offset_max_length: int = 65536,
    score_aggregation: str = "mean",
    pruning_seed: int = 0,
    metadata_expectations: Dict[str, object] | None = None,
    normalize_scores: bool = False,
    count_prompt_tokens: bool = False,
    allow_prefill_compression: bool = False,
    divide_length: int = 128,
    use_slack_trigger: bool = False,
    disable_mlr: bool = False,
    disable_trig: bool = False,
    prefix_coverage_ratio: float = 0.0625,
    recent_window_size: int = 128,
    protected_core_ratio: float = 0.5,
    min_protected_core_tokens: int = 1,
    core_selection_mode: str = "vote",
    merge_operator: str = "keepkv",
    merge_local_window: int = 32,
    merge_similarity_threshold: float = 0.985,
    value_projection_threshold: float | None = None,
    representative_mode: str = "score_max_source",
    promotion_score_ratio: float | None = None,
    merge_score_mass_ratio_threshold: float | None = None,
    use_phase_markers: bool = True,
    phase_marker_token_ids: Optional[Tuple[Tuple[int, ...], ...]] = None,
    score_dump_dir: Path | None = None,
    score_dump_max_events: int | None = None,
) -> None:
    """Apply CASK on top of the TriAttention scoring path."""
    device = next(model.parameters()).device
    dtype = torch.float32

    config = CASKConfig(
        stats_path=stats_path,
        model_path=model_path,
        device=device,
        dtype=dtype,
        budget=kv_budget,
        offset_max_length=offset_max_length,
        score_aggregation=score_aggregation,
        seed=pruning_seed,
        metadata_expectations=metadata_expectations,
        normalize_scores=normalize_scores,
        count_prompt_tokens=count_prompt_tokens,
        allow_prefill_compression=allow_prefill_compression,
        divide_length=divide_length,
        use_slack_trigger=use_slack_trigger,
        per_head_pruning=False,
        per_layer_perhead_pruning=False,
        disable_mlr=disable_mlr,
        disable_trig=disable_trig,
        horizon_mode="fixed",
        norm_mode="tri",
        prefix_coverage_ratio=prefix_coverage_ratio,
        recent_window_size=recent_window_size,
        protected_core_ratio=protected_core_ratio,
        min_protected_core_tokens=min_protected_core_tokens,
        core_selection_mode=core_selection_mode,
        merge_operator=merge_operator,
        merge_local_window=merge_local_window,
        merge_similarity_threshold=merge_similarity_threshold,
        value_projection_threshold=value_projection_threshold,
        representative_mode=representative_mode,
        promotion_score_ratio=promotion_score_ratio,
        merge_score_mass_ratio_threshold=merge_score_mass_ratio_threshold,
        use_phase_markers=use_phase_markers,
        phase_marker_token_ids=phase_marker_token_ids,
        score_dump_dir=score_dump_dir,
        score_dump_max_events=score_dump_max_events,
    )

    compressor = CASK(config)
    compressor.attach_runtime_assets(model, phase_marker_token_ids=phase_marker_token_ids)

    model_rotary_emb = None
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            if len(layers) > 0 and hasattr(layers[0], "self_attn"):
                attn = layers[0].self_attn
                if hasattr(attn, "rotary_emb"):
                    model_rotary_emb = attn.rotary_emb
    except Exception:
        pass

    if model_rotary_emb is not None:
        verify_rotary_alignment(compressor.rotary, model_rotary_emb)
    else:
        print("[CASK] WARNING: Could not locate model rotary_emb for alignment verification.")

    model._cask_compressor = compressor

    orig_forward = model.forward

    def cask_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        comp = self._cask_compressor
        cache_position_override = cache_position
        position_ids_override = position_ids
        attention_mask_override = attention_mask

        is_empty_cache = True
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                if past_key_values.get_seq_length() > 0:
                    is_empty_cache = False
            elif isinstance(past_key_values, (tuple, list)):
                if len(past_key_values) > 0 and past_key_values[0][0].shape[2] > 0:
                    is_empty_cache = False

        if is_empty_cache:
            comp.reset_compression_state()

        if past_key_values is not None and input_ids is not None and not is_empty_cache:
            bsz, step = input_ids.shape
            start_pos = comp.absolute_position
            abs_positions = torch.arange(
                start_pos,
                start_pos + step,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            if bsz > 1:
                abs_positions = abs_positions.expand(bsz, -1)
            position_ids_override = abs_positions

            current_cache_len = None
            if isinstance(past_key_values, Cache) and hasattr(past_key_values, "get_seq_length"):
                current_cache_len = int(past_key_values.get_seq_length())
            elif isinstance(past_key_values, (tuple, list)) and past_key_values:
                current_cache_len = int(past_key_values[0][0].shape[2])

            if current_cache_len is not None:
                cache_position_override = torch.arange(
                    current_cache_len,
                    current_cache_len + step,
                    device=input_ids.device,
                    dtype=torch.long,
                )
            attention_mask_override = None
        else:
            cache_position_override = None

        outputs = orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask_override,
            position_ids=position_ids_override,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position_override,
            **kwargs,
        )

        if getattr(outputs, "past_key_values", None) is None:
            return outputs

        pkv = outputs.past_key_values
        pkv_tuple = _cache_to_legacy_tuple(pkv)
        if not pkv_tuple:
            return outputs

        seq_len = pkv_tuple[0][0].shape[2]
        cached_len = len(comp.cache_positions)
        is_decode_step = False

        if cached_len == 0:
            comp.cache_positions = list(range(seq_len))
            comp.merge_counts = [1] * seq_len
            comp.merge_supports = [1.0] * seq_len
            comp.merge_spans = [(pos, pos) for pos in comp.cache_positions]
            comp.phase_ids = [-1] * seq_len
            comp.current_phase_id = 0
            comp.decode_token_history = []
            comp.absolute_position = seq_len
            comp.prefix_length = seq_len
        elif cached_len < seq_len:
            is_decode_step = True
            added = seq_len - cached_len
            new_positions = list(range(comp.absolute_position, comp.absolute_position + added))
            new_token_ids = input_ids[0, -added:].detach().to("cpu").tolist()
            comp.append_decode_tokens(new_token_ids, new_positions)
            comp.absolute_position += added

        effective_size = seq_len
        if not comp.config.count_prompt_tokens:
            effective_size = max(0, seq_len - comp.prefix_length)

        if comp.use_slack_trigger:
            trigger_threshold = comp.budget + comp.divide_length
            should_compress = is_decode_step and effective_size >= trigger_threshold
        else:
            trigger_threshold = comp.budget
            should_compress = (
                is_decode_step
                and effective_size >= trigger_threshold
                and (comp.absolute_position % comp.divide_length == 0)
            )

        if should_compress:
            pkv_tuple = comp.compress_pkv(pkv_tuple, prefix_length=comp.prefix_length)

        new_cache = _legacy_tuple_to_cache(pkv_tuple, outputs.past_key_values)
        return CausalLMOutputWithPast(
            loss=getattr(outputs, "loss", None),
            logits=outputs.logits,
            past_key_values=new_cache,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    model.forward = MethodType(cask_forward, model)
    print(
        f"[CASK] Applied two-stage compression (budget={kv_budget}, "
        f"prefix_coverage_ratio={prefix_coverage_ratio}, recent_window={recent_window_size}, core_ratio={protected_core_ratio}, "
        f"core_selection_mode={core_selection_mode}, "
        f"merge_operator={merge_operator}, "
        f"merge_local_window={merge_local_window}, similarity_threshold={merge_similarity_threshold}, "
        f"value_projection_threshold={value_projection_threshold}, representative_mode={representative_mode}, "
        f"promotion_score_ratio={promotion_score_ratio}, "
        f"merge_score_mass_ratio_threshold={merge_score_mass_ratio_threshold}, "
        f"use_phase_markers={use_phase_markers})"
    )
