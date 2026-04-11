#!/usr/bin/env python3
"""Calibrate frequency-domain statistics for TriAttention.

Runs a single forward pass on plain text input, hooks into every attention
layer to capture query states, inverts RoPE, and computes per-head frequency
statistics.  The resulting .pt file can be loaded directly by
``cask.pruning_utils.load_head_frequency_stats``.

Usage
-----
    python scripts/calibrate.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --input calibration_text.txt \
        --output calibration/qwen7b_stats.pt \
        --max-length 32768 \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers imported from cask (kept local to make script self-contained)
# ---------------------------------------------------------------------------

def _determine_rope_style(config: AutoConfig) -> str:
    model_type = getattr(config, "model_type", "")
    if "llama" in model_type:
        return "half"
    return "half"


def _rotate_half(x: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    if style == "interleaved":
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _invert_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
    *,
    style: str = "half",
) -> torch.Tensor:
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    if style == "interleaved":
        even = base[..., ::2]
        odd = base[..., 1::2]
        cos_even = cos_unit[..., ::2]
        cos_odd = cos_unit[..., 1::2]
        sin_even = sin_unit[..., ::2]
        sin_odd = sin_unit[..., 1::2]
        det = cos_even * cos_odd + sin_even * sin_odd
        det = det.clamp_min(1e-12)
        orig_even = (even * cos_odd + odd * sin_even) / det
        orig_odd = (odd * cos_even - even * sin_odd) / det
        restored = torch.empty_like(base)
        restored[..., ::2] = orig_even
        restored[..., 1::2] = orig_odd
        return restored
    return base * cos_unit - _rotate_half(base, style=style) * sin_unit


def _to_complex_pairs(tensor: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        real = tensor_real[..., ::2].contiguous()
        imag = tensor_real[..., 1::2].contiguous()
        return torch.complex(real, imag)
    freq_count = tensor.shape[-1] // 2
    real = tensor_real[..., :freq_count].contiguous()
    imag = tensor_real[..., freq_count:].contiguous()
    return torch.complex(real, imag)


def _build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def _compute_effective_weights(
    q_mean_complex: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    q_mean_abs = torch.abs(q_mean_complex).to(dtype=torch.float32)
    return q_mean_abs / q_mean_abs.sum().clamp_min(float(eps))


def _solve_variational_horizon_qp(
    tau_complex: torch.Tensor,
    weights: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    *,
    eps: float = 1e-8,
    maxiter: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    from scipy.optimize import Bounds, LinearConstraint, minimize

    tau_fp32 = tau_complex.to(dtype=torch.complex64)
    weights_fp32 = weights.to(dtype=torch.float32)
    omega_fp32 = omega.to(dtype=torch.float32)
    offsets_fp32 = offsets.to(dtype=torch.float32)

    phases = offsets_fp32.unsqueeze(1) * omega_fp32.unsqueeze(0)
    basis_complex = torch.polar(torch.ones_like(phases), phases)  # [J, F]
    basis_real = basis_complex.real.T.contiguous()
    basis_imag = basis_complex.imag.T.contiguous()
    tau_real = tau_fp32.real.to(dtype=torch.float32)
    tau_imag = tau_fp32.imag.to(dtype=torch.float32)

    weighted_basis_real = weights_fp32.unsqueeze(1) * basis_real
    weighted_basis_imag = weights_fp32.unsqueeze(1) * basis_imag

    hessian = (
        basis_real.T @ weighted_basis_real
        + basis_imag.T @ weighted_basis_imag
    ).cpu().numpy()
    linear = (
        basis_real.T @ (weights_fp32 * tau_real)
        + basis_imag.T @ (weights_fp32 * tau_imag)
    ).cpu().numpy()

    num_offsets = int(offsets.numel())
    x0 = torch.full((num_offsets,), 1.0 / max(1, num_offsets), dtype=torch.float64).cpu().numpy()

    def objective(x):
        return float(x @ hessian @ x - 2.0 * linear @ x)

    def gradient(x):
        return 2.0 * (hessian @ x - linear)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(0.0, float("inf")),
        constraints=[
            LinearConstraint(
                torch.ones(num_offsets, dtype=torch.float64).cpu().numpy(),
                lb=1.0,
                ub=1.0,
            )
        ],
        options={"maxiter": int(maxiter), "ftol": float(eps), "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Variational horizon QP failed: {result.message}")

    pi = torch.from_numpy(result.x).to(device=offsets.device, dtype=torch.float32)
    pi = pi.clamp_min(0.0)
    pi = pi / pi.sum().clamp_min(float(eps))
    kappa_complex = (pi.unsqueeze(1) * basis_complex.to(device=offsets.device)).sum(dim=0)
    return pi, kappa_complex.to(dtype=torch.complex64)


def _sample_query_positions(seq_len: int, sample_count: Optional[int], device: torch.device) -> torch.Tensor:
    if seq_len <= 1:
        return torch.empty(0, device=device, dtype=torch.long)
    all_queries = torch.arange(1, seq_len, device=device, dtype=torch.long)
    if sample_count is None or sample_count <= 0 or sample_count >= all_queries.numel():
        return all_queries
    lin = torch.linspace(0, all_queries.numel() - 1, steps=sample_count, device=device)
    indices = torch.unique(lin.round().to(dtype=torch.long), sorted=True)
    return all_queries.index_select(0, indices)


def _estimate_tau_from_attention(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    omega: torch.Tensor,
    *,
    num_kv_heads: int,
    query_sample_count: Optional[int],
    key_chunk_size: int,
) -> torch.Tensor:
    """Estimate tau_h,f from sampled raw attention using streamed softmax blocks."""
    num_heads, seq_len, head_dim = q_rot.shape
    if seq_len <= 1:
        return torch.zeros(num_heads, omega.numel(), device=q_rot.device, dtype=torch.complex64)

    query_positions = _sample_query_positions(seq_len, query_sample_count, q_rot.device)
    if query_positions.numel() == 0:
        return torch.zeros(num_heads, omega.numel(), device=q_rot.device, dtype=torch.complex64)

    gqa_ratio = max(1, num_heads // max(1, num_kv_heads))
    scale = float(head_dim) ** -0.5
    freq_count = int(omega.numel())
    tau_values: List[torch.Tensor] = []

    full_key_positions = torch.arange(seq_len, device=q_rot.device, dtype=torch.long)
    query_pos_fp32 = query_positions.to(dtype=torch.float32)
    omega_fp32 = omega.to(device=q_rot.device, dtype=torch.float32)

    for head_idx in range(num_heads):
        kv_head = min(num_kv_heads - 1, head_idx // gqa_ratio)
        q_queries = q_rot[head_idx].index_select(0, query_positions).to(dtype=torch.float32)  # [Q, D]
        k_head = k_rot[kv_head].to(dtype=torch.float32)  # [L, D]

        q_count = q_queries.shape[0]
        running_max = torch.full((q_count,), float("-inf"), device=q_rot.device, dtype=torch.float32)
        running_norm = torch.zeros(q_count, device=q_rot.device, dtype=torch.float32)
        running_past_mass = torch.zeros(q_count, device=q_rot.device, dtype=torch.float32)
        running_real = torch.zeros(q_count, freq_count, device=q_rot.device, dtype=torch.float32)
        running_imag = torch.zeros(q_count, freq_count, device=q_rot.device, dtype=torch.float32)

        for start in range(0, seq_len, key_chunk_size):
            end = min(start + key_chunk_size, seq_len)
            key_positions = full_key_positions[start:end]
            logits = torch.matmul(q_queries, k_head[start:end].T) * scale
            allowed_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            logits = logits.masked_fill(~allowed_mask, float("-inf"))

            block_max = logits.max(dim=1).values
            new_max = torch.maximum(running_max, block_max)

            prev_scale = torch.exp(running_max - new_max)
            running_norm = running_norm * prev_scale
            running_past_mass = running_past_mass * prev_scale
            running_real = running_real * prev_scale.unsqueeze(1)
            running_imag = running_imag * prev_scale.unsqueeze(1)

            block_exp = torch.exp(logits - new_max.unsqueeze(1))
            running_norm = running_norm + block_exp.sum(dim=1)

            past_mask = key_positions.unsqueeze(0) < query_positions.unsqueeze(1)
            past_exp = block_exp * past_mask.to(dtype=block_exp.dtype)
            running_past_mass = running_past_mass + past_exp.sum(dim=1)

            if past_mask.any():
                delta = query_pos_fp32.unsqueeze(1) - key_positions.to(dtype=torch.float32).unsqueeze(0)
                delta = delta.masked_fill(~past_mask, 0.0)
                phase = delta.unsqueeze(-1) * omega_fp32.view(1, 1, -1)
                weighted = past_exp.unsqueeze(-1)
                running_real = running_real + (weighted * torch.cos(phase)).sum(dim=1)
                running_imag = running_imag + (weighted * torch.sin(phase)).sum(dim=1)

            running_max = new_max

        valid = running_past_mass > 0
        if not bool(valid.any()):
            tau_values.append(torch.zeros(freq_count, device=q_rot.device, dtype=torch.complex64))
            continue

        normalized_complex = torch.complex(
            running_real[valid] / running_norm[valid].unsqueeze(1).clamp_min(1e-12),
            running_imag[valid] / running_norm[valid].unsqueeze(1).clamp_min(1e-12),
        )
        normalized_mass = running_past_mass[valid] / running_norm[valid].clamp_min(1e-12)
        tau_head = (
            normalized_complex * normalized_mass.unsqueeze(1).to(dtype=normalized_complex.dtype)
        ).sum(dim=0) / normalized_mass.sum().clamp_min(1e-12)
        tau_values.append(tau_head.to(dtype=torch.complex64))

    return torch.stack(tau_values, dim=0)


# ---------------------------------------------------------------------------
# Main calibration logic
# ---------------------------------------------------------------------------

def _find_attention_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Return the list of attention sub-modules in layer order."""
    layers = []
    # Common HF naming: model.model.layers[i].self_attn
    backbone = getattr(model, "model", model)
    layer_list = getattr(backbone, "layers", None)
    if layer_list is None:
        raise RuntimeError(
            "Cannot locate transformer layers. Expected model.model.layers."
        )
    for layer_module in layer_list:
        attn = getattr(layer_module, "self_attn", None)
        if attn is None:
            raise RuntimeError("Layer missing self_attn attribute.")
        layers.append(attn)
    return layers


def calibrate(
    model_name_or_path: str,
    input_path: str,
    output_path: str,
    max_length: int = 32768,
    device: str = "cuda",
    attn_implementation: str = "flash_attention_2",
    build_variational_horizon: bool = False,
    variational_query_samples: int | None = 128,
    variational_key_chunk_size: int = 512,
    variational_offset_max_length: int = 65536,
) -> None:
    device_obj = torch.device(device)
    dtype = torch.bfloat16

    # --- Load config, tokenizer, model ---
    print(f"Loading model: {model_name_or_path}", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    rope_style = _determine_rope_style(config)

    # --- Build rotary for RoPE inversion ---
    attn_layers = _find_attention_layers(model)
    backbone = getattr(model, "model", model)
    # rotary_emb may live on backbone (Qwen2) or on individual attn layers
    if hasattr(backbone, "rotary_emb"):
        rotary = backbone.rotary_emb
    else:
        rotary = attn_layers[0].rotary_emb
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # --- Read and tokenize input ---
    print(f"Reading input: {input_path}", file=sys.stderr)
    text = Path(input_path).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = input_ids.to(device_obj)
    seq_len = input_ids.shape[1]
    print(f"Tokenized length: {seq_len}", file=sys.stderr)

    # --- Pre-compute cos/sin tables ---
    position_ids = torch.arange(seq_len, device=device_obj).unsqueeze(0)
    probe = torch.zeros(1, seq_len, head_dim, device=device_obj, dtype=dtype)
    cos_table, sin_table = rotary(probe, position_ids)
    # cos_table, sin_table: [1, seq_len, head_dim]

    # --- Register hooks to capture Q ---
    captured_q: Dict[int, torch.Tensor] = {}
    captured_k: Dict[int, torch.Tensor] = {}

    def _make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return
            # Compute Q projection manually
            attn = module
            bsz, q_len, _ = hidden_states.shape
            q = attn.q_proj(hidden_states)
            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hidden_states)
            k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            # Apply RoPE
            pos_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            p = torch.zeros(1, q_len, head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            cos, sin = rotary(p, pos_ids)
            q_rot = (q * cos.unsqueeze(1)) + (_rotate_half(q, style=rope_style) * sin.unsqueeze(1))
            k_rot = (k * cos.unsqueeze(1)) + (_rotate_half(k, style=rope_style) * sin.unsqueeze(1))
            q_rot = q_rot * attn_scale
            k_rot = k_rot * attn_scale
            captured_q[layer_idx] = q_rot.detach()
            captured_k[layer_idx] = k_rot.detach()
        return hook_fn

    handles = []
    for layer_idx, attn in enumerate(attn_layers):
        h = attn.register_forward_pre_hook(_make_pre_hook(layer_idx), with_kwargs=True)
        handles.append(h)

    # --- Forward pass ---
    print("Running forward pass...", file=sys.stderr)
    with torch.no_grad():
        model(input_ids)
    print("Forward pass complete.", file=sys.stderr)

    # Remove hooks
    for h in handles:
        h.remove()

    # --- Compute per-head frequency statistics ---
    print("Computing frequency statistics...", file=sys.stderr)
    sampled_heads: List[Tuple[int, int]] = []
    stats_dict: Dict[str, Dict[str, torch.Tensor]] = {}
    variational_offsets = None
    if build_variational_horizon:
        variational_offsets = _build_geometric_offsets(
            variational_offset_max_length,
            device_obj,
        )

    for layer_idx in range(num_layers):
        q_rot = captured_q.get(layer_idx)
        if q_rot is None:
            print(f"  [warn] No Q captured for layer {layer_idx}, skipping.", file=sys.stderr)
            continue
        k_rot = captured_k.get(layer_idx)
        if build_variational_horizon and k_rot is None:
            raise RuntimeError(f"Missing K capture for layer {layer_idx} while building variational horizons.")

        # q_rot: [1, num_heads, seq_len, head_dim]
        # Invert RoPE to get base Q
        cos = cos_table[:, :seq_len, :].unsqueeze(1)  # [1, 1, seq_len, head_dim]
        sin = sin_table[:, :seq_len, :].unsqueeze(1)
        q_base = _invert_rope(q_rot, cos, sin, attn_scale, style=rope_style)
        oracle_tau_by_head = None
        if build_variational_horizon and k_rot is not None and variational_offsets is not None:
            inv_freq = rotary.inv_freq.to(device=device_obj, dtype=torch.float32)
            freq_count = head_dim // 2
            omega = inv_freq[:freq_count]
            oracle_tau_by_head = _estimate_tau_from_attention(
                q_rot[0],
                k_rot[0],
                omega,
                num_kv_heads=num_kv_heads,
                query_sample_count=variational_query_samples,
                key_chunk_size=variational_key_chunk_size,
            )

        for head_idx in range(num_heads):
            q_head = q_base[0, head_idx]  # [seq_len, head_dim]
            q_complex = _to_complex_pairs(q_head, style=rope_style)  # [seq_len, freq_count]

            q_mean_complex = q_complex.mean(dim=0)  # [freq_count]
            q_abs_mean = q_complex.abs().mean(dim=0)  # [freq_count]
            q_sq_abs_mean = q_complex.abs().square().mean(dim=0)  # [freq_count]
            oracle_tau_complex = None
            oracle_pi = None
            oracle_horizon_mean_complex = None
            if oracle_tau_by_head is not None and variational_offsets is not None:
                inv_freq = rotary.inv_freq.to(device=device_obj, dtype=torch.float32)
                freq_count = head_dim // 2
                omega = inv_freq[:freq_count]
                oracle_tau_complex = oracle_tau_by_head[head_idx]
                weights = _compute_effective_weights(q_mean_complex)
                oracle_pi, oracle_horizon_mean_complex = _solve_variational_horizon_qp(
                    oracle_tau_complex,
                    weights,
                    omega,
                    variational_offsets,
                )

            key = f"layer{layer_idx:02d}_head{head_idx:02d}"
            stats_dict[key] = {
                "q_mean_real": q_mean_complex.real.cpu(),
                "q_mean_imag": q_mean_complex.imag.cpu(),
                "q_abs_mean": q_abs_mean.cpu(),
                "q_sq_abs_mean": q_sq_abs_mean.cpu(),
            }
            if oracle_tau_complex is not None:
                stats_dict[key]["oracle_tau_real"] = oracle_tau_complex.real.cpu()
                stats_dict[key]["oracle_tau_imag"] = oracle_tau_complex.imag.cpu()
            if oracle_pi is not None:
                stats_dict[key]["oracle_pi"] = oracle_pi.cpu()
            if oracle_horizon_mean_complex is not None:
                stats_dict[key]["oracle_horizon_mean_real"] = oracle_horizon_mean_complex.real.cpu()
                stats_dict[key]["oracle_horizon_mean_imag"] = oracle_horizon_mean_complex.imag.cpu()
            sampled_heads.append((layer_idx, head_idx))

        # Free memory
        del captured_q[layer_idx]
        if layer_idx in captured_k:
            del captured_k[layer_idx]

    # --- Determine rope_type ---
    rope_scaling = getattr(config, "rope_scaling", {}) or {}
    rope_type = (
        rope_scaling.get("rope_type")
        or rope_scaling.get("type")
        or getattr(config, "rope_type", "default")
        or "default"
    )

    # --- Build metadata ---
    metadata = {
        "stats_version": 3,
        "num_traces": 1,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "use_chat_template": False,
        "system_prompt": "",
        "attn_implementation": attn_implementation,
        "rope_style": rope_style,
        "rope_type": rope_type,
        "sampled_heads": [[int(l), int(h)] for l, h in sampled_heads],
        "build_variational_horizon": bool(build_variational_horizon),
        "variational_query_samples": None if variational_query_samples is None else int(variational_query_samples),
        "variational_key_chunk_size": int(variational_key_chunk_size),
        "variational_offset_max_length": int(variational_offset_max_length),
    }
    if variational_offsets is not None:
        metadata["oracle_offsets"] = variational_offsets.cpu()

    payload = {
        "metadata": metadata,
        "stats": stats_dict,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"Saved stats to {out} ({len(sampled_heads)} heads)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate CASK / TriAttention frequency statistics from plain text."
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Plain text file for calibration.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .pt file path for stats.",
    )
    parser.add_argument(
        "--max-length", type=int, default=32768,
        help="Maximum token length (default: 32768).",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to run on (default: cuda).",
    )
    parser.add_argument(
        "--attn-implementation", default="flash_attention_2",
        help="Attention implementation (default: flash_attention_2).",
    )
    parser.add_argument(
        "--build-variational-horizon",
        action="store_true",
        help="Estimate oracle horizon targets from sampled attention and solve the offline variational QP.",
    )
    parser.add_argument(
        "--variational-query-samples",
        type=int,
        default=128,
        help="Number of query positions sampled per head when estimating the oracle horizon target.",
    )
    parser.add_argument(
        "--variational-key-chunk-size",
        type=int,
        default=512,
        help="Chunk size used for streamed key-side softmax accumulation during oracle horizon estimation.",
    )
    parser.add_argument(
        "--variational-offset-max-length",
        type=int,
        default=65536,
        help="Maximum dyadic offset length used by the offline variational horizon solver.",
    )
    args = parser.parse_args()
    calibrate(
        model_name_or_path=args.model,
        input_path=args.input,
        output_path=args.output,
        max_length=args.max_length,
        device=args.device,
        attn_implementation=args.attn_implementation,
        build_variational_horizon=args.build_variational_horizon,
        variational_query_samples=args.variational_query_samples,
        variational_key_chunk_size=args.variational_key_chunk_size,
        variational_offset_max_length=args.variational_offset_max_length,
    )


if __name__ == "__main__":
    main()


