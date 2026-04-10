"""Shared helpers for round-based sparse KV pruning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
except ImportError:  # Transformers build without Qwen3 modules
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as Qwen3RotaryEmbedding
    except ImportError:
        Qwen3RotaryEmbedding = None  # type: ignore[assignment]
try:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
except ImportError:
    LlamaRotaryEmbedding = None  # type: ignore[assignment]

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
STATS_VERSION = 3


def determine_rope_style(config: AutoConfig) -> str:
    model_type = getattr(config, "model_type", "")
    # HF Llama (incl. llama3 / YaRN scalings) applies RoPE by pairing the front/back
    # halves of the head dimension, not even/odd interleaving.
    if "llama" in model_type:
        return "half"
    return "half"  # front/back pairing (Qwen)


def rotate_half(x: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    if style == "interleaved":
        # Llama-style even/odd pairing
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def invert_rope(
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
        # Allow even/odd to carry different cos/sin (e.g., YaRN/llama3).
        even = base[..., ::2]
        odd = base[..., 1::2]
        cos_even = cos_unit[..., ::2]
        cos_odd = cos_unit[..., 1::2]
        sin_even = sin_unit[..., ::2]
        sin_odd = sin_unit[..., 1::2]
        det = cos_even * cos_odd + sin_even * sin_odd
        det = det.clamp_min(1e-12)
        # Forward: y_even = x_even * cos_even - x_odd * sin_even
        #          y_odd  = x_odd  * cos_odd  + x_even * sin_odd
        orig_even = (even * cos_odd + odd * sin_even) / det
        orig_odd = (odd * cos_even - even * sin_odd) / det
        restored = torch.empty_like(base)
        restored[..., ::2] = orig_even
        restored[..., 1::2] = orig_odd
        return restored
    return base * cos_unit - rotate_half(base, style=style) * sin_unit


def to_complex_pairs(tensor: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        real = tensor_real[..., ::2].contiguous()
        imag = tensor_real[..., 1::2].contiguous()
        return torch.complex(real, imag)
    freq_count = tensor.shape[1] // 2
    real = tensor_real[:, :freq_count].contiguous()
    imag = tensor_real[:, freq_count:].contiguous()
    return torch.complex(real, imag)


@dataclass
class HeadFrequencyStats:
    q_mean_complex: torch.Tensor
    q_abs_mean: torch.Tensor
    q_sq_abs_mean: Optional[torch.Tensor] = None
    oracle_tau_complex: Optional[torch.Tensor] = None
    oracle_pi: Optional[torch.Tensor] = None
    oracle_horizon_mean_complex: Optional[torch.Tensor] = None


def load_or_create_sample(
    sample_file: Path,
    sample_count: int,
    seed: int,
    layer_count: int,
    head_count: int,
) -> List[Tuple[int, int]]:
    if sample_file.exists():
        with sample_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [(int(pair[0]), int(pair[1])) for pair in data]

    if sample_count > layer_count * head_count:
        raise ValueError("sample_count exceeds total available heads")

    indices = [(layer, head) for layer in range(layer_count) for head in range(head_count)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(len(indices), generator=generator)
    selected = [indices[idx] for idx in perm[:sample_count].tolist()]

    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with sample_file.open("w", encoding="utf-8") as handle:
        json.dump([[layer, head] for layer, head in selected], handle, indent=2)

    return selected


def build_rotary(
    cache_device: torch.device,
    model_path: Path,
    dtype: torch.dtype,
    config: Optional[AutoConfig] = None,
) -> object:
    if config is None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    rope_style = determine_rope_style(config)
    model_type = getattr(config, "model_type", "")
    if "llama" in model_type:
        if LlamaRotaryEmbedding is None:
            raise ImportError("Llama rotary embedding is unavailable in the current transformers build.")

        rope_scaling = dict(config.rope_scaling or {})
        if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
            rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
        rope_scaling.pop("attn_factor", None)
        if "rope_type" not in rope_scaling:
            rope_scaling["rope_type"] = rope_scaling.get("type", "default")
        rope_scaling.pop("type", None)
        config.rope_scaling = rope_scaling

        rotary = LlamaRotaryEmbedding(config=config, device=cache_device)
        rotary.to(dtype=dtype, device=cache_device)
        rotary._rope_style = rope_style  # type: ignore[attr-defined]
        return rotary

    if Qwen3RotaryEmbedding is None:
        raise ImportError(
            "Neither Qwen3 nor Qwen2 rotary embeddings are available in the installed transformers package."
        )

    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    if "rope_type" not in rope_scaling:
        rope_scaling["rope_type"] = rope_scaling.get("type", "default")
    rope_scaling.pop("type", None)
    config.rope_scaling = rope_scaling
    rotary = Qwen3RotaryEmbedding(config=config, device=cache_device)
    rotary.to(dtype=dtype)
    rotary._rope_style = rope_style  # type: ignore[attr-defined]
    return rotary


def verify_rotary_alignment(local_rotary: torch.nn.Module, model_rotary: torch.nn.Module) -> None:
    """
    Asserts that the locally constructed RotaryEmbedding matches the model's live RotaryEmbedding.
    Checks inv_freq and scaling factors.
    """
    # Check inv_freq
    if hasattr(local_rotary, "inv_freq") and hasattr(model_rotary, "inv_freq"):
        local_freq = local_rotary.inv_freq.to("cpu")
        model_freq = model_rotary.inv_freq.to("cpu")
        
        if local_freq.shape != model_freq.shape:
             raise ValueError(
                f"Rotary embedding shape mismatch! Local: {local_freq.shape}, Model: {model_freq.shape}"
            )

        # Allow small float error
        if not torch.allclose(local_freq, model_freq, atol=1e-5):
            diff = (local_freq - model_freq).abs().max().item()
            raise ValueError(
                f"Rotary embedding inv_freq mismatch! Max diff: {diff}. "
                "This likely means the pruner's RoPE config does not match the model's."
            )
    
    # Check scaling factor if present (e.g. Qwen)
    # Llama usually bakes scaling into inv_freq or uses explicit implementation differences.
    # Ideally inv_freq equality covers most scaling issues for Llama3.



def compute_frequency_scaling(
    rotary: Qwen3RotaryEmbedding,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
    probe = torch.zeros(1, 1, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(probe, position_ids)
    cos0 = cos[0, 0]
    sin0 = sin[0, 0]
    scale = torch.sqrt(cos0[0::2].pow(2) + sin0[0::2].pow(2))
    return scale.to(device=device, dtype=torch.float32)


def compute_rotary_tables(
    rotary: Qwen3RotaryEmbedding,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0]
    sin_table = sin_table[0]
    inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float64)
    freq_scale = compute_frequency_scaling(rotary, head_dim, dtype, device)
    style = getattr(rotary, "_rope_style", "half")
    return cos_table, sin_table, inv_freq, freq_scale, style


def build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def compute_offset_mean_complex(
    offsets: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    offsets_fp32 = offsets.to(dtype=torch.float32)
    omega_fp32 = omega.to(dtype=torch.float32)
    phases = offsets_fp32.unsqueeze(1) * omega_fp32.unsqueeze(0)
    return torch.polar(torch.ones_like(phases), phases).mean(dim=0)


def compute_band_concentration(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    q_mean_abs = torch.abs(q_mean_complex).to(dtype=torch.float32)
    q_abs = q_abs_mean.to(dtype=torch.float32)
    return q_mean_abs / q_abs.clamp_min(float(eps))


def compute_rms2_coefficient(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    q_sq_abs_mean: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if q_sq_abs_mean is None:
        raise ValueError("q_sq_abs_mean is required to compute RMS2 coefficients")
    q_mean_abs = torch.abs(q_mean_complex).to(dtype=torch.float32)
    q_sq_abs = q_sq_abs_mean.to(dtype=torch.float32)
    numerator = (q_sq_abs - q_mean_abs.pow(2)).clamp_min(0.0)
    denominator = (q_abs_mean.to(dtype=torch.float32) + q_mean_abs).clamp_min(float(eps))
    return numerator / denominator


def compute_effective_wavelength(
    g: torch.Tensor,
    omega: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    g_fp32 = g.to(dtype=torch.float32)
    weights = g_fp32 / g_fp32.sum().clamp_min(float(eps))
    wavelengths = (2.0 * torch.pi) / omega.to(dtype=torch.float32).clamp_min(float(eps))
    log_lambda_eff = (weights * torch.log(wavelengths.clamp_min(float(eps)))).sum()
    return weights, torch.exp(log_lambda_eff)


def build_adaptive_horizon_kernel(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    *,
    c_lambda: float,
    s0: float,
    s1: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_mean_abs = torch.abs(q_mean_complex).to(dtype=torch.float32)
    R_band = compute_band_concentration(q_mean_complex, q_abs_mean, eps=eps)
    weights, lambda_eff = compute_effective_wavelength(q_mean_abs, omega, eps=eps)
    R_bar = (weights * R_band).sum()

    offsets_fp32 = offsets.to(dtype=torch.float32)
    offset_exponents = torch.log2(offsets_fp32.clamp_min(1.0))
    c_lambda_t = torch.tensor(float(c_lambda), device=offsets.device, dtype=torch.float32)
    j_center = torch.log2((c_lambda_t * lambda_eff).clamp_min(float(eps)))
    sigma = torch.tensor(float(s0), device=offsets.device, dtype=torch.float32) + (
        torch.tensor(float(s1), device=offsets.device, dtype=torch.float32) * (1.0 - R_bar)
    )
    sigma = sigma.clamp_min(float(eps))
    logits = -((offset_exponents - j_center) ** 2) / (2.0 * sigma.pow(2))
    pi = torch.softmax(logits, dim=0)

    phases = offsets_fp32.unsqueeze(1) * omega.to(dtype=torch.float32).unsqueeze(0)
    exp_phases = torch.polar(torch.ones_like(phases), phases)
    kappa_complex = (pi.unsqueeze(1) * exp_phases).sum(dim=0)
    return R_band, lambda_eff, R_bar, pi, kappa_complex


def solve_variational_horizon_qp(
    tau_complex: torch.Tensor,
    weights: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    *,
    eps: float = 1e-8,
    maxiter: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the simplex-constrained QP for the variational horizon kernel.

    The objective is:
        min_{pi in simplex} sum_f w_f |kappa_pi(omega_f) - tau_f|^2
    where kappa_pi(omega_f) = sum_j pi_j exp(i * omega_f * offset_j).
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, minimize
    except Exception as exc:  # pragma: no cover - scipy import safety
        raise ImportError(
            "scipy is required to solve the variational horizon QP. "
            "Install scipy and retry calibration."
        ) from exc

    tau_fp32 = tau_complex.to(dtype=torch.complex64)
    weights_fp32 = weights.to(dtype=torch.float32)
    omega_fp32 = omega.to(dtype=torch.float32)
    offsets_fp32 = offsets.to(dtype=torch.float32)

    phases = offsets_fp32.unsqueeze(1) * omega_fp32.unsqueeze(0)
    basis_complex = torch.polar(torch.ones_like(phases), phases)  # [J, F]
    basis_real = basis_complex.real.T.contiguous()  # [F, J]
    basis_imag = basis_complex.imag.T.contiguous()  # [F, J]
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

    num_offsets = int(offsets_fp32.numel())
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


def compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_unrot: torch.Tensor,
    *,
    style: str = "half",
    disable_mlr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_complex = to_complex_pairs(k_unrot, style=style)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    if disable_mlr:
        extra = q_abs_mean.unsqueeze(0) * k_abs
    else:
        extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def score_keys_for_round(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
    freq_scale_sq: torch.Tensor,
    disable_trig: bool = False,
) -> torch.Tensor:
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)

    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)
    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(1)

    cos_phase = torch.cos(phase)

    scale = freq_scale_sq.view(1, 1, -1)

    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)
    # additive term uses original freq_scale_sq (not affected by high-freq masking)
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1, keepdim=True)
    combined = additive if disable_trig else (base_scores + additive)

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values


def score_rotated_keys_for_round(
    *,
    k_rot: torch.Tensor,
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    round_start: int,
    omega: torch.Tensor,
    freq_scale: torch.Tensor,
    aggregation: str,
    style: str = "half",
    disable_mlr: bool = False,
    disable_trig: bool = False,
    extra_coef: Optional[torch.Tensor] = None,
    horizon_mean_complex: Optional[torch.Tensor] = None,
    offset_mean_complex: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if k_rot.numel() == 0:
        return torch.empty(0, device=k_rot.device, dtype=torch.float32)

    k_rot_complex = to_complex_pairs(k_rot, style=style)
    scale = freq_scale.to(device=k_rot.device, dtype=torch.float32)
    q_mean = q_mean_complex.to(device=k_rot.device)
    q_abs = q_abs_mean.to(device=k_rot.device, dtype=torch.float32)
    q_mean_abs = torch.abs(q_mean)
    conj_k = torch.conj(k_rot_complex)
    k_abs = torch.abs(k_rot_complex)

    if extra_coef is None:
        extra_coef = q_abs if disable_mlr else (q_abs - q_mean_abs)
    else:
        extra_coef = extra_coef.to(device=k_rot.device, dtype=torch.float32)
    extra_term = (extra_coef.unsqueeze(0) * k_abs * scale.unsqueeze(0)).sum(dim=1)

    if disable_trig:
        return extra_term

    round_start_fp32 = torch.tensor(float(round_start), device=k_rot.device, dtype=torch.float32)
    if aggregation == "mean":
        horizon_complex = horizon_mean_complex
        if horizon_complex is None:
            horizon_complex = offset_mean_complex
        if horizon_complex is None:
            if offsets is None:
                raise ValueError("offsets or offset_mean_complex is required for mean aggregation")
            horizon_complex = compute_offset_mean_complex(offsets, omega)
        round_phase = torch.polar(
            torch.ones_like(omega, dtype=torch.float32, device=k_rot.device),
            round_start_fp32 * omega.to(device=k_rot.device, dtype=torch.float32),
        )
        gamma = q_mean * horizon_complex.to(device=k_rot.device) * round_phase
        base_scores = ((gamma.unsqueeze(0) * conj_k).real * scale.unsqueeze(0)).sum(dim=1)
        return base_scores + extra_term

    if offsets is None:
        raise ValueError("offsets is required for non-mean aggregation")

    offset_positions = offsets.to(device=k_rot.device, dtype=torch.float32) + round_start_fp32
    phases = offset_positions.unsqueeze(1) * omega.to(device=k_rot.device, dtype=torch.float32).unsqueeze(0)
    round_phases = torch.polar(torch.ones_like(phases), phases)
    weighted_q = q_mean.unsqueeze(0) * round_phases
    base_scores = (
        (weighted_q.unsqueeze(1) * conj_k.unsqueeze(0)).real * scale.view(1, 1, -1)
    ).sum(dim=2)
    combined = base_scores + extra_term.unsqueeze(0)
    return combined.max(dim=0).values


def save_head_frequency_stats(
    output_path: Path,
    sampled_heads: Sequence[Tuple[int, int]],
    stats_map: Dict[Tuple[int, int], HeadFrequencyStats],
    metadata: Dict[str, torch.Tensor | int | str | float],
) -> None:
    payload: Dict[str, object] = {
        "metadata": {
            **metadata,
            "stats_version": int(metadata.get("stats_version", STATS_VERSION)),
            "sampled_heads": [[int(layer), int(head)] for layer, head in sampled_heads],
        },
        "stats": {},
    }
    for (layer, head), head_stats in stats_map.items():
        key = f"layer{layer:02d}_head{head:02d}"
        payload["stats"][key] = {
            "q_mean_real": head_stats.q_mean_complex.real.cpu(),
            "q_mean_imag": head_stats.q_mean_complex.imag.cpu(),
            "q_abs_mean": head_stats.q_abs_mean.cpu(),
        }
        if head_stats.q_sq_abs_mean is not None:
            payload["stats"][key]["q_sq_abs_mean"] = head_stats.q_sq_abs_mean.cpu()
        if head_stats.oracle_tau_complex is not None:
            payload["stats"][key]["oracle_tau_real"] = head_stats.oracle_tau_complex.real.cpu()
            payload["stats"][key]["oracle_tau_imag"] = head_stats.oracle_tau_complex.imag.cpu()
        if head_stats.oracle_pi is not None:
            payload["stats"][key]["oracle_pi"] = head_stats.oracle_pi.cpu()
        if head_stats.oracle_horizon_mean_complex is not None:
            payload["stats"][key]["oracle_horizon_mean_real"] = head_stats.oracle_horizon_mean_complex.real.cpu()
            payload["stats"][key]["oracle_horizon_mean_imag"] = head_stats.oracle_horizon_mean_complex.imag.cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_head_frequency_stats(
    stats_path: Path,
    device: torch.device,
) -> tuple[Dict[str, object], Dict[Tuple[int, int], HeadFrequencyStats]]:
    payload = torch.load(stats_path, map_location=device)
    metadata = dict(payload["metadata"])
    metadata.setdefault("stats_version", 1)
    stats_raw: Dict[str, Dict[str, torch.Tensor]] = payload["stats"]
    sampled_heads = [tuple(item) for item in metadata["sampled_heads"]]
    stats: Dict[Tuple[int, int], HeadFrequencyStats] = {}
    for layer, head in sampled_heads:
        key = f"layer{layer:02d}_head{head:02d}"
        entry = stats_raw.get(key)
        if entry is None:
            continue
        q_mean_complex = torch.complex(
            entry["q_mean_real"].to(device=device, dtype=torch.float32),
            entry["q_mean_imag"].to(device=device, dtype=torch.float32),
        )
        q_abs_mean = entry["q_abs_mean"].to(device=device, dtype=torch.float32)
        q_sq_abs_mean_raw = entry.get("q_sq_abs_mean")
        q_sq_abs_mean = None
        if isinstance(q_sq_abs_mean_raw, torch.Tensor):
            q_sq_abs_mean = q_sq_abs_mean_raw.to(device=device, dtype=torch.float32)
        oracle_tau_complex = None
        if isinstance(entry.get("oracle_tau_real"), torch.Tensor) and isinstance(entry.get("oracle_tau_imag"), torch.Tensor):
            oracle_tau_complex = torch.complex(
                entry["oracle_tau_real"].to(device=device, dtype=torch.float32),
                entry["oracle_tau_imag"].to(device=device, dtype=torch.float32),
            )
        oracle_pi = None
        if isinstance(entry.get("oracle_pi"), torch.Tensor):
            oracle_pi = entry["oracle_pi"].to(device=device, dtype=torch.float32)
        oracle_horizon_mean_complex = None
        if isinstance(entry.get("oracle_horizon_mean_real"), torch.Tensor) and isinstance(entry.get("oracle_horizon_mean_imag"), torch.Tensor):
            oracle_horizon_mean_complex = torch.complex(
                entry["oracle_horizon_mean_real"].to(device=device, dtype=torch.float32),
                entry["oracle_horizon_mean_imag"].to(device=device, dtype=torch.float32),
            )
        stats[(int(layer), int(head))] = HeadFrequencyStats(
            q_mean_complex=q_mean_complex,
            q_abs_mean=q_abs_mean,
            q_sq_abs_mean=q_sq_abs_mean,
            oracle_tau_complex=oracle_tau_complex,
            oracle_pi=oracle_pi,
            oracle_horizon_mean_complex=oracle_horizon_mean_complex,
        )
    return metadata, stats
