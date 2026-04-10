#!/usr/bin/env python3
"""Benchmark exact rotated-key TriAttention scoring against the legacy path."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triattention.methods.pruning_utils import (
    build_rotary,
    compute_frequency_scaling,
    compute_frequency_statistics_from_means,
    compute_offset_mean_complex,
    invert_rope,
    load_head_frequency_stats,
    score_keys_for_round,
    score_rotated_keys_for_round,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0, help="Layer index to benchmark.")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[199, 1024, 4096],
        help="Decode candidate lengths to benchmark.",
    )
    parser.add_argument("--round-start", type=int, default=384, help="Query round start position.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations per seq len.")
    parser.add_argument("--iters", type=int, default=80, help="Measured iterations per seq len.")
    parser.add_argument("--device", type=str, default="cuda", help="Benchmark device.")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def make_rotated_keys(
    *,
    rotary,
    seq_len: int,
    head_dim: int,
    num_kv_heads: int,
    attn_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.arange(185, 185 + seq_len, device=device, dtype=torch.long)
    probe = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(probe, positions.unsqueeze(0))
    cos = cos[0]
    sin = sin[0]
    k_unrot = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    k_rot = (k_unrot * cos.unsqueeze(0)) + (rotate_half(k_unrot) * sin.unsqueeze(0))
    return positions, cos, sin, k_rot * attn_scale


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    metadata, stats_map = load_head_frequency_stats(args.stats_path, device)
    rotary = build_rotary(device, args.model_path, torch.float32, config=config)
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))
    freq_scale = compute_frequency_scaling(rotary, config.head_dim, torch.float32, device)
    freq_scale_sq = freq_scale.pow(2)
    omega = rotary.inv_freq.to(device=device, dtype=torch.float32)[: config.head_dim // 2]
    offsets = torch.tensor(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0],
        device=device,
        dtype=torch.float32,
    )
    offset_mean_complex = compute_offset_mean_complex(offsets, omega)

    num_attention_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    num_kv_groups = num_attention_heads // num_kv_heads
    layer_idx = int(args.layer)
    layer_heads = [(layer_idx, head_idx) for head_idx in range(num_attention_heads)]

    def old_layer_scores(positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_rot: torch.Tensor) -> torch.Tensor:
        out = []
        for _, head in layer_heads:
            stats = stats_map[(layer_idx, head)]
            kv_head = min(num_kv_heads - 1, head // num_kv_groups)
            k_values = k_rot[kv_head]
            k_unrot = invert_rope(k_values, cos, sin, attn_scale, style="half")
            amp, phi, extra = compute_frequency_statistics_from_means(
                stats.q_mean_complex,
                stats.q_abs_mean,
                k_unrot,
                style="half",
                disable_mlr=False,
            )
            scores = score_keys_for_round(
                key_indices=positions,
                round_start=args.round_start,
                amp=amp,
                phi=phi,
                omega=omega,
                extra=extra,
                offsets=offsets,
                aggregation="mean",
                freq_scale_sq=freq_scale_sq,
                disable_trig=False,
            )
            out.append(scores)
        return torch.stack(out, dim=0)

    def new_layer_scores(positions: torch.Tensor, k_rot: torch.Tensor) -> torch.Tensor:
        out = []
        for _, head in layer_heads:
            stats = stats_map[(layer_idx, head)]
            kv_head = min(num_kv_heads - 1, head // num_kv_groups)
            scores = score_rotated_keys_for_round(
                k_rot=k_rot[kv_head],
                q_mean_complex=stats.q_mean_complex,
                q_abs_mean=stats.q_abs_mean,
                round_start=args.round_start,
                omega=omega,
                freq_scale=freq_scale,
                aggregation="mean",
                style="half",
                disable_mlr=False,
                disable_trig=False,
                offset_mean_complex=offset_mean_complex,
            )
            out.append(scores)
        return torch.stack(out, dim=0)

    results = []
    for seq_len in args.seq_lens:
        positions, cos, sin, k_rot = make_rotated_keys(
            rotary=rotary,
            seq_len=seq_len,
            head_dim=config.head_dim,
            num_kv_heads=num_kv_heads,
            attn_scale=attn_scale,
            device=device,
            dtype=torch.float32,
        )

        old_scores = old_layer_scores(positions, cos, sin, k_rot)
        new_scores = new_layer_scores(positions, k_rot)
        max_abs_diff = float((old_scores - new_scores).abs().max().item())

        for _ in range(args.warmup):
            old_layer_scores(positions, cos, sin, k_rot)
            new_layer_scores(positions, k_rot)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(args.iters):
            old_layer_scores(positions, cos, sin, k_rot)
        if device.type == "cuda":
            torch.cuda.synchronize()
        old_ms = (time.perf_counter() - start) * 1000.0 / args.iters

        start = time.perf_counter()
        for _ in range(args.iters):
            new_layer_scores(positions, k_rot)
        if device.type == "cuda":
            torch.cuda.synchronize()
        new_ms = (time.perf_counter() - start) * 1000.0 / args.iters

        result = {
            "seq_len": int(seq_len),
            "layer": layer_idx,
            "old_ms_per_layer": old_ms,
            "new_ms_per_layer": new_ms,
            "speedup": old_ms / new_ms,
            "max_abs_diff": max_abs_diff,
        }
        results.append(result)
        print(json.dumps(result))

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
