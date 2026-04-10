#!/usr/bin/env python3
"""Run Qwen3 vLLM comparisons for baseline, TriAttention, or HorizonKV under Linux or WSL."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triattention.common.prompt_utils import build_plain_prompt, extract_question_from_record


DEFAULT_MODEL_PATH = REPO_ROOT / "experiments" / "models" / "Qwen3-8B"
DEFAULT_STATS_PATH = (
    REPO_ROOT / "triattention" / "calibration" / "for_aime25_experiment" / "qwen3_8b.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("pair", "baseline", "triattention", "horizonkv"),
        default="pair",
        help="Run both modes in subprocesses or a single concrete mode.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("triattention", "horizonkv"),
        default="horizonkv",
        help="Candidate mode used in pair runs.",
    )
    parser.add_argument("--dataset", default="aime24", help="Dataset name under data/*.jsonl.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Explicit dataset jsonl path.")
    parser.add_argument("--limit", type=int, default=1, help="Number of prompts to run.")
    parser.add_argument(
        "--record-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional dataset record ids to keep.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--cpu-offload-gb", type=float, default=6.0)
    parser.add_argument("--dtype", default="half", choices=("half", "bfloat16"))
    parser.add_argument("--kv-budget", type=int, default=256)
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--triattention-horizon-mode", choices=("fixed", "adaptive", "variational"), default=None)
    parser.add_argument("--triattention-norm-mode", choices=("tri", "rms2"), default=None)
    parser.add_argument("--triattention-kernel-c-lambda", type=float, default=None)
    parser.add_argument("--triattention-kernel-s0", type=float, default=None)
    parser.add_argument("--triattention-kernel-s1", type=float, default=None)
    parser.add_argument("--triattention-norm-lambda", type=float, default=None)
    parser.add_argument(
        "--filler-repeat",
        type=int,
        default=260,
        help="Repeat count for context filler inserted ahead of the real problem.",
    )
    parser.add_argument(
        "--filler-token",
        default="alpha",
        help="Token repeated in the context filler block.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Result file for single-mode or pair summary.",
    )
    parser.add_argument(
        "--run-tag",
        default="wsl_pair",
        help="Tag used to name per-mode result files in pair mode.",
    )
    return parser.parse_args()


def stats_supports_rms2(stats_path: Path) -> bool:
    payload = torch.load(stats_path, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    stats_version = int(metadata.get("stats_version", 1))
    if stats_version >= 2:
        return True
    stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
    if not isinstance(stats, dict):
        return False
    return any(isinstance(entry, dict) and "q_sq_abs_mean" in entry for entry in stats.values())


def stats_supports_variational_horizon(stats_path: Path) -> bool:
    payload = torch.load(stats_path, map_location="cpu")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if bool(metadata.get("build_variational_horizon", False)):
        return True
    stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
    if not isinstance(stats, dict):
        return False
    return any(
        isinstance(entry, dict)
        and "oracle_horizon_mean_real" in entry
        and "oracle_horizon_mean_imag" in entry
        for entry in stats.values()
    )


def validate_stats_args(args: argparse.Namespace) -> None:
    if args.stats_path is None:
        raise ValueError("--stats-path is required for vLLM pair runs.")
    if not args.stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")

    requires_rms2 = False
    if args.mode == "horizonkv":
        requires_rms2 = True
    elif args.mode == "pair" and args.candidate_mode == "horizonkv":
        requires_rms2 = True
    elif args.triattention_norm_mode == "rms2":
        requires_rms2 = True

    if requires_rms2 and not stats_supports_rms2(args.stats_path):
        raise ValueError(
            f"Stats file {args.stats_path} does not contain q_sq_abs_mean and cannot be used "
            "with HorizonKV / norm_mode='rms2'. Regenerate v2 stats with scripts/calibrate.py "
            "and pass the result via --stats-path."
        )

    requires_variational = False
    if args.triattention_horizon_mode == "variational":
        requires_variational = True
    if requires_variational and not stats_supports_variational_horizon(args.stats_path):
        raise ValueError(
            f"Stats file {args.stats_path} does not contain offline variational horizon tensors and cannot be used "
            "with horizon_mode='variational'. Regenerate stats with scripts/calibrate.py "
            "--build-variational-horizon and pass the result via --stats-path."
        )


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path is not None:
        return args.dataset_path.resolve()
    return (REPO_ROOT / "data" / f"{args.dataset}.jsonl").resolve()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_prompt(question: str, filler_repeat: int, filler_token: str) -> str:
    base_prompt = build_plain_prompt(question)
    if filler_repeat <= 0:
        return base_prompt
    filler = ((filler_token + " ") * filler_repeat).strip()
    return (
        "Ignore the following scratch context until the actual problem begins.\n\n"
        f"{filler}\n\n"
        "The actual problem begins now.\n\n"
        f"{base_prompt}"
    )


def load_prompts(args: argparse.Namespace) -> list[dict]:
    dataset_path = resolve_dataset_path(args)
    wanted_ids = set(args.record_ids or [])
    prompts: list[dict] = []
    for index, row in enumerate(iter_jsonl(dataset_path)):
        record_id = int(row.get("id", index))
        if wanted_ids and record_id not in wanted_ids:
            continue
        question = extract_question_from_record(row, fallback_keys=("question", "problem"))
        prompt = build_prompt(question, args.filler_repeat, args.filler_token)
        prompts.append(
            {
                "dataset_index": index,
                "record_id": record_id,
                "question": question,
                "prompt": prompt,
            }
        )
        if len(prompts) >= args.limit:
            break
    if not prompts:
        raise SystemExit(f"No prompts selected from {dataset_path}")
    return prompts


def read_new_trace_lines(trace_path: Path, offset: int) -> list[dict]:
    if not trace_path.exists():
        return []
    rows: list[dict] = []
    with trace_path.open(encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_trace(rows: list[dict]) -> dict:
    event_counts = Counter()
    applied_steps: list[int] = []
    skipped_reasons = Counter()
    for row in rows:
        event = str(row.get("event"))
        event_counts[event] += 1
        if event != "runner_compression_result":
            continue
        if row.get("applied"):
            step = row.get("step")
            if isinstance(step, int):
                applied_steps.append(step)
        else:
            skipped_reasons[str(row.get("reason"))] += 1
    return {
        "trace_event_counts": dict(event_counts),
        "compression_events": len(applied_steps),
        "compression_steps": applied_steps,
        "compression_first_step": applied_steps[0] if applied_steps else None,
        "compression_last_step": applied_steps[-1] if applied_steps else None,
        "compression_skipped_reasons": dict(skipped_reasons),
    }


def default_result_path(args: argparse.Namespace, mode: str) -> Path:
    safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in args.run_tag).strip("-")
    if not safe_tag:
        safe_tag = "wsl_pair"
    out_dir = REPO_ROOT / "experiments" / "vllm_pair"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe_tag}_{args.dataset}_{mode}.json"


def build_sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        ignore_eos=bool(args.ignore_eos),
    )


def apply_mode_env(args: argparse.Namespace, mode: str, trace_path: Path, compression_log_path: Path) -> None:
    os.environ["TRIATTENTION_INTERFACE"] = "runtime"
    os.environ["TRIATTENTION_QUIET"] = "1"
    os.environ["TRIATTN_RUNTIME_TRACE_PATH"] = str(trace_path)
    if mode in {"triattention", "horizonkv"}:
        os.environ["ENABLE_TRIATTENTION"] = "true"
        os.environ["TRIATTN_RUNTIME_KV_BUDGET"] = str(args.kv_budget)
        os.environ["TRIATTN_RUNTIME_DIVIDE_LENGTH"] = str(args.divide_length)
        os.environ["TRIATTN_RUNTIME_WINDOW_SIZE"] = str(args.window_size)
        os.environ["TRIATTN_RUNTIME_SPARSE_STATS_PATH"] = str(args.stats_path.resolve())
        os.environ["TRIATTN_RUNTIME_MODEL_PATH"] = str(args.model_path.resolve())
        os.environ["TRIATTN_RUNTIME_DEBUG_COMPRESSION_LOG"] = "true"
        os.environ["TRIATTN_RUNTIME_DEBUG_COMPRESSION_LOG_PATH"] = str(compression_log_path)
        os.environ["TRIATTN_RUNTIME_DEBUG_COMPRESSION_LOG_CONTEXT_TOKENS"] = "16"
        horizon_mode = args.triattention_horizon_mode
        norm_mode = args.triattention_norm_mode
        if mode == "horizonkv":
            horizon_mode = horizon_mode or "adaptive"
            norm_mode = norm_mode or "rms2"
        if horizon_mode:
            os.environ["TRIATTN_RUNTIME_SPARSE_HORIZON_MODE"] = str(horizon_mode)
        else:
            os.environ.pop("TRIATTN_RUNTIME_SPARSE_HORIZON_MODE", None)
        if norm_mode:
            os.environ["TRIATTN_RUNTIME_SPARSE_NORM_MODE"] = str(norm_mode)
        else:
            os.environ.pop("TRIATTN_RUNTIME_SPARSE_NORM_MODE", None)
        if args.triattention_kernel_c_lambda is not None:
            os.environ["TRIATTN_RUNTIME_SPARSE_KERNEL_C_LAMBDA"] = str(args.triattention_kernel_c_lambda)
        else:
            os.environ.pop("TRIATTN_RUNTIME_SPARSE_KERNEL_C_LAMBDA", None)
        if args.triattention_kernel_s0 is not None:
            os.environ["TRIATTN_RUNTIME_SPARSE_KERNEL_S0"] = str(args.triattention_kernel_s0)
        else:
            os.environ.pop("TRIATTN_RUNTIME_SPARSE_KERNEL_S0", None)
        if args.triattention_kernel_s1 is not None:
            os.environ["TRIATTN_RUNTIME_SPARSE_KERNEL_S1"] = str(args.triattention_kernel_s1)
        else:
            os.environ.pop("TRIATTN_RUNTIME_SPARSE_KERNEL_S1", None)
        if args.triattention_norm_lambda is not None:
            os.environ["TRIATTN_RUNTIME_SPARSE_NORM_LAMBDA"] = str(args.triattention_norm_lambda)
        else:
            os.environ.pop("TRIATTN_RUNTIME_SPARSE_NORM_LAMBDA", None)
    else:
        os.environ["ENABLE_TRIATTENTION"] = "false"
        for key in [
            "TRIATTN_RUNTIME_KV_BUDGET",
            "TRIATTN_RUNTIME_DIVIDE_LENGTH",
            "TRIATTN_RUNTIME_WINDOW_SIZE",
            "TRIATTN_RUNTIME_SPARSE_STATS_PATH",
            "TRIATTN_RUNTIME_SPARSE_HORIZON_MODE",
            "TRIATTN_RUNTIME_SPARSE_NORM_MODE",
            "TRIATTN_RUNTIME_SPARSE_KERNEL_C_LAMBDA",
            "TRIATTN_RUNTIME_SPARSE_KERNEL_S0",
            "TRIATTN_RUNTIME_SPARSE_KERNEL_S1",
            "TRIATTN_RUNTIME_SPARSE_NORM_LAMBDA",
            "TRIATTN_RUNTIME_MODEL_PATH",
            "TRIATTN_RUNTIME_DEBUG_COMPRESSION_LOG",
            "TRIATTN_RUNTIME_DEBUG_COMPRESSION_LOG_PATH",
            "TRIATTN_RUNTIME_DEBUG_COMPRESSION_LOG_CONTEXT_TOKENS",
        ]:
            os.environ.pop(key, None)


def run_single_mode(args: argparse.Namespace) -> dict:
    mode = args.mode
    prompts = load_prompts(args)

    trace_path = Path(tempfile.gettempdir()) / f"triattn_{mode}_{os.getpid()}_{int(time.time())}.jsonl"
    compression_log_path = (
        Path(tempfile.gettempdir()) / f"triattn_{mode}_{os.getpid()}_{int(time.time())}.log"
    )
    for path in (trace_path, compression_log_path):
        if path.exists():
            path.unlink()

    apply_mode_env(args, mode, trace_path, compression_log_path)

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    sampling_params = build_sampling_params(args)
    llm = LLM(
        model=str(args.model_path),
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    records: list[dict] = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_generation_seconds = 0.0

    for item in prompts:
        prompt = item["prompt"]
        prompt_token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        total_prompt_tokens += len(prompt_token_ids)
        trace_offset = trace_path.stat().st_size if trace_path.exists() else 0
        started = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        generation_seconds = time.perf_counter() - started
        output = outputs[0].outputs[0]
        output_text = output.text
        if hasattr(output, "token_ids") and output.token_ids is not None:
            output_tokens = int(len(output.token_ids))
        else:
            output_tokens = int(
                len(tokenizer(output_text, add_special_tokens=False)["input_ids"])
            )
        total_output_tokens += output_tokens
        total_generation_seconds += generation_seconds
        trace_rows = read_new_trace_lines(trace_path, trace_offset)
        trace_summary = summarize_trace(trace_rows)
        records.append(
            {
                **item,
                "prompt_tokens": len(prompt_token_ids),
                "output_tokens": output_tokens,
                "generation_seconds": generation_seconds,
                "output_tokens_per_second": (
                    output_tokens / generation_seconds if generation_seconds > 0 else None
                ),
                "output_preview": output_text[:200],
                **trace_summary,
            }
        )

    summary = {
        "mode": mode,
        "dataset": args.dataset,
        "dataset_path": str(resolve_dataset_path(args)),
        "model_path": str(args.model_path.resolve()),
        "stats_path": str(args.stats_path.resolve()),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "kv_budget": args.kv_budget,
        "divide_length": args.divide_length,
        "window_size": args.window_size,
        "filler_repeat": args.filler_repeat,
        "records": records,
        "aggregate": {
            "num_prompts": len(records),
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "generation_seconds": total_generation_seconds,
            "output_tokens_per_second": (
                total_output_tokens / total_generation_seconds
                if total_generation_seconds > 0
                else None
            ),
            "total_tokens_per_second": (
                (total_prompt_tokens + total_output_tokens) / total_generation_seconds
                if total_generation_seconds > 0
                else None
            ),
            "compression_events": sum(int(r["compression_events"]) for r in records),
        },
        "trace_path": str(trace_path),
        "compression_log_path": str(compression_log_path),
    }
    return summary


def run_pair(args: argparse.Namespace) -> dict:
    summary_dir = (args.result_json.parent if args.result_json is not None else default_result_path(args, "pair").parent)
    summary_dir.mkdir(parents=True, exist_ok=True)
    baseline_json = summary_dir / f"{args.run_tag}_{args.dataset}_baseline.json"
    candidate_json = summary_dir / f"{args.run_tag}_{args.dataset}_{args.candidate_mode}.json"

    common_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        args.dataset,
        "--model-path",
        str(args.model_path),
        "--stats-path",
        str(args.stats_path),
        "--limit",
        str(args.limit),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--min-tokens",
        str(args.min_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cpu-offload-gb",
        str(args.cpu_offload_gb),
        "--dtype",
        args.dtype,
        "--kv-budget",
        str(args.kv_budget),
        "--divide-length",
        str(args.divide_length),
        "--window-size",
        str(args.window_size),
        "--candidate-mode",
        args.candidate_mode,
        "--filler-repeat",
        str(args.filler_repeat),
        "--filler-token",
        args.filler_token,
        "--run-tag",
        args.run_tag,
    ]
    if args.triattention_horizon_mode is not None:
        common_args.extend(["--triattention-horizon-mode", args.triattention_horizon_mode])
    if args.triattention_norm_mode is not None:
        common_args.extend(["--triattention-norm-mode", args.triattention_norm_mode])
    if args.triattention_kernel_c_lambda is not None:
        common_args.extend(["--triattention-kernel-c-lambda", str(args.triattention_kernel_c_lambda)])
    if args.triattention_kernel_s0 is not None:
        common_args.extend(["--triattention-kernel-s0", str(args.triattention_kernel_s0)])
    if args.triattention_kernel_s1 is not None:
        common_args.extend(["--triattention-kernel-s1", str(args.triattention_kernel_s1)])
    if args.triattention_norm_lambda is not None:
        common_args.extend(["--triattention-norm-lambda", str(args.triattention_norm_lambda)])
    if args.ignore_eos:
        common_args.append("--ignore-eos")
    if args.dataset_path is not None:
        common_args.extend(["--dataset-path", str(args.dataset_path)])
    if args.record_ids:
        common_args.append("--record-ids")
        common_args.extend(str(x) for x in args.record_ids)

    subprocess.check_call(common_args + ["--mode", "baseline", "--result-json", str(baseline_json)])
    subprocess.check_call(common_args + ["--mode", args.candidate_mode, "--result-json", str(candidate_json)])

    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_json.read_text(encoding="utf-8"))
    base_tps = baseline["aggregate"]["output_tokens_per_second"]
    candidate_tps = candidate["aggregate"]["output_tokens_per_second"]
    total_base_tps = baseline["aggregate"]["total_tokens_per_second"]
    total_candidate_tps = candidate["aggregate"]["total_tokens_per_second"]
    pair_summary = {
        "candidate_mode": args.candidate_mode,
        "baseline": baseline,
        args.candidate_mode: candidate,
        "output_tps_speedup": (candidate_tps / base_tps) if base_tps not in (None, 0) else None,
        "total_tps_speedup": (
            total_candidate_tps / total_base_tps
            if total_base_tps not in (None, 0)
            else None
        ),
        "compression_events_delta": (
            candidate["aggregate"]["compression_events"]
            - baseline["aggregate"]["compression_events"]
        ),
    }
    return pair_summary


def main() -> None:
    args = parse_args()
    validate_stats_args(args)
    if args.mode == "pair":
        payload = run_pair(args)
    else:
        payload = run_single_mode(args)

    output_path = args.result_json or default_result_path(args, args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result_json": str(output_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
