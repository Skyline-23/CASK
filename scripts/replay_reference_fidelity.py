"""Replay reference continuations under a candidate KV policy and measure fidelity.

This script takes a completed reference run (typically FullKV or a high-budget
reference policy), replays each reference continuation token-by-token under a
candidate method, and reports:

- teacher-forced next-token agreement against the reference continuation
- teacher-forced target log-prob / NLL / perplexity
- first mismatch / strict prefix agreement
- terminal KV savings observed along the exact reference continuation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compare_experiment_runs import load_jsonl, resolve_merged_jsonl_path
from cask.integration.monkeypatch import replace_llama, replace_qwen2, replace_qwen3
from worker import (
    build_cask_phase_marker_token_ids,
    configure_tokenizer,
    resolve_torch_dtype,
    resolve_under_rkv,
    set_seed,
    str2bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher-force a reference run under a candidate KV policy."
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Reference run root or merged.jsonl path.",
    )
    parser.add_argument("--model-path", required=True, help="HF model path or name.")
    parser.add_argument(
        "--method",
        required=True,
        choices=["fullkv", "triattention", "horizonkv", "cask", "r1kv", "snapkv", "expectedattention"],
        help="Candidate method to replay under.",
    )
    parser.add_argument("--budget", type=int, default=None, help="KV budget for candidate method.")
    parser.add_argument(
        "--triattention-stats-file",
        dest="triattention_stats_file",
        type=str,
        default=None,
        help="Stats file for cask/horizonkv/cask.",
    )
    parser.add_argument("--load-dtype", default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV per-record report path.",
    )

    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--first-tokens", type=int, default=4)
    parser.add_argument("--mix-lambda", type=float, default=0.1)
    parser.add_argument("--retain-ratio", type=float, default=0.2)
    parser.add_argument("--update-kv", type=str2bool, default=True)
    parser.add_argument("--fp32-topk", type=str2bool, default=False)
    parser.add_argument("--protect-prefill", type=str2bool, default=False)
    parser.add_argument(
        "--retain-direction",
        choices=["last", "first"],
        default="last",
    )
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--slack-budget-trigger", type=str2bool, default=False)
    parser.add_argument("--count-prompt-tokens", type=str2bool, default=False)
    parser.add_argument("--allow-prefill-compression", type=str2bool, default=False)
    parser.add_argument("--disable-mlr", type=str2bool, default=False)
    parser.add_argument("--disable-trig", type=str2bool, default=False)
    parser.add_argument(
        "--triattention-score-aggregation",
        choices=["sum", "mean"],
        default="mean",
    )
    parser.add_argument("--triattention-frequency-window", type=int, default=65536)
    parser.add_argument(
        "--triattention-normalize-scores",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--triattention-horizon-mode",
        choices=["fixed", "adaptive", "variational"],
        default="fixed",
    )
    parser.add_argument(
        "--triattention-norm-mode",
        choices=["tri", "rms2", "mad2"],
        default="tri",
    )
    parser.add_argument("--triattention-kernel-c-lambda", type=float, default=1.0)
    parser.add_argument("--triattention-kernel-s0", type=float, default=1.0)
    parser.add_argument("--triattention-kernel-s1", type=float, default=1.0)
    parser.add_argument("--triattention-norm-lambda", type=float, default=1.0)

    parser.add_argument("--cask-protected-core-ratio", type=float, default=0.5)
    parser.add_argument("--cask-prefix-coverage-ratio", type=float, default=0.0625)
    parser.add_argument("--cask-decode-merge-enabled", type=str2bool, default=True)
    parser.add_argument("--cask-min-protected-core-tokens", type=int, default=1)
    parser.add_argument(
        "--cask-core-selection-mode",
        choices=["vote", "score"],
        default="vote",
    )
    parser.add_argument(
        "--cask-merge-operator",
        choices=["keepkv", "mean"],
        default="keepkv",
    )
    parser.add_argument("--cask-merge-local-window", type=int, default=32)
    parser.add_argument("--cask-similarity-threshold", type=float, default=0.985)
    parser.add_argument("--cask-value-projection-threshold", type=float, default=None)
    parser.add_argument(
        "--cask-representative-mode",
        choices=["weighted_latest", "score_max_source"],
        default="score_max_source",
    )
    parser.add_argument("--cask-promotion-score-ratio", type=float, default=None)
    parser.add_argument("--cask-merge-score-mass-ratio-threshold", type=float, default=None)
    parser.add_argument("--cask-use-phase-markers", type=str2bool, default=True)
    parser.add_argument("--expectedattention-n-future-positions", type=int, default=512)
    parser.add_argument("--expectedattention-n-sink", type=int, default=4)
    parser.add_argument("--expectedattention-use-covariance", type=str2bool, default=True)
    parser.add_argument("--expectedattention-use-vnorm", type=str2bool, default=True)
    parser.add_argument("--expectedattention-epsilon", type=float, default=0.0)
    return parser.parse_args()


def load_reference_records(reference: Path, max_records: int | None) -> tuple[Path, list[dict[str, Any]]]:
    merged_path = resolve_merged_jsonl_path(reference)
    if merged_path is None:
        raise FileNotFoundError(f"Could not resolve merged.jsonl under reference path: {reference}")
    records = list(load_jsonl(merged_path))
    if max_records is not None:
        records = records[: max_records]
    return merged_path, records


def apply_candidate_method(
    model,
    tokenizer,
    args: argparse.Namespace,
) -> None:
    method = args.method.lower()
    if method == "fullkv":
        return

    if args.budget is None:
        raise ValueError(f"--budget is required for {method}.")

    local_model_path = Path(args.model_path)
    model_path_value: str | Path
    if local_model_path.exists():
        model_path_value = local_model_path
    else:
        model_path_value = args.model_path.replace("\\", "/")

    if method in {"triattention", "horizonkv", "cask"}:
        if args.triattention_stats_file is None:
            raise ValueError(f"--triattention-stats-file is required for {method}.")
        stats_path = resolve_under_rkv(args.triattention_stats_file)
        if not stats_path.exists():
            raise FileNotFoundError(f"TriAttention stats file not found: {stats_path}")
    else:
        stats_path = None

    if method == "cask":
        from cask.methods.cask import apply_cask_patch

        phase_marker_token_ids = (
            build_cask_phase_marker_token_ids(tokenizer)
            if args.cask_use_phase_markers
            else ()
        )
        apply_cask_patch(
            model,
            stats_path=stats_path,
            model_path=model_path_value,
            kv_budget=int(args.budget),
            offset_max_length=args.triattention_frequency_window,
            score_aggregation=args.triattention_score_aggregation,
            pruning_seed=args.seed,
            metadata_expectations={},
            normalize_scores=args.triattention_normalize_scores,
            count_prompt_tokens=args.count_prompt_tokens,
            allow_prefill_compression=args.allow_prefill_compression,
            divide_length=args.divide_length,
            use_slack_trigger=args.slack_budget_trigger,
            disable_mlr=args.disable_mlr,
            disable_trig=args.disable_trig,
            prefix_coverage_ratio=args.cask_prefix_coverage_ratio,
            decode_merge_enabled=args.cask_decode_merge_enabled,
            recent_window_size=args.window_size,
            protected_core_ratio=args.cask_protected_core_ratio,
            min_protected_core_tokens=args.cask_min_protected_core_tokens,
            core_selection_mode=args.cask_core_selection_mode,
            merge_operator=args.cask_merge_operator,
            merge_local_window=args.cask_merge_local_window,
            merge_similarity_threshold=args.cask_similarity_threshold,
            value_projection_threshold=args.cask_value_projection_threshold,
            representative_mode=args.cask_representative_mode,
            promotion_score_ratio=args.cask_promotion_score_ratio,
            merge_score_mass_ratio_threshold=args.cask_merge_score_mass_ratio_threshold,
            use_phase_markers=args.cask_use_phase_markers,
            phase_marker_token_ids=phase_marker_token_ids,
            score_dump_dir=None,
            score_dump_max_events=None,
        )
        return

    if method in {"triattention", "horizonkv"}:
        from cask.methods.triattention import apply_triattention_patch

        apply_triattention_patch(
            model,
            stats_path=stats_path,
            model_path=model_path_value,
            kv_budget=int(args.budget),
            offset_max_length=args.triattention_frequency_window,
            score_aggregation=args.triattention_score_aggregation,
            pruning_seed=args.seed,
            metadata_expectations={},
            normalize_scores=args.triattention_normalize_scores,
            count_prompt_tokens=args.count_prompt_tokens,
            allow_prefill_compression=args.allow_prefill_compression,
            divide_length=args.divide_length,
            use_slack_trigger=args.slack_budget_trigger,
            disable_mlr=args.disable_mlr,
            disable_trig=args.disable_trig,
            horizon_mode=args.triattention_horizon_mode,
            norm_mode=args.triattention_norm_mode,
            kernel_c_lambda=args.triattention_kernel_c_lambda,
            kernel_s0=args.triattention_kernel_s0,
            kernel_s1=args.triattention_kernel_s1,
            norm_lambda=args.triattention_norm_lambda,
            score_dump_dir=None,
            score_dump_max_events=None,
        )
        return

    if method == "expectedattention":
        method_config: Dict[str, Any] = {
            "budget": int(args.budget),
            "window_size": args.window_size,
            "n_future_positions": args.expectedattention_n_future_positions,
            "n_sink": args.expectedattention_n_sink,
            "use_covariance": args.expectedattention_use_covariance,
            "use_vnorm": args.expectedattention_use_vnorm,
            "epsilon": args.expectedattention_epsilon,
            "protect_prefill": args.protect_prefill,
            "model_path": model_path_value,
        }
    else:
        method_config = {
            "budget": int(args.budget),
            "window_size": args.window_size,
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "first_tokens": args.first_tokens,
            "fp32_topk": args.fp32_topk,
            "protect_prefill": args.protect_prefill,
        }

    compression_config = {
        "method": method,
        "method_config": method_config,
        "compression": None,
        "update_kv": args.update_kv,
    }
    model_path_lower = str(args.model_path).lower()
    if "llama" in model_path_lower:
        replace_llama(compression_config)
    elif "qwen3" in model_path_lower:
        replace_qwen3(compression_config)
    elif "qwen" in model_path_lower:
        replace_qwen2(compression_config)
    else:
        raise ValueError(f"Unsupported model family for {method}: {args.model_path}")


def get_active_compressor(model):
    compressor = getattr(model, "_triattention_compressor", None)
    if compressor is None:
        compressor = getattr(model, "_cask_compressor", None)
    return compressor


def get_runtime_summary(
    model,
    *,
    total_reference_tokens: int,
    past_key_values=None,
) -> Dict[str, Any]:
    # At the final teacher-forced scoring step, the KV cache contains the replay
    # prefix plus all previously-consumed target tokens, but not the final target
    # token itself. Use this active horizon consistently for saved-ratio accounting.
    active_reference_tokens = max(0, int(total_reference_tokens) - 1)
    compressor = get_active_compressor(model)
    if compressor is None:
        cache_tokens = None
        if past_key_values is not None:
            for cache_attr in ("key_cache",):
                cache_obj = getattr(past_key_values, cache_attr, None)
                if isinstance(cache_obj, list) and cache_obj:
                    layer_cache = cache_obj[0]
                    if layer_cache is not None:
                        cache_tokens = int(layer_cache.shape[-2])
                        break
            if cache_tokens is None and isinstance(past_key_values, (tuple, list)) and past_key_values:
                first_layer = past_key_values[0]
                if isinstance(first_layer, (tuple, list)) and first_layer:
                    first_key = first_layer[0]
                    if first_key is not None:
                        cache_tokens = int(first_key.shape[-2])
        if cache_tokens is None:
            cache_tokens = int(active_reference_tokens)
        reference_saved_tokens = 0
        return {
            "compression_events": None,
            "current_cache_tokens": int(cache_tokens),
            "current_total_cardinality": int(active_reference_tokens),
            "terminal_saved_tokens": max(0, int(active_reference_tokens) - int(cache_tokens)),
            "terminal_saved_ratio": (
                float(max(0, int(active_reference_tokens) - int(cache_tokens)) / active_reference_tokens)
                if active_reference_tokens > 0
                else 0.0
            ),
            "terminal_cache_ratio": (
                float(int(cache_tokens) / active_reference_tokens) if active_reference_tokens > 0 else 1.0
            ),
            "reference_terminal_saved_tokens": max(0, int(active_reference_tokens) - int(cache_tokens)),
            "reference_terminal_saved_ratio": (
                float(max(0, int(active_reference_tokens) - int(cache_tokens)) / active_reference_tokens)
                if active_reference_tokens > 0
                else 0.0
            ),
            "reference_terminal_cache_ratio": (
                float(int(cache_tokens) / active_reference_tokens) if active_reference_tokens > 0 else 1.0
            ),
        }

    if hasattr(compressor, "get_runtime_summary"):
        summary = dict(compressor.get_runtime_summary())
        cache_tokens = int(summary.get("current_cache_tokens", total_reference_tokens))
        total_cardinality = int(summary.get("current_total_cardinality", total_reference_tokens))
    else:
        cache_tokens = int(len(getattr(compressor, "cache_positions", [])))
        total_cardinality = int(active_reference_tokens)
        summary = {
            "compression_events": None,
            "current_cache_tokens": cache_tokens,
            "current_total_cardinality": total_cardinality,
        }

    if total_cardinality <= 0:
        total_cardinality = int(active_reference_tokens)
    saved_tokens = max(0, total_cardinality - cache_tokens)
    summary["terminal_saved_tokens"] = int(saved_tokens)
    summary["terminal_saved_ratio"] = float(saved_tokens / total_cardinality) if total_cardinality > 0 else 0.0
    summary["terminal_cache_ratio"] = float(cache_tokens / total_cardinality) if total_cardinality > 0 else 1.0
    reference_saved_tokens = max(0, int(active_reference_tokens) - cache_tokens)
    summary["reference_terminal_saved_tokens"] = int(reference_saved_tokens)
    summary["reference_terminal_saved_ratio"] = (
        float(reference_saved_tokens / active_reference_tokens) if active_reference_tokens > 0 else 0.0
    )
    summary["reference_terminal_cache_ratio"] = (
        float(cache_tokens / active_reference_tokens) if active_reference_tokens > 0 else 1.0
    )
    return summary


def encode_reference_continuation(
    tokenizer,
    prompt: str,
    reference_output: str,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    full_ids = tokenizer(
        prompt + reference_output,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"][0]
    used_combined_boundary = False
    continuation_ids: torch.Tensor
    if (
        full_ids.numel() >= prompt_ids.numel()
        and torch.equal(full_ids[: prompt_ids.numel()], prompt_ids)
    ):
        continuation_ids = full_ids[prompt_ids.numel() :]
        used_combined_boundary = True
    else:
        continuation_ids = tokenizer(
            reference_output,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
    return prompt_ids.to(device), continuation_ids.to(device), used_combined_boundary


@torch.inference_mode()
def replay_record(
    model,
    tokenizer,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = str(record.get("prompt", ""))
    reference_output = str(record.get("output", ""))
    record_index = record.get("index", record.get("sample_idx"))
    record_id = record.get("id", record_index)

    device = next(model.parameters()).device
    prompt_ids, continuation_ids, used_combined_boundary = encode_reference_continuation(
        tokenizer,
        prompt,
        reference_output,
        device=device,
    )

    if continuation_ids.numel() == 0:
        total_reference_tokens = int(prompt_ids.numel())
        runtime_summary = get_runtime_summary(model, total_reference_tokens=total_reference_tokens)
        return {
            "index": record_index,
            "id": record_id,
            "reference_output_tokens": 0,
            "used_combined_boundary_tokenization": used_combined_boundary,
            "target_top1_match_rate": None,
            "target_top5_match_rate": None,
            "strict_prefix_top1_ratio": None,
            "first_top1_mismatch_step": None,
            "mean_target_logprob": None,
            "mean_target_nll": None,
            "perplexity": None,
            **runtime_summary,
        }

    prefill_outputs = model(
        input_ids=prompt_ids.unsqueeze(0),
        use_cache=True,
        return_dict=True,
    )
    past_key_values = prefill_outputs.past_key_values
    next_token_logits = prefill_outputs.logits[:, -1, :]

    top1_matches = 0
    top5_matches = 0
    strict_prefix_len = 0
    first_mismatch_step: int | None = None
    logprob_values: list[float] = []

    continuation_cpu = continuation_ids.detach().to("cpu")
    for step_idx in range(int(continuation_ids.numel())):
        target_token = continuation_ids[step_idx : step_idx + 1]
        target_token_id = int(target_token.item())
        logits_last = next_token_logits[0]
        log_probs = torch.log_softmax(logits_last, dim=-1)
        logprob_values.append(float(log_probs[target_token_id].item()))

        pred_top1 = int(torch.argmax(logits_last).item())
        if pred_top1 == target_token_id:
            top1_matches += 1
            if first_mismatch_step is None:
                strict_prefix_len += 1
        elif first_mismatch_step is None:
            first_mismatch_step = step_idx

        top5 = torch.topk(logits_last, k=min(5, logits_last.shape[-1]), dim=-1).indices
        if bool((top5 == target_token_id).any().item()):
            top5_matches += 1

        if step_idx == int(continuation_ids.numel()) - 1:
            break

        step_outputs = model(
            input_ids=target_token.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = step_outputs.past_key_values
        next_token_logits = step_outputs.logits[:, -1, :]

    total_reference_tokens = int(prompt_ids.numel() + continuation_ids.numel())
    runtime_summary = get_runtime_summary(model, total_reference_tokens=total_reference_tokens)

    total_steps = int(continuation_ids.numel())
    mean_logprob = mean(logprob_values)
    mean_nll = -mean_logprob
    perplexity = math.exp(mean_nll) if mean_nll < 20 else float("inf")
    if first_mismatch_step is None:
        first_mismatch_step = total_steps

    return {
        "index": record_index,
        "id": record_id,
        "reference_output_tokens": total_steps,
        "used_combined_boundary_tokenization": used_combined_boundary,
        "target_top1_match_rate": float(top1_matches / total_steps),
        "target_top5_match_rate": float(top5_matches / total_steps),
        "strict_prefix_top1_ratio": float(strict_prefix_len / total_steps),
        "first_top1_mismatch_step": int(first_mismatch_step),
        "mean_target_logprob": float(mean_logprob),
        "mean_target_nll": float(mean_nll),
        "perplexity": float(perplexity),
        "reference_prompt_tokens": int(prompt_ids.numel()),
        "reference_total_tokens": int(total_reference_tokens),
        **runtime_summary,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    def _collect(key: str) -> list[float]:
        values: list[float] = []
        for item in results:
            value = item.get(key)
            if value is None:
                continue
            values.append(float(value))
        return values

    def _mean_or_none(key: str) -> float | None:
        values = _collect(key)
        if not values:
            return None
        return float(mean(values))

    guard_reason_counts: dict[str, int] = {}
    for item in results:
        reason_counts = item.get("guard_reason_counts")
        if isinstance(reason_counts, dict):
            for key, value in reason_counts.items():
                try:
                    guard_reason_counts[str(key)] = guard_reason_counts.get(str(key), 0) + int(value)
                except Exception:
                    continue

    return {
        "records_compared": len(results),
        "mean_target_top1_match_rate": _mean_or_none("target_top1_match_rate"),
        "mean_target_top5_match_rate": _mean_or_none("target_top5_match_rate"),
        "mean_strict_prefix_top1_ratio": _mean_or_none("strict_prefix_top1_ratio"),
        "mean_first_top1_mismatch_step": _mean_or_none("first_top1_mismatch_step"),
        "mean_target_logprob": _mean_or_none("mean_target_logprob"),
        "mean_target_nll": _mean_or_none("mean_target_nll"),
        "mean_perplexity": _mean_or_none("perplexity"),
        "mean_terminal_saved_tokens": _mean_or_none("terminal_saved_tokens"),
        "mean_terminal_saved_ratio": _mean_or_none("terminal_saved_ratio"),
        "mean_terminal_cache_ratio": _mean_or_none("terminal_cache_ratio"),
        "mean_reference_terminal_saved_tokens": _mean_or_none("reference_terminal_saved_tokens"),
        "mean_reference_terminal_saved_ratio": _mean_or_none("reference_terminal_saved_ratio"),
        "mean_reference_terminal_cache_ratio": _mean_or_none("reference_terminal_cache_ratio"),
        "records_with_positive_terminal_savings": sum(
            1 for item in results if float(item.get("terminal_saved_tokens", 0) or 0) > 0
        ),
        "records_with_positive_reference_terminal_savings": sum(
            1 for item in results if float(item.get("reference_terminal_saved_tokens", 0) or 0) > 0
        ),
        "records_with_guard_triggered": sum(
            1 for item in results if bool(item.get("guard_triggered"))
        ),
        "guard_reason_counts": guard_reason_counts,
        "mean_compression_events": _mean_or_none("compression_events"),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    reference_merged_path, reference_records = load_reference_records(args.reference, args.max_records)

    dtype = resolve_torch_dtype(args.load_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, padding_side="left")
    tokenizer = configure_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    apply_candidate_method(model, tokenizer, args)

    results: list[dict[str, Any]] = []
    for record in reference_records:
        results.append(replay_record(model, tokenizer, record))

    summary = summarize_results(results)
    report = {
        "reference_merged_jsonl": str(reference_merged_path.resolve()),
        "candidate_method": args.method,
        "candidate_budget": args.budget,
        "model_path": args.model_path,
        "summary": summary,
        "records": results,
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.csv_output is not None:
        write_csv(args.csv_output, results)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

