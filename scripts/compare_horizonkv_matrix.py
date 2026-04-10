#!/usr/bin/env python3
"""Compare baseline and HorizonKV paper-matrix outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare_eval_runs import (
    compare_question_scores,
    load_json,
    load_question_scores,
    paired_bootstrap_delta_ci,
    resolve_eval_jsonl_path,
    resolve_metrics_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-aime24", type=Path, required=True)
    parser.add_argument("--candidate-aime24", type=Path, required=True)
    parser.add_argument("--baseline-aime25", type=Path, required=True)
    parser.add_argument("--candidate-aime25", type=Path, required=True)
    parser.add_argument("--baseline-math500", type=Path, required=True)
    parser.add_argument("--candidate-math500", type=Path, required=True)
    parser.add_argument("--baseline-dfs", type=Path, required=True)
    parser.add_argument("--candidate-dfs", type=Path, required=True)
    parser.add_argument("--baseline-longbench", type=Path, required=True)
    parser.add_argument("--candidate-longbench", type=Path, required=True)
    parser.add_argument("--baseline-ruler", type=Path, default=None)
    parser.add_argument("--candidate-ruler", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def find_single(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.rglob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {pattern} under {path}: {matches}")
    return matches[0]


def resolve_named_json(path: Path, filename: str) -> Path:
    if path.is_file():
        return path
    match = find_single(path, filename)
    if match is None:
        raise FileNotFoundError(f"No {filename} found under {path}")
    return match


def compare_reasoning_run(
    baseline: Path,
    candidate: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_metrics_path = resolve_metrics_path(baseline)
    candidate_metrics_path = resolve_metrics_path(candidate)
    baseline_metrics = load_json(baseline_metrics_path)
    candidate_metrics = load_json(candidate_metrics_path)

    baseline_acc = float(baseline_metrics.get("acc", 0.0))
    candidate_acc = float(candidate_metrics.get("acc", 0.0))
    acc_delta = candidate_acc - baseline_acc

    baseline_scores = load_question_scores(resolve_eval_jsonl_path(baseline))
    candidate_scores = load_question_scores(resolve_eval_jsonl_path(candidate))
    compared, improved, regressed, unchanged = compare_question_scores(
        baseline_scores,
        candidate_scores,
    )
    bootstrap = paired_bootstrap_delta_ci(
        baseline_scores,
        candidate_scores,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "baseline_acc": baseline_acc,
        "candidate_acc": candidate_acc,
        "acc_delta": acc_delta,
        "questions_compared": compared,
        "questions_improved": improved,
        "questions_regressed": regressed,
        "questions_unchanged": unchanged,
        "paired_bootstrap": bootstrap,
        "baseline_metrics_path": str(baseline_metrics_path.resolve()),
        "candidate_metrics_path": str(candidate_metrics_path.resolve()),
    }


def compare_scalar_json(
    baseline: Path,
    candidate: Path,
    *,
    filename: str,
    value_path: tuple[str, ...],
) -> dict[str, Any]:
    baseline_path = resolve_named_json(baseline, filename)
    candidate_path = resolve_named_json(candidate, filename)
    baseline_json = load_json(baseline_path)
    candidate_json = load_json(candidate_path)

    baseline_value: Any = baseline_json
    candidate_value: Any = candidate_json
    for key in value_path:
        baseline_value = baseline_value[key]
        candidate_value = candidate_value[key]
    baseline_value = float(baseline_value)
    candidate_value = float(candidate_value)
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": candidate_value - baseline_value,
        "baseline_path": str(baseline_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
    }


def compare_ruler(
    baseline: Path,
    candidate: Path,
) -> dict[str, Any]:
    baseline_path = resolve_named_json(baseline, "ruler_lengths_summary.json")
    candidate_path = resolve_named_json(candidate, "ruler_lengths_summary.json")
    baseline_json = load_json(baseline_path)
    candidate_json = load_json(candidate_path)
    shared_lengths = sorted(set(baseline_json) & set(candidate_json), key=lambda x: int(x))
    length_deltas: dict[str, float] = {}
    baseline_values = []
    candidate_values = []
    for length in shared_lengths:
        base = float(baseline_json[length]["overall_average"])
        cand = float(candidate_json[length]["overall_average"])
        baseline_values.append(base)
        candidate_values.append(cand)
        length_deltas[length] = cand - base
    baseline_mean = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0
    candidate_mean = sum(candidate_values) / len(candidate_values) if candidate_values else 0.0
    return {
        "baseline": baseline_mean,
        "candidate": candidate_mean,
        "delta": candidate_mean - baseline_mean,
        "length_deltas": length_deltas,
        "baseline_path": str(baseline_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    reasoning = {
        "aime24": compare_reasoning_run(
            args.baseline_aime24,
            args.candidate_aime24,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "aime25": compare_reasoning_run(
            args.baseline_aime25,
            args.candidate_aime25,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "math500": compare_reasoning_run(
            args.baseline_math500,
            args.candidate_math500,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    reasoning_mean_delta = sum(item["acc_delta"] for item in reasoning.values()) / len(reasoning)
    clear_win = (
        reasoning["aime24"]["acc_delta"] >= 2.0
        or reasoning["aime25"]["acc_delta"] >= 2.0
        or reasoning["math500"]["acc_delta"] >= 1.0
    )

    dfs = compare_scalar_json(
        args.baseline_dfs,
        args.candidate_dfs,
        filename="dfs_summary.json",
        value_path=("rates", "fully_correct"),
    )
    longbench = compare_scalar_json(
        args.baseline_longbench,
        args.candidate_longbench,
        filename="longbench_eval.json",
        value_path=("overall_average",),
    )
    ruler = None
    if args.baseline_ruler is not None and args.candidate_ruler is not None:
        ruler = compare_ruler(args.baseline_ruler, args.candidate_ruler)

    summary = {
        "reasoning": reasoning,
        "reasoning_mean_delta": reasoning_mean_delta,
        "dfs": dfs,
        "longbench": longbench,
        "ruler": ruler,
        "success_criteria": {
            "reasoning_mean_positive": reasoning_mean_delta > 0.0,
            "clear_win": clear_win,
            "longbench_non_regression": longbench["delta"] >= -0.5,
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
