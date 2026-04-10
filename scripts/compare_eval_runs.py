#!/usr/bin/env python3
"""Compare evaluation outputs from two experiment runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline eval dir or metrics json path.")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate eval dir or metrics json path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000, help="Number of paired bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap resampling.")
    return parser.parse_args()


def find_single(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.rglob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {pattern} under {path}: {matches}")
    return matches[0]


def resolve_metrics_path(path: Path) -> Path:
    if path.is_file():
        return path
    metrics = find_single(path, "*_metrics.json")
    if metrics is None:
        raise FileNotFoundError(f"No *_metrics.json found under {path}")
    return metrics


def resolve_eval_jsonl_path(path: Path) -> Path | None:
    if path.is_file():
        sibling = path.with_name(path.name.replace("_cot_metrics.json", ".jsonl"))
        return sibling if sibling.exists() else None
    return find_single(path, "*_eval.jsonl")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def load_question_scores(path: Path | None) -> Dict[int, float]:
    if path is None:
        return {}
    scores: Dict[int, float] = {}
    for item in load_jsonl(path):
        idx = item.get("idx")
        if idx is None:
            continue
        pass_at_1 = item.get("pass_at_1")
        if pass_at_1 is not None:
            scores[int(idx)] = float(pass_at_1)
            continue
        score = item.get("score")
        if isinstance(score, list) and score:
            scores[int(idx)] = float(any(bool(v) for v in score))
    return scores


def compare_question_scores(
    baseline_scores: Dict[int, float],
    candidate_scores: Dict[int, float],
) -> Tuple[int, int, int, int]:
    improved = 0
    regressed = 0
    unchanged = 0
    compared = 0
    for idx in sorted(set(baseline_scores) & set(candidate_scores)):
        compared += 1
        base = baseline_scores[idx]
        cand = candidate_scores[idx]
        if cand > base:
            improved += 1
        elif cand < base:
            regressed += 1
        else:
            unchanged += 1
    return compared, improved, regressed, unchanged


def paired_bootstrap_delta_ci(
    baseline_scores: Dict[int, float],
    candidate_scores: Dict[int, float],
    *,
    samples: int,
    seed: int,
) -> dict | None:
    shared_ids = sorted(set(baseline_scores) & set(candidate_scores))
    if not shared_ids:
        return None

    baseline = np.array([baseline_scores[idx] for idx in shared_ids], dtype=np.float64)
    candidate = np.array([candidate_scores[idx] for idx in shared_ids], dtype=np.float64)
    deltas = candidate - baseline
    observed = float(deltas.mean())

    rng = np.random.default_rng(seed)
    resampled = np.empty(samples, dtype=np.float64)
    n = len(deltas)
    for i in range(samples):
        indices = rng.integers(0, n, size=n)
        resampled[i] = deltas[indices].mean()

    ci_low, ci_high = np.percentile(resampled, [2.5, 97.5])
    p_nonpositive = float((resampled <= 0.0).mean())
    p_nonnegative = float((resampled >= 0.0).mean())
    return {
        "questions": n,
        "observed_delta": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_nonpositive": p_nonpositive,
        "p_nonnegative": p_nonnegative,
    }


def main() -> None:
    args = parse_args()
    baseline_metrics_path = resolve_metrics_path(args.baseline)
    candidate_metrics_path = resolve_metrics_path(args.candidate)
    baseline_metrics = load_json(baseline_metrics_path)
    candidate_metrics = load_json(candidate_metrics_path)

    baseline_acc = float(baseline_metrics.get("acc", 0.0))
    candidate_acc = float(candidate_metrics.get("acc", 0.0))
    acc_delta = candidate_acc - baseline_acc

    baseline_eval_jsonl = resolve_eval_jsonl_path(args.baseline)
    candidate_eval_jsonl = resolve_eval_jsonl_path(args.candidate)
    baseline_scores = load_question_scores(baseline_eval_jsonl)
    candidate_scores = load_question_scores(candidate_eval_jsonl)
    compared, improved, regressed, unchanged = compare_question_scores(baseline_scores, candidate_scores)
    bootstrap = paired_bootstrap_delta_ci(
        baseline_scores,
        candidate_scores,
        samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )

    summary = {
        "baseline_metrics": str(baseline_metrics_path.resolve()),
        "candidate_metrics": str(candidate_metrics_path.resolve()),
        "baseline_acc": baseline_acc,
        "candidate_acc": candidate_acc,
        "acc_delta": acc_delta,
        "baseline_num_scores": baseline_metrics.get("num_scores"),
        "candidate_num_scores": candidate_metrics.get("num_scores"),
        "questions_compared": compared,
        "questions_improved": improved,
        "questions_regressed": regressed,
        "questions_unchanged": unchanged,
        "paired_bootstrap": bootstrap,
    }

    print(f"baseline_acc={baseline_acc:.4f}")
    print(f"candidate_acc={candidate_acc:.4f}")
    print(f"acc_delta={acc_delta:+.4f}")
    print(f"questions_compared={compared}")
    print(f"questions_improved={improved}")
    print(f"questions_regressed={regressed}")
    print(f"questions_unchanged={unchanged}")
    if bootstrap is not None:
        print(
            "paired_bootstrap_ci95="
            f"[{bootstrap['ci95_low']:+.4f}, {bootstrap['ci95_high']:+.4f}]"
        )
        print(f"paired_bootstrap_p_nonpositive={bootstrap['p_nonpositive']:.4f}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
