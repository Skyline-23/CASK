#!/usr/bin/env python3
"""Compare accuracy and throughput for two experiment runs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Baseline run root, merged jsonl, eval dir, or metrics json.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Candidate run root, merged jsonl, eval dir, or metrics json.",
    )
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


def find_single(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.rglob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {pattern} under {path}: {matches}")
    return matches[0]


def resolve_metrics_path(path: Path) -> Path | None:
    if path.is_file():
        if path.name.endswith("_metrics.json"):
            return path
        return None
    direct = path / "eval"
    if direct.exists():
        metrics = find_single(direct, "*_metrics.json")
        if metrics is not None:
            return metrics
    return find_single(path, "*_metrics.json")


def resolve_eval_jsonl_path(path: Path) -> Path | None:
    if path.is_file():
        if path.name.endswith("_eval.jsonl"):
            return path
        return None
    direct = path / "eval"
    if direct.exists():
        eval_jsonl = find_single(direct, "*_eval.jsonl")
        if eval_jsonl is not None:
            return eval_jsonl
    return find_single(path, "*_eval.jsonl")


def resolve_merged_jsonl_path(path: Path) -> Path | None:
    if path.is_file():
        if path.suffix == ".jsonl":
            return path
        return None
    direct = path / "merged" / "merged.jsonl"
    if direct.exists():
        return direct
    alt = path / "merged.jsonl"
    if alt.exists():
        return alt
    return find_single(path, "merged.jsonl")


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
) -> Dict[str, int]:
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
    return {
        "questions_compared": compared,
        "questions_improved": improved,
        "questions_regressed": regressed,
        "questions_unchanged": unchanged,
    }


def summarize_records(path: Path | None) -> dict:
    if path is None:
        return {
            "merged_jsonl": None,
            "records": 0,
            "timed_records": 0,
            "prefill_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "generation_seconds": 0.0,
            "output_tokens_per_second": None,
            "total_tokens_per_second": None,
            "median_output_tokens_per_second": None,
            "median_total_tokens_per_second": None,
        }

    records = 0
    timed_records = 0
    prefill_tokens = 0
    output_tokens = 0
    total_tokens = 0
    generation_seconds = 0.0
    output_tps_values: List[float] = []
    total_tps_values: List[float] = []

    for item in load_jsonl(path):
        records += 1
        prefill = int(item.get("prefill_tokens", 0) or 0)
        output = int(item.get("output_tokens", 0) or 0)
        total = int(item.get("total_tokens", 0) or 0)
        prefill_tokens += prefill
        output_tokens += output
        total_tokens += total

        seconds = item.get("generation_seconds")
        if seconds is None:
            continue
        seconds_value = float(seconds)
        if seconds_value <= 0:
            continue
        timed_records += 1
        generation_seconds += seconds_value
        output_tps_values.append(output / seconds_value)
        total_tps_values.append(total / seconds_value)

    output_tps = None
    total_tps = None
    if generation_seconds > 0:
        output_tps = output_tokens / generation_seconds
        total_tps = total_tokens / generation_seconds

    return {
        "merged_jsonl": str(path.resolve()),
        "records": records,
        "timed_records": timed_records,
        "prefill_tokens": prefill_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "generation_seconds": generation_seconds,
        "output_tokens_per_second": output_tps,
        "total_tokens_per_second": total_tps,
        "median_output_tokens_per_second": statistics.median(output_tps_values) if output_tps_values else None,
        "median_total_tokens_per_second": statistics.median(total_tps_values) if total_tps_values else None,
    }


def summarize_run(path: Path) -> dict:
    metrics_path = resolve_metrics_path(path)
    eval_jsonl_path = resolve_eval_jsonl_path(path)
    merged_jsonl_path = resolve_merged_jsonl_path(path)

    metrics = load_json(metrics_path) if metrics_path is not None else {}
    records = summarize_records(merged_jsonl_path)
    question_scores = load_question_scores(eval_jsonl_path)

    return {
        "input_path": str(path.resolve()),
        "metrics_path": str(metrics_path.resolve()) if metrics_path is not None else None,
        "eval_jsonl_path": str(eval_jsonl_path.resolve()) if eval_jsonl_path is not None else None,
        "accuracy": float(metrics.get("acc", 0.0)) if metrics else None,
        "num_scores": metrics.get("num_scores"),
        "question_scores": question_scores,
        "records": records,
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    baseline = summarize_run(args.baseline)
    candidate = summarize_run(args.candidate)

    question_compare = compare_question_scores(
        baseline["question_scores"],
        candidate["question_scores"],
    )

    baseline_acc = baseline["accuracy"]
    candidate_acc = candidate["accuracy"]
    acc_delta = None
    if baseline_acc is not None and candidate_acc is not None:
        acc_delta = candidate_acc - baseline_acc

    baseline_output_tps = baseline["records"]["output_tokens_per_second"]
    candidate_output_tps = candidate["records"]["output_tokens_per_second"]
    baseline_total_tps = baseline["records"]["total_tokens_per_second"]
    candidate_total_tps = candidate["records"]["total_tokens_per_second"]

    summary = {
        "baseline": baseline,
        "candidate": candidate,
        "acc_delta": acc_delta,
        "output_tps_speedup": ratio(candidate_output_tps, baseline_output_tps),
        "total_tps_speedup": ratio(candidate_total_tps, baseline_total_tps),
        **question_compare,
    }

    print(f"baseline_acc={fmt_float(baseline_acc)}")
    print(f"candidate_acc={fmt_float(candidate_acc)}")
    print(f"acc_delta={fmt_float(acc_delta, digits=4) if acc_delta is not None else 'n/a'}")
    print(f"baseline_output_tps={fmt_float(baseline_output_tps, digits=2)}")
    print(f"candidate_output_tps={fmt_float(candidate_output_tps, digits=2)}")
    print(f"output_tps_speedup={fmt_float(summary['output_tps_speedup'], digits=3)}")
    print(f"baseline_total_tps={fmt_float(baseline_total_tps, digits=2)}")
    print(f"candidate_total_tps={fmt_float(candidate_total_tps, digits=2)}")
    print(f"total_tps_speedup={fmt_float(summary['total_tps_speedup'], digits=3)}")
    print(f"baseline_records={baseline['records']['records']}")
    print(f"candidate_records={candidate['records']['records']}")
    print(f"baseline_timed_records={baseline['records']['timed_records']}")
    print(f"candidate_timed_records={candidate['records']['timed_records']}")
    print(f"questions_compared={question_compare['questions_compared']}")
    print(f"questions_improved={question_compare['questions_improved']}")
    print(f"questions_regressed={question_compare['questions_regressed']}")
    print(f"questions_unchanged={question_compare['questions_unchanged']}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
