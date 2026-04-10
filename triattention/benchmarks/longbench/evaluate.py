"""Evaluate LongBench predictions produced by the worker path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from triattention.benchmarks.longbench.constants import (
    CATEGORY_TO_DATASETS,
    LENGTH_BUCKETS,
    LONG_BENCH_DATASETS,
    LONG_BENCH_E_DATASETS,
    SPECIAL_FIRST_LINE_DATASETS,
)
from triattention.benchmarks.longbench.metrics import DATASET_TO_METRIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing per-task jsonl predictions.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--longbench-e", action="store_true", help="Interpret the task set as LongBench-E.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task subset.")
    return parser.parse_args()


def resolve_tasks(use_e: bool, tasks: Iterable[str] | None) -> list[str]:
    available = LONG_BENCH_E_DATASETS if use_e else LONG_BENCH_DATASETS
    if tasks is None:
        return list(available)
    requested = list(tasks)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        scope = "LongBench-E" if use_e else "LongBench"
        raise ValueError(f"Unsupported {scope} tasks: {unknown}")
    return requested


def normalize_prediction(task: str, prediction: str) -> str:
    if task in SPECIAL_FIRST_LINE_DATASETS:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def bucket_name(length: int | float | None) -> str:
    if length is None:
        return "8k+"
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def score_item(task: str, prediction: str, ground_truths: list[str], all_classes) -> float:
    metric = DATASET_TO_METRIC[task]
    prediction = normalize_prediction(task, prediction)
    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, metric(prediction, ground_truth, all_classes=all_classes))
    return score


def load_task_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_task(task: str, records: list[dict], *, use_e: bool) -> dict:
    scores: list[float] = []
    bucket_scores = {bucket: [] for bucket in LENGTH_BUCKETS}
    for record in records:
        prediction = record.get("output", record.get("pred", ""))
        ground_truths = record.get("answers", [])
        all_classes = record.get("all_classes")
        score = score_item(task, prediction, ground_truths, all_classes)
        scores.append(score)
        bucket_scores[bucket_name(record.get("length"))].append(score)
    result = {
        "overall": round(100 * float(np.mean(scores)), 2) if scores else 0.0,
        "num_examples": len(scores),
    }
    if use_e:
        result["length_buckets"] = {
            bucket: round(100 * float(np.mean(values)), 2) if values else 0.0
            for bucket, values in bucket_scores.items()
        }
    return result


def summarize_categories(task_results: dict[str, dict]) -> dict[str, float]:
    category_scores: dict[str, float] = {}
    for category, tasks in CATEGORY_TO_DATASETS.items():
        values = [task_results[task]["overall"] for task in tasks if task in task_results]
        if values:
            category_scores[category] = round(float(np.mean(values)), 2)
    return category_scores


def summarize_length_buckets(task_results: dict[str, dict]) -> dict[str, float]:
    bucket_values = {bucket: [] for bucket in LENGTH_BUCKETS}
    for result in task_results.values():
        for bucket, value in result.get("length_buckets", {}).items():
            bucket_values[bucket].append(value)
    return {
        bucket: round(float(np.mean(values)), 2) if values else 0.0
        for bucket, values in bucket_values.items()
    }


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.longbench_e, args.tasks)
    task_results: dict[str, dict] = {}
    for task in tasks:
        pred_path = args.pred_dir / f"{task}.jsonl"
        if not pred_path.exists():
            continue
        task_results[task] = evaluate_task(task, load_task_predictions(pred_path), use_e=args.longbench_e)

    summary = {
        "task_set": "longbench_e" if args.longbench_e else "longbench",
        "num_tasks": len(task_results),
        "task_scores": task_results,
        "overall_average": round(float(np.mean([value["overall"] for value in task_results.values()])), 2)
        if task_results
        else 0.0,
        "category_averages": summarize_categories(task_results),
    }
    if args.longbench_e:
        summary["length_bucket_averages"] = summarize_length_buckets(task_results)

    output_path = args.output or (args.pred_dir / "longbench_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

