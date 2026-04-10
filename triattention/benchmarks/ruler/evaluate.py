"""Evaluate RULER predictions produced by the worker path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from triattention.benchmarks.ruler.constants import CATEGORY_TO_TASKS, DEFAULT_TASKS, PARTIAL_MATCH_TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing per-task jsonl predictions.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--tasks", nargs="*", default=None)
    return parser.parse_args()


def string_match_part(predictions: list[str], references: list[list[str]]) -> float:
    score = sum(
        max([1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs])
        for pred, refs in zip(predictions, references)
    )
    return round(score / len(predictions) * 100, 2) if predictions else 0.0


def string_match_all(predictions: list[str], references: list[list[str]]) -> float:
    score = sum(
        sum([1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs]) / len(refs)
        for pred, refs in zip(predictions, references)
    )
    return round(score / len(predictions) * 100, 2) if predictions else 0.0


def load_task_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_task(task: str, records: list[dict]) -> dict:
    predictions = [record.get("output", record.get("pred", "")).strip() for record in records]
    references = [record.get("outputs", [record.get("output", "")]) for record in records]
    metric = string_match_part if task in PARTIAL_MATCH_TASKS else string_match_all
    return {
        "score": metric(predictions, references),
        "nulls": f"{sum(1 for value in predictions if not value)}/{len(predictions)}",
        "num_examples": len(predictions),
    }


def summarize_categories(task_results: dict[str, dict]) -> dict[str, float]:
    category_scores: dict[str, float] = {}
    for category, tasks in CATEGORY_TO_TASKS.items():
        values = [task_results[task]["score"] for task in tasks if task in task_results]
        if values:
            category_scores[category] = round(float(np.mean(values)), 2)
    return category_scores


def main() -> None:
    args = parse_args()
    tasks = list(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    task_results: dict[str, dict] = {}
    for task in tasks:
        pred_path = args.pred_dir / f"{task}.jsonl"
        if not pred_path.exists():
            continue
        task_results[task] = evaluate_task(task, load_task_predictions(pred_path))

    summary = {
        "num_tasks": len(task_results),
        "task_scores": task_results,
        "overall_average": round(float(np.mean([value["score"] for value in task_results.values()])), 2)
        if task_results
        else 0.0,
        "category_averages": summarize_categories(task_results),
    }
    output_path = args.output or (args.pred_dir / "ruler_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

