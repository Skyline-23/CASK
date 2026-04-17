#!/usr/bin/env python3
"""Build a compact bridge/system summary from per-row comparison outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fidelity_path in sorted(input_dir.glob("*_fidelity.json")):
        stem = fidelity_path.name.removesuffix("_fidelity.json")
        throughput_path = input_dir / f"{stem}_throughput.json"
        task_eval_path = input_dir / f"{stem}_task_eval.json"

        fidelity = load_json(fidelity_path)
        throughput = load_json(throughput_path) if throughput_path.exists() else {}
        task_eval = load_json(task_eval_path) if task_eval_path.exists() else {}

        parts = stem.split("_")
        if len(parts) < 3:
            task = stem
            method = None
            budget = None
        else:
            task = "_".join(parts[:-2])
            method = parts[-2]
            budget = parts[-1]

        task_score = None
        if isinstance(task_eval.get("task_scores"), dict):
            task_row = task_eval["task_scores"].get(task)
            if isinstance(task_row, dict):
                task_score = task_row.get("overall")

        candidate_records = throughput.get("candidate", {}).get("records", {})
        rows.append(
            {
                "label": stem,
                "task": task,
                "method": method,
                "budget": budget,
                "sequence_ratio": fidelity.get("fidelity", {}).get("mean_sequence_ratio"),
                "semantic_similarity": fidelity.get("fidelity", {}).get("mean_semantic_similarity"),
                "final_answer_match_rate": fidelity.get("fidelity", {}).get("final_answer_match_rate"),
                "task_metric": task_score,
                "terminal_saved_ratio": fidelity.get("savings", {}).get("mean_terminal_saved_ratio"),
                "compression_events": fidelity.get("savings", {}).get("mean_compression_events"),
                "output_tokens_per_second": candidate_records.get("output_tokens_per_second"),
                "total_tokens_per_second": candidate_records.get("total_tokens_per_second"),
                "generation_seconds": candidate_records.get("generation_seconds"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = discover_rows(args.input_dir)
    payload = {
        "input_dir": str(args.input_dir.resolve()),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "task",
        "method",
        "budget",
        "sequence_ratio",
        "semantic_similarity",
        "final_answer_match_rate",
        "task_metric",
        "terminal_saved_ratio",
        "compression_events",
        "output_tokens_per_second",
        "total_tokens_per_second",
        "generation_seconds",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    main()
