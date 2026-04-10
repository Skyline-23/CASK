#!/usr/bin/env python3
"""Summarize accuracy-vs-budget frontier runs for CASK comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from compare_eval_runs import load_json, load_jsonl, resolve_eval_jsonl_path, resolve_metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-method", default="triattention")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def summarize_compression(eval_jsonl_path: Path | None) -> dict[str, float] | None:
    if eval_jsonl_path is None or not eval_jsonl_path.exists():
        return None
    numeric_keys = (
        "compression_events",
        "total_protected_core_tokens",
        "total_scratch_descriptors",
        "total_scratch_source_tokens",
        "total_scratch_merged_groups",
        "total_scratch_saved_tokens",
        "current_cache_tokens",
        "current_prefix_tokens",
        "current_total_cardinality",
    )
    totals = {key: 0.0 for key in numeric_keys}
    records = 0
    for item in load_jsonl(eval_jsonl_path):
        summary = item.get("compression_summary")
        if not isinstance(summary, dict):
            continue
        records += 1
        for key in numeric_keys:
            value = summary.get(key)
            if value is None:
                continue
            totals[key] += float(value)
    if records == 0:
        return None
    output = {"records_with_summary": float(records)}
    for key, total in totals.items():
        output[f"mean_{key}"] = total / float(records)
    return output


def resolve_trace_jsonl_path(run_dir: Path) -> Path | None:
    merged_path = run_dir / "merged" / "merged.jsonl"
    if merged_path.exists():
        return merged_path
    shards_path = run_dir / "shards"
    if shards_path.exists():
        shard_matches = sorted(shards_path.glob("*.jsonl"))
        if len(shard_matches) == 1:
            return shard_matches[0]
    return None


def iter_rows(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Manifest does not contain a list of entries.")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        run_dir_value = entry.get("run_dir")
        if not run_dir_value:
            continue
        run_dir = Path(str(run_dir_value))
        metrics_path = resolve_metrics_path(run_dir)
        metrics = load_json(metrics_path)
        eval_jsonl_path = resolve_eval_jsonl_path(run_dir)
        trace_jsonl_path = resolve_trace_jsonl_path(run_dir)
        compression = summarize_compression(trace_jsonl_path)
        row: dict[str, Any] = {
            "dataset": str(entry.get("dataset")),
            "method": str(entry.get("method")),
            "budget": entry.get("budget"),
            "run_tag": str(entry.get("run_tag")),
            "run_dir": str(run_dir.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "eval_jsonl_path": str(eval_jsonl_path.resolve()) if eval_jsonl_path is not None else None,
            "trace_jsonl_path": str(trace_jsonl_path.resolve()) if trace_jsonl_path is not None else None,
            "acc": float(metrics.get("acc", 0.0)),
            "num_scores": metrics.get("num_scores"),
        }
        if compression is not None:
            row.update(compression)
        yield row


def sort_key(row: dict[str, Any]) -> tuple[str, str, float]:
    budget = row.get("budget")
    budget_value = float(budget) if budget is not None else float("inf")
    return str(row.get("dataset")), str(row.get("method")), budget_value


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = sorted(iter_rows(manifest), key=sort_key)

    baseline_map: Dict[tuple[str, Any], dict[str, Any]] = {}
    for row in rows:
        if row["method"] == args.baseline_method:
            baseline_map[(row["dataset"], row["budget"])] = row

    for row in rows:
        baseline = baseline_map.get((row["dataset"], row["budget"]))
        if baseline is not None:
            row["delta_vs_baseline"] = float(row["acc"]) - float(baseline["acc"])
        else:
            row["delta_vs_baseline"] = None

    frontier: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        dataset_frontier = frontier.setdefault(str(row["dataset"]), {})
        dataset_frontier.setdefault(str(row["method"]), []).append(
            {
                "budget": row["budget"],
                "acc": row["acc"],
                "delta_vs_baseline": row["delta_vs_baseline"],
                "mean_total_scratch_saved_tokens": row.get("mean_total_scratch_saved_tokens"),
                "mean_total_scratch_merged_groups": row.get("mean_total_scratch_merged_groups"),
            }
        )

    summary = {
        "manifest": str(args.manifest.resolve()),
        "baseline_method": args.baseline_method,
        "rows": rows,
        "frontier": frontier,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset",
            "method",
            "budget",
            "acc",
            "delta_vs_baseline",
            "num_scores",
            "mean_compression_events",
            "mean_total_protected_core_tokens",
            "mean_total_scratch_descriptors",
            "mean_total_scratch_source_tokens",
            "mean_total_scratch_merged_groups",
            "mean_total_scratch_saved_tokens",
            "mean_current_cache_tokens",
            "mean_current_prefix_tokens",
            "mean_current_total_cardinality",
            "run_dir",
            "metrics_path",
            "eval_jsonl_path",
            "trace_jsonl_path",
        ]
        with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    main()
