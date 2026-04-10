#!/usr/bin/env python3
"""Compare TriAttention score dump directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--json-output", default=None, type=Path, help="Optional path for JSON summary output.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum differing events to print.")
    parser.add_argument("--fail-on-diff", action="store_true", help="Exit with status 1 when any difference is found.")
    return parser.parse_args()


def collect_dump_files(root: Path) -> Dict[str, Path]:
    return {
        file.relative_to(root).as_posix(): file
        for file in sorted(root.rglob("*.pt"))
    }


def load_dump(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def tensor_diff_stats(left: torch.Tensor, right: torch.Tensor) -> Dict[str, object]:
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    if left.dtype != right.dtype:
        right = right.to(left.dtype)
    if left.numel() == 0:
        return {"shape_match": True, "equal": True, "max_abs_diff": 0.0}
    if left.dtype.is_floating_point:
        diff = (left - right).abs()
        return {
            "shape_match": True,
            "equal": bool(torch.equal(left, right)),
            "max_abs_diff": float(diff.max().item()),
        }
    unequal = int((left != right).sum().item())
    return {
        "shape_match": True,
        "equal": unequal == 0,
        "unequal_values": unequal,
    }


def compare_event(left_path: Path, right_path: Path) -> Tuple[bool, Dict[str, object]]:
    left = load_dump(left_path)
    right = load_dump(right_path)
    left_meta = dict(left.get("metadata", {}))
    right_meta = dict(right.get("metadata", {}))
    left_tensors = dict(left.get("tensors", {}))
    right_tensors = dict(right.get("tensors", {}))

    event_result: Dict[str, object] = {
        "metadata_match": left_meta == right_meta,
        "metadata_left": left_meta,
        "metadata_right": right_meta,
        "tensor_diffs": {},
    }
    matched = True

    left_keys = set(left_tensors.keys())
    right_keys = set(right_tensors.keys())
    if left_keys != right_keys:
        matched = False
        event_result["tensor_keys_left"] = sorted(left_keys)
        event_result["tensor_keys_right"] = sorted(right_keys)

    tensor_diffs = event_result["tensor_diffs"]
    assert isinstance(tensor_diffs, dict)
    for key in sorted(left_keys & right_keys):
        diff_stats = tensor_diff_stats(left_tensors[key], right_tensors[key])
        tensor_diffs[key] = diff_stats
        if not diff_stats.get("equal", False):
            matched = False

    if not event_result["metadata_match"]:
        matched = False
    return matched, event_result


def main() -> None:
    args = parse_args()
    baseline_files = collect_dump_files(args.baseline_dir)
    candidate_files = collect_dump_files(args.candidate_dir)

    summary: Dict[str, object] = {
        "baseline_dir": str(args.baseline_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "baseline_only": sorted(set(baseline_files) - set(candidate_files)),
        "candidate_only": sorted(set(candidate_files) - set(baseline_files)),
        "matched_events": 0,
        "different_events": [],
    }

    different_events = summary["different_events"]
    assert isinstance(different_events, list)
    for key in sorted(set(baseline_files) & set(candidate_files)):
        equal, event_result = compare_event(baseline_files[key], candidate_files[key])
        if equal:
            summary["matched_events"] = int(summary["matched_events"]) + 1
            continue
        different_events.append({"event": key, **event_result})

    print(f"baseline_only={len(summary['baseline_only'])}")
    print(f"candidate_only={len(summary['candidate_only'])}")
    print(f"matched_events={summary['matched_events']}")
    print(f"different_events={len(summary['different_events'])}")

    for item in different_events[: args.limit]:
        print(f"[diff] {item['event']}")
        tensor_diffs = item.get("tensor_diffs", {})
        if isinstance(tensor_diffs, dict):
            for key, stats in tensor_diffs.items():
                print(f"  {key}: {stats}")
        if not item.get("metadata_match", True):
            print("  metadata differs")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    has_diff = bool(summary["baseline_only"]) or bool(summary["candidate_only"]) or bool(summary["different_events"])
    if args.fail_on_diff and has_diff:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
