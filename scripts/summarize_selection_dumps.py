#!/usr/bin/env python3
"""Summarize CASK token-selection dump files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def min_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(min(values))


def max_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def load_dump(path: Path) -> dict[str, Any] | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("dump_kind") != "cask_selection":
        return None

    tensors = payload.get("tensors", {})
    candidate_indices = tensors.get("candidate_indices")
    candidate_scores = tensors.get("candidate_scores")
    protected_core_indices = tensors.get("protected_core_indices")
    if candidate_indices is None or candidate_scores is None or protected_core_indices is None:
        return None

    candidate_index_list = [int(x) for x in candidate_indices.tolist()]
    candidate_score_tensor = candidate_scores.reshape(-1).to(dtype=torch.float32)
    protected_core_set = {int(x) for x in protected_core_indices.tolist()}
    core_offsets = [offset for offset, idx in enumerate(candidate_index_list) if idx in protected_core_set]
    core_count = len(core_offsets)
    top_offsets = (
        torch.topk(candidate_score_tensor, k=core_count, largest=True).indices.tolist()
        if core_count > 0
        else []
    )
    core_mass = float(candidate_score_tensor[core_offsets].sum().item()) if core_offsets else 0.0
    top_mass = float(candidate_score_tensor[top_offsets].sum().item()) if top_offsets else 0.0

    scratch_descriptors = payload.get("scratch_descriptors", [])
    group_sizes = [len(item.get("source_indices", [])) for item in scratch_descriptors]
    merged_group_count = sum(1 for size in group_sizes if size > 1)
    merge_trace = payload.get("merge_trace", [])
    similarities = [float(item["similarity"]) for item in merge_trace if item.get("similarity") is not None]
    gaps = [float(item["gap"]) for item in merge_trace if item.get("gap") is not None]
    projection_distances = [
        float(item["projection_distance"])
        for item in merge_trace
        if item.get("projection_distance") is not None
    ]

    return {
        "file": str(path.resolve()),
        "event_index": int(metadata.get("event_index", -1)),
        "sample_idx": int(metadata.get("sample_idx", -1)),
        "record_id": metadata.get("record_id"),
        "absolute_position": int(metadata.get("absolute_position", -1)),
        "budget": int(metadata.get("budget", -1)),
        "kv_cache_len": int(metadata.get("kv_cache_len", -1)),
        "candidate_count": int(metadata.get("candidate_count", len(candidate_index_list))),
        "available_slots": int(metadata.get("available_slots", -1)),
        "core_count": int(core_count),
        "core_topk_overlap": int(len(set(core_offsets) & set(top_offsets))),
        "core_topk_overlap_ratio": (
            float(len(set(core_offsets) & set(top_offsets)) / core_count) if core_count > 0 else None
        ),
        "core_score_mass": core_mass,
        "oracle_topk_score_mass": top_mass,
        "core_score_mass_ratio": (float(core_mass / top_mass) if top_mass > 0 else None),
        "scratch_descriptor_count": int(len(scratch_descriptors)),
        "scratch_saved_tokens": int(metadata.get("scratch_saved_tokens", 0)),
        "merged_group_count": int(merged_group_count),
        "mean_group_size": mean_or_none([float(size) for size in group_sizes]),
        "max_group_size": max_or_none([float(size) for size in group_sizes]),
        "merge_steps": int(len(merge_trace)),
        "merge_similarity_mean": mean_or_none(similarities),
        "merge_similarity_min": min_or_none(similarities),
        "merge_similarity_max": max_or_none(similarities),
        "merge_gap_mean": mean_or_none(gaps),
        "merge_gap_max": max_or_none(gaps),
        "projection_distance_mean": mean_or_none(projection_distances),
        "projection_distance_max": max_or_none(projection_distances),
        "phase_boundary_relaxed": bool(metadata.get("phase_boundary_relaxed", False)),
    }


def build_summary(rows: list[dict[str, Any]], dump_dir: Path) -> dict[str, Any]:
    numeric_keys = [
        "candidate_count",
        "available_slots",
        "core_count",
        "core_topk_overlap",
        "core_topk_overlap_ratio",
        "core_score_mass",
        "oracle_topk_score_mass",
        "core_score_mass_ratio",
        "scratch_descriptor_count",
        "scratch_saved_tokens",
        "merged_group_count",
        "mean_group_size",
        "max_group_size",
        "merge_steps",
        "merge_similarity_mean",
        "merge_similarity_min",
        "merge_similarity_max",
        "merge_gap_mean",
        "merge_gap_max",
        "projection_distance_mean",
        "projection_distance_max",
    ]
    aggregate: dict[str, Any] = {
        "dump_dir": str(dump_dir.resolve()),
        "events": len(rows),
        "rows": rows,
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        aggregate[f"mean_{key}"] = mean_or_none(values)
    aggregate["phase_boundary_relaxed_events"] = sum(
        1 for row in rows if bool(row.get("phase_boundary_relaxed"))
    )
    return aggregate


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.dump_dir.glob("*.pt")):
        row = load_dump(path)
        if row is not None:
            rows.append(row)
    summary = build_summary(rows, args.dump_dir)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
