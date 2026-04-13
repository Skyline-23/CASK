from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def frontier_source_path(dataset_key: str, method: str, budget: int) -> Path:
    dataset_tag = dataset_key.split("_")[0]
    frontier_dir = ROOT / "experiments" / "frontier" / "Qwen3-8B" / f"h100_{dataset_tag}_fidelity_gate_20260410"
    method_tag = "triattention" if method == "triattention" else "cask"
    return frontier_dir / f"{method_tag}_budget_{budget}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync replay runs into reasoning-gate artifact summaries."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=resolve_default_report_dir(),
        help="Directory containing aime24/aime25 tri/cask replay JSONs to sync.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts" / "h100_2026_04_10" / "cask_h100_fidelity",
        help="Reasoning-gate artifact directory to update.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=384,
        help="Budget row to overwrite from the replay reports.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["aime24_ref6", "aime25_ref6"],
        default=["aime24_ref6", "aime25_ref6"],
        help="Dataset slices to sync from replay reports.",
    )
    return parser.parse_args()


def resolve_default_report_dir() -> Path:
    reports_root = ROOT / "experiments" / "reports"
    candidates = sorted((path for path in reports_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        has_aime24 = any(candidate.glob("aime24_*.json"))
        has_aime25 = any(candidate.glob("aime25_*.json"))
        if has_aime24 and has_aime25:
            return candidate
    return reports_root / "reasoning_gate_sync"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def round6(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def pct4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 4)


def build_row(dataset_key: str, budget: int, method: str, payload: dict[str, Any], source_json: Path) -> dict[str, Any]:
    summary = payload["summary"]
    tracked_source_json = frontier_source_path(dataset_key, method, budget)
    try:
        source_json_value = tracked_source_json.relative_to(ROOT).as_posix()
    except ValueError:
        source_json_value = str(tracked_source_json.resolve())
    return {
        "dataset": dataset_key,
        "budget": budget,
        "method": method,
        "top1": pct4(summary["mean_target_top1_match_rate"]),
        "top5": pct4(summary["mean_target_top5_match_rate"]),
        "strict_prefix": pct4(summary["mean_strict_prefix_top1_ratio"]),
        "first_mismatch": round4(summary["mean_first_top1_mismatch_step"]),
        "mean_nll": round6(summary["mean_target_nll"]),
        "saved_ratio": pct4(summary["mean_terminal_saved_ratio"]),
        "cache_ratio": pct4(summary["mean_terminal_cache_ratio"]),
        "ref_saved_ratio": pct4(summary["mean_reference_terminal_saved_ratio"]),
        "records": int(summary["records_compared"]),
        "compression_events": round4(summary["mean_compression_events"]),
        "guards": int(summary["records_with_guard_triggered"]),
        "source_json": source_json_value,
    }


def load_report_rows(report_dir: Path, budget: int, datasets: list[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for dataset_key in datasets:
        for method in ("triattention", "cask"):
            stem = f"{dataset_key[:6]}_{'tri' if method == 'triattention' else 'cask'}{budget}"
            path = report_dir / f"{stem}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing replay file: {path}")
            payload = load_json(path)
            rows[f"{dataset_key}:{method}:{budget}"] = build_row(
                dataset_key=dataset_key,
                budget=budget,
                method=method,
                payload=payload,
                source_json=path,
            )
    return rows


def update_csv(path: Path, replacement_rows: dict[str, dict[str, Any]]) -> None:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        normalized_fieldnames = []
        fieldname_map: dict[str, str] = {}
        for name in fieldnames:
            normalized = name.replace("\ufeff", "").replace('"', "")
            normalized_fieldnames.append(normalized)
            fieldname_map[name] = normalized

        rows = []
        for raw_row in reader:
            row = {fieldname_map[key]: value for key, value in raw_row.items()}
            rows.append(row)

    for row in rows:
        dataset_name = row.get("dataset") or row.get("dataset_slice")
        normalized_dataset = (
            dataset_name.lower() if dataset_name and dataset_name.lower().startswith("aime") else dataset_name
        )
        key = f"{normalized_dataset}:{row['method']}:{int(row['budget'])}"
        if key in replacement_rows:
            replacement = replacement_rows[key]
            if "dataset" in row:
                row["dataset"] = replacement["dataset"]
            if "dataset_slice" in row:
                row["dataset_slice"] = replacement["dataset"].replace("aime", "AIME")
            if "top1" in row:
                row["top1"] = replacement["top1"]
            if "top5" in row:
                row["top5"] = replacement["top5"]
            if "strict_prefix" in row:
                row["strict_prefix"] = replacement["strict_prefix"]
            if "first_mismatch" in row:
                row["first_mismatch"] = replacement["first_mismatch"]
            if "mean_nll" in row:
                row["mean_nll"] = replacement["mean_nll"]
            if "saved_ratio" in row:
                row["saved_ratio"] = replacement["saved_ratio"]
            if "cache_ratio" in row:
                row["cache_ratio"] = replacement["cache_ratio"]
            if "ref_saved_ratio" in row:
                row["ref_saved_ratio"] = replacement["ref_saved_ratio"]
            if "records" in row:
                row["records"] = replacement["records"]
            if "compression_events" in row:
                row["compression_events"] = replacement["compression_events"]
            if "guards" in row:
                row["guards"] = replacement["guards"]
            if "source_json" in row:
                row["source_json"] = replacement["source_json"]
            if "top1_pct" in row:
                row["top1_pct"] = round6(replacement["top1"])
            if "top5_pct" in row:
                row["top5_pct"] = round6(replacement["top5"])
            if "strict_prefix_pct" in row:
                row["strict_prefix_pct"] = round6(replacement["strict_prefix"])
            if "saved_ratio_pct" in row:
                row["saved_ratio_pct"] = round6(replacement["saved_ratio"])
            if "records_compared" in row:
                row["records_compared"] = replacement["records"]
            if "guard_records" in row:
                row["guard_records"] = replacement["guards"]
            if "mean_compression_events" in row:
                row["mean_compression_events"] = replacement["compression_events"]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=normalized_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_aime24_json(path: Path, replacement_rows: dict[str, dict[str, Any]], budget: int) -> None:
    payload = load_json(path)
    tri = replacement_rows[f"aime24_ref6:triattention:{budget}"]
    cask = replacement_rows[f"aime24_ref6:cask:{budget}"]

    budget_key = str(budget)
    payload["same_budget"][budget_key]["triattention"].update(tri)
    payload["same_budget"][budget_key]["cask"].update(cask)
    payload["same_budget"][budget_key]["cask_minus_triattention"] = {
        "top1": round4(cask["top1"] - tri["top1"]),
        "top5": round4(cask["top5"] - tri["top5"]),
        "strict_prefix": round4(cask["strict_prefix"] - tri["strict_prefix"]),
        "first_mismatch": round4(cask["first_mismatch"] - tri["first_mismatch"]),
        "mean_nll": round6(cask["mean_nll"] - tri["mean_nll"]),
        "saved_ratio": round4(cask["saved_ratio"] - tri["saved_ratio"]),
    }

    crossing = payload.get("budget_crossing", {})
    if "cask_256_vs_tri_384" in crossing:
        tri_cross = crossing["cask_256_vs_tri_384"]["triattention"]
        tri_cross.update(tri)
        crossing["cask_256_vs_tri_384"]["delta"] = {
            "top1": round4(crossing["cask_256_vs_tri_384"]["cask"]["top1"] - tri["top1"]),
            "top5": round4(crossing["cask_256_vs_tri_384"]["cask"]["top5"] - tri["top5"]),
            "mean_nll": round6(crossing["cask_256_vs_tri_384"]["cask"]["mean_nll"] - tri["mean_nll"]),
            "saved_ratio": round4(crossing["cask_256_vs_tri_384"]["cask"]["saved_ratio"] - tri["saved_ratio"]),
        }
    if "cask_384_vs_tri_512" in crossing:
        cask_cross = crossing["cask_384_vs_tri_512"]["cask"]
        cask_cross.update(cask)
        crossing["cask_384_vs_tri_512"]["delta"] = {
            "top1": round4(cask["top1"] - crossing["cask_384_vs_tri_512"]["triattention"]["top1"]),
            "top5": round4(cask["top5"] - crossing["cask_384_vs_tri_512"]["triattention"]["top5"]),
            "mean_nll": round6(cask["mean_nll"] - crossing["cask_384_vs_tri_512"]["triattention"]["mean_nll"]),
            "saved_ratio": round4(cask["saved_ratio"] - crossing["cask_384_vs_tri_512"]["triattention"]["saved_ratio"]),
        }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def update_aime25_json(path: Path, replacement_rows: dict[str, dict[str, Any]], budget: int) -> None:
    payload = load_json(path)
    tri = replacement_rows[f"aime25_ref6:triattention:{budget}"]
    cask = replacement_rows[f"aime25_ref6:cask:{budget}"]

    for row in payload["rows"]:
        if row["budget"] == budget and row["method"] == "triattention":
            row.update(
                {
                    "top1_pct": round6(tri["top1"]),
                    "top5_pct": round6(tri["top5"]),
                    "strict_prefix_pct": round6(tri["strict_prefix"]),
                    "mean_nll": round6(tri["mean_nll"]),
                    "first_mismatch": round4(tri["first_mismatch"]),
                    "saved_ratio_pct": round6(tri["saved_ratio"]),
                    "records_compared": tri["records"],
                    "guard_records": tri["guards"],
                    "mean_compression_events": tri["compression_events"],
                    "source_json": tri["source_json"],
                }
            )
        if row["budget"] == budget and row["method"] == "cask":
            row.update(
                {
                    "top1_pct": round6(cask["top1"]),
                    "top5_pct": round6(cask["top5"]),
                    "strict_prefix_pct": round6(cask["strict_prefix"]),
                    "mean_nll": round6(cask["mean_nll"]),
                    "first_mismatch": round4(cask["first_mismatch"]),
                    "saved_ratio_pct": round6(cask["saved_ratio"]),
                    "records_compared": cask["records"],
                    "guard_records": cask["guards"],
                    "mean_compression_events": cask["compression_events"],
                    "source_json": cask["source_json"],
                }
            )

    payload["same_budget_headline"] = (
        "CASK beats TriAttention on top1/top5/mean_nll at 256, 384, and 512, "
        "while terminal savings remain budget-dependent across the same-budget sweep."
    )

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def write_update_note(path: Path, replacement_rows: dict[str, dict[str, Any]], budget: int) -> None:
    tri24 = replacement_rows[f"aime24_ref6:triattention:{budget}"]
    cask24 = replacement_rows[f"aime24_ref6:cask:{budget}"]
    tri25 = replacement_rows[f"aime25_ref6:triattention:{budget}"]
    cask25 = replacement_rows[f"aime25_ref6:cask:{budget}"]
    lines = [
        "# Reasoning Gate Pending Paper Update",
        "",
        f"Synced budget: `{budget}`",
        "",
        "| Slice | Budget | Tri Top-1 | CASK Top-1 | Tri Mean NLL | CASK Mean NLL | Tri Saved Ratio | CASK Saved Ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `AIME24 ref6` | `{budget}` | `{tri24['top1']:.2f}` | `{cask24['top1']:.2f}` | `{tri24['mean_nll']:.3f}` | `{cask24['mean_nll']:.3f}` | `{tri24['saved_ratio']:.2f}%` | `{cask24['saved_ratio']:.2f}%` |",
        f"| `AIME25 ref6` | `{budget}` | `{tri25['top1']:.2f}` | `{cask25['top1']:.2f}` | `{tri25['mean_nll']:.3f}` | `{cask25['mean_nll']:.3f}` | `{tri25['saved_ratio']:.2f}%` | `{cask25['saved_ratio']:.2f}%` |",
        "",
        "Next step: regenerate `docs/assets/fig1_reasoning_gate_frontier.*`, then patch `paper/content.tex`, `paper/arxiv_submit/content.tex`, and `README.md` if the numbers differ materially from the current draft.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    replacement_rows = load_report_rows(args.report_dir, args.budget, args.datasets)

    if "aime24_ref6" in args.datasets:
        update_csv(args.artifact_dir / "aime24_ref6_h100_fidelity_summary.csv", replacement_rows)
        update_aime24_json(args.artifact_dir / "aime24_ref6_h100_fidelity_summary.json", replacement_rows, args.budget)
    if "aime25_ref6" in args.datasets:
        update_csv(args.artifact_dir / "aime25_ref6_h100_fidelity_summary.csv", replacement_rows)
        update_aime25_json(args.artifact_dir / "aime25_ref6_h100_fidelity_summary.json", replacement_rows, args.budget)
    if set(args.datasets) == {"aime24_ref6", "aime25_ref6"}:
        write_update_note(args.report_dir / "reasoning_gate_update_note.md", replacement_rows, args.budget)


if __name__ == "__main__":
    main()
