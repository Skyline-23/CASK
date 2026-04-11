from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FRONTIER_ROOT = ROOT / "experiments" / "frontier" / "Qwen3-8B"
DEFAULT_OUTPUT_DIR = ROOT / "paper_artifacts" / "h100_2026_04_11" / "promptheavy_saved_ratio_audit"


def resolve_default_manifest() -> Path:
    candidates = sorted(
        FRONTIER_ROOT.glob("**/overnight_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No overnight_manifest.json found under {FRONTIER_ROOT}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit prompt-heavy replay saved-ratio fields without rerunning the model."
    )
    parser.add_argument("--manifest", type=Path, default=resolve_default_manifest())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_event(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(value))


def corrected_reference_fields(record: dict[str, Any]) -> dict[str, Any]:
    active_reference_tokens = max(0, int(record.get("reference_total_tokens", 0)) - 1)
    current_cache_tokens = int(record.get("current_cache_tokens", 0))
    corrected_saved_tokens = max(0, active_reference_tokens - current_cache_tokens)
    corrected_saved_ratio = (
        float(corrected_saved_tokens / active_reference_tokens) if active_reference_tokens > 0 else 0.0
    )
    corrected_cache_ratio = (
        float(current_cache_tokens / active_reference_tokens) if active_reference_tokens > 0 else 1.0
    )
    return {
        "active_reference_tokens": active_reference_tokens,
        "corrected_reference_saved_tokens": corrected_saved_tokens,
        "corrected_reference_saved_ratio": corrected_saved_ratio,
        "corrected_reference_cache_ratio": corrected_cache_ratio,
    }


def build_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        if job.get("kind") != "longbench":
            continue
        task = str(job.get("name", "")).split(":", 1)[-1]
        for replay_path_str in job.get("replay_ready_paths", []):
            replay_path = Path(replay_path_str)
            if not replay_path.exists():
                continue
            payload = load_json(replay_path)
            summary = payload.get("summary", {})
            records = payload.get("records", [])
            if not records:
                continue
            record = records[0]
            corrected = corrected_reference_fields(record)
            current_total_cardinality = int(record.get("current_total_cardinality", 0))
            terminal_saved_tokens = int(record.get("terminal_saved_tokens", 0))
            candidate_saved_ratio_recomputed = (
                float(terminal_saved_tokens / current_total_cardinality)
                if current_total_cardinality > 0
                else 0.0
            )
            rows.append(
                {
                    "task": task,
                    "method": payload.get("candidate_method"),
                    "budget": payload.get("candidate_budget"),
                    "top1": summary.get("mean_target_top1_match_rate"),
                    "top5": summary.get("mean_target_top5_match_rate"),
                    "mean_nll": summary.get("mean_target_nll"),
                    "first_mismatch": summary.get("mean_first_top1_mismatch_step"),
                    "prefix_events": record.get("prefix_compression_events"),
                    "decode_events": record.get("compression_events"),
                    "current_cache_tokens": record.get("current_cache_tokens"),
                    "current_total_cardinality": current_total_cardinality,
                    "reference_total_tokens": record.get("reference_total_tokens"),
                    "active_reference_tokens": corrected["active_reference_tokens"],
                    "candidate_saved_ratio_reported": summary.get("mean_terminal_saved_ratio"),
                    "candidate_saved_ratio_recomputed": candidate_saved_ratio_recomputed,
                    "reference_saved_ratio_reported": summary.get("mean_reference_terminal_saved_ratio"),
                    "reference_saved_ratio_corrected": corrected["corrected_reference_saved_ratio"],
                    "reference_saved_ratio_delta": (
                        corrected["corrected_reference_saved_ratio"]
                        - float(summary.get("mean_reference_terminal_saved_ratio", 0.0))
                    ),
                    "report_path": str(replay_path.resolve()),
                }
            )
    rows.sort(key=lambda row: (str(row["task"]), str(row["method"]), int(row["budget"])))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: list[dict[str, Any]], manifest_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Prompt-Heavy Saved-Ratio Audit")
    lines.append("")
    lines.append(f"| Source Manifest | `{manifest_path.resolve()}` |")
    lines.append("| --- | --- |")
    lines.append("")
    lines.append("| Task | Method | Budget | Top1 | Top5 | NLL | FirstMismatch | Candidate Saved | Ref Saved | Prefix | Decode |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| `{row['task']}` | `{row['method']}` | `{row['budget']}` | "
            f"`{fmt_pct(row['top1'])}` | `{fmt_pct(row['top5'])}` | `{fmt_num(row['mean_nll'])}` | "
            f"`{int(row['first_mismatch'])}` | `{fmt_pct(row['candidate_saved_ratio_recomputed'])}` | "
            f"`{fmt_pct(row['reference_saved_ratio_corrected'])}` | "
            f"`{fmt_event(row['prefix_events'])}` | `{fmt_event(row['decode_events'])}` |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    rows = build_rows(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "promptheavy_saved_ratio_audit.json"
    csv_path = args.output_dir / "promptheavy_saved_ratio_audit.csv"
    md_path = args.output_dir / "promptheavy_saved_ratio_audit.md"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    md_path.write_text(build_markdown(rows, args.manifest), encoding="utf-8")

    print(str(json_path.resolve()))
    print(str(csv_path.resolve()))
    print(str(md_path.resolve()))


if __name__ == "__main__":
    main()
