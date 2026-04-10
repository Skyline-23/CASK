from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triattention.benchmarks.longbench.metrics import DATASET_TO_METRIC

REPORTS_DIR = ROOT / "experiments" / "reports"
ARTIFACT_DIR = ROOT / "paper_artifacts" / "rtx5070ti_2026_04_10" / "cask_v2_fidelity"


TEACHER_FORCED_SPECS = [
    {
        "task": "multi_news",
        "variant": "tri_384",
        "label": "TriAttention @ 384",
        "report": "multi_news_teacher_forced_fidelity_vs_fullkv_tri384.json",
    },
    {
        "task": "multi_news",
        "variant": "cask_384_cov",
        "label": "CASK @ 384 (coverage 0.0625)",
        "report": "multi_news_teacher_forced_fidelity_vs_fullkv_cask384.json",
    },
    {
        "task": "qasper",
        "variant": "tri_384",
        "label": "TriAttention @ 384",
        "report": "qasper_teacher_forced_fidelity_vs_fullkv_tri384.json",
    },
    {
        "task": "qasper",
        "variant": "tri_512",
        "label": "TriAttention @ 512",
        "report": "qasper_teacher_forced_fidelity_vs_fullkv_tri512.json",
    },
    {
        "task": "qasper",
        "variant": "cask_384",
        "label": "CASK @ 384",
        "report": "qasper_teacher_forced_fidelity_vs_fullkv_cask384.json",
    },
    {
        "task": "qasper",
        "variant": "cask_384_cov",
        "label": "CASK @ 384 (coverage 0.0625)",
        "report": "qasper_teacher_forced_fidelity_vs_fullkv_cask384_cov.json",
    },
    {
        "task": "qasper",
        "variant": "cask_256",
        "label": "CASK @ 256",
        "report": "qasper_teacher_forced_fidelity_vs_fullkv_cask256.json",
    },
    {
        "task": "2wikimqa",
        "variant": "tri_512",
        "label": "TriAttention @ 512",
        "report": "2wikimqa_teacher_forced_fidelity_vs_fullkv_tri512.json",
    },
    {
        "task": "2wikimqa",
        "variant": "cask_384",
        "label": "CASK @ 384",
        "report": "2wikimqa_teacher_forced_fidelity_vs_fullkv_cask384.json",
    },
    {
        "task": "2wikimqa",
        "variant": "cask_384_cov",
        "label": "CASK @ 384 (coverage 0.0625)",
        "report": "2wikimqa_teacher_forced_fidelity_vs_fullkv_cask384_cov.json",
    },
    {
        "task": "2wikimqa",
        "variant": "cask_384_cov0125",
        "label": "CASK @ 384 (coverage 0.125)",
        "report": "2wikimqa_teacher_forced_fidelity_vs_fullkv_cask384_cov0125.json",
    },
]


OUTPUT_FIDELITY_SPECS = [
    {
        "task": "multi_news",
        "label": "TriAttention @ 384",
        "report": "multi_news_output_fidelity_vs_fullkv_tri384.json",
    },
    {
        "task": "multi_news",
        "label": "CASK @ 384",
        "report": "multi_news_output_fidelity_vs_fullkv_cask384.json",
    },
    {
        "task": "qasper",
        "label": "TriAttention @ 512",
        "report": "qasper_output_fidelity_vs_fullkv_tri512.json",
    },
    {
        "task": "qasper",
        "label": "CASK @ 512",
        "report": "qasper_output_fidelity_vs_fullkv_cask512.json",
    },
    {
        "task": "2wikimqa",
        "label": "TriAttention @ 512",
        "report": "2wikimqa_output_fidelity_vs_fullkv_tri512.json",
    },
    {
        "task": "2wikimqa",
        "label": "CASK @ 384",
        "report": "2wikimqa_output_fidelity_vs_fullkv_cask384.json",
    },
]


TASK_METRIC_SPECS = [
    {
        "task": "qasper",
        "label": "TriAttention @ 512",
        "merged_jsonl": ROOT / "experiments" / "outputs" / "longbench_qasper" / "Qwen3-8B" / "sample1" / "triattention" / "merged" / "merged.jsonl",
    },
    {
        "task": "qasper",
        "label": "CASK @ 512",
        "merged_jsonl": ROOT / "experiments" / "outputs" / "longbench_qasper" / "Qwen3-8B" / "sample1" / "cask" / "merged" / "merged.jsonl",
    },
    {
        "task": "2wikimqa",
        "label": "TriAttention @ 512",
        "merged_jsonl": ROOT / "experiments" / "outputs" / "longbench_2wikimqa" / "Qwen3-8B" / "sample1" / "triattention" / "merged" / "merged.jsonl",
    },
    {
        "task": "2wikimqa",
        "label": "CASK @ 384",
        "merged_jsonl": ROOT / "experiments" / "outputs" / "longbench_2wikimqa" / "Qwen3-8B" / "sample1" / "cask" / "merged" / "merged.jsonl",
    },
    {
        "task": "multi_news",
        "label": "TriAttention @ 384",
        "merged_jsonl": ROOT / "experiments" / "outputs" / "longbench_multi_news" / "Qwen3-8B" / "sample1" / "triattention" / "merged" / "merged.jsonl",
    },
    {
        "task": "multi_news",
        "label": "CASK @ 384",
        "merged_jsonl": ROOT / "experiments" / "outputs" / "longbench_multi_news" / "Qwen3-8B" / "sample1" / "cask" / "merged" / "merged.jsonl",
    },
    {
        "task": "multi_news",
        "label": "FullKV",
        "merged_jsonl": ROOT / "experiments" / "outputs" / "longbench_multi_news" / "Qwen3-8B" / "sample1" / "fullkv" / "merged" / "merged.jsonl",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_stage_profile(method: str, prefix_events: int | None, decode_events: int | None) -> str:
    if method == "triattention":
        return "eviction_only_baseline"
    if (prefix_events or 0) > 0 and (decode_events or 0) > 0:
        return "two_stage_active"
    if (prefix_events or 0) > 0:
        return "prefix_only_active"
    if (decode_events or 0) > 0:
        return "decode_only_active"
    return "inactive_or_guarded"


def build_teacher_forced_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in TEACHER_FORCED_SPECS:
        data = load_json(REPORTS_DIR / spec["report"])
        summary = data["summary"]
        record = data["records"][0]
        row = {
            "task": spec["task"],
            "variant": spec["variant"],
            "label": spec["label"],
            "method": data["candidate_method"],
            "budget": data["candidate_budget"],
            "prefix_coverage_ratio": record.get("prefix_coverage_ratio"),
            "top1": summary["mean_target_top1_match_rate"],
            "top5": summary["mean_target_top5_match_rate"],
            "strict_prefix": summary["mean_strict_prefix_top1_ratio"],
            "mean_nll": summary["mean_target_nll"],
            "first_mismatch": summary["mean_first_top1_mismatch_step"],
            "candidate_terminal_saved_ratio": summary["mean_terminal_saved_ratio"],
            "reference_terminal_saved_ratio": summary["mean_reference_terminal_saved_ratio"],
            "prefix_events": record.get("prefix_compression_events"),
            "decode_events": record.get("compression_events"),
            "stage_profile": infer_stage_profile(
                data["candidate_method"],
                record.get("prefix_compression_events"),
                record.get("compression_events"),
            ),
            "report_path": str((REPORTS_DIR / spec["report"]).resolve()),
        }
        rows.append(row)
    return rows


def build_output_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in OUTPUT_FIDELITY_SPECS:
        data = load_json(REPORTS_DIR / spec["report"])
        fidelity = data["fidelity"]
        savings = data.get("savings", {})
        row = {
            "task": spec["task"],
            "label": spec["label"],
            "final_answer_match": fidelity["final_answer_match_rate"],
            "exact_output_match": fidelity["exact_output_match_rate"],
            "normalized_output_match": fidelity["normalized_output_match_rate"],
            "sequence_ratio": fidelity["mean_sequence_ratio"],
            "prefix_token_ratio": fidelity["mean_prefix_token_ratio"],
            "prefix_char_ratio": fidelity["mean_prefix_char_ratio"],
            "output_token_ratio": fidelity["mean_output_token_ratio"],
            "output_token_delta": fidelity["mean_output_token_delta"],
            "candidate_terminal_saved_ratio": savings.get("mean_terminal_saved_ratio"),
            "candidate_compression_events": savings.get("mean_compression_events"),
            "report_path": str((REPORTS_DIR / spec["report"]).resolve()),
        }
        rows.append(row)
    return rows


def build_task_metric_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in TASK_METRIC_SPECS:
        merged_path = spec["merged_jsonl"]
        record = json.loads(merged_path.read_text(encoding="utf-8").splitlines()[0])
        metric_fn = DATASET_TO_METRIC[spec["task"]]
        answers = record.get("answers") or []
        prediction = record.get("output", "")
        score = max(
            (
                metric_fn(prediction, answer, all_classes=record.get("all_classes"))
                for answer in answers
            ),
            default=None,
        )
        rows.append(
            {
                "task": spec["task"],
                "label": spec["label"],
                "task_metric": score,
                "merged_jsonl": str(merged_path.resolve()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def build_markdown(teacher_rows: list[dict], output_rows: list[dict], task_metric_rows: list[dict]) -> str:
    multi_news_rows = [row for row in teacher_rows if row["task"] == "multi_news"]
    qasper_rows = [row for row in teacher_rows if row["task"] == "qasper"]
    wiki_rows = [row for row in teacher_rows if row["task"] == "2wikimqa"]
    lines: list[str] = []
    lines.append("# Prompt-Heavy Stage and Coverage Summary")
    lines.append("")
    lines.append("This file promotes the prompt-heavy evidence from raw replay reports into tracked paper artifacts.")
    lines.append("")
    lines.append("## 1. Decode-active prompt-heavy witness")
    lines.append("")
    lines.append("| Task | Method | Budget | Top-1 | Top-5 | Mean NLL | First Mismatch | Ref-Length Saved Ratio | Prefix Events | Decode Events | Stage Profile |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in multi_news_rows:
        lines.append(
            f"| `multi_news` | `{row['label']}` | `{row['budget']}` | `{fmt_pct(row['top1'])}` | "
            f"`{fmt_pct(row['top5'])}` | `{fmt_num(row['mean_nll'])}` | `{int(row['first_mismatch'])}` | "
            f"`{fmt_pct(row['reference_terminal_saved_ratio'])}` | `{row['prefix_events'] or 0}` | "
            f"`{row['decode_events'] or 0}` | `{row['stage_profile']}` |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `multi_news @ 384` is the clean decode-active prompt-heavy witness.")
    lines.append("- `cask` fires both stage-1 prefix eviction and stage-2 decode consolidation (`prefix_events = 2`, `decode_events = 2`).")
    lines.append("- At the same physical budget, `cask` substantially improves `top1`, `top5`, and `mean_nll` over `triattention`.")
    lines.append("")
    lines.append("## 2. Prompt-heavy stage contribution")
    lines.append("")
    lines.append("| Task | Method | Budget | Top-1 | Top-5 | Mean NLL | First Mismatch | Ref-Length Saved Ratio | Prefix Events | Decode Events | Stage Profile |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in qasper_rows:
        lines.append(
            f"| `qasper` | `{row['label']}` | `{row['budget']}` | `{fmt_pct(row['top1'])}` | "
            f"`{fmt_pct(row['top5'])}` | `{fmt_num(row['mean_nll'])}` | `{int(row['first_mismatch'])}` | "
            f"`{fmt_pct(row['reference_terminal_saved_ratio'])}` | `{row['prefix_events'] or 0}` | "
            f"`{row['decode_events'] or 0}` | `{row['stage_profile']}` |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `qasper` is a strong prompt-heavy crossing witness.")
    lines.append("- Every `cask` row above is `prefix_only_active`; decode merge does not fire on this witness.")
    lines.append("- The crossing therefore comes from the two-stage prompt-aware prefix policy, not from decode-stage consolidation.")
    lines.append("")
    lines.append("## 3. Prefix coverage reserve ablation on `2wikimqa`")
    lines.append("")
    lines.append("| Variant | Coverage Ratio | Top-1 | Top-5 | Mean NLL | First Mismatch | Stage Profile |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in wiki_rows:
        coverage = row["prefix_coverage_ratio"]
        lines.append(
            f"| `{row['label']}` | `{coverage if coverage is not None else 0}` | `{fmt_pct(row['top1'])}` | "
            f"`{fmt_pct(row['top5'])}` | `{fmt_num(row['mean_nll'])}` | `{int(row['first_mismatch'])}` | "
            f"`{row['stage_profile']}` |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `2wikimqa` remains a boundary case under teacher-forced `top1`.")
    lines.append("- A small coverage reserve (`0.0625`) improves `top1` and `mean_nll` relative to score-only prefix eviction.")
    lines.append("- Increasing the reserve to `0.125` does not help further, which suggests the correction is real but should stay small.")
    lines.append("")
    lines.append("## 4. Output-level sanity")
    lines.append("")
    lines.append("| Task | Method | Final Answer Match | Sequence Ratio | Prefix Token Ratio |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in output_rows:
        lines.append(
            f"| `{row['task']}` | `{row['label']}` | `{fmt_pct(row['final_answer_match'])}` | "
            f"`{fmt_num(row['sequence_ratio'])}` | `{fmt_num(row['prefix_token_ratio'])}` |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- The prompt-heavy story is not just teacher-forced: under actual greedy decoding, `cask` stays materially closer to the `fullkv` continuation than `triattention` on both tracked tasks.")
    lines.append("- `qasper` is the clean crossing witness.")
    lines.append("- `2wikimqa` is the mixed but informative boundary case that motivated the coverage-reserve correction.")
    lines.append("")
    lines.append("## 5. Single-example task metric sanity")
    lines.append("")
    lines.append("| Task | Method | Task Metric |")
    lines.append("| --- | --- | ---: |")
    for row in task_metric_rows:
        lines.append(
            f"| `{row['task']}` | `{row['label']}` | `{fmt_num(row['task_metric'])}` |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- These single-example task metrics are not a substitute for a full benchmark matrix.")
    lines.append("- They do show that the observed fidelity gains on `qasper` and `multi_news` correspond to better task-visible outputs, while `2wikimqa` remains the honest boundary case.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    teacher_rows = build_teacher_forced_rows()
    output_rows = build_output_rows()
    task_metric_rows = build_task_metric_rows()

    teacher_json = ARTIFACT_DIR / "prompt_heavy_stage_summary.json"
    teacher_csv = ARTIFACT_DIR / "prompt_heavy_stage_summary.csv"
    output_json = ARTIFACT_DIR / "prompt_heavy_output_sanity.json"
    output_csv = ARTIFACT_DIR / "prompt_heavy_output_sanity.csv"
    task_metric_json = ARTIFACT_DIR / "prompt_heavy_task_metrics.json"
    task_metric_csv = ARTIFACT_DIR / "prompt_heavy_task_metrics.csv"
    markdown_path = ARTIFACT_DIR / "prompt_heavy_stage_and_output_summary.md"

    teacher_json.write_text(json.dumps(teacher_rows, indent=2), encoding="utf-8")
    output_json.write_text(json.dumps(output_rows, indent=2), encoding="utf-8")
    task_metric_json.write_text(json.dumps(task_metric_rows, indent=2), encoding="utf-8")
    write_csv(teacher_csv, teacher_rows)
    write_csv(output_csv, output_rows)
    write_csv(task_metric_csv, task_metric_rows)
    markdown_path.write_text(build_markdown(teacher_rows, output_rows, task_metric_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
