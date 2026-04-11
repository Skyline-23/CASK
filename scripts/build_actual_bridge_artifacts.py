from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "experiments" / "frontier" / "Qwen3-8B" / "h100_actual_bridge_metrics_20260411"
OUTPUT_DIR = ROOT / "artifacts" / "h100_2026_04_11" / "cask_h100_actual_bridge"


ACTUAL_BRIDGE_SPECS = [
    {
        "task": "qasper",
        "variant": "TriAttention @ 512",
        "method": "triattention",
        "budget": 512,
        "bridge_type": "budget_crossing_baseline",
        "metrics_json": METRICS_DIR / "qasper_tri512.json",
        "eval_json": ROOT / "experiments" / "longbench_h100_actual_bridge_20260411" / "qasper_tri512" / "longbench" / "Qwen3-8B" / "longbench_eval.json",
    },
    {
        "task": "qasper",
        "variant": "CASK @ 256",
        "method": "cask",
        "budget": 256,
        "bridge_type": "budget_crossing_candidate",
        "metrics_json": METRICS_DIR / "qasper_cask256.json",
        "eval_json": ROOT / "experiments" / "longbench_h100_actual_bridge_20260411" / "qasper_cask256" / "longbench" / "Qwen3-8B" / "longbench_eval.json",
    },
    {
        "task": "multi_news",
        "variant": "TriAttention @ 384",
        "method": "triattention",
        "budget": 384,
        "bridge_type": "same_budget_baseline",
        "metrics_json": METRICS_DIR / "multi_news_tri384.json",
        "eval_json": ROOT / "experiments" / "longbench_h100_actual_bridge_20260411" / "multi_news_tri384" / "longbench" / "Qwen3-8B" / "longbench_eval.json",
    },
    {
        "task": "multi_news",
        "variant": "CASK @ 384",
        "method": "cask",
        "budget": 384,
        "bridge_type": "same_budget_candidate",
        "metrics_json": METRICS_DIR / "multi_news_cask384.json",
        "eval_json": ROOT / "experiments" / "longbench_h100_actual_bridge_20260411" / "multi_news_cask384" / "longbench" / "Qwen3-8B" / "longbench_eval.json",
    },
    {
        "task": "hotpotqa",
        "variant": "TriAttention @ 256",
        "method": "triattention",
        "budget": 256,
        "bridge_type": "same_budget_baseline",
        "metrics_json": METRICS_DIR / "hotpotqa_tri256.json",
        "eval_json": ROOT / "experiments" / "longbench_h100_actual_bridge_20260411" / "hotpotqa_tri256" / "longbench" / "Qwen3-8B" / "longbench_eval.json",
    },
    {
        "task": "hotpotqa",
        "variant": "CASK @ 256",
        "method": "cask",
        "budget": 256,
        "bridge_type": "same_budget_candidate",
        "metrics_json": METRICS_DIR / "hotpotqa_cask256.json",
        "eval_json": ROOT / "experiments" / "longbench_h100_actual_bridge_20260411" / "hotpotqa_cask256" / "longbench" / "Qwen3-8B" / "longbench_eval.json",
    },
]


STAGE_ABLATION_SPECS = [
    {
        "task": "multi_news",
        "variant": "CASK @ 384 (full two-stage)",
        "stage_mode": "two_stage",
        "budget": 384,
        "metrics_json": METRICS_DIR / "multi_news_cask384.json",
        "eval_json": ROOT / "experiments" / "longbench_h100_actual_bridge_20260411" / "multi_news_cask384" / "longbench" / "Qwen3-8B" / "longbench_eval.json",
    },
    {
        "task": "multi_news",
        "variant": "CASK @ 384 (stage 1 only)",
        "stage_mode": "stage1_only",
        "budget": 384,
        "metrics_json": METRICS_DIR / "multi_news_cask384_stage1only.json",
        "eval_json": ROOT / "experiments" / "longbench_h100_stage_ablation" / "longbench" / "Qwen3-8B" / "longbench_eval.json",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task_metric(path: Path, task: str) -> float | None:
    payload = load_json(path)
    return payload.get("task_scores", {}).get(task, {}).get("overall")


def build_row(spec: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(spec["metrics_json"])
    fidelity = payload.get("fidelity", {})
    savings = payload.get("savings", {})
    return {
        "task": spec["task"],
        "variant": spec["variant"],
        "method": spec.get("method"),
        "bridge_type": spec.get("bridge_type"),
        "stage_mode": spec.get("stage_mode"),
        "budget": spec["budget"],
        "records_compared": fidelity.get("records_compared"),
        "sequence_ratio": fidelity.get("mean_sequence_ratio"),
        "prefix_token_ratio": fidelity.get("mean_prefix_token_ratio"),
        "prefix_char_ratio": fidelity.get("mean_prefix_char_ratio"),
        "output_token_ratio": fidelity.get("mean_output_token_ratio"),
        "semantic_similarity": fidelity.get("mean_semantic_similarity"),
        "final_answer_match_rate": fidelity.get("final_answer_match_rate"),
        "exact_output_match_rate": fidelity.get("exact_output_match_rate"),
        "normalized_output_match_rate": fidelity.get("normalized_output_match_rate"),
        "task_metric": load_task_metric(spec["eval_json"], spec["task"]),
        "compression_events": savings.get("mean_compression_events"),
        "terminal_saved_ratio": savings.get("mean_terminal_saved_ratio"),
        "terminal_cache_ratio": savings.get("mean_terminal_cache_ratio"),
        "cumulative_saved_tokens": savings.get("mean_cumulative_saved_tokens"),
        "source_json": str(spec["metrics_json"].resolve()),
        "source_eval_json": str(spec["eval_json"].resolve()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.{digits}f}%"


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def build_bridge_markdown(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Actual-Bridge Summary")
    lines.append("")
    lines.append("| Task | Variant | Budget | Sequence Ratio | Prefix Ratio | Semantic Sim. | Output Ratio | Task Metric | Terminal Saved | Compression Events |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| `{row['task']}` | `{row['variant']}` | `{row['budget']}` | "
            f"`{fmt_pct(row['sequence_ratio'])}` | `{fmt_pct(row['prefix_token_ratio'])}` | "
            f"`{fmt_num(row['semantic_similarity'], 3)}` | `{fmt_pct(row['output_token_ratio'])}` | `{fmt_num(row['task_metric'], 2)}` | "
            f"`{fmt_pct(row['terminal_saved_ratio'])}` | `{fmt_num(row['compression_events'], 1)}` |"
        )
    lines.append("")
    lines.append("## Readout")
    lines.append("")
    lines.append("- `qasper`: `CASK @ 256` beats `TriAttention @ 512` on `sequence_ratio`, official task metric, and semantic similarity, giving the cleanest actual-output budget crossing in the current H100 package.")
    lines.append("- `multi_news`: `CASK @ 384` recovers a non-zero actual-output bridge where `TriAttention @ 384` collapses on both lexical overlap and task metric; semantic similarity shows the same direction even more strongly.")
    lines.append("- `hotpotqa`: `CASK @ 256` matches `TriAttention @ 256` exactly at the output level on this witness while still ending with a `97.6%` terminal saved ratio.")
    return "\n".join(lines)


def build_stage_markdown(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Stage-Ablation Summary")
    lines.append("")
    lines.append("| Task | Variant | Budget | Sequence Ratio | Prefix Ratio | Task Metric | Terminal Saved | Compression Events | Cumulative Saved Tokens |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| `{row['task']}` | `{row['variant']}` | `{row['budget']}` | "
            f"`{fmt_pct(row['sequence_ratio'])}` | `{fmt_pct(row['prefix_token_ratio'])}` | "
            f"`{fmt_num(row['task_metric'], 2)}` | `{fmt_pct(row['terminal_saved_ratio'])}` | "
            f"`{fmt_num(row['compression_events'], 1)}` | `{fmt_num(row['cumulative_saved_tokens'], 1)}` |"
        )
    lines.append("")
    lines.append("## Readout")
    lines.append("")
    lines.append("- `multi_news` stage ablation is useful as a guardrail, not as a standalone stage-2 headline.")
    lines.append("- On this single witness, `stage 1 only` and `full two-stage` have nearly identical output-level similarity; the current paper should claim the overall two-stage policy, not a large standalone decode-merge output gain from this one case.")
    return "\n".join(lines)


def build_readme(bridge_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# CASK H100 Actual-Bridge Assets")
    lines.append("")
    lines.append("This directory packages the H100 output-level evidence that complements the replay-fidelity gates in `artifacts/h100_2026_04_10/cask_h100_fidelity/`.")
    lines.append("")
    lines.append("Tracked here:")
    lines.append("- actual-output bridge rows for `qasper`, `multi_news`, and `hotpotqa`")
    lines.append("- the `multi_news` stage-ablation row needed to keep the stage-2 claim honest")
    lines.append("- clean CSV / JSON / Markdown tables with direct provenance links")
    lines.append("- output-level evaluation reported on official task metric, lexical overlap, and semantic/reference similarity")
    lines.append("")
    lines.append("Primary files:")
    lines.append("- `actual_bridge_summary.csv`")
    lines.append("- `actual_bridge_summary.json`")
    lines.append("- `actual_bridge_summary.md`")
    lines.append("- `stage_ablation_summary.csv`")
    lines.append("- `stage_ablation_summary.json`")
    lines.append("- `stage_ablation_summary.md`")
    lines.append("")
    lines.append("Artifact builders:")
    lines.append("- `scripts/build_actual_bridge_artifacts.py`")
    lines.append("- `scripts/build_promptheavy_saved_ratio_audit.py`")
    lines.append("")
    lines.append("Experiment roots:")
    lines.append("- replay metrics: `experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/`")
    lines.append("- actual-generation runs: `experiments/longbench_h100_actual_bridge_20260411/`")
    lines.append("- stage-ablation generation run: `experiments/longbench_h100_stage_ablation/`")
    lines.append("- stage-ablation replay metrics: `experiments/frontier/Qwen3-8B/h100_promptheavy_stage_ablation_20260411/`")
    lines.append("")
    lines.append("Headline read:")
    lines.append("- `qasper`: `CASK @ 256` crosses above `TriAttention @ 512` on lexical overlap, semantic similarity, and task metric.")
    lines.append("- `multi_news`: `CASK @ 384` beats `TriAttention @ 384` on lexical overlap, semantic similarity, and task metric at the same budget.")
    lines.append("- `hotpotqa`: `CASK @ 256` is an output-parity non-regression witness rather than a win case.")
    lines.append("- `multi_news` stage ablation stays in the package as a claim boundary: it does not justify a large standalone stage-2 output gain.")
    lines.append("")
    lines.append(f"- bridge rows packaged: `{len(bridge_rows)}`")
    lines.append(f"- stage-ablation rows packaged: `{len(stage_rows)}`")
    return "\n".join(lines)


def main() -> None:
    bridge_rows = [build_row(spec) for spec in ACTUAL_BRIDGE_SPECS]
    stage_rows = [build_row(spec) for spec in STAGE_ABLATION_SPECS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "actual_bridge_summary.json").write_text(
        json.dumps(bridge_rows, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "stage_ablation_summary.json").write_text(
        json.dumps(stage_rows, indent=2), encoding="utf-8"
    )
    write_csv(OUTPUT_DIR / "actual_bridge_summary.csv", bridge_rows)
    write_csv(OUTPUT_DIR / "stage_ablation_summary.csv", stage_rows)
    (OUTPUT_DIR / "actual_bridge_summary.md").write_text(
        build_bridge_markdown(bridge_rows), encoding="utf-8"
    )
    (OUTPUT_DIR / "stage_ablation_summary.md").write_text(
        build_stage_markdown(stage_rows), encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(
        build_readme(bridge_rows, stage_rows), encoding="utf-8"
    )

    print(str((OUTPUT_DIR / "README.md").resolve()))


if __name__ == "__main__":
    main()

