from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "paper_artifacts" / "h100_2026_04_11" / "figures"

CASK_COLOR = "#2563EB"
TRI_COLOR = "#DC2626"
CASK_MARKER = "s"
TRI_MARKER = "o"

BASE_PROMPT_TASKS = ["hotpotqa", "multi_news", "qasper", "musique", "2wikimqa"]
WITNESS_TASKS = [
    ("hotpotqa", "strongest\nsame-budget", "#059669"),
    ("multi_news", "decode-\nactive", "#2563EB"),
    ("qasper", "prefix-\ndominant", "#7C3AED"),
    ("musique", "weaker\ngain", "#D97706"),
    ("2wikimqa", "retained\nboundary", "#DC2626"),
    ("vcsum", "decode-active\nfollow-up", "#0891B2"),
    ("qmsum", "prefix-only\nboundary", "#6B7280"),
    ("gov_report", "failure\nboundary", "#7F1D1D"),
]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing H100 figure pack.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory for PNG/PDF figures (default: {DEFAULT_OUT})",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def resolve_repo_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    normalized = str(raw_path).replace("/", "\\")
    for marker in ("experiments\\", "paper_artifacts\\", "docs\\", "scripts\\"):
        idx = normalized.lower().find(marker)
        if idx != -1:
            suffix = normalized[idx:].replace("\\", "/")
            return ROOT / suffix
    return None


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png")
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)


def load_reasoning_slice(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    rows = load_csv(path)
    result: dict[str, dict[int, dict[str, float]]] = {}
    for row in rows:
        method = row["method"]
        budget = int(row["budget"])
        result.setdefault(method, {})[budget] = {
            "top1": float(row.get("top1", row.get("top1_pct"))),
            "top5": float(row.get("top5", row.get("top5_pct"))),
            "mean_nll": float(row["mean_nll"]),
            "first_mismatch": float(row["first_mismatch"]),
        }
    return result


def build_reasoning_gate_figure(outdir: Path) -> str:
    budgets = [256, 384, 512]
    aime24 = load_reasoning_slice(
        ROOT / "paper_artifacts" / "h100_2026_04_10" / "cask_h100_fidelity" / "aime24_ref6_h100_fidelity_summary.csv"
    )
    aime25 = load_reasoning_slice(
        ROOT / "paper_artifacts" / "h100_2026_04_10" / "cask_h100_fidelity" / "aime25_ref6_h100_fidelity_summary.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    slices = [
        ("AIME24 (6-example)", aime24),
        ("AIME25 (6-example)", aime25),
    ]

    for col, (slice_name, rows) in enumerate(slices):
        tri_top1 = [rows["triattention"][budget]["top1"] for budget in budgets]
        cask_top1 = [rows["cask"][budget]["top1"] for budget in budgets]
        tri_nll = [rows["triattention"][budget]["mean_nll"] for budget in budgets]
        cask_nll = [rows["cask"][budget]["mean_nll"] for budget in budgets]

        top_ax = axes[0, col]
        top_ax.plot(
            budgets,
            tri_top1,
            marker=TRI_MARKER,
            color=TRI_COLOR,
            linewidth=1.5,
            markersize=6,
            label="TriAttention",
            linestyle="--",
        )
        top_ax.plot(
            budgets,
            cask_top1,
            marker=CASK_MARKER,
            color=CASK_COLOR,
            linewidth=1.5,
            markersize=6,
            label="CASK",
            linestyle="-",
        )
        top_ax.set_ylabel("Top-1 Agreement (%)")
        top_ax.set_title(slice_name, fontweight="bold")
        top_ax.legend(loc="lower right", framealpha=0.9)
        top_ax.set_ylim(84, 93)

        nll_ax = axes[1, col]
        nll_ax.plot(
            budgets,
            tri_nll,
            marker=TRI_MARKER,
            color=TRI_COLOR,
            linewidth=1.5,
            markersize=6,
            label="TriAttention",
            linestyle="--",
        )
        nll_ax.plot(
            budgets,
            cask_nll,
            marker=CASK_MARKER,
            color=CASK_COLOR,
            linewidth=1.5,
            markersize=6,
            label="CASK",
            linestyle="-",
        )
        nll_ax.set_ylabel("Mean NLL")
        nll_ax.set_xlabel("KV Budget")
        nll_ax.legend(loc="upper right", framealpha=0.9)
        nll_ax.invert_yaxis()
        nll_ax.set_ylim(0.55, 0.20)

        for ax in (top_ax, nll_ax):
            ax.set_xticks(budgets)

        if col == 0:
            top_ax.annotate(
                "",
                xy=(384, cask_top1[1]),
                xytext=(512, tri_top1[2]),
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.2, ls="dotted"),
            )
            top_ax.text(
                450,
                (cask_top1[1] + tri_top1[2]) / 2 + 0.3,
                "crossing",
                fontsize=7.5,
                color="#6B7280",
                ha="center",
                style="italic",
            )

    fig.suptitle(
        "Figure 1. H100 Reasoning Replay Gate: Top-1 Agreement and Mean NLL vs Budget",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, outdir, "fig1_reasoning_gate_frontier")
    return "Figure 1. H100 reasoning replay gate frontier."


def load_promptheavy_rows() -> list[dict[str, Any]]:
    return load_json(
        ROOT / "paper_artifacts" / "h100_2026_04_11" / "promptheavy_saved_ratio_audit" / "promptheavy_saved_ratio_audit.json"
    )


def load_output_tokens(row: dict[str, Any]) -> int:
    report_path = resolve_repo_path(row.get("report_path"))
    if report_path is None:
        raise FileNotFoundError(f"Could not resolve report path for {row}")
    report = load_json(report_path)
    if report.get("records"):
        return int(report["records"][0]["reference_output_tokens"])
    raise KeyError(f"reference_output_tokens missing in {report_path}")


def build_promptheavy_aggregate_figure(outdir: Path) -> str:
    rows = load_promptheavy_rows()
    weights: dict[tuple[str, int], dict[str, float]] = {}

    for method in ("triattention", "cask"):
        for budget in (256, 384):
            total_tokens = 0
            top1_sum = 0.0
            top5_sum = 0.0
            nll_sum = 0.0
            for task in BASE_PROMPT_TASKS:
                row = next(
                    item
                    for item in rows
                    if item["task"] == task and item["method"] == method and int(item["budget"]) == budget
                )
                output_tokens = load_output_tokens(row)
                total_tokens += output_tokens
                top1_sum += float(row["top1"]) * output_tokens
                top5_sum += float(row["top5"]) * output_tokens
                nll_sum += float(row["mean_nll"]) * output_tokens
            weights[(method, budget)] = {
                "top1_pct": 100.0 * top1_sum / total_tokens,
                "top5_pct": 100.0 * top5_sum / total_tokens,
                "mean_nll": nll_sum / total_tokens,
            }

    methods = ["TriAttention\n@256", "CASK\n@256", "TriAttention\n@384", "CASK\n@384"]
    ordered = [
        weights[("triattention", 256)],
        weights[("cask", 256)],
        weights[("triattention", 384)],
        weights[("cask", 384)],
    ]
    w_top1 = [item["top1_pct"] for item in ordered]
    w_top5 = [item["top5_pct"] for item in ordered]
    w_nll = [item["mean_nll"] for item in ordered]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = list(range(len(methods)))
    width = 0.35
    bar_colors = [TRI_COLOR, CASK_COLOR, TRI_COLOR, CASK_COLOR]

    lighter_top1 = ["#FECACA" if color == TRI_COLOR else CASK_COLOR for color in bar_colors]
    lighter_top5 = ["#FCA5A5" if color == TRI_COLOR else CASK_COLOR for color in bar_colors]
    edge_colors = [TRI_COLOR if color == TRI_COLOR else CASK_COLOR for color in bar_colors]

    ax1.bar(
        [value - width / 2 for value in x],
        w_top1,
        width,
        label="Weighted Top-1",
        color=lighter_top1,
        edgecolor=edge_colors,
        linewidth=0.8,
    )
    ax1.bar(
        [value + width / 2 for value in x],
        w_top5,
        width,
        label="Weighted Top-5",
        color=lighter_top5,
        edgecolor=edge_colors,
        linewidth=0.8,
        alpha=0.7,
    )
    ax1.set_ylabel("Agreement (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_ylim(50, 95)
    ax1.set_title("Weighted Top-1 & Top-5", fontweight="bold")

    bars = ax2.bar(x, w_nll, 0.5, color=bar_colors, edgecolor="#374151", linewidth=0.8, alpha=0.85)
    ax2.set_ylabel("Weighted Mean NLL")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_title("Weighted Mean NLL", fontweight="bold")
    ax2.set_ylim(1.0, 2.2)
    for bar, value in zip(bars, w_nll):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 2. Prompt-Heavy Replay: Weighted Aggregate Comparison", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, outdir, "fig2_promptheavy_aggregate")
    return "Figure 2. H100 prompt-heavy weighted aggregate."


def promptheavy_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {(item["task"], int(item["budget"]), item["method"]): item for item in rows}


def load_replay_summary_json(relpath: str) -> dict[str, Any]:
    data = load_json(ROOT / relpath)
    record = data["records"][0]
    summary = data["summary"]
    guard_reason = None
    if record.get("guard_triggered"):
        if record.get("last_guard") and record["last_guard"].get("reason"):
            guard_reason = record["last_guard"]["reason"]
        elif summary.get("guard_reason_counts"):
            guard_reason = next(iter(summary["guard_reason_counts"]))
        else:
            guard_reason = "guarded"
    return {
        "top1": float(summary["mean_target_top1_match_rate"]),
        "top5": float(summary["mean_target_top5_match_rate"]),
        "mean_nll": float(summary["mean_target_nll"]),
        "first_mismatch": float(summary["mean_first_top1_mismatch_step"]),
        "prefix_events": int(record.get("prefix_compression_events") or 0),
        "decode_events": int(record.get("compression_events") or 0),
        "guard_reason": guard_reason,
    }


def build_witness_map_figure(outdir: Path) -> str:
    audit_rows = load_promptheavy_rows()
    indexed = promptheavy_index(audit_rows)

    extra_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for task in ("vcsum", "qmsum", "gov_report"):
        for budget in (256, 384):
            for method in ("triattention", "cask"):
                extra_rows[(task, budget, method)] = load_replay_summary_json(
                    f"experiments/frontier/Qwen3-8B/h100_decode_probe_{task}_20260411_{task}_replay/{method}_budget_{budget}.json"
                )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.4))
    y_positions = list(range(len(WITNESS_TASKS)))

    for idx, (task, role, color) in enumerate(WITNESS_TASKS):
        if task in {"vcsum", "qmsum", "gov_report"}:
            tri_256 = extra_rows[(task, 256, "triattention")]
            tri_384 = extra_rows[(task, 384, "triattention")]
            cask_256 = extra_rows[(task, 256, "cask")]
            cask_384 = extra_rows[(task, 384, "cask")]
        else:
            tri_256 = indexed[(task, 256, "triattention")]
            tri_384 = indexed[(task, 384, "triattention")]
            cask_256 = indexed[(task, 256, "cask")]
            cask_384 = indexed[(task, 384, "cask")]

        delta_top1_256 = 100.0 * (float(cask_256["top1"]) - float(tri_256["top1"]))
        delta_top1_384 = 100.0 * (float(cask_384["top1"]) - float(tri_384["top1"]))
        delta_nll_256 = float(cask_256["mean_nll"]) - float(tri_256["mean_nll"])
        delta_nll_384 = float(cask_384["mean_nll"]) - float(tri_384["mean_nll"])

        ax1.scatter(delta_top1_256, idx - 0.12, marker="o", s=80, color=color, zorder=5, edgecolors="white", linewidth=0.5)
        ax1.scatter(delta_top1_384, idx + 0.12, marker="D", s=60, color=color, zorder=5, edgecolors="white", linewidth=0.5, alpha=0.7)
        ax2.scatter(delta_nll_256, idx - 0.12, marker="o", s=80, color=color, zorder=5, edgecolors="white", linewidth=0.5)
        ax2.scatter(delta_nll_384, idx + 0.12, marker="D", s=60, color=color, zorder=5, edgecolors="white", linewidth=0.5, alpha=0.7)

        cask_384_decode = int(cask_384.get("decode_events") or 0)
        cask_384_prefix = int(cask_384.get("prefix_events") or 0)
        guard_reason = cask_384.get("guard_reason")
        if cask_384_decode > 0:
            ax1.annotate(
                f"p={cask_384_prefix}, d={cask_384_decode}",
                xy=(delta_top1_384 + 0.2, idx + 0.12),
                fontsize=6.8,
                color="#2563EB",
                style="italic",
            )
        elif guard_reason:
            ax1.annotate(
                guard_reason,
                xy=(delta_top1_384 + 0.2, idx + 0.12),
                fontsize=6.5,
                color="#7F1D1D",
                style="italic",
            )

    ax1.axvline(x=0, color="#9CA3AF", linewidth=0.8, linestyle="-")
    ax2.axvline(x=0, color="#9CA3AF", linewidth=0.8, linestyle="-")
    labels = [f"{task}\n({role})" for task, role, _ in WITNESS_TASKS]
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(labels, fontsize=8)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Delta Top-1 (%p)")
    ax1.set_title("Top-1 Delta (CASK - TriAttention)", fontweight="bold")
    ax1.legend(["Budget 256", "Budget 384"], loc="lower right", framealpha=0.9, fontsize=8)
    ax1.set_xlim(-8.0, 18.5)

    ax2.set_xlabel("Delta Mean NLL")
    ax2.set_title("Mean NLL Delta (CASK - TriAttention, lower is better)", fontweight="bold")
    ax2.set_xlim(-3.5, 1.1)

    fig.suptitle("Figure 3. Prompt-Heavy Witness Map: Active Wins, Prefix-Only Wins, and Boundaries", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, outdir, "fig3_promptheavy_witness_map")
    return "Figure 3. Prompt-heavy witness map with active, prefix-only, and boundary regimes."


def read_longbench_task_metric(task: str, variant_dir: str) -> float:
    path = ROOT / "experiments" / "longbench_h100_actual_bridge_20260411" / variant_dir / "longbench" / "Qwen3-8B" / "longbench_eval.json"
    payload = load_json(path)
    return float(payload["task_scores"][task]["overall"])


def build_actual_bridge_figure(outdir: Path) -> str:
    rows = load_json(
        ROOT / "paper_artifacts" / "h100_2026_04_11" / "cask_h100_actual_bridge" / "actual_bridge_summary.json"
    )
    indexed = {(row["task"], row["method"]): row for row in rows}

    vcsum_tri = load_json(ROOT / "experiments" / "frontier" / "Qwen3-8B" / "h100_actual_bridge_metrics_20260411" / "vcsum_tri384.json")
    vcsum_cask = load_json(ROOT / "experiments" / "frontier" / "Qwen3-8B" / "h100_actual_bridge_metrics_20260411" / "vcsum_cask384.json")

    bridge_rows = [
        {
            "task": "qasper",
            "role": "crossing",
            "tri": indexed[("qasper", "triattention")],
            "cask": indexed[("qasper", "cask")],
            "tri_budget": "@512",
            "cask_budget": "@256",
        },
        {
            "task": "multi_news",
            "role": "same-budget\nbridge",
            "tri": indexed[("multi_news", "triattention")],
            "cask": indexed[("multi_news", "cask")],
            "tri_budget": "@384",
            "cask_budget": "@384",
        },
        {
            "task": "hotpotqa",
            "role": "parity",
            "tri": indexed[("hotpotqa", "triattention")],
            "cask": indexed[("hotpotqa", "cask")],
            "tri_budget": "@256",
            "cask_budget": "@256",
        },
        {
            "task": "vcsum",
            "role": "lexical-semantic\nsplit",
            "tri": {
                "sequence_ratio": vcsum_tri["fidelity"]["mean_sequence_ratio"],
                "semantic_similarity": vcsum_tri["fidelity"]["mean_semantic_similarity"],
                "task_metric": read_longbench_task_metric("vcsum", "vcsum_tri384"),
            },
            "cask": {
                "sequence_ratio": vcsum_cask["fidelity"]["mean_sequence_ratio"],
                "semantic_similarity": vcsum_cask["fidelity"]["mean_semantic_similarity"],
                "task_metric": read_longbench_task_metric("vcsum", "vcsum_cask384"),
            },
            "tri_budget": "@384",
            "cask_budget": "@384",
        },
    ]

    x = list(range(len(bridge_rows)))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))

    tri_seq = [100.0 * float(row["tri"]["sequence_ratio"]) for row in bridge_rows]
    cask_seq = [100.0 * float(row["cask"]["sequence_ratio"]) for row in bridge_rows]
    tri_sem = [float(row["tri"]["semantic_similarity"]) for row in bridge_rows]
    cask_sem = [float(row["cask"]["semantic_similarity"]) for row in bridge_rows]
    tri_metric = [float(row["tri"]["task_metric"]) for row in bridge_rows]
    cask_metric = [float(row["cask"]["task_metric"]) for row in bridge_rows]
    labels = [f"{row['task']}\n({row['role']})" for row in bridge_rows]

    def draw_pair(ax: plt.Axes, tri_values: list[float], cask_values: list[float], ylabel: str, title: str, ylim: tuple[float, float], budget_offset: float) -> None:
        bars_tri = ax.bar([value - width / 2 for value in x], tri_values, width, label="TriAttention", color=TRI_COLOR, alpha=0.7, edgecolor="#374151", linewidth=0.8)
        bars_cask = ax.bar([value + width / 2 for value in x], cask_values, width, label="CASK", color=CASK_COLOR, alpha=0.85, edgecolor="#374151", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.2)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(*ylim)
        for idx, row in enumerate(bridge_rows):
            ax.text(idx - width / 2, tri_values[idx] + budget_offset, f"Tri{row['tri_budget']}", ha="center", fontsize=6.2, color=TRI_COLOR)
            ax.text(idx + width / 2, cask_values[idx] + budget_offset, f"CASK{row['cask_budget']}", ha="center", fontsize=6.2, color=CASK_COLOR)
        return bars_tri, bars_cask

    draw_pair(axes[0], tri_seq, cask_seq, "Sequence Ratio (%)", "Lexical Overlap vs Full-KV", (0, 110), 2.0)
    bars_tri, bars_cask = draw_pair(axes[1], tri_sem, cask_sem, "Semantic Similarity", "Semantic / Reference Similarity", (0.35, 1.08), 0.015)
    draw_pair(axes[2], tri_metric, cask_metric, "Task Metric", "Official Task Metric", (0, 30), 0.7)
    axes[0].legend(framealpha=0.9, loc="upper left")

    fig.suptitle("Figure 4. Actual-Output Bridge: Lexical, Semantic, and Task-Level Readouts", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    save_figure(fig, outdir, "fig4_actual_output_bridge")
    return "Figure 4. Actual-output bridge across lexical, semantic, and task-level metrics."


def build_method_overview_figure(outdir: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    boxes = [
        (0.5, 1.0, 2.2, 1.5, "#FEF3C7", "#D97706", "Stage 1\nPrefix\nEviction"),
        (3.2, 1.0, 2.2, 1.5, "#DBEAFE", "#2563EB", "Stage 2\nCore / Scratch\nDecomposition"),
        (5.9, 1.0, 2.2, 1.5, "#D1FAE5", "#059669", "Selective\nScratch\nConsolidation"),
        (8.6, 1.0, 2.2, 1.5, "#F3E8FF", "#7C3AED", "Final\nRepresentative\nSet"),
    ]

    for x, y, width, height, face_color, edge_color, label in boxes:
        rect = plt.Rectangle((x, y), width, height, facecolor=face_color, edgecolor=edge_color, linewidth=1.5, zorder=2, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9, fontweight="bold", color=edge_color, zorder=3)

    arrow_props = dict(arrowstyle="->", color="#374151", lw=1.5)
    for x_start, x_end in ((2.7, 3.2), (5.4, 5.9), (8.1, 8.6)):
        ax.annotate("", xy=(x_end, 1.75), xytext=(x_start, 1.75), arrowprops=arrow_props)

    ax.text(1.6, 2.7, "Prefix tokens", ha="center", fontsize=8, style="italic", color="#92400E")
    ax.text(4.3, 2.7, "Protected core versus scratch", ha="center", fontsize=8, style="italic", color="#1E40AF")
    ax.text(7.0, 2.7, "Merge scratch only", ha="center", fontsize=8, style="italic", color="#065F46")
    ax.text(9.7, 2.7, "C subseteq R, |R| <= B", ha="center", fontsize=8, style="italic", color="#5B21B6")
    ax.text(
        5.5,
        0.3,
        "Two-stage compression: prefix eviction -> decode-time core-aware selective consolidation",
        ha="center",
        fontsize=10,
        color="#374151",
        fontweight="bold",
    )

    fig.suptitle("Figure 5. CASK Method Overview", fontsize=12, fontweight="bold", y=0.98)
    save_figure(fig, outdir, "fig5_method_overview")
    return "Figure 5. Method overview for two-stage CASK."


def write_figure_index(outdir: Path, captions: list[tuple[str, str]]) -> None:
    lines = [
        "# H100 Figure Pack",
        "",
        "This directory stores the current paper-facing figure pack generated from the tracked H100 artifact summaries.",
        "",
        "## Files",
        "",
    ]
    for stem, caption in captions:
        lines.append(f"- [`{stem}.png`]({stem}.png) / [`{stem}.pdf`]({stem}.pdf): {caption}")
    lines.extend(
        [
            "",
            "## Source packages",
            "",
            "- [`paper_artifacts/h100_2026_04_10/cask_h100_fidelity/`](../h100_2026_04_10/cask_h100_fidelity/)",
            "- [`paper_artifacts/h100_2026_04_11/promptheavy_saved_ratio_audit/`](../promptheavy_saved_ratio_audit/)",
            "- [`paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/`](../cask_h100_actual_bridge/)",
            "- [`paper_artifacts/h100_2026_04_11/decode_active_replay_probe.md`](../decode_active_replay_probe.md)",
            "- [`paper_artifacts/h100_2026_04_11/coverage_followup_probe.md`](../coverage_followup_probe.md)",
            "",
            "## Regeneration",
            "",
            "```bash",
            "python scripts/build_paper_figures.py",
            "```",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    captions = [
        ("fig1_reasoning_gate_frontier", build_reasoning_gate_figure(outdir)),
        ("fig2_promptheavy_aggregate", build_promptheavy_aggregate_figure(outdir)),
        ("fig3_promptheavy_witness_map", build_witness_map_figure(outdir)),
        ("fig4_actual_output_bridge", build_actual_bridge_figure(outdir)),
        ("fig5_method_overview", build_method_overview_figure(outdir)),
    ]
    write_figure_index(outdir, captions)

    print(f"Wrote {len(captions)} figures to {outdir}")


if __name__ == "__main__":
    main()
