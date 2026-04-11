# H100 Actual-Output Bridge

This package contains the output-level evidence that complements the replay gate. Open this when you want to answer: "does the replay advantage show up in actual generation too?"

## What This Package Answers

| Question | Answer source |
| --- | --- |
| Is there a clean output-level budget crossing? | `actual_bridge_summary.md` |
| Is there a same-budget output bridge on a prompt-heavy task? | `actual_bridge_summary.md` |
| Do we have a decode-stage claim boundary? | `stage_ablation_summary.md` |

## Read This First

1. `actual_bridge_summary.md`
2. `stage_ablation_summary.md`
3. `actual_bridge_summary.csv` / `stage_ablation_summary.csv` if you want the table source

## Headline Read

| Task | Main read |
| --- | --- |
| `qasper` | clean output-level budget crossing: `CASK @ 256` beats `TriAttention @ 512` on both `sequence_ratio` and `task_metric` |
| `multi_news` | strongest same-budget output bridge: `CASK @ 384` beats `TriAttention @ 384` and is also the cleanest decode-active output witness |
| `hotpotqa` | non-regression parity witness rather than a win case |
| `multi_news` stage ablation | useful as a claim boundary; it does **not** justify a large standalone stage-2 output claim |

## File Guide

| File | What it contains | When to open it |
| --- | --- | --- |
| `actual_bridge_summary.md` | paper-facing readout of the output-level bridge rows | first stop |
| `actual_bridge_summary.csv` | compact table for the same rows | when you want copyable numbers |
| `actual_bridge_summary.json` | same rows plus raw provenance | when tracing back to raw experiments |
| `stage_ablation_summary.md` | output-level boundary for `multi_news` stage ablation | when discussing decode-stage caution |
| `stage_ablation_summary.csv` | compact ablation table | when you want the raw values |
| `stage_ablation_summary.json` | ablation rows with raw provenance | when auditing inputs |

## Scope

| Field | Value |
| --- | --- |
| Model | `Qwen3-8B` |
| Hardware | `H100 PCIe` |
| Main tasks | `qasper`, `multi_news`, `hotpotqa` |
| Metric family | `sequence_ratio`, `task_metric`, `terminal_saved_ratio` |

## Raw Provenance

- replay metrics: `experiments/frontier/Qwen3-8B/h100_actual_bridge_metrics_20260411/`
- generation runs: `experiments/longbench_h100_actual_bridge_20260411/`
- stage-ablation generation run: `experiments/longbench_h100_stage_ablation/`
- stage-ablation replay metrics: `experiments/frontier/Qwen3-8B/h100_promptheavy_stage_ablation_20260411/`
