# Local CASK v2 Fidelity Package

This is the cheapest paper-facing evidence package in the repo. Open it when you want local replay sweeps, a compact reasoning-side bridge witness, or a quick sanity check before paying for larger H100 runs.

## What This Package Answers

| Question | Best file |
| --- | --- |
| How does CASK behave on the local replay budget sweep? | `teacher_forced_budget_sweep_summary.csv` |
| Do we have a compact answer-flip bridge witness? | `math_actual_accuracy_subset.md` |
| What was the local prompt-heavy story before the H100 packages took over? | `prompt_heavy_stage_and_output_summary.md` |
| What extra sanity checks were used during submission gating? | `submission_gate_checks.md` |

## Read This First

1. `teacher_forced_budget_sweep_summary.csv`
2. `math_actual_accuracy_subset.md`
3. `submission_gate_checks.md`

## Headline Read

| Witness | Main read |
| --- | --- |
| `geometry248` | CASK wins on `top1` and `mean_nll` across the tracked local budgets |
| `geometry434` | CASK wins at most tracked budgets; `160` is close-to-parity |
| `hexagon` | compact reasoning-side answer flip: `triattention@104` fails while `cask@104` still reaches the correct answer |

## File Guide

| File | What it contains | When to open it |
| --- | --- | --- |
| `teacher_forced_budget_sweep_summary.csv` | main local replay table | first stop for local numbers |
| `teacher_forced_budget_sweep_summary.json` | same rows plus provenance | when tracing local reports |
| `math_actual_accuracy_subset.md` | compact local answer-accuracy bridge note | when you need a small answer-flip example |
| `prompt_heavy_stage_and_output_summary.md` | local prompt-heavy consolidation note | historical context only; H100 packages are stronger |
| `submission_gate_checks.md` | extra local sanity checks used before larger runs | when reconstructing decision flow |

## Scope

| Field | Value |
| --- | --- |
| GPU | `RTX 5070 Ti 16 GB` |
| Model | `Qwen3-8B` |
| Methods | `triattention`, `cask` |
| Main local budgets | `104`, `128`, `160`, `192` |
| Main witnesses | `hexagon`, `geometry248`, `geometry434` |

## Caveat

This package is useful for local bridge examples and cheap debugging, but the H100 packages are the stronger sources for headline paper claims.

## Raw Provenance

Raw per-run replay reports were generated under `experiments/reports/` and are intentionally not tracked in git. This directory contains the tracked paper-facing extraction of those local reports.
