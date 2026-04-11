# Paper Artifacts Guide

This directory contains the paper-facing evidence packages for CASK. Start here if you want to understand which files matter, what each package proves, and what to cite in the paper.

Command trace:
[`COMMAND_MAP.md`](COMMAND_MAP.md)

## Read Order

| If you want to answer this question | Open this first | Then drill into |
| --- | --- | --- |
| Does CASK beat TriAttention on the main reasoning replay gate? | [`h100_2026_04_10/cask_h100_fidelity/README.md`](h100_2026_04_10/cask_h100_fidelity/README.md) | [`aime24_ref6_h100_fidelity_summary.csv`](h100_2026_04_10/cask_h100_fidelity/aime24_ref6_h100_fidelity_summary.csv), [`aime25_ref6_h100_fidelity_summary.csv`](h100_2026_04_10/cask_h100_fidelity/aime25_ref6_h100_fidelity_summary.csv) |
| Does CASK preserve output quality, not just replay fidelity? | [`h100_2026_04_11/cask_h100_actual_bridge/README.md`](h100_2026_04_11/cask_h100_actual_bridge/README.md) | [`actual_bridge_summary.md`](h100_2026_04_11/cask_h100_actual_bridge/actual_bridge_summary.md), [`stage_ablation_summary.md`](h100_2026_04_11/cask_h100_actual_bridge/stage_ablation_summary.md) |
| What is the prompt-heavy story, including decode-active witnesses and failure boundaries? | [`h100_2026_04_11/README.md`](h100_2026_04_11/README.md) | [`decode_active_replay_probe.md`](h100_2026_04_11/decode_active_replay_probe.md), [`coverage_followup_probe.md`](h100_2026_04_11/coverage_followup_probe.md), [`promptheavy_replay_readout.md`](h100_2026_04_11/promptheavy_saved_ratio_audit/promptheavy_replay_readout.md) |
| What is the cheapest local evidence package for debugging and compact bridge examples? | [`rtx5070ti_2026_04_10/cask_v2_fidelity/README.md`](rtx5070ti_2026_04_10/cask_v2_fidelity/README.md) | [`teacher_forced_budget_sweep_summary.csv`](rtx5070ti_2026_04_10/cask_v2_fidelity/teacher_forced_budget_sweep_summary.csv), [`math_actual_accuracy_subset.md`](rtx5070ti_2026_04_10/cask_v2_fidelity/math_actual_accuracy_subset.md), [`submission_gate_checks.md`](rtx5070ti_2026_04_10/cask_v2_fidelity/submission_gate_checks.md) |

## Package Map

| Package | Main role | Strongest paper-facing use |
| --- | --- | --- |
| [`h100_2026_04_10/cask_h100_fidelity/`](h100_2026_04_10/cask_h100_fidelity/) | H100 reasoning replay gate | same-budget replay advantage and partial budget crossing on `AIME24` / `AIME25` |
| [`h100_2026_04_11/cask_h100_actual_bridge/`](h100_2026_04_11/cask_h100_actual_bridge/) | H100 output-level bridge | `qasper` budget crossing, `multi_news` same-budget output bridge, `hotpotqa` parity |
| [`h100_2026_04_11/`](h100_2026_04_11/) | H100 prompt-heavy follow-up package | decode-active replay witnesses plus `prefix_budget_exhausted` boundaries |
| [`rtx5070ti_2026_04_10/cask_v2_fidelity/`](rtx5070ti_2026_04_10/cask_v2_fidelity/) | local bridge and budget sweep package | `hexagon` answer-flip bridge and cheap local replay debugging |

## What Not To Cite As A Headline

| Item | Why not headline it |
| --- | --- |
| `2wikimqa` | retained boundary case, not a clean prompt-heavy win |
| `qmsum` / `gov_report` | useful for regime separation, not decode-active evidence |
| local-only one-off sanity checks | useful for diagnosis, weaker than the H100 packages for headline claims |

## Provenance

Every package README points to tracked summary files that already carry raw provenance fields such as `source_json` or `source_eval_json`. For paper writing, cite the packaged summaries first, then use [`COMMAND_MAP.md`](COMMAND_MAP.md) plus those provenance fields to trace each row back to its generating experiment.
