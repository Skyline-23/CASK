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
| Which finalized paper figures should I paste into the draft? | [`../docs/assets/README.md`](../docs/assets/README.md) | `fig1_reasoning_gate_frontier`, `fig2_promptheavy_aggregate`, `fig3_promptheavy_witness_map`, `fig4_actual_output_bridge`, `fig5_method_overview`, `fig6_mass_recovery_hexagon`, `fig7_cluster_collapse_hexagon` |

## Package Map

| Package | Main role | Strongest paper-facing use |
| --- | --- | --- |
| [`h100_2026_04_10/cask_h100_fidelity/`](h100_2026_04_10/cask_h100_fidelity/) | H100 reasoning replay gate | same-budget replay advantage and partial budget crossing on `AIME24` / `AIME25` |
| [`h100_2026_04_11/cask_h100_actual_bridge/`](h100_2026_04_11/cask_h100_actual_bridge/) | H100 output-level bridge | `qasper` budget crossing, `multi_news` same-budget output bridge, `hotpotqa` parity |
| [`h100_2026_04_11/`](h100_2026_04_11/) | H100 prompt-heavy follow-up package | decode-active replay witnesses plus `prefix_budget_exhausted` boundaries |
| [`../docs/assets/`](../docs/assets/) | paper-ready figure bundle | current PNG/PDF figures synchronized to the tracked H100 artifact summaries |

## What Not To Cite As A Headline

| Item | Why not headline it |
| --- | --- |
| `2wikimqa` | retained boundary case, not a clean prompt-heavy win |
| `qmsum` / `gov_report` | useful for regime separation, not decode-active evidence |

## Provenance

Every package README points to tracked summary files that already carry raw provenance fields such as `source_json` or `source_eval_json`. For paper writing, cite the packaged summaries first, then use [`COMMAND_MAP.md`](COMMAND_MAP.md) plus those provenance fields to trace each row back to its generating experiment.

## Raw Data Policy

The repository keeps raw experiment data in-tree, but not every raw file belongs in the PDF itself.

Use this split:

| Layer | What belongs there | Where it lives |
| --- | --- | --- |
| Main paper | headline tables, compact figures, claim-supporting summaries | `paper/`, `docs/assets/` |
| Appendix / audit | measured counts, witness-level audit matrices, boundary-case readout | `paper/` Appendix C-D, package README files |
| Raw provenance | JSON/CSV reports, manifests, merged references, eval roots, generation roots | `experiments/frontier/`, `experiments/outputs/`, `experiments/longbench_h100_*` |

If a reviewer asks where a row came from, the intended path is:

1. Open the packaged summary in `artifacts/...`
2. Follow its `source_json` or `source_eval_json`
3. Use [`COMMAND_MAP.md`](COMMAND_MAP.md) to recover the exact script, manifest, and raw directory
