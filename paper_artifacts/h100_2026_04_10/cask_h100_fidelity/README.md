# H100 Reasoning Replay Gate

This package is the main H100 replay-level evidence for the paper's reasoning story.

## What This Package Answers

| Question | Answer source |
| --- | --- |
| Does CASK beat TriAttention at the same budget on reasoning replay? | `aime24_ref6_h100_fidelity_summary.csv`, `aime25_ref6_h100_fidelity_summary.csv` |
| Is there any budget crossing? | same CSV files; compare `cask@256` vs `triattention@384`, and `cask@384` vs `triattention@512` |
| Are the gains fidelity-only or broad enough to matter for the paper? | this package establishes the replay gate, not final benchmark accuracy |

## Read This First

1. `aime24_ref6_h100_fidelity_summary.csv`
2. `aime25_ref6_h100_fidelity_summary.csv`
3. `aime24_ref6_h100_fidelity_summary.json` / `aime25_ref6_h100_fidelity_summary.json` if you need per-row provenance

## Headline Read

| Slice | Main read |
| --- | --- |
| `AIME24 ref6` | CASK wins the same-budget replay gate at `256`, `384`, and `512`, and shows two clean crossing points: `cask@256 > triattention@384` and `cask@384 > triattention@512` |
| `AIME25 ref6` | CASK again wins the same-budget replay gate at `256`, `384`, and `512`; the crossing is weaker than on `AIME24`, but `cask@384 > triattention@512` still holds |

## File Guide

| File | What it contains | When to open it |
| --- | --- | --- |
| `aime24_ref6_h100_fidelity_summary.csv` | compact replay summary for the 6-example `AIME24` slice | first stop for main table numbers |
| `aime25_ref6_h100_fidelity_summary.csv` | compact replay summary for the 6-example `AIME25` slice | first stop for main table numbers |
| `aime24_ref6_h100_fidelity_summary.json` | same data plus explicit `source_json` provenance | when you need to trace a row back to raw outputs |
| `aime25_ref6_h100_fidelity_summary.json` | same data plus explicit `source_json` provenance | when you need raw-path traceability |
| `MANIFEST.sha256` | integrity manifest for tracked files | only if you need a frozen artifact audit |

## Scope

| Field | Value |
| --- | --- |
| Model | `Qwen3-8B` |
| Hardware | `H100 PCIe` |
| Methods | `triattention`, `cask` |
| Budgets | `256`, `384`, `512` |
| Metric family | replay `top1`, `top5`, `mean_nll`, `first_mismatch`, `saved_ratio` |

## Caveat

This is a **replay-fidelity** package. It should support the full-KV similarity story and the reasoning gate, but it should not be cited as if it were already the final benchmark-accuracy table.

## Raw Provenance

- `experiments/frontier/Qwen3-8B/h100_aime24_fidelity_gate_20260410/`
- `experiments/frontier/Qwen3-8B/h100_aime25_fidelity_gate_20260410/`
