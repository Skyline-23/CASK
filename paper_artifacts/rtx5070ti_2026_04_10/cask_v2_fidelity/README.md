# CASK v2 Teacher-Forced Fidelity Sweep

This directory snapshots the local RTX 5070 Ti teacher-forced fidelity sweep
used to close the CASK v2 paper-facing verification loop.

Scope:
- model: `Qwen3-8B`
- reference: `fullkv`
- candidates: `triattention`, `cask`
- budgets: `104`, `128`, `160`, `192`
- witnesses: `hexagon`, `geometry248`, `geometry434`

Primary files:
- `teacher_forced_budget_sweep_summary.csv`
- `teacher_forced_budget_sweep_summary.json`

Columns:
- `problem`: local witness label
- `budget`: KV budget used during replay
- `method`: `triattention` or `cask`
- `first_mismatch`: first step where top-1 next-token prediction diverges from the full-KV reference
- `top1`, `top5`: teacher-forced next-token agreement rates
- `strict_prefix`: fraction of the continuation preserved before the first top-1 mismatch
- `mean_nll`: teacher-forced negative log-likelihood on the reference continuation
- `saved_ratio`: terminal KV savings ratio relative to the replayed candidate state
- `cache_ratio`: terminal cache footprint ratio
- `prefix_compression_events`: number of prefix-stage v2 evictions recorded by CASK
- `compression_events`: number of decode-stage merge events

High-level takeaways:
- `geometry248`: CASK v2 beats TriAttention on `top1` and `mean_nll` at every tested budget, while TriAttention keeps slightly higher terminal savings.
- `geometry434`: CASK v2 wins on `104`, `128`, and `192`; `160` is effectively parity and should be treated as a tradeoff point.
- `hexagon`: CASK clearly wins at `104` and `192`, while `128` and `160` are close-to-parity budgets.
- `first_mismatch` is useful for charts but should not be interpreted alone; on `geometry248` it is non-monotonic across budgets, so the chart should be paired with `top1` or `mean_nll`.

Provenance:
- Raw per-run replay reports were generated under `experiments/reports/` and are intentionally ignored from git.
- This directory contains the tracked summary extracted from those local reports for paper preparation.
