# CASK v2 Teacher-Forced Fidelity Sweep

This directory snapshots the local RTX 5070 Ti teacher-forced fidelity sweep
used to close the CASK v2 paper-facing verification loop.

Scope:
- model: `Qwen3-8B`
- reference: `fullkv`
- candidates: `triattention`, `cask`
- budgets: `104`, `128`, `160`, `192`
- witnesses: `hexagon`, `geometry248`, `geometry434`
- additional prompt-heavy LongBench witnesses: `qasper`, `multi_news`
- archived prompt-heavy boundary case: `2wikimqa`

Primary files:
- `teacher_forced_budget_sweep_summary.csv`
- `teacher_forced_budget_sweep_summary.json`
- `longbench_qasper_prompt_heavy_witness.md`
- `prompt_heavy_stage_summary.csv`
- `prompt_heavy_stage_summary.json`
- `prompt_heavy_output_sanity.csv`
- `prompt_heavy_output_sanity.json`
- `prompt_heavy_task_metrics.csv`
- `prompt_heavy_task_metrics.json`
- `math_actual_accuracy_subset.md`
- `prompt_heavy_stage_and_output_summary.md`
- `submission_gate_checks.md`

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
- `multi_news`: the H100 rerun keeps this as the clean prompt-heavy **decode-active** witness. At both `256` and `384`, CASK improves `top1`, `top5`, and `mean_nll` over TriAttention.
- `hotpotqa`: the H100 rerun adds the strongest prompt-heavy same-budget witness. At `384`, CASK reaches `top1 = 96.9%`, `top5 = 100.0%`, and `mean_nll = 0.110` against TriAttention's `81.3%`, `90.6%`, and `1.344`.
- `qasper`: the H100 rerun confirms same-budget prompt-heavy wins at `256` and `384`; the older local witness still supplies the clean budget-crossing story.
- `musique`: the H100 rerun shows smaller but consistent same-budget fidelity gains.
- `2wikimqa`: under the H100 prompt-heavy matrix, this remains the boundary case. `cask @ 256` wins, while `cask @ 384` improves `top5` and `mean_nll` but does not flip `top1`.
- `hexagon`: on the reasoning-side actual witness, `triattention @ 104` fails while `cask @ 104` still produces the correct answer `42`, giving a compact example where the fidelity gap maps to a real answer flip.
- `math_actual_accuracy_subset.md`: the smallest math-side bridge check now shows `cask` doubling draw-level exact match (`2/12 -> 4/12`) on a 3-witness subset at `budget = 104`, driven by a `hexagon` robustness gain (`2/4 -> 4/4`) even though subset-level `pass@4` stays tied.
- `prompt_heavy_stage_and_output_summary.md`: consolidates the prompt-heavy stage decomposition, the archived `2wikimqa` local ablation, and the output-level sanity table into one tracked note.
- `submission_gate_checks.md` consolidates the extra prompt-heavy witness, one output-level sanity check, and one representative-mode ablation used to judge submission readiness.
- `first_mismatch` is useful for charts but should not be interpreted alone; on `geometry248` it is non-monotonic across budgets, so the chart should be paired with `top1` or `mean_nll`.

Provenance:
- Raw per-run replay reports were generated under `experiments/reports/` and are intentionally ignored from git.
- This directory contains the tracked summary extracted from those local reports for paper preparation.
