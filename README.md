CASK is built on top of TriAttention's codebase and scoring framework.

# CASK

Core-Aware Selective KV Compression for Reasoning Traces

This repository now treats **CASK** as the main paper direction. The central claim is no longer that a better RoPE scorer alone wins; it is that reasoning traces contain a **protected core** plus **mergeable scratch**, and that preserving the core while selectively consolidating scratch can preserve full-KV behavior better than pure eviction at the same budget.

The repository name and the Python namespace still reflect the historical `HorizonKV` / `triattention` lineage. Paper-facing method naming, however, is now **CASK**.

## Main Idea

CASK uses **2-stage compression**:

1. **Stage 1: prefix compression**
   Prompt-heavy runs first shrink the prefix with **TriAttention eviction plus a small coverage reserve** so the decode path is not dead on arrival and long prompts do not collapse to a purely score-ranked prefix.
2. **Stage 2: decode compression**
   Decode-time tokens are split into a **protected core** and **mergeable scratch**. Core tokens are preserved. Scratch tokens are locally consolidated with the KeepKV-style merge operator under CASK's selection policy.

The main internal validation axis is **teacher-forced reference fidelity**, not raw accuracy alone. The question is:

> Given the same prompt, how closely does the compressed policy track a full-KV continuation, and how much physical KV does it save while doing so?

## Code Mapping

- `fullkv`: unconstrained reference.
- `triattention`: original TriAttention eviction baseline.
- `cask`: **CASK mainline implementation**.
- `expectedattention`: closest prior-style baseline kept for comparison.
- `horizonkv`: archived Phase 1 scorer-engineering direction.

If you are running the current paper candidate, use `--method cask`.

## Current Status

- CASK mainline is implemented on the HuggingFace path as `cask`.
- Prompt-heavy runs now use the **v2 two-stage path**: prefix eviction followed by decode-stage consolidation.
- Invalid prompt-heavy regimes are explicitly marked with runtime guards instead of being silently treated as successful merge runs.
- The local RTX 5070 Ti teacher-forced fidelity sweep is tracked under:
  - `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/`
- Historical HorizonKV / adaptive-horizon work is retained in-tree as Phase 1 archive material, not as the current paper mainline.

## Installation

Python 3.10+ is required.

```bash
git clone https://github.com/Skyline-23/CASK.git
cd CASK
pip install -e .
```

For Linux benchmarking and vLLM runtime work, install the matching CUDA / FlashAttention / vLLM stack separately. The local fidelity harnesses in this repository also run on a single 16 GB consumer GPU with HuggingFace + `sdpa`.

## Quick Start

Full-KV reference:

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset math500 \
    --method fullkv
```

TriAttention baseline:

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset math500 \
    --method triattention \
    --budget 104 \
    --stats-path triattention/calibration/for_aime24_experiment/qwen3_8b.pt
```

CASK mainline:

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset math500 \
    --method cask \
    --budget 104 \
    --stats-path triattention/calibration/for_aime24_experiment/qwen3_8b.pt
```

## Fidelity-First Evaluation

The main replay harness is:

- `scripts/replay_reference_fidelity.py`

It replays a full-KV reference continuation under a candidate KV policy and reports:

- `top1`, `top5`
- `strict_prefix`
- `first_mismatch`
- `mean_nll`
- terminal `saved_ratio`

This is the paper-facing metric family for local verification.

Example:

```bash
python scripts/replay_reference_fidelity.py \
    --reference experiments/outputs/math500/Qwen3-8B/sample1/fullkv/fullkv_selection_geometry_reference \
    --model-path experiments/models/Qwen3-8B \
    --method cask \
    --budget 104 \
    --triattention-stats-file triattention/calibration/for_aime24_experiment/qwen3_8b.pt \
    --attn-implementation sdpa \
    --count-prompt-tokens true \
    --slack-budget-trigger true \
    --json-output experiments/reports/geometry_teacher_forced_fidelity_vs_fullkv_sm104_v2.json
```

## Tracked Local Evidence

The tracked local summary lives here:

- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/README.md`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/teacher_forced_budget_sweep_summary.csv`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/teacher_forced_budget_sweep_summary.json`
- `paper_artifacts/h100_2026_04_10/cask_h100_fidelity/README.md`
- `paper_artifacts/h100_2026_04_10/cask_h100_fidelity/aime24_ref6_h100_fidelity_summary.csv`
- `paper_artifacts/h100_2026_04_10/cask_h100_fidelity/aime24_ref6_h100_fidelity_summary.json`
- `paper_artifacts/h100_2026_04_10/cask_h100_fidelity/aime25_ref6_h100_fidelity_summary.csv`
- `paper_artifacts/h100_2026_04_10/cask_h100_fidelity/aime25_ref6_h100_fidelity_summary.json`
- `paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/README.md`
- `paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/actual_bridge_summary.csv`
- `paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/actual_bridge_summary.json`
- `paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/stage_ablation_summary.csv`
- `paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/stage_ablation_summary.json`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/longbench_qasper_prompt_heavy_witness.md`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/prompt_heavy_stage_summary.csv`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/prompt_heavy_output_sanity.csv`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/prompt_heavy_task_metrics.csv`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/math_actual_accuracy_subset.md`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/prompt_heavy_stage_and_output_summary.md`
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/submission_gate_checks.md`

That sweep covers:

- witnesses: `hexagon`, `geometry248`, `geometry434`
- budgets: `104`, `128`, `160`, `192`
- methods: `triattention`, `cask`
- plus one prompt-heavy LongBench witness: `qasper @ 512`

High-level read:

- `geometry248`: CASK beats TriAttention on `top1` and `mean_nll` at every tested budget.
- `geometry434`: CASK wins at `104`, `128`, and `192`; `160` is essentially parity.
- `hexagon`: CASK is clearly stronger at `104` and `192`, with near-parity at `128` and `160`.
- `multi_news`: the H100 rerun keeps this as the prompt-heavy **decode-active** witness. At both `256` and `384`, both-stage CASK improves `top1`, `top5`, and `mean_nll` over `triattention`.
- `multi_news`: the H100 actual-output bridge also survives generation. At `384`, `TriAttention` falls to `sequence_ratio = 0.000` and task metric `0.00`, while `CASK` reaches `sequence_ratio = 0.169` and task metric `15.16`.
- `hotpotqa`: the H100 rerun adds the strongest prompt-heavy same-budget witness. At `384`, `cask` reaches `top1 = 96.9%`, `top5 = 100.0%`, and `mean_nll = 0.110` against `triattention`'s `81.3%`, `90.6%`, and `1.344`.
- `qasper`: the H100 actual-output package now gives the cleanest budget-crossing witness. `CASK @ 256` exceeds `TriAttention @ 512` on both `sequence_ratio` (`0.238 > 0.173`) and task metric (`12.77 > 11.94`), while ending with a `90.9%` terminal saved ratio.
- `hotpotqa`: the H100 actual-output package should be read as non-regression parity, not as a win case. `CASK @ 256` matches `TriAttention @ 256` at `sequence_ratio = 1.000` and task metric `27.27`, while preserving a `97.6%` terminal saved ratio.
- `musique`: the H100 rerun shows smaller but consistent same-budget fidelity gains.
- `2wikimqa`: the later H100 saved-ratio audit reclassified this as an **inactive regime** under the current `divide_length=128` trigger semantics. Neither TriAttention nor CASK fires a compression event before the 32-token continuation ends, so this task should not be cited as prompt-heavy compression evidence or as a saved-ratio comparison.
- `2wikimqa`: the older local coverage-reserve ablation is retained only as archived reverse-engineering context for output drift; it is not part of the current compression headline.
- `hexagon`: on a tracked math witness, `triattention @ 104` fails while `cask @ 104` still produces the correct answer `42`, giving a compact reasoning-side answer-flip example.
- `AIME24 H100 ref6`: `cask` now wins the same-budget full-KV replay gate at `256`, `384`, and `512`, and also shows two explicit crossing points: `cask @ 256 > triattention @ 384` and `cask @ 384 > triattention @ 512`.
- `AIME25 H100 ref6`: `cask` again wins the same-budget full-KV replay gate at `256`, `384`, and `512`. The crossing is weaker than on `AIME24`, but `cask @ 384 > triattention @ 512` still holds on `top1`, `top5`, and `mean_nll`.
- `math_actual_accuracy_subset.md`: small `math500` bridge check showing that `cask` doubles draw-level exact match (`2/12 -> 4/12`) on a 3-witness subset at `budget = 104`, driven by a `hexagon` robustness gain (`2/4 -> 4/4`).
- `prompt_heavy_stage_and_output_summary.md`: consolidates the prompt-heavy stage decomposition, the archived `2wikimqa` local ablation, and the output-level sanity table.
- `paper_artifacts/h100_2026_04_11/cask_h100_actual_bridge/`: packages the clean H100 output-level bridge and the `multi_news` stage-ablation boundary in one place, so the paper can cite actual-output evidence without mixing it with the replay-only gate.
- `submission_gate_checks.md`: small add-on package covering the active LongBench witnesses, one representative-mode ablation, and the inactive-regime boundary checks.

`first_mismatch` is useful, but it should be plotted together with `top1` or `mean_nll`, not interpreted alone.

## Repository Layout

- `scripts/cli.py`: high-level experiment wrapper.
- `scripts/worker.py`: HuggingFace execution path for all supported methods.
- `scripts/replay_reference_fidelity.py`: teacher-forced fidelity harness.
- `scripts/compare_kv_fidelity.py`: output-level fidelity comparison helper.
- `scripts/build_prompt_heavy_artifacts.py`: promotes selected prompt-heavy replay reports into tracked paper-facing tables.
- `triattention/methods/triattention.py`: TriAttention baseline.
- `triattention/methods/cask.py`: CASK mainline implementation.
- `paper_artifacts/`: tracked paper-facing summaries and frozen experiment snapshots.
- `docs/`: historical notes and archived Phase 1 materials.

## Provenance

This codebase started from the TriAttention implementation and now serves as the active research repository for CASK. Historical file paths and namespaces are retained where changing them would break reproducibility or internal tooling.

## License

Apache 2.0. See [LICENSE](LICENSE).
