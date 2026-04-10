CASK is built on top of TriAttention's codebase and scoring framework.

# CASK

Core-Aware Selective KV Compression for Reasoning Traces

This repository now treats **CASK** as the main paper direction. The central claim is no longer that a better RoPE scorer alone wins; it is that reasoning traces contain a **protected core** plus **mergeable scratch**, and that preserving the core while selectively consolidating scratch can preserve full-KV behavior better than pure eviction at the same budget.

The repository name and the Python namespace still reflect the historical `HorizonKV` / `triattention` lineage. Paper-facing method naming, however, is now **CASK**.

## Main Idea

CASK uses **2-stage compression**:

1. **Stage 1: prefix compression**
   Prompt-heavy runs first shrink the prefix with **TriAttention eviction** so the decode path is not dead on arrival.
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
- `paper_artifacts/rtx5070ti_2026_04_10/cask_v2_fidelity/longbench_qasper_prompt_heavy_witness.md`

That sweep covers:

- witnesses: `hexagon`, `geometry248`, `geometry434`
- budgets: `104`, `128`, `160`, `192`
- methods: `triattention`, `cask`
- plus one prompt-heavy LongBench witness: `qasper @ 512`

High-level read:

- `geometry248`: CASK beats TriAttention on `top1` and `mean_nll` at every tested budget.
- `geometry434`: CASK wins at `104`, `128`, and `192`; `160` is essentially parity.
- `hexagon`: CASK is clearly stronger at `104` and `192`, with near-parity at `128` and `160`.
- `qasper @ 512`: CASK improves same-budget prompt-heavy fidelity over TriAttention, but this witness is prefix-stage-only and should be read as two-stage coverage evidence rather than decode-merge evidence.

`first_mismatch` is useful, but it should be plotted together with `top1` or `mean_nll`, not interpreted alone.

## Repository Layout

- `scripts/cli.py`: high-level experiment wrapper.
- `scripts/worker.py`: HuggingFace execution path for all supported methods.
- `scripts/replay_reference_fidelity.py`: teacher-forced fidelity harness.
- `scripts/compare_kv_fidelity.py`: output-level fidelity comparison helper.
- `triattention/methods/triattention.py`: TriAttention baseline.
- `triattention/methods/cask.py`: CASK mainline implementation.
- `paper_artifacts/`: tracked paper-facing summaries and frozen experiment snapshots.
- `docs/`: historical notes and archived Phase 1 materials.

## Historical Archive

The repository still contains the earlier HorizonKV / adaptive-horizon direction. Those materials are retained as archive context and baseline infrastructure, not as the current paper claim.

In particular:

- `docs/head_adaptive_horizon_averaging_for_kv_cache_compression.md`
- `docs/head_adaptive_horizon_implementation_spec.md`

should be read as **Phase 1 archive**, while the current paper mainline is **CASK**, exposed as `cask` in the CLI.

## Provenance

This codebase started from the TriAttention implementation and now serves as the active research repository for CASK. Historical file paths and namespaces are retained where changing them would break reproducibility or internal tooling.

## License

Apache 2.0. See [LICENSE](LICENSE).
