# CASK

Core-Aware Selective KV Compression for Reasoning Traces

CASK treats reasoning-time KV compression as a **behavior-preserving selective consolidation** problem rather than a pure scoring problem. The paper-facing method is `cask`: protect a small **core** of reasoning states, selectively consolidate redundant **scratch** states, and use a **two-stage** policy for prompt-heavy runs.

## At A Glance

| Item | Current answer |
| --- | --- |
| Main method | `cask` |
| Main baseline | `triattention` |
| Main metric family | teacher-forced reference fidelity |
| Headline claim | CASK improves the **minimum usable budget frontier** rather than trying to be the most aggressive compressor at every setting |
| Prompt-heavy policy | Stage 1 prefix eviction, then Stage 2 decode consolidation |
| Primary runner | `python scripts/cli.py run-one ... --method cask` |
| Main replay harness | `scripts/replay_reference_fidelity.py` |
| Command provenance | `artifacts/COMMAND_MAP.md` |
| Figure assets | `docs/assets/` |
| Paper source | `paper/` |

## Method Summary

| Stage | What happens | Why it exists |
| --- | --- | --- |
| Stage 1: prefix compression | TriAttention-style eviction plus a small coverage reserve | Prevent prompt-heavy runs from exhausting the budget before decode starts |
| Stage 2: decode compression | Split decode states into `protected core` and `mergeable scratch`, then consolidate scratch only | Preserve answer-critical states while compressing redundant reasoning work |

## Safety Hook

CASK is built around two paper-facing diagnostics: `lost representative mass` and `kappa-dispersion`.
If a scratch group preserves most of the horizon-weighted mass and its members stay close to the representative under the calibrated `kappa` geometry, then replacing that group with one representative changes the next-token score only by a small additive amount.
This is the main reason the repo reports fidelity and saved ratio together rather than treating compression as a pure ranking problem.

## Mode Map

| Mode | Purpose | Status |
| --- | --- | --- |
| `fullkv` | Uncompressed reference run | primary reference |
| `triattention` | Original eviction baseline | primary baseline |
| `cask` | CASK mainline implementation | primary paper method |
| `expectedattention` | Closest prior-style comparison kept in-tree | optional baseline |
| `horizonkv` | Legacy internal alias for archived Phase 1 scorer work | not paper-facing |

If you are running the current paper candidate, use `--method cask`.

## Benchmark Snapshot

The tables below report both fidelity and **terminal saved ratio** so the quality-memory tradeoff is visible at a glance.

### H100 Reasoning Replay Gate

| Slice | Budget | Tri Top-1 | CASK Top-1 | Tri Mean NLL | CASK Mean NLL | Tri Saved Ratio | CASK Saved Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AIME24 ref6` | `256` | `86.10` | `88.43` | `0.463` | `0.359` | `65.31%` | `65.28%` |
| `AIME24 ref6` | `384` | `88.25` | `90.69` | `0.383` | `0.268` | `61.55%` | `61.55%` |
| `AIME24 ref6` | `512` | `89.42` | `91.68` | `0.333` | `0.233` | `43.57%` | `43.52%` |
| `AIME25 ref6` | `256` | `85.71` | `86.77` | `0.500` | `0.504` | `63.37%` | `55.87%` |
| `AIME25 ref6` | `384` | `89.13` | `90.32` | `0.357` | `0.313` | `59.52%` | `63.26%` |
| `AIME25 ref6` | `512` | `89.94` | `91.68` | `0.321` | `0.254` | `44.76%` | `37.25%` |

Detail:
[H100 reasoning replay package](artifacts/h100_2026_04_10/cask_h100_fidelity/README.md)

### Prompt-Heavy Replay Highlights

| Dataset | Budget | Tri Top-1 | CASK Top-1 | Tri Mean NLL | CASK Mean NLL | CASK Saved Ratio | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qasper` | `256` | `67.19` | `71.09` | `1.315` | `1.247` | `90.90%` | same-budget replay win |
| `multi_news` | `384` | `53.71` | `61.33` | `2.052` | `1.540` | `84.07%` | decode-active replay win |
| `hotpotqa` | `384` | `81.25` | `96.88` | `1.344` | `0.110` | `96.49%` | strongest same-budget witness |
| `2wikimqa` | `384` | `59.38` | `56.25` | `3.415` | `2.397` | `94.41%` | retained boundary |

Detail:
[Prompt-heavy replay readout](artifacts/h100_2026_04_11/promptheavy_saved_ratio_audit/promptheavy_replay_readout.md)

### Output-Level Bridge Highlights

| Task | Comparison | Lexical / Sequence | Semantic Sim. | Official Metric | CASK Terminal Saved Ratio | Read |
| --- | --- | --- | ---: | --- | ---: | --- |
| `qasper` | `CASK @ 256` vs `TriAttention @ 512` | `0.238 > 0.173` | `0.791 > 0.678` | `12.77 > 11.94` | `90.90%` | clean budget crossing on all three axes |
| `multi_news` | `CASK @ 384` vs `TriAttention @ 384` | `0.169 > 0.000` | `0.952 > 0.452` | `15.16 > 0.00` | `84.07%` | strongest decode-active output bridge |
| `hotpotqa` | `CASK @ 256` vs `TriAttention @ 256` | `1.000 = 1.000` | `1.000 = 1.000` | `27.27 = 27.27` | `97.57%` | non-regression parity |

Detail:
[H100 actual-output bridge package](artifacts/h100_2026_04_11/cask_h100_actual_bridge/README.md)

## Current Read

| Axis | Current read |
| --- | --- |
| Reasoning replay | CASK beats TriAttention at the same budget on the tracked H100 gate and shows partial crossing |
| Prompt-heavy replay | CASK has a strong same-budget replay package, with `multi_news` and `vcsum` as the current replay-level decode-active witnesses |
| Output-level bridge | `multi_news` remains the strongest decode-active output bridge; `vcsum` is a lexical-vs-semantic boundary rather than a clean headline win |
| Savings interpretation | The claim is **not** "always compress more"; it is "keep full-KV behavior alive at a lower usable budget" |
| Claim boundary | Active decode regime and `prefix_budget_exhausted` regime must be separated explicitly |

## Artifact Index

Start here:
[artifacts/README.md](artifacts/README.md)

Command trace:
[artifacts/COMMAND_MAP.md](artifacts/COMMAND_MAP.md)

### Which package should you open?

| If you want to know... | Open this | Why this is the right package |
| --- | --- | --- |
| whether CASK wins the main reasoning replay gate | [H100 reasoning replay gate](artifacts/h100_2026_04_10/cask_h100_fidelity/README.md) | contains the `AIME24` / `AIME25` synchronized replay tables and crossing read |
| whether replay gains show up in actual generation | [H100 actual-output bridge](artifacts/h100_2026_04_11/cask_h100_actual_bridge/README.md) | contains the tracked `qasper`, `multi_news`, and `hotpotqa` output bridge rows |
| how to read the full prompt-heavy story | [H100 prompt-heavy follow-up](artifacts/h100_2026_04_11/README.md) | separates replay-level decode-active wins from output-level bridge rows and `prefix_budget_exhausted` boundaries |
| which figures are current and ready to paste into the draft | [Figure asset pack](docs/assets/README.md) | bundles the synchronized PNG/PDF versions of the reasoning gate, prompt-heavy, bridge, and method-overview figures |

### Package Roles

| Package | Use it for | Do not use it for |
| --- | --- | --- |
| H100 reasoning replay gate | main reasoning replay headline | final benchmark-accuracy headline |
| H100 actual-output bridge | showing replay-to-output linkage | broad decode-stage generalization claims by itself |
| H100 prompt-heavy follow-up | regime separation and prompt-heavy narrative | pretending every prompt-heavy task is decode-active |

## Installation

Python 3.10+ is required.

```bash
git clone https://github.com/Skyline-23/CASK.git
cd CASK
pip install -e .
```

For Linux benchmarking and vLLM runtime work, install the matching CUDA, FlashAttention, and vLLM stack separately. The HuggingFace replay harnesses also run on a single 16 GB consumer GPU with `sdpa`.

## Quick Start

| Task | Command |
| --- | --- |
| FullKV reference | `python scripts/cli.py run-one --model Qwen3-8B --dataset math500 --method fullkv` |
| TriAttention baseline | `python scripts/cli.py run-one --model Qwen3-8B --dataset math500 --method triattention --budget 104 --stats-path cask/calibration/for_aime24_experiment/qwen3_8b.pt` |
| CASK mainline | `python scripts/cli.py run-one --model Qwen3-8B --dataset math500 --method cask --budget 104 --stats-path cask/calibration/for_aime24_experiment/qwen3_8b.pt` |
| Teacher-forced replay | `python scripts/replay_reference_fidelity.py --reference ... --model-path ... --method cask --budget 104 --triattention-stats-file ...` |

Replay example:

```bash
python scripts/replay_reference_fidelity.py \
    --reference experiments/outputs/math500/Qwen3-8B/sample1/fullkv/fullkv_selection_geometry_reference \
    --model-path experiments/models/Qwen3-8B \
    --method cask \
    --budget 104 \
    --triattention-stats-file cask/calibration/for_aime24_experiment/qwen3_8b.pt \
    --attn-implementation sdpa \
    --count-prompt-tokens true \
    --slack-budget-trigger true \
    --json-output experiments/reports/geometry_teacher_forced_fidelity_vs_fullkv_sm104_v2.json
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `scripts/cli.py` | high-level experiment wrapper |
| `scripts/worker.py` | HuggingFace execution path |
| `scripts/replay_reference_fidelity.py` | teacher-forced replay harness |
| `scripts/run_replay_fidelity_frontier.py` | batch replay frontier launcher |
| `scripts/run_promptheavy_pack.py` | generic prompt-heavy fullkv + replay package planner/launcher |
| `scripts/run_replay_queue.ps1` | PowerShell queue wrapper for replay configs |
| `scripts/run_longbench_suite.py` | LongBench generation harness |
| `scripts/compare_kv_fidelity.py` | output-level comparison helper |
| `scripts/build_promptheavy_saved_ratio_audit.py` | package prompt-heavy replay summaries |
| `scripts/build_actual_bridge_artifacts.py` | package actual-output bridge summaries |
| `scripts/sync_reasoning_gate.py` | sync replay reports into the tracked reasoning-gate summaries |
| `scripts/build_paper_figures.py` | generate the current paper-facing figure pack under `docs/assets/` |
| `scripts/refresh_paper_figures.ps1` | optional sync + figure refresh wrapper |
| `paper/` | NeurIPS paper source, bibliography, and style files |
| `cask/methods/triattention.py` | TriAttention baseline implementation |
| `cask/methods/cask.py` | CASK implementation |
| `artifacts/` | tracked paper-facing summaries |
| `docs/assets/` | paper-facing rendered figures |

## Provenance

This codebase started from the TriAttention implementation and now serves as the active research repository for CASK. Some internal names remain for compatibility with existing tooling, but the paper-facing method is CASK.

## License

Apache 2.0. See [LICENSE](LICENSE).

