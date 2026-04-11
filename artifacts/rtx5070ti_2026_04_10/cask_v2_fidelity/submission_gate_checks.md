# Submission Gate Checks

This note tracks the current submission-facing evidence package after the
H100 prompt-heavy two-stage rerun.

## 1. H100 Prompt-Heavy Matrix

Authoritative source:

- `experiments/frontier/Qwen3-8B/h100_promptheavy_twostage_rerun_20260411/overnight_status.md`

| Task | Budget | TriAttention Top-1 | CASK Top-1 | TriAttention Top-5 | CASK Top-5 | TriAttention NLL | CASK NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qasper` | `256` | `67.2%` | `71.1%` | `89.8%` | `93.8%` | `1.315` | `1.247` |
| `qasper` | `384` | `66.4%` | `73.4%` | `90.6%` | `92.2%` | `1.398` | `1.297` |
| `multi_news` | `256` | `54.7%` | `60.0%` | `82.0%` | `87.9%` | `2.060` | `1.652` |
| `multi_news` | `384` | `53.7%` | `61.3%` | `81.6%` | `89.5%` | `2.052` | `1.540` |
| `hotpotqa` | `256` | `81.3%` | `93.8%` | `90.6%` | `100.0%` | `1.374` | `0.151` |
| `hotpotqa` | `384` | `81.3%` | `96.9%` | `90.6%` | `100.0%` | `1.344` | `0.110` |
| `musique` | `256` | `59.4%` | `65.6%` | `71.9%` | `75.0%` | `2.697` | `2.713` |
| `musique` | `384` | `53.1%` | `62.5%` | `75.0%` | `75.0%` | `2.862` | `2.650` |
| `2wikimqa` | `256` | `59.4%` | `62.5%` | `68.8%` | `81.3%` | `3.368` | `2.375` |
| `2wikimqa` | `384` | `59.4%` | `56.3%` | `68.8%` | `81.3%` | `3.415` | `2.397` |

Readout:

- `qasper`, `multi_news`, `hotpotqa`, and `musique` all show same-budget
  fidelity advantage for CASK.
- `hotpotqa` is the strongest prompt-heavy same-budget witness.
- `2wikimqa` remains the boundary case.

## 2. Stage Attribution

| Task | `cask @ 256` decode events | `cask @ 384` decode events | Interpretation |
| --- | ---: | ---: | --- |
| `qasper` | `0` | `0` | prefix-dominant |
| `multi_news` | `3` | `3` | decode-active |
| `hotpotqa` | `0` | `0` | prefix-dominant |
| `musique` | `0` | `0` | prefix-dominant |
| `2wikimqa` | `0` | `0` | prefix-dominant boundary case |

Readout:

- `multi_news` is still the clean decode-active witness.
- The rest of the H100 prompt-heavy package is primarily telling a
  stage-1 prefix-aware story.

## 3. Output-Level Sanity

Local single-example output checks still support the replay story:

| Task | Method | Sequence Ratio | Prefix Token Ratio | Task Metric |
| --- | --- | ---: | ---: | ---: |
| `multi_news` | `TriAttention @ 384` | `0.000` | `0.000` | `0.000` |
| `multi_news` | `CASK @ 384` | `0.169` | `0.081` | `0.139` |
| `qasper` | `TriAttention @ 512` | `0.042` | `0.018` | `0.015` |
| `qasper` | `CASK @ 512` | `0.174` | `0.045` | `0.075` |
| `2wikimqa` | `TriAttention @ 512` | `0.083` | `0.125` | `0.000` |
| `2wikimqa` | `CASK @ 384` | `0.703` | `0.625` | `0.000` |

Readout:

- `multi_news` is the strongest same-budget prompt-heavy bridge from replay
  fidelity to task-visible output quality.
- `qasper` remains a useful prompt-heavy output-level sanity check.
- `2wikimqa` remains boundary analysis, not headline evidence.

## 4. Reasoning-Side Bridge

### `hexagon @ 104`

| Method | Extracted Answer | Correct |
| --- | --- | ---: |
| `fullkv` | `42` | `1` |
| `triattention @ 104` | repeated `7...` tail | `0` |
| `cask @ 104` | `42` | `1` |

### `math500` 3-witness subset, `budget=104`, `sample4`

| Witness | TriAttention | CASK |
| --- | ---: | ---: |
| `hexagon` | `2/4` | `4/4` |
| `geometry/248` | `0/4` | `0/4` |
| `geometry/434` | `0/4` | `0/4` |

| Metric | TriAttention | CASK |
| --- | ---: | ---: |
| draw-level exact match | `2/12` | `4/12` |
| problem-level `pass@4` | `1/3` | `1/3` |

Readout:

- The strongest reasoning-side bridge is still `hexagon`.
- The subset bridge remains small, but it is not zero.

## 5. Representative-Mode Ablation

| Representative Mode | Top-1 | Top-5 | Mean NLL | First Mismatch |
| --- | ---: | ---: | ---: | ---: |
| `score_max_source` | `0.8639` | `0.9827` | `0.4976` | `160` |
| `weighted_latest` | `0.8553` | `0.9827` | `0.5152` | `160` |

Readout:

- `score_max_source` remains the better default.

## Bottom Line

The current package now supports the following submission-facing claim:

> CASK improves the same-budget full-KV fidelity frontier for reasoning traces,
> and the two-stage path extends that advantage into prompt-heavy
> regimes.

The strongest prompt-heavy evidence is now:

- a decode-active witness: `multi_news`
- a very strong same-budget witness: `hotpotqa`
- a stable prompt-heavy same-budget win: `qasper`
- a retained boundary case: `2wikimqa`
