# Submission Gate Checks

This note records the extra checks added after the initial witness sweep to
estimate whether the current CASK paper package is strong enough for
submission.

## 1. LongBench prompt-heavy witness: `qasper` budget crossing

Teacher-forced replay against `fullkv`:

| Method | Top-1 | Top-5 | Strict Prefix | Mean NLL | First Mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cask @ 256` | `0.6953` | `0.9531` | `0.0313` | `1.2583` | `4` |
| `cask @ 384` | `0.7656` | `0.9531` | `0.0469` | `1.0135` | `6` |
| `triattention @ 384` | `0.6719` | `0.9063` | `0.0156` | `1.3659` | `2` |
| `triattention @ 512` | `0.6563` | `0.9063` | `0.0156` | `1.3839` | `2` |
| `cask @ 512` | `0.7578` | `0.9453` | `0.0469` | `1.1091` | `6` |

Interpretation:
- `cask @ 384` beats `triattention @ 512` on every tracked fidelity metric while
  using `25%` less physical budget.
- `cask @ 256` still beats `triattention @ 512`, giving a `50%` budget-crossing
  example on this prompt-heavy LongBench witness.
- This witness is **prefix-stage dominated**:
  - `prefix_compression_events = 1`
  - `compression_events = 0`
- It should be cited as **two-stage coverage evidence**, not as decode-merge
  evidence.

## 2. Output-level sanity on `qasper @ 512`

Greedy generation compared to the `fullkv` output:

| Method | Final Answer Match | Sequence Ratio | Prefix Token Ratio |
| --- | ---: | ---: | ---: |
| `triattention @ 512` | `0.0` | `0.0420` | `0.0182` |
| `cask @ 512` | `0.0` | `0.1735` | `0.0455` |

Observed outputs:
- `fullkv`: repeats the correct article-grounded explanation.
- `triattention`: collapses into repeated `The ground ...`.
- `cask`: keeps the correct semantic gist much longer, but still loops and does
  not exactly match the `fullkv` final string.

Interpretation:
- This is **not** a clean final-answer win.
- It is still useful as a sanity check because `cask` remains materially closer
  to `fullkv` than `triattention` in actual greedy decoding.

## 3. Boundary-case prompt-heavy witness: `2wikimqa`

Teacher-forced replay against `fullkv`:

| Method | Top-1 | Top-5 | Mean NLL | First Mismatch |
| --- | ---: | ---: | ---: | ---: |
| `triattention @ 512` | `0.5938` | `0.6875` | `3.5010` | `2` |
| `cask @ 384` | `0.5625` | `0.8438` | `2.4323` | `2` |

Greedy output comparison:

| Method | Sequence Ratio vs `fullkv` | Prefix Token Ratio vs `fullkv` |
| --- | ---: | ---: |
| `triattention @ 512` | `0.0833` | `0.1250` |
| `cask @ 384` | `0.7027` | `0.6250` |

Observed outputs:
- `fullkv`: `Based on Passage 2, the wife of Francis I Rákóczi, Jelena Zrinska, was born in Gyulafehérv`
- `triattention @ 512`: collapses into repeated `Based on on on ...`
- `cask @ 384`: keeps the answer structure and supporting relation, but still drifts to an incorrect surface form

Interpretation:
- This is a **boundary case**, not a clean win.
- Under teacher-forced replay, `cask` improves `top5` and `mean_nll` but does
  not beat `triattention` on `top1`.
- Under actual greedy decoding, `cask` is dramatically closer to the `fullkv`
  output than `triattention`.
- This task is therefore best used to show the current limit of the prefix-only
  prompt-heavy story: the gain is real, but not every task yields the same kind
  of win.

## 4. Representative-mode ablation on `hexagon @ 104`

Teacher-forced replay against `fullkv`:

| Representative Mode | Top-1 | Top-5 | Mean NLL | First Mismatch |
| --- | ---: | ---: | ---: | ---: |
| `score_max_source` | `0.8639` | `0.9827` | `0.4976` | `160` |
| `weighted_latest` | `0.8553` | `0.9827` | `0.5152` | `160` |

Interpretation:
- The `score_max_source` representative anchor is modestly but consistently
  better than `weighted_latest`.
- This is a small but useful ablation that supports the current default.

## Bottom line

The current package is enough for a **submission-facing draft**:
- same-budget fidelity advantage exists across the tracked math witnesses,
  plus one strong LongBench prompt-heavy crossing witness
- two-stage coverage is now supported by an actual LongBench example
- one implementation ablation supports the current representative default
- one boundary-case prompt-heavy task shows `cask` can still be much closer to
  `fullkv` under actual greedy decoding even when teacher-forced `top1` does
  not flip

It is still not a fully closed large-scale empirical package. The strongest
current claim remains:

> CASK improves the same-budget full-KV fidelity frontier for reasoning traces,
> and two-stage compression extends that advantage into prompt-heavy regimes.
