# Submission Gate Checks

This note records the extra checks added after the initial witness sweep to
estimate whether the current CASK paper package is strong enough for
submission.

## 1. LongBench prompt-heavy witness: `qasper @ 512`

Teacher-forced replay against `fullkv`:

| Method | Top-1 | Top-5 | Strict Prefix | Mean NLL | First Mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `triattention @ 512` | `0.6563` | `0.9063` | `0.0156` | `1.3839` | `2` |
| `cask @ 512` | `0.7578` | `0.9453` | `0.0469` | `1.1091` | `6` |

Interpretation:
- `cask` is clearly better on same-budget prompt-heavy fidelity.
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

## 3. Representative-mode ablation on `hexagon @ 104`

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
  plus one LongBench prompt-heavy witness
- two-stage coverage is now supported by an actual LongBench example
- one implementation ablation supports the current representative default
- one output-level sanity check shows `cask` remains closer to `fullkv` than
  `triattention`, even when exact answer match is not achieved

It is still not a fully closed large-scale empirical package. The strongest
current claim remains:

> CASK improves the same-budget full-KV fidelity frontier for reasoning traces,
> and two-stage compression extends that advantage into prompt-heavy regimes.
