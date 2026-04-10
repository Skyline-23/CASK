# Submission Gate Checks

This note records the extra checks added after the initial witness sweep to
estimate whether the current CASK paper package is strong enough for
submission.

## 1. Prompt-heavy decode-active witness: `multi_news`

Teacher-forced replay against `fullkv`:

| Method | Top-1 | Top-5 | Mean NLL | First Mismatch | Ref-Length Saved Ratio | Prefix Events | Decode Events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `triattention @ 384` | `0.5371` | `0.8125` | `2.0676` | `2` | `0.8408` | `0` | `0` |
| `cask @ 384` | `0.6582` | `0.9043` | `1.4737` | `2` | `0.8411` | `2` | `2` |

Interpretation:
- This is the cleanest prompt-heavy witness where **both** CASK stages are active.
- At the same physical budget, CASK substantially improves `top1`, `top5`, and
  `mean_nll` over TriAttention.
- This closes the gap left by `qasper`: the prompt-heavy story is no longer
  limited to prefix-only evidence.

## 2. LongBench prompt-heavy witness: `qasper` budget crossing

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

Stage-contribution summary:

| Method | Budget | Ref-Length Saved Ratio | Prefix Events | Decode Events | Stage Profile |
| --- | ---: | ---: | ---: | ---: | --- |
| `triattention @ 384` | `384` | `0.8786` | `0` | `0` | `eviction_only_baseline` |
| `triattention @ 512` | `512` | `0.8481` | `0` | `0` | `eviction_only_baseline` |
| `cask @ 384` | `384` | `0.8786` | `1` | `0` | `prefix_only_active` |
| `cask @ 384 (coverage 0.0625)` | `384` | `0.8786` | `1` | `0` | `prefix_only_active` |
| `cask @ 256` | `256` | `0.9090` | `1` | `0` | `prefix_only_active` |

This is the cleanest evidence that the current prompt-heavy gain comes from the
two-stage prefix policy rather than from decode-stage merge.

## 3. Output-level sanity on `qasper @ 512`

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

## 4. Boundary-case prompt-heavy witness: `2wikimqa`

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

Coverage-reserve ablation:

| Variant | Coverage Ratio | Top-1 | Top-5 | Mean NLL | First Mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cask @ 384` | `0` | `0.5313` | `0.8438` | `2.5322` | `2` |
| `cask @ 384` | `0.0625` | `0.5625` | `0.8438` | `2.4323` | `2` |
| `cask @ 384` | `0.125` | `0.5625` | `0.8438` | `2.5400` | `2` |

This ablation supports the current reverse-engineering story:
- score-only prefix eviction was leaving `2wikimqa` under-covered
- a small coverage reserve fixes part of that drift
- pushing the reserve further does not keep helping

## 5. Representative-mode ablation on `hexagon @ 104`

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
  plus both a decode-active prompt-heavy witness and a strong prompt-heavy
  crossing witness
- two-stage coverage is now supported by actual LongBench examples in both the
  prefix-only and decode-active regimes
- prompt-heavy stage contribution is now explicitly decomposed into prefix-only,
  decode-active, and boundary-case behavior
- one implementation ablation supports the current representative default
- one boundary-case prompt-heavy task shows `cask` can still be much closer to
  `fullkv` under actual greedy decoding even when teacher-forced `top1` does
  not flip

It is still not a fully closed large-scale empirical package. The strongest
current claim remains:

> CASK improves the same-budget full-KV fidelity frontier for reasoning traces,
> and two-stage compression extends that advantage into prompt-heavy regimes.
