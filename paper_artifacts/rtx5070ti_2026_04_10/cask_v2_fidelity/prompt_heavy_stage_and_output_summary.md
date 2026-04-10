# Prompt-Heavy Stage and Coverage Summary

This file promotes the prompt-heavy evidence from raw replay reports into tracked paper artifacts.

## 1. Decode-active prompt-heavy witness

| Task | Method | Budget | Top-1 | Top-5 | Mean NLL | First Mismatch | Ref-Length Saved Ratio | Prefix Events | Decode Events | Stage Profile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `multi_news` | `TriAttention @ 384` | `384` | `53.7%` | `81.2%` | `2.068` | `2` | `84.1%` | `0` | `0` | `eviction_only_baseline` |
| `multi_news` | `CASK @ 384 (coverage 0.0625)` | `384` | `65.8%` | `90.4%` | `1.474` | `2` | `84.1%` | `2` | `2` | `two_stage_active` |

Interpretation:
- `multi_news @ 384` is the clean decode-active prompt-heavy witness.
- `cask` fires both stage-1 prefix eviction and stage-2 decode consolidation (`prefix_events = 2`, `decode_events = 2`).
- At the same physical budget, `cask` substantially improves `top1`, `top5`, and `mean_nll` over `triattention`.

## 2. Prompt-heavy stage contribution

| Task | Method | Budget | Top-1 | Top-5 | Mean NLL | First Mismatch | Ref-Length Saved Ratio | Prefix Events | Decode Events | Stage Profile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qasper` | `TriAttention @ 384` | `384` | `67.2%` | `90.6%` | `1.366` | `2` | `87.9%` | `0` | `0` | `eviction_only_baseline` |
| `qasper` | `TriAttention @ 512` | `512` | `65.6%` | `90.6%` | `1.384` | `2` | `84.8%` | `0` | `0` | `eviction_only_baseline` |
| `qasper` | `CASK @ 384` | `384` | `76.6%` | `95.3%` | `1.014` | `6` | `87.9%` | `1` | `0` | `prefix_only_active` |
| `qasper` | `CASK @ 384 (coverage 0.0625)` | `384` | `76.6%` | `94.5%` | `0.985` | `6` | `87.9%` | `1` | `0` | `prefix_only_active` |
| `qasper` | `CASK @ 256` | `256` | `69.5%` | `95.3%` | `1.258` | `4` | `90.9%` | `1` | `0` | `prefix_only_active` |

Interpretation:
- `qasper` is a strong prompt-heavy crossing witness.
- Every `cask` row above is `prefix_only_active`; decode merge does not fire on this witness.
- The crossing therefore comes from the two-stage prompt-aware prefix policy, not from decode-stage consolidation.

## 3. Prefix coverage reserve ablation on `2wikimqa`

| Variant | Coverage Ratio | Top-1 | Top-5 | Mean NLL | First Mismatch | Stage Profile |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `TriAttention @ 512` | `0` | `59.4%` | `68.8%` | `3.501` | `2` | `eviction_only_baseline` |
| `CASK @ 384` | `0` | `53.1%` | `84.4%` | `2.532` | `2` | `prefix_only_active` |
| `CASK @ 384 (coverage 0.0625)` | `0.0625` | `56.2%` | `84.4%` | `2.432` | `2` | `prefix_only_active` |
| `CASK @ 384 (coverage 0.125)` | `0.125` | `56.2%` | `84.4%` | `2.540` | `2` | `prefix_only_active` |

Interpretation:
- `2wikimqa` remains a boundary case under teacher-forced `top1`.
- A small coverage reserve (`0.0625`) improves `top1` and `mean_nll` relative to score-only prefix eviction.
- Increasing the reserve to `0.125` does not help further, which suggests the correction is real but should stay small.

## 4. Output-level sanity

| Task | Method | Final Answer Match | Sequence Ratio | Prefix Token Ratio |
| --- | --- | ---: | ---: | ---: |
| `multi_news` | `TriAttention @ 384` | `0.0%` | `0.000` | `0.000` |
| `multi_news` | `CASK @ 384` | `0.0%` | `0.169` | `0.081` |
| `qasper` | `TriAttention @ 512` | `0.0%` | `0.042` | `0.018` |
| `qasper` | `CASK @ 512` | `0.0%` | `0.174` | `0.045` |
| `2wikimqa` | `TriAttention @ 512` | `0.0%` | `0.083` | `0.125` |
| `2wikimqa` | `CASK @ 384` | `0.0%` | `0.703` | `0.625` |

Interpretation:
- The prompt-heavy story is not just teacher-forced: under actual greedy decoding, `cask` stays materially closer to the `fullkv` continuation than `triattention` on both tracked tasks.
- `qasper` is the clean crossing witness.
- `2wikimqa` is the mixed but informative boundary case that motivated the coverage-reserve correction.

## 5. Single-example task metric sanity

| Task | Method | Task Metric |
| --- | --- | ---: |
| `qasper` | `TriAttention @ 512` | `0.015` |
| `qasper` | `CASK @ 512` | `0.075` |
| `2wikimqa` | `TriAttention @ 512` | `0.000` |
| `2wikimqa` | `CASK @ 384` | `0.000` |
| `multi_news` | `TriAttention @ 384` | `0.000` |
| `multi_news` | `CASK @ 384` | `0.139` |
| `multi_news` | `FullKV` | `0.178` |

Interpretation:
- These single-example task metrics are not a substitute for a full benchmark matrix.
- They do show that the observed fidelity gains on `qasper` and `multi_news` correspond to better task-visible outputs, while `2wikimqa` remains the honest boundary case.
