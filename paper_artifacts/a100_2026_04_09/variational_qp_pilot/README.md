# Variational/QP Horizon Pilot

Date: 2026-04-09

Purpose: quick A100 gate for the offline variational horizon mode after
`adaptive + tri` tied `fixed + tri` on the first paired probes.

Calibration artifact used on the server, not checked in here:

```bash
experiments/stats/Qwen3-8B/a100_qp32_v3_len32768.pt
```

Calibration command:

```bash
.venv/bin/python scripts/calibrate.py \
  --model experiments/models/Qwen3-8B \
  --input experiments/calibration/wikitext103_train_excerpt.txt \
  --output experiments/stats/Qwen3-8B/a100_qp32_v3_len32768.pt \
  --max-length 32768 \
  --attn-implementation sdpa \
  --build-variational-horizon \
  --variational-query-samples 32 \
  --variational-key-chunk-size 1024 \
  --variational-offset-max-length 65536
```

## MATH-500 pilot15 / budget384 / sample1

| method | correct | accuracy | correct idx |
| --- | ---: | ---: | --- |
| fixed + tri | 3/15 | 20.0% | 3, 5, 13 |
| adaptive + tri | 3/15 | 20.0% | 5, 6, 13 |
| variational/QP + tri | 4/15 | 26.7% | 3, 5, 6, 13 |

Observation: on this tiny pilot, the variational/QP mode preserved the
previously split fixed/adaptive wins.

## AIME25 idx13 / budget544 / sample8

| method | pass count | pass rate | predictions |
| --- | ---: | ---: | --- |
| fixed + tri | 5/8 | 62.5% | 60, 0, 6.062, 60, 60, 60, 60, 0 |
| adaptive + tri | 5/8 | 62.5% | 60, 14, 2, 60, 60, 60, 60, 0 |
| variational/QP + tri | 4/8 | 50.0% | 60, 0, 14, 60, long malformed text, 60, 60, 0 |

Observation: QP is not a monotone fix. It improved the small MATH pilot but
regressed this targeted AIME25 stress case.

## Status

This is a directional pilot only. It is useful for deciding the next focused
run, but it is not a paper-quality aggregate.

## Kernel-Space Diagnostic

Artifact:

```bash
kernel_distance_summary.json
churn_probe_idx0_budget512_summary.json
```

Weighted kernel distance used normalized `|mu_f|` as the band weight and
compared each head's complex frequency response `kappa(omega_f)`.

| comparison | median absolute distance | median distance relative to fixed response magnitude |
| --- | ---: | ---: |
| fixed vs adaptive | 0.466 | 0.750 |
| fixed vs variational/QP | 0.222 | 0.355 |
| adaptive vs variational/QP | 0.401 | n/a |

Interpretation: in this v3 calibration artifact, adaptive does not collapse to
fixed in kernel-response space. QP is substantially closer to fixed than
adaptive is.

Churn probe used one AIME25 idx0 generation with budget512/max_new700 and
recorded the first two compression events.

| event | comparison | unique keep-set churn | per-KV-row mean churn | score Pearson |
| ---: | --- | ---: | ---: | ---: |
| 0 | fixed vs adaptive | 7.0% | 16.1% | 0.798 |
| 0 | fixed vs variational/QP | 3.4% | 6.6% | 0.938 |
| 1 | fixed vs adaptive | 9.9% | 30.0% | 0.115 |
| 1 | fixed vs variational/QP | 7.1% | 19.4% | 0.243 |

Interpretation: the scorer can change a lot, especially by the second
compression event, but the flattened retained-token set remains mostly shared
with fixed. The current failure mode is not simply "all horizon kernels are
identical"; it is closer to "the changed horizon response only moderately
changes eviction, and the changed decisions have not produced stable quality
gain."
