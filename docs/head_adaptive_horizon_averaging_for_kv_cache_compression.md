# Head-Adaptive Horizon Averaging for KV Cache Compression

This note captures the current follow-up direction beyond the exact scoring rewrite.
The exact rewrite is useful engineering, but it is not strong enough as the main
paper contribution on its own. The research contribution needs to come from
changing the score, not only accelerating an equivalent one.

For concrete repo-level implementation details, see
[`head_adaptive_horizon_implementation_spec.md`](./head_adaptive_horizon_implementation_spec.md).

## Working Thesis

TriAttention can be generalized from a fixed geometric future-offset rule to a
head-adaptive horizon kernel, while keeping the same RoPE-aware scoring structure.
The most promising extension is:

- adaptive horizon kernel `pi_h`
- offline variational horizon fit `pi_h*` for upper-bound comparison
- RMS2 norm refinement
- exact rotated-key rewrite retained as a systems optimization

The exact rewrite should be presented as supporting infrastructure, not as the
headline claim.

## Claim Boundary

Strong claims:

- There is a shared RoPE horizon-averaging primitive behind TriAttention-style
  future offset averaging.
- A head-adaptive horizon kernel is an algorithmic generalization of the fixed
  geometric offset rule.
- RMS2 is a strict refinement of the original norm coefficient under the
  intended interpretation.

Careful claims:

- The relationship to Expected Attention is a shared primitive, not full-score
  equivalence.

Claims to avoid:

- "All RoPE KV compression methods are one family."
- "Expected Attention is subsumed by our method."

## Mathematical Direction

### 1. Horizon-Averaging Primitive

For a head `h`, define

`S^{mean}_{h,pi}(k,Delta) = Re sum_f mu_{h,f} conj(k_{h,f}) kappa_{pi_h}(omega_f) exp(i omega_f Delta)`

with

`kappa_{pi_h}(omega) = E_{delta ~ pi_h}[exp(i omega delta)]`.

Under this view, TriAttention's fixed geometric offset set is a special case of
an empirical horizon kernel.

### 1.1 Practical And Variational Kernel Split

The practical runtime method remains the Gaussian adaptive parameterization used
to define HorizonKV. In parallel, calibration may estimate an oracle target
`tau_h` from sampled raw attention and solve the dyadic simplex QP for
`pi_h*`. The solved kernel is not the default deployed method; it is the
comparison mode that tells us whether the Gaussian parameterization is leaving
quality on the table.

### 2. RMS2 Refinement

The preferred refinement is

`alpha^{RMS2}_f = (E ||q_f||^2 - ||mu_f||^2) / (E ||q_f|| + ||mu_f|| + eps)`.

This should be framed as a refinement of the original coefficient, not as a
heuristic replacement. In the constant-norm limit it reduces to the original
TriAttention coefficient.

### 3. Expected Attention Positioning

Expected Attention is the closest prior baseline and must be treated directly in
both writing and experiments. The clean connection is the horizon-averaging
primitive, while keeping the distinction that Expected Attention uses a
matrix-averaged RoPE approximation with covariance corrections.

## Main Risks

1. Adaptive `pi_h` may fail to beat the fixed geometric baseline by enough to
   matter.
2. The new method introduces additional knobs, which can trigger "won by tuning"
   criticism.
3. Expected Attention is too close a prior to omit from the main comparison.
4. Small deltas on AIME24 and AIME25 can be swallowed by variance if the
   protocol is not fixed in advance.

## Experimental Principles

- Expected Attention baseline is mandatory.
- Tune on AIME24 or a calibration-only split, and report held-out numbers on
  AIME25, MATH-500, and LongBench.
- Treat quality as the headline metric and latency as supporting evidence.
- Match KV budget, model, and decode settings across compressed methods.
- Keep Full Attention as an unconstrained upper bound, not a budget-matched row.
- Report headline deltas with paired bootstrap confidence intervals over
  problems.

## Success Criteria

The follow-up is worth writing up only if all of the following hold:

1. Positive average delta over `{AIME24, AIME25, MATH-500}`.
2. At least one clear win:
   - `+2.0` or more on AIME24 or AIME25, or
   - `+1.0` or more on MATH-500.
3. LongBench average regression stays within `-0.5` of the TriAttention
   baseline.

## Implementation Plan In This Repo

### Phase 1. Extend Calibration Statistics

Current implementation status:

- `q_mean_complex`
- `q_abs_mean`
- `q_sq_abs_mean`
- optional variational tensors: `oracle_tau`, `oracle_pi`, `oracle_horizon_mean_complex`

To support RMS2 and variational kernel comparison cleanly, calibration now also
supports sampled raw-attention estimation and the offline dyadic simplex QP.

Primary files:

- [`scripts/calibrate.py`](../scripts/calibrate.py)
- [`triattention/methods/pruning_utils.py`](../triattention/methods/pruning_utils.py)

### Phase 2. Separate Kernel Choice From Norm Choice

The HF path should expose scorer variants explicitly, instead of overloading the
current fixed formula:

- `horizon_mode = fixed | adaptive | variational`
- `norm_mode = tri | rms2`

Primary file:

- [`triattention/methods/triattention.py`](../triattention/methods/triattention.py)

### Phase 3. Evaluate In HF First

Before large-scale runs, validate score behavior and accuracy in the HF pipeline
using the existing `run-one`, score-dump, and comparison harnesses. This repo
already supports `adaptive + rms2` as the default HorizonKV mode and
`variational + rms2` as the explicit offline-QP comparison mode.

### Phase 4. Port Stable Variants To vLLM

The current repo already carries the new statistics schema into the vLLM
runtime. `fixed + tri` is still the Triton fast path; `adaptive + rms2` and
`variational + rms2` currently use the PyTorch scorer fallback.

Primary files:

- [`triattention/vllm/core/scoring.py`](../triattention/vllm/core/scoring.py)
- [`triattention/vllm/core/utils.py`](../triattention/vllm/core/utils.py)

## Immediate Next Steps

1. Reproduce `fullkv`, `triattention`, and `expectedattention`.
2. Run `adaptive + rms2` against:
   - TriAttention
   - Expected Attention
   - Full Attention
3. If Gaussian adaptive underperforms, run the `variational + rms2` comparison mode.
4. Decide whether the Gaussian parameterization is tight enough to headline.
5. Promote the stable configuration to the full benchmark matrix.

## One-Line Summary

The follow-up paper should live or die on whether adaptive `pi_h` beats both
fixed geometric horizons and Expected Attention on held-out quality, not on the
exact rewrite alone.
