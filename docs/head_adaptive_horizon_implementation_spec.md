# Head-Adaptive Horizon Averaging Implementation Spec

This document is the implementation companion to
[`head_adaptive_horizon_averaging_for_kv_cache_compression.md`](./head_adaptive_horizon_averaging_for_kv_cache_compression.md).
Its job is not to motivate the paper. Its job is to make the next coding pass
unambiguous.

## Goal

Implement a paper-grade scorer variant in this repo that can be A/B tested
against the current TriAttention scorer.

The default candidate variant is:

- `horizon_mode = adaptive`
- `norm_mode = rms2`

The repo now also carries an explicit offline-QP comparison mode:

- `horizon_mode = variational`
- `norm_mode = rms2`

The exact rotated-key rewrite remains in place as infrastructure. It is not the
main algorithmic change.

## Current Implementation Status

Implemented today:

- `horizon_mode = fixed | adaptive | variational`
- `norm_mode = tri | rms2`
- stats v3 with `q_sq_abs_mean`, `oracle_tau`, `oracle_pi`, and `oracle_horizon_mean_complex`
- offline variational horizon fitting in `scripts/calibrate.py`
- HF runtime support for `variational`
- vLLM stats-loader / scorer-state support for `variational`

Method identity rule:

- `horizonkv` defaults to `adaptive + rms2`
- `variational + rms2` is the explicit comparison mode, enabled with `--triattention-horizon-mode variational`

## Current Baseline In This Repo

The current HF TriAttention path already has:

- exact rotated-key scoring on cached post-RoPE keys
- fixed geometric future offsets
- original TriAttention norm term
- end-to-end HF evaluation harness
- vLLM runtime path with Linux compatibility patches

Primary files for the current scorer:

- [`triattention/methods/triattention.py`](../triattention/methods/triattention.py)
- [`triattention/methods/pruning_utils.py`](../triattention/methods/pruning_utils.py)
- [`scripts/calibrate.py`](../scripts/calibrate.py)

## Repository Gaps To Keep In Mind

These are not theory issues. They are implementation and evaluation gaps.

- `Expected Attention` now has an HF baseline implementation.
- LongBench, RULER, and DFS worker-path harnesses now exist, plus a one-shot
  bundle runner for paper-matrix execution.
- The repo already supports AIME24, AIME25, MATH-500, and DFS-style memory
  retention evaluation.

That means:

- HF scorer implementation can proceed immediately.
- The remaining implementation gaps are now runtime parity, reporting glue, and
  final large-scale experiment execution.

## Target Scorer Decomposition

The new score should be exposed as a clean product of two choices:

- horizon kernel choice
- norm coefficient choice

Concretely:

- `horizon_mode = fixed | adaptive | variational`
- `norm_mode = tri | rms2`

This allows four main combinations:

- `fixed + tri`: current paper baseline
- `fixed + rms2`: isolated norm ablation
- `adaptive + tri`: isolated horizon ablation
- `adaptive + rms2`: proposed full method

and one explicit comparison mode:

- `variational + rms2`: offline-QP comparison against the Gaussian adaptive parameterization

## Variable Mapping

The paper notation must map directly to repo tensors.

### Per-head query statistics

For each sampled head `h` and frequency band `f`:

- `mu[h, f]`
  - repo field: `q_mean_complex`
  - shape: `[freq_count]` in HF sampled-head stats
- `E_abs[h, f]`
  - repo field: `q_abs_mean`
  - shape: `[freq_count]`
- `E_sq_abs[h, f]`
  - new repo field: `q_sq_abs_mean`
  - shape: `[freq_count]`

Derived quantities:

- `mu_abs = abs(mu)`
- `R_band = mu_abs / (E_abs + eps)`
- `alpha_tri = E_abs - mu_abs`
- `alpha_rms2 = (E_sq_abs - mu_abs^2) / (E_abs + mu_abs + eps)`

### Per-head horizon statistics

Recommended first implementation:

- `g[h, f] = mu_abs[h, f]`
- `w[h, f] = g[h, f] / sum_f g[h, f]`
- `lambda_eff[h] = exp(sum_f w[h, f] * log(2*pi / omega[f]))`
- `R_bar[h] = sum_f w[h, f] * R_band[h, f]`

This is the concrete default to implement first. If a later ablation wants a
different `g[h, f]`, that can be added afterward.

### Adaptive dyadic kernel

Let the support be the dyadic offset set already used by TriAttention:

- `G = {2^0, 2^1, ..., 2^J}`

where `J` is determined by the existing offset builder.

For each head `h`:

- `j_center[h] = log2(c_lambda * lambda_eff[h])`
- `sigma[h] = max(eps, s0 + s1 * (1 - R_bar[h]))`
- `pi_logits[h, j] = -((j - j_center[h])^2) / (2 * sigma[h]^2)`
- `pi[h, :] = softmax(pi_logits[h, :])`

Then:

- `kappa[h, f] = sum_j pi[h, j] * exp(i * omega[f] * 2^j)`

This `kappa[h, f]` is the object that should actually enter the scorer.

## Runtime Formula To Implement

### Mean scorer

For the rotated-key HF scorer, the target mean-path formula is:

`S_h(k) = Re sum_f mu[h,f] * conj(k_rot[f]) * kappa[h,f] * exp(i * omega[f] * round_start) + lambda_norm * sum_f alpha[h,f] * |k_rot[f]|`

where:

- `alpha = alpha_tri` when `norm_mode=tri`
- `alpha = alpha_rms2` when `norm_mode=rms2`

Important implementation point:

- `kappa[h, f]` is head-specific but key-independent
- therefore it can be precomputed once at `TriAttention.__init__`
- the mean scorer remains an `O(NF)` bandwise dot-product path

This is the runtime-feasibility sentence that must not get lost.

### Max scorer

For `score_aggregation = max`, keep the current exact rotated-key structure and
apply the adaptive kernel by weighting each offset with `pi[h, j]` before the
max reduction only if a stable and defensible formulation is chosen.

Initial implementation rule:

- implement `adaptive` only for `score_aggregation = mean`
- keep `max` available only for `fixed`

This avoids mixing the paper idea with a half-specified non-mean aggregation.

## Stats Schema Changes

The stats file needs a version bump.

### Metadata additions

Add:

- `stats_version = 3`

### Per-head field additions

Current fields:

- `q_mean_real`
- `q_mean_imag`
- `q_abs_mean`

Add:

- `q_sq_abs_mean`
- `oracle_tau_real`
- `oracle_tau_imag`
- `oracle_pi`
- `oracle_horizon_mean_real`
- `oracle_horizon_mean_imag`

Definition:

- `q_sq_abs_mean[f] = E[|q_f|^2]`

### Backward compatibility rule

Old stats files should remain usable for:

- `norm_mode=tri`
- `horizon_mode=fixed`

Old stats files should fail loudly for:

- `norm_mode=rms2`
- `horizon_mode=variational`

Reason:

- `q_sq_abs_mean` cannot be reconstructed exactly from existing first-order
  statistics.

## File-by-File Work Plan

### 1. `scripts/calibrate.py`

Implemented:

- second-order query statistics
- sampled raw-attention estimation for `tau[h, f]`
- offline dyadic simplex QP

New tensors per head:

- `q_sq_abs_mean = q_complex.abs().square().mean(dim=0)`
- `oracle_tau_complex`
- `oracle_pi`
- `oracle_horizon_mean_complex`

### 2. `triattention/methods/pruning_utils.py`

Extend the stats container and add the new helpers.

Required changes:

- extend `HeadFrequencyStats` with `q_sq_abs_mean`
- extend `HeadFrequencyStats` with optional `oracle_tau_complex`, `oracle_pi`, and `oracle_horizon_mean_complex`
- update `save_head_frequency_stats`
- update `load_head_frequency_stats`

Add helper functions:

- `compute_band_concentration(...)`
- `compute_rms2_coefficient(...)`
- `compute_effective_wavelength(...)`
- `build_adaptive_horizon_kernel(...)`
- `solve_variational_horizon_qp(...)`

Recommended helper outputs:

- `R_band: [freq_count]`
- `lambda_eff: scalar`
- `R_bar: scalar`
- `pi: [num_offsets]`
- `kappa_complex: [freq_count]`
- `oracle_tau_complex: [freq_count]`

### 3. `triattention/methods/triattention.py`

This is the primary HF implementation site.

Add config fields to `TriAttentionConfig`:

- `horizon_mode: str = "fixed"`
- `norm_mode: str = "tri"`
- `kernel_c_lambda: float = 1.0`
- `kernel_s0: float = 1.0`
- `kernel_s1: float = 1.0`
- `norm_lambda: float = 1.0`
- `adaptive_mean_only: bool = True`

Add per-head cached state in `TriAttention.__init__`:

- `head_alpha_tri[(layer, head)]`
- `head_alpha_rms2[(layer, head)]`
- `head_R_band[(layer, head)]`
- `head_lambda_eff[(layer, head)]`
- `head_R_bar[(layer, head)]`
- `head_pi[(layer, head)]`
- `head_kappa_complex[(layer, head)]`

Modify `_compute_layer_head_scores(...)`:

- when `horizon_mode=fixed`, keep existing path
- when `horizon_mode=adaptive`, use `kappa_complex` for mean aggregation
- select norm coefficient by `norm_mode`

### 4. `scripts/worker.py`

Thread the new HF scorer flags through the worker CLI.

Add arguments:

- `--triattention-horizon-mode`
- `--triattention-norm-mode`
- `--triattention-kernel-c-lambda`
- `--triattention-kernel-s0`
- `--triattention-kernel-s1`
- `--triattention-norm-lambda`

### 5. `scripts/cli.py`

Expose the same knobs through `run-one`, so A/B runs can be launched without
editing code.

### 6. `triattention/vllm/core/utils.py`

Port the new stats field into the vLLM stats loader after HF validation.

Required later changes:

- load `q_sq_abs_mean` if present
- preserve backward compatibility for old stats
- fail clearly when `rms2` is requested without version-2 stats

### 7. `triattention/vllm/core/scoring.py`

Port the winner from HF after it is validated.

Do not start here. HF is the reference path.

## Testing And Validation Order

The order matters.

### Stage A. Compatibility checks

Goal:

- prove that `fixed + tri` still matches the current exact scorer

Required checks:

- synthetic equivalence test for the existing mean scorer
- HF smoke on tiny model
- Qwen3 smoke on one sample

### Stage B. RMS2-only ablation

Goal:

- isolate the effect of the norm refinement without changing the horizon kernel

Required checks:

- `fixed + tri` vs `fixed + rms2`
- constant-norm synthetic sanity check:
  - RMS2 should reduce to the original coefficient up to numerical tolerance

### Stage C. Adaptive-kernel-only ablation

Goal:

- isolate the horizon contribution without changing the norm term

Required checks:

- `adaptive + tri` vs `fixed + tri`
- kernel sanity:
  - `pi` sums to 1
  - `sigma > 0`
  - `|kappa[h, f]| <= 1`

### Stage D. Full method

Goal:

- evaluate `adaptive + rms2`

Required checks:

- AIME24 tuning or calibration-only tuning
- held-out reporting on AIME25 and MATH-500
- DFS non-regression

### Stage E. Paper-complete evaluation

Infrastructure status:

- `Expected Attention` HF baseline exists
- LongBench, DFS, and RULER harnesses exist
- matrix reporting exists

Do not treat the paper as complete until the full matrix has actually been run.

## Experiment Matrix To Run

Once the HF scorer is wired, the minimum comparison table is:

- Full Attention
- TriAttention
- Fixed horizon + RMS2
- Adaptive horizon + original norm
- Adaptive horizon + RMS2
- Variational horizon + RMS2

When the missing baseline is added:

- Expected Attention

## Non-Negotiable Logging

For every A/B run, record:

- `horizon_mode`
- `norm_mode`
- `kernel_c_lambda`
- `kernel_s0`
- `kernel_s1`
- `norm_lambda`
- `stats_version`
- model
- dataset
- budget
- sample count

These fields should appear in run metadata and score dumps so comparisons remain
auditable.

## Recommended Default Hyperparameters For The First Pass

Use these only as the initial implementation defaults:

- `kernel_c_lambda = 1.0`
- `kernel_s0 = 1.0`
- `kernel_s1 = 1.0`
- `norm_lambda = 1.0`

These are not paper claims. They are just a stable starting point for wiring
the method into the codebase.

## Immediate Coding Order

1. Extend stats schema to version 3.
2. Keep `fixed + tri` behavior exactly stable after the schema change.
3. Add `norm_mode=rms2`.
4. Add `horizon_mode=adaptive` for mean aggregation only.
5. Add `horizon_mode=variational` as an explicit offline-QP comparison mode.
6. Add CLI and worker flags.
7. Run HF smoke and small A/B.
8. Run the full matrix in HF and vLLM.

## One-Line Summary

The implementation should keep `adaptive + rms2` as the clean default
HorizonKV variant, while preserving `variational + rms2` as the explicit
comparison mode for the Gaussian approximation, with explicit modes, versioned
stats, and auditable run metadata.
