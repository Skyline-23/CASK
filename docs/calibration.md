# Calibration Guide

HorizonKV and TriAttention both rely on precomputed frequency statistics captured from model query states. The main distinction is the stats version:

- legacy TriAttention `fixed + tri` runs can use the packaged v1 stats already checked into `triattention/calibration/` or `triattention/vllm/stats/`
- HorizonKV `adaptive + rms2` runs require v2 stats that include `q_sq_abs_mean`
- HorizonKV `variational + rms2` runs require v3 stats that also include offline horizon tensors from the simplex QP

## When You Need New Stats

Generate a new stats file when:

- you are running `--method horizonkv`
- you want to use `norm_mode=rms2`
- you want to compare the Gaussian adaptive kernel against the offline variational/QP horizon
- you are evaluating a model that does not already have a matching calibrated stats file
- you want calibration data that better matches your tuning or deployment regime

## Generating v2 Stats

```bash
python scripts/calibrate.py \
    --model /path/to/model \
    --input /path/to/calibration.txt \
    --output /path/to/model_stats_v2.pt
```

The calibration script runs a forward pass on plain text input, captures query states from every attention layer, inverts RoPE, and computes per-head frequency statistics. For HorizonKV, the resulting file includes second-order magnitude statistics such as `q_sq_abs_mean` in addition to the original mean and norm terms.

## Generating v3 Stats For Variational Horizon Runs

```bash
python scripts/calibrate.py \
    --model /path/to/model \
    --input /path/to/calibration.txt \
    --output /path/to/model_stats_v3.pt \
    --build-variational-horizon
```

With `--build-variational-horizon`, calibration additionally:

- samples raw attention over past tokens for each head to estimate the oracle target `tau_h`
- solves the dyadic simplex QP offline
- stores `oracle_tau`, `oracle_pi`, and `oracle_horizon_mean_complex` in the stats file

## Using The Stats File

### HuggingFace Path

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method horizonkv \
    --budget 2048 \
    --stats-path /path/to/model_stats_v2.pt
```

### vLLM Path

```bash
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/path/to/model_stats_v2.pt
export TRIATTN_RUNTIME_SPARSE_HORIZON_MODE=adaptive
export TRIATTN_RUNTIME_SPARSE_NORM_MODE=rms2
```

For the offline variational comparison mode:

```bash
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/path/to/model_stats_v3.pt
export TRIATTN_RUNTIME_SPARSE_HORIZON_MODE=variational
export TRIATTN_RUNTIME_SPARSE_NORM_MODE=rms2
```

## Packaged Legacy Stats

This repository still carries packaged legacy stats for the original TriAttention experiments. They are useful for:

- reproducing the original `fixed + tri` baseline
- quick local smoke tests
- backwards-compatible worker and runtime runs that do not require RMS2

They are not sufficient for HorizonKV RMS2 or variational runs unless they were regenerated as v2/v3 stats.

## Practical Recommendation

For any serious HorizonKV experiment:

1. create a fresh calibration text file
2. generate a new v2 stats file with `scripts/calibrate.py`
3. if you need the offline QP comparison, generate a v3 stats file with `--build-variational-horizon`
4. pass that file explicitly with `--stats-path` or `TRIATTN_RUNTIME_SPARSE_STATS_PATH`
5. avoid relying on packaged stats unless you are intentionally reproducing the historical TriAttention baseline
