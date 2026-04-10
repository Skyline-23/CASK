# Reproduction Guide

This guide describes the current HorizonKV experiment entry points. The repository still contains the original TriAttention baselines, but the paper-facing method and canonical repo identity are now HorizonKV.

## Prerequisites

```bash
git clone https://github.com/Skyline-23/horizonkv.git
cd horizonkv
pip install -e .
pip install flash-attn --no-build-isolation
```

For serious throughput experiments, use native Linux with a compatible vLLM installation. HuggingFace runs are sufficient for functional verification and many quality experiments.

## Step 1: Generate HorizonKV Stats

```bash
python scripts/calibrate.py \
    --model /path/to/Qwen3-8B \
    --input /path/to/calibration.txt \
    --output /path/to/qwen3_stats_v2.pt
```

For the offline variational/QP comparison mode, generate v3 stats instead:

```bash
python scripts/calibrate.py \
    --model /path/to/Qwen3-8B \
    --input /path/to/calibration.txt \
    --output /path/to/qwen3_stats_v3.pt \
    --build-variational-horizon
```

## Step 2: Run Reference Methods

### FullKV

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method fullkv
```

### TriAttention

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --budget 2048 \
    --stats-path /path/to/legacy_or_v2_stats.pt
```

### Expected Attention

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method expectedattention \
    --budget 2048
```

### HorizonKV

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method horizonkv \
    --budget 2048 \
    --stats-path /path/to/qwen3_stats_v2.pt
```

### HorizonKV With Offline Variational Horizon

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method horizonkv \
    --budget 2048 \
    --stats-path /path/to/qwen3_stats_v3.pt \
    --triattention-horizon-mode variational \
    --triattention-norm-mode rms2
```

## Step 3: Run The Full Benchmark Bundle

```bash
python scripts/run_horizonkv_benchmark_bundle.py \
    --cli-model Qwen3-8B \
    --model-path /path/to/Qwen3-8B \
    --method horizonkv \
    --suite-kv-budget 2048 \
    --stats-path /path/to/qwen3_stats_v2.pt \
    --run-tag paper_bundle
```

This bundle runner orchestrates:

- reasoning runs for AIME24, AIME25, and MATH-500
- DFS State Query
- LongBench / LongBench-E
- RULER

## Step 4: Compare The Final Matrix

```bash
python scripts/compare_horizonkv_matrix.py \
    --baseline-aime24 /path/to/baseline/aime24/eval \
    --candidate-aime24 /path/to/candidate/aime24/eval \
    --baseline-aime25 /path/to/baseline/aime25/eval \
    --candidate-aime25 /path/to/candidate/aime25/eval \
    --baseline-math500 /path/to/baseline/math500/eval \
    --candidate-math500 /path/to/candidate/math500/eval \
    --baseline-dfs /path/to/baseline/dfs \
    --candidate-dfs /path/to/candidate/dfs \
    --baseline-longbench /path/to/baseline/longbench \
    --candidate-longbench /path/to/candidate/longbench
```

## vLLM Reproduction

For vLLM, set runtime env vars explicitly and pass HorizonKV v2 stats:

```bash
export TRIATTN_RUNTIME_KV_BUDGET=2048
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/path/to/qwen3_stats_v2.pt
export TRIATTN_RUNTIME_SPARSE_HORIZON_MODE=adaptive
export TRIATTN_RUNTIME_SPARSE_NORM_MODE=rms2
```

For the offline variational comparison:

```bash
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=/path/to/qwen3_stats_v3.pt
export TRIATTN_RUNTIME_SPARSE_HORIZON_MODE=variational
export TRIATTN_RUNTIME_SPARSE_NORM_MODE=rms2
```

Then either launch a server:

```bash
vllm serve /path/to/model \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enforce-eager \
    --trust-remote-code \
    --enable-prefix-caching false
```

or use the pair runner:

```bash
python scripts/run_vllm_qwen3_pair.py \
    --mode horizonkv \
    --model-path /path/to/Qwen3-8B \
    --stats-path /path/to/qwen3_stats_v2.pt
```

## Notes

- Packaged legacy stats are acceptable for reproducing the historical TriAttention `fixed + tri` baseline, but not for HorizonKV RMS2 or variational runs.
- The current vLLM fast path is still optimized for `fixed + tri`; HorizonKV `adaptive + rms2` and `variational + rms2` run through the PyTorch scorer fallback.
- Historical upstream automation under `scripts/experiments/` remains available, but the repo-level recommended entry points are the HorizonKV commands above.
