#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-experiments/models/Qwen3-8B}"
STATS_PATH="${STATS_PATH:-cask/calibration/for_aime25_experiment/qwen3_8b.pt}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DTYPE="${DTYPE:-bfloat16}"
REASONING_DATASET="${REASONING_DATASET:-aime24}"
PROMPT_TASK="${PROMPT_TASK:-multi_news}"
MAX_EXAMPLES="${MAX_EXAMPLES:-6}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_RECORDS="${MAX_RECORDS:-1}"
BUDGET="${BUDGET:-384}"
TAG_PREFIX="${TAG_PREFIX:-v2_group3}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-experiments/analysis/v2/group3}"
CORE_RATIOS_STR="${CORE_RATIOS_STR:-0.35 0.50 0.65}"
PREFIX_COVERAGES_STR="${PREFIX_COVERAGES_STR:-0.0 0.0625 0.125}"
SIM_THRESHOLDS_STR="${SIM_THRESHOLDS_STR:-0.975 0.985 0.992}"

read -r -a CORE_RATIOS <<< "$CORE_RATIOS_STR"
read -r -a PREFIX_COVERAGES <<< "$PREFIX_COVERAGES_STR"
read -r -a SIM_THRESHOLDS <<< "$SIM_THRESHOLDS_STR"

mkdir -p "$ANALYSIS_ROOT"

for core_ratio in "${CORE_RATIOS[@]}"; do
  core_tag="$(printf '%s' "$core_ratio" | tr '.' '_')"
  "$PYTHON_BIN" scripts/run_cask_frontier.py \
    --model "$MODEL_ALIAS" \
    --datasets "$REASONING_DATASET" \
    --methods cask \
    --budgets "$BUDGET" \
    --frontier-tag "${TAG_PREFIX}_${REASONING_DATASET}_core_${core_tag}" \
    --stats-path "$STATS_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --max-examples "$MAX_EXAMPLES" \
    --job-parallel 1 \
    --attn-implementation "$ATTN_IMPL" \
    --load-dtype "$DTYPE" \
    --cask-protected-core-ratio "$core_ratio"
done

"$PYTHON_BIN" scripts/run_promptheavy_pack.py \
  --tag "${TAG_PREFIX}_${PROMPT_TASK}" \
  --stage refs \
  --main-tasks "$PROMPT_TASK" \
  --methods triattention cask snapkv \
  --budgets "$BUDGET" \
  --max-examples 1 \
  --ref-parallel 1 \
  --replay-parallel 1 \
  --replay-inner-parallel 1 \
  --skip-existing

PROMPT_REF="experiments/${TAG_PREFIX}_${PROMPT_TASK}_refs/longbench/Qwen3-8B/runs/${PROMPT_TASK}/merged/merged.jsonl"

for coverage in "${PREFIX_COVERAGES[@]}"; do
  coverage_tag="$(printf '%s' "$coverage" | tr '.' '_')"
  "$PYTHON_BIN" scripts/replay_reference_fidelity.py \
    --reference "$PROMPT_REF" \
    --model-path "$MODEL_PATH" \
    --method cask \
    --budget "$BUDGET" \
    --triattention-stats-file "$STATS_PATH" \
    --max-records "$MAX_RECORDS" \
    --attn-implementation "$ATTN_IMPL" \
    --load-dtype "$DTYPE" \
    --count-prompt-tokens true \
    --slack-budget-trigger true \
    --allow-prefill-compression false \
    --cask-prefix-coverage-ratio "$coverage" \
    --json-output "${ANALYSIS_ROOT}/${PROMPT_TASK}_coverage_${coverage_tag}.json" \
    --csv-output "${ANALYSIS_ROOT}/${PROMPT_TASK}_coverage_${coverage_tag}.csv"
done

for threshold in "${SIM_THRESHOLDS[@]}"; do
  threshold_tag="$(printf '%s' "$threshold" | tr '.' '_')"
  "$PYTHON_BIN" scripts/replay_reference_fidelity.py \
    --reference "$PROMPT_REF" \
    --model-path "$MODEL_PATH" \
    --method cask \
    --budget "$BUDGET" \
    --triattention-stats-file "$STATS_PATH" \
    --max-records "$MAX_RECORDS" \
    --attn-implementation "$ATTN_IMPL" \
    --load-dtype "$DTYPE" \
    --count-prompt-tokens true \
    --slack-budget-trigger true \
    --allow-prefill-compression false \
    --cask-similarity-threshold "$threshold" \
    --json-output "${ANALYSIS_ROOT}/${PROMPT_TASK}_similarity_${threshold_tag}.json" \
    --csv-output "${ANALYSIS_ROOT}/${PROMPT_TASK}_similarity_${threshold_tag}.csv"
done
