#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-experiments/models/Qwen3-8B}"
MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-8B}"
STATS_PATH="${STATS_PATH:-cask/calibration/for_aime25_experiment/qwen3_8b.pt}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DTYPE="${DTYPE:-bfloat16}"
TAG_PREFIX="${TAG_PREFIX:-v2_group6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/actual_bridge_v2/${TAG_PREFIX}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-experiments/analysis/v2/group6}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEMANTIC_MODEL="${SEMANTIC_MODEL:-}"
BRIDGE_ROWS="${BRIDGE_ROWS:-qasper|triattention|512;qasper|cask|256;multi_news|triattention|384;multi_news|cask|384;hotpotqa|triattention|256;hotpotqa|cask|256}"

IFS=';' read -r -a ROWS <<< "$BRIDGE_ROWS"

mkdir -p "$OUTPUT_ROOT" "$ANALYSIS_ROOT"

declare -A SEEN_TASKS=()

for row in "${ROWS[@]}"; do
  IFS='|' read -r TASK METHOD BUDGET <<< "$row"
  if [[ -z "${TASK:-}" || -z "${METHOD:-}" || -z "${BUDGET:-}" ]]; then
    echo "[error] invalid BRIDGE_ROWS row: $row" >&2
    exit 1
  fi
  SEEN_TASKS["$TASK"]=1
done

for TASK in "${!SEEN_TASKS[@]}"; do
  REF_ROOT="${OUTPUT_ROOT}/${TASK}_fullkv"
  "$PYTHON_BIN" scripts/run_longbench_suite.py \
    --model-path "$MODEL_PATH" \
    --output-root "$REF_ROOT" \
    --tasks "$TASK" \
    --max-examples "$MAX_EXAMPLES" \
    --num-samples "$NUM_SAMPLES" \
    --stats-path "$STATS_PATH" \
    --method fullkv \
    --attn-implementation "$ATTN_IMPL" \
    --load-dtype "$DTYPE"
done

for row in "${ROWS[@]}"; do
  IFS='|' read -r TASK METHOD BUDGET <<< "$row"
  CAND_ROOT="${OUTPUT_ROOT}/${TASK}_${METHOD}${BUDGET}"
  REF_ROOT="${OUTPUT_ROOT}/${TASK}_fullkv"
  LABEL="${TASK}_${METHOD}_${BUDGET}"

  "$PYTHON_BIN" scripts/run_longbench_suite.py \
    --model-path "$MODEL_PATH" \
    --output-root "$CAND_ROOT" \
    --tasks "$TASK" \
    --max-examples "$MAX_EXAMPLES" \
    --num-samples "$NUM_SAMPLES" \
    --stats-path "$STATS_PATH" \
    --method "$METHOD" \
    --kv-budget "$BUDGET" \
    --attn-implementation "$ATTN_IMPL" \
    --load-dtype "$DTYPE"

  FIDELITY_CMD=(
    "$PYTHON_BIN" scripts/compare_kv_fidelity.py
    --reference "$REF_ROOT"
    --candidate "$CAND_ROOT"
    --json-output "${ANALYSIS_ROOT}/${LABEL}_fidelity.json"
    --csv-output "${ANALYSIS_ROOT}/${LABEL}_fidelity.csv"
  )
  if [[ -n "$SEMANTIC_MODEL" ]]; then
    FIDELITY_CMD+=(--semantic-model "$SEMANTIC_MODEL")
  fi
  "${FIDELITY_CMD[@]}"

  "$PYTHON_BIN" scripts/compare_experiment_runs.py \
    --baseline "$REF_ROOT" \
    --candidate "$CAND_ROOT" \
    --json-output "${ANALYSIS_ROOT}/${LABEL}_throughput.json"

  cp "${CAND_ROOT}/longbench/${MODEL_ALIAS}/longbench_eval.json" \
    "${ANALYSIS_ROOT}/${LABEL}_task_eval.json"
done

"$PYTHON_BIN" scripts/build_v2_bridge_system_summary.py \
  --input-dir "$ANALYSIS_ROOT" \
  --output-json "${ANALYSIS_ROOT}/bridge_system_summary.json" \
  --output-csv "${ANALYSIS_ROOT}/bridge_system_summary.csv"
