#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DTYPE="${DTYPE:-bfloat16}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
TAG_PREFIX="${TAG_PREFIX:-v2_group4}"
MODEL_SPECS="${MODEL_SPECS:-Qwen3-8B|experiments/models/Qwen3-8B|cask/calibration/for_aime25_experiment/qwen3_8b.pt;DeepSeek-R1-Distill-Qwen-7B|experiments/models/DeepSeek-R1-Distill-Qwen-7B|cask/calibration/for_aime25_experiment/ds_qwen7b.pt;DeepSeek-R1-Distill-Llama-8B|experiments/models/DeepSeek-R1-Distill-Llama-8B|cask/calibration/for_aime25_experiment/ds_llama8b.pt}"
REASONING_DATASETS_STR="${REASONING_DATASETS_STR:-aime24 aime25 math500}"
REASONING_METHODS_STR="${REASONING_METHODS_STR:-triattention cask snapkv expectedattention}"
REASONING_BUDGETS_STR="${REASONING_BUDGETS_STR:-256 384}"
MAX_EXAMPLES="${MAX_EXAMPLES:-6}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
JOB_PARALLEL="${JOB_PARALLEL:-1}"
ENABLE_PROMPTHEAVY="${ENABLE_PROMPTHEAVY:-true}"
PROMPT_TASKS_STR="${PROMPT_TASKS_STR:-qasper hotpotqa multi_news}"
PROMPT_METHODS_STR="${PROMPT_METHODS_STR:-triattention cask snapkv}"
PROMPT_BUDGETS_STR="${PROMPT_BUDGETS_STR:-256 384}"
PROMPT_MAX_EXAMPLES="${PROMPT_MAX_EXAMPLES:-1}"
PROMPT_MAX_RECORDS="${PROMPT_MAX_RECORDS:-1}"
PROMPT_REF_PARALLEL="${PROMPT_REF_PARALLEL:-1}"
PROMPT_REPLAY_PARALLEL="${PROMPT_REPLAY_PARALLEL:-1}"
PROMPT_REPLAY_INNER_PARALLEL="${PROMPT_REPLAY_INNER_PARALLEL:-1}"

read -r -a REASONING_DATASETS <<< "$REASONING_DATASETS_STR"
read -r -a REASONING_METHODS <<< "$REASONING_METHODS_STR"
read -r -a REASONING_BUDGETS <<< "$REASONING_BUDGETS_STR"
read -r -a PROMPT_TASKS <<< "$PROMPT_TASKS_STR"
read -r -a PROMPT_METHODS <<< "$PROMPT_METHODS_STR"
read -r -a PROMPT_BUDGETS <<< "$PROMPT_BUDGETS_STR"
IFS=';' read -r -a MODEL_ROWS <<< "$MODEL_SPECS"

PIDS=()

refresh_pids() {
  local live=()
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      live+=("$pid")
    fi
  done
  PIDS=("${live[@]:-}")
}

run_with_limit() {
  "$@" &
  PIDS+=("$!")
  while ((${#PIDS[@]} >= MAX_PARALLEL)); do
    wait -n
    refresh_pids
  done
}

wait_all() {
  local failed=0
  for pid in "${PIDS[@]:-}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if ((failed != 0)); then
    return 1
  fi
}

for row in "${MODEL_ROWS[@]}"; do
  IFS='|' read -r MODEL_ALIAS MODEL_PATH STATS_PATH <<< "$row"
  if [[ -z "${MODEL_ALIAS:-}" || -z "${MODEL_PATH:-}" ]]; then
    echo "[error] invalid MODEL_SPECS row: $row" >&2
    exit 1
  fi

  run_with_limit "$PYTHON_BIN" scripts/run_cask_frontier.py \
    --model "$MODEL_ALIAS" \
    --datasets "${REASONING_DATASETS[@]}" \
    --methods "${REASONING_METHODS[@]}" \
    --budgets "${REASONING_BUDGETS[@]}" \
    --frontier-tag "${TAG_PREFIX}_reasoning_${MODEL_ALIAS}" \
    --stats-path "$STATS_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --max-examples "$MAX_EXAMPLES" \
    --job-parallel "$JOB_PARALLEL" \
    --attn-implementation "$ATTN_IMPL" \
    --load-dtype "$DTYPE" \
    --skip-existing
done

wait_all

if [[ "$ENABLE_PROMPTHEAVY" == "true" ]]; then
  for row in "${MODEL_ROWS[@]}"; do
    IFS='|' read -r MODEL_ALIAS MODEL_PATH STATS_PATH <<< "$row"
    run_with_limit "$PYTHON_BIN" scripts/run_promptheavy_pack.py \
      --tag "${TAG_PREFIX}_prompt_${MODEL_ALIAS}" \
      --model-path "$MODEL_PATH" \
      --stats-path "$STATS_PATH" \
      --stage all \
      --main-tasks "${PROMPT_TASKS[@]}" \
      --probe-tasks \
      --methods "${PROMPT_METHODS[@]}" \
      --budgets "${PROMPT_BUDGETS[@]}" \
      --max-examples "$PROMPT_MAX_EXAMPLES" \
      --max-records "$PROMPT_MAX_RECORDS" \
      --ref-parallel "$PROMPT_REF_PARALLEL" \
      --replay-parallel "$PROMPT_REPLAY_PARALLEL" \
      --replay-inner-parallel "$PROMPT_REPLAY_INNER_PARALLEL" \
      --skip-existing
  done

  wait_all
fi
