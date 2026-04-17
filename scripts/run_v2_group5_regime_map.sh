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
TAG_PREFIX="${TAG_PREFIX:-v2_group5}"
PACK_TAG="${PACK_TAG:-${TAG_PREFIX}_promptheavy}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-experiments/analysis/v2/group5}"
MAIN_TASKS_STR="${MAIN_TASKS_STR:-qasper hotpotqa multi_news musique 2wikimqa}"
PROBE_TASKS_STR="${PROBE_TASKS_STR:-vcsum qmsum gov_report}"
METHODS_STR="${METHODS_STR:-triattention cask snapkv}"
BUDGETS_STR="${BUDGETS_STR:-256 384}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1}"
MAX_RECORDS="${MAX_RECORDS:-1}"
REF_PARALLEL="${REF_PARALLEL:-1}"
REPLAY_PARALLEL="${REPLAY_PARALLEL:-1}"
REPLAY_INNER_PARALLEL="${REPLAY_INNER_PARALLEL:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
SWEEP_TASKS_STR="${SWEEP_TASKS_STR:-multi_news qmsum gov_report}"
PREFIX_COVERAGES_STR="${PREFIX_COVERAGES_STR:-0.0 0.03125 0.0625 0.125}"
SIM_THRESHOLDS_STR="${SIM_THRESHOLDS_STR:-0.975 0.985 0.992}"

read -r -a MAIN_TASKS <<< "$MAIN_TASKS_STR"
read -r -a PROBE_TASKS <<< "$PROBE_TASKS_STR"
read -r -a METHODS <<< "$METHODS_STR"
read -r -a BUDGETS <<< "$BUDGETS_STR"
read -r -a SWEEP_TASKS <<< "$SWEEP_TASKS_STR"
read -r -a PREFIX_COVERAGES <<< "$PREFIX_COVERAGES_STR"
read -r -a SIM_THRESHOLDS <<< "$SIM_THRESHOLDS_STR"

mkdir -p "$ANALYSIS_ROOT"

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

"$PYTHON_BIN" scripts/run_promptheavy_pack.py \
  --tag "$PACK_TAG" \
  --model-path "$MODEL_PATH" \
  --stats-path "$STATS_PATH" \
  --stage all \
  --main-tasks "${MAIN_TASKS[@]}" \
  --probe-tasks "${PROBE_TASKS[@]}" \
  --methods "${METHODS[@]}" \
  --budgets "${BUDGETS[@]}" \
  --max-examples "$MAX_EXAMPLES" \
  --max-records "$MAX_RECORDS" \
  --ref-parallel "$REF_PARALLEL" \
  --replay-parallel "$REPLAY_PARALLEL" \
  --replay-inner-parallel "$REPLAY_INNER_PARALLEL" \
  --skip-existing

for task in "${SWEEP_TASKS[@]}"; do
  REF_PATH="experiments/${PACK_TAG}_refs/longbench/${MODEL_ALIAS}/runs/${task}/merged/merged.jsonl"
  if [[ ! -f "$REF_PATH" ]]; then
    echo "[error] missing replay reference for task=$task path=$REF_PATH" >&2
    exit 1
  fi

  for budget in "${BUDGETS[@]}"; do
    for coverage in "${PREFIX_COVERAGES[@]}"; do
      coverage_tag="$(printf '%s' "$coverage" | tr '.' '_')"
      run_with_limit "$PYTHON_BIN" scripts/replay_reference_fidelity.py \
        --reference "$REF_PATH" \
        --model-path "$MODEL_PATH" \
        --method cask \
        --budget "$budget" \
        --triattention-stats-file "$STATS_PATH" \
        --max-records "$MAX_RECORDS" \
        --attn-implementation "$ATTN_IMPL" \
        --load-dtype "$DTYPE" \
        --count-prompt-tokens true \
        --slack-budget-trigger true \
        --allow-prefill-compression false \
        --cask-prefix-coverage-ratio "$coverage" \
        --json-output "${ANALYSIS_ROOT}/${task}_budget${budget}_coverage_${coverage_tag}.json" \
        --csv-output "${ANALYSIS_ROOT}/${task}_budget${budget}_coverage_${coverage_tag}.csv"
    done

    for threshold in "${SIM_THRESHOLDS[@]}"; do
      threshold_tag="$(printf '%s' "$threshold" | tr '.' '_')"
      run_with_limit "$PYTHON_BIN" scripts/replay_reference_fidelity.py \
        --reference "$REF_PATH" \
        --model-path "$MODEL_PATH" \
        --method cask \
        --budget "$budget" \
        --triattention-stats-file "$STATS_PATH" \
        --max-records "$MAX_RECORDS" \
        --attn-implementation "$ATTN_IMPL" \
        --load-dtype "$DTYPE" \
        --count-prompt-tokens true \
        --slack-budget-trigger true \
        --allow-prefill-compression false \
        --cask-similarity-threshold "$threshold" \
        --json-output "${ANALYSIS_ROOT}/${task}_budget${budget}_similarity_${threshold_tag}.json" \
        --csv-output "${ANALYSIS_ROOT}/${task}_budget${budget}_similarity_${threshold_tag}.csv"
    done
  done
done

wait_all
