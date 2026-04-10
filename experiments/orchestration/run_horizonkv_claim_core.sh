#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/elicer/HorizonKV"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
CLI_SCRIPT="${REPO_ROOT}/scripts/cli.py"
CALIBRATE_SCRIPT="${REPO_ROOT}/scripts/calibrate.py"
COMPARE_PAIR_SCRIPT="${REPO_ROOT}/scripts/compare_experiment_runs.py"
LONG_BENCH_SCRIPT="${REPO_ROOT}/scripts/run_longbench_suite.py"
DFS_SCRIPT="${REPO_ROOT}/scripts/run_dfs_suite.py"

MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/experiments/models/Qwen3-8B}"
CALIBRATION_INPUT="${CALIBRATION_INPUT:-${REPO_ROOT}/experiments/calibration/wikitext103_train_excerpt.txt}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-a100_claim_core}"
export RUN_TAG_PREFIX

SUITE_ROOT="${REPO_ROOT}/experiments/paper_support/${RUN_TAG_PREFIX}"
STATUS_PATH="${SUITE_ROOT}/status.txt"
COMPARE_ROOT="${SUITE_ROOT}/compare"
SUMMARY_ROOT="${SUITE_ROOT}/summaries"
STATS_DIR="${REPO_ROOT}/experiments/stats/${MODEL_ALIAS}"
STATS_V2_PATH="${STATS_V2_PATH:-${STATS_DIR}/${RUN_TAG_PREFIX}_v2.pt}"

HEADLINE_AIME_BUDGET="${HEADLINE_AIME_BUDGET:-512}"
HEADLINE_MATH_BUDGET="${HEADLINE_MATH_BUDGET:-512}"
LONGCTX_BUDGET="${LONGCTX_BUDGET:-1024}"
HEADLINE_AIME_SAMPLES="${HEADLINE_AIME_SAMPLES:-8}"
HEADLINE_MATH_SAMPLES="${HEADLINE_MATH_SAMPLES:-1}"
HEADLINE_AIME_MAX_EXAMPLES="${HEADLINE_AIME_MAX_EXAMPLES:-}"
HEADLINE_MATH_MAX_EXAMPLES="${HEADLINE_MATH_MAX_EXAMPLES:-}"

HEADLINE_AIME_MAX_NEW_TOKENS="${HEADLINE_AIME_MAX_NEW_TOKENS:-512}"
HEADLINE_MATH_MAX_NEW_TOKENS="${HEADLINE_MATH_MAX_NEW_TOKENS:-512}"

LONG_BENCH_MAX_EXAMPLES="${LONG_BENCH_MAX_EXAMPLES:-5}"
LONG_BENCH_TASKS=(${LONG_BENCH_TASKS:-qasper hotpotqa multi_news passage_count repobench-p})
DFS_MAX_EXAMPLES="${DFS_MAX_EXAMPLES:-50}"
DFS_MAX_NEW_TOKENS="${DFS_MAX_NEW_TOKENS:-512}"

STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"
CALIBRATION_MAX_LENGTH="${CALIBRATION_MAX_LENGTH:-32768}"
NONREG_METHODS=(triattention horizonkv)

mkdir -p "${SUITE_ROOT}" "${COMPARE_ROOT}" "${SUMMARY_ROOT}" "${STATS_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(timestamp)] $*"
}

write_status() {
  cat >"${STATUS_PATH}" <<EOF
timestamp=$(timestamp)
status=$1
detail=$2
EOF
}

sample_dir_for_count() {
  echo "sample$1"
}

run_root_for() {
  local dataset="$1"
  local method="$2"
  local tag="$3"
  local budget="$4"
  local num_samples="$5"
  local sample_dir
  sample_dir="$(sample_dir_for_count "${num_samples}")"
  if [[ "${method}" == "fullkv" ]]; then
    echo "${REPO_ROOT}/experiments/outputs/${dataset}/${MODEL_ALIAS}/${sample_dir}/fullkv/full_${tag}"
  else
    echo "${REPO_ROOT}/experiments/outputs/${dataset}/${MODEL_ALIAS}/${sample_dir}/${method}/budget_${budget}_${tag}"
  fi
}

build_stats_v2() {
  if [[ -f "${STATS_V2_PATH}" ]]; then
    log "reuse v2 stats ${STATS_V2_PATH}"
    return
  fi
  write_status "running" "build stats v2"
  log "start build stats v2 -> ${STATS_V2_PATH}"
  "${PYTHON_BIN}" "${CALIBRATE_SCRIPT}" \
    --model "${MODEL_PATH}" \
    --input "${CALIBRATION_INPUT}" \
    --output "${STATS_V2_PATH}" \
    --max-length "${CALIBRATION_MAX_LENGTH}" \
    --attn-implementation sdpa
  log "done build stats v2"
}

run_reasoning_variant() {
  local dataset="$1"
  local label="$2"
  local method="$3"
  local budget="$4"
  local num_samples="$5"
  local max_new_tokens="$6"
  local max_examples="$7"
  local tag="${RUN_TAG_PREFIX}_${dataset}_${label}"

  write_status "running" "reasoning ${dataset} ${label}"
  log "start reasoning dataset=${dataset} label=${label} method=${method} budget=${budget} num_samples=${num_samples} max_new_tokens=${max_new_tokens} max_examples=${max_examples}"

  local -a cmd=(
    "${PYTHON_BIN}" "${CLI_SCRIPT}" run-one
    --model "${MODEL_ALIAS}"
    --dataset "${dataset}"
    --method "${method}"
    --run-tag "${tag}"
    --gpus 0
    --num-shards 1
    --attn-implementation sdpa
    --load-dtype bfloat16
    --num-samples "${num_samples}"
    --max-new-tokens "${max_new_tokens}"
  )
  if [[ "${method}" != "fullkv" ]]; then
    cmd+=(--budget "${budget}" --stats-path "${STATS_V2_PATH}")
  fi
  if [[ -n "${max_examples}" ]]; then
    cmd+=(--max-examples "${max_examples}")
  fi
  "${cmd[@]}"
  log "done reasoning dataset=${dataset} label=${label}"
}

compare_pair() {
  local baseline="$1"
  local candidate="$2"
  local output_json="$3"
  mkdir -p "$(dirname "${output_json}")"
  "${PYTHON_BIN}" "${COMPARE_PAIR_SCRIPT}" \
    --baseline "${baseline}" \
    --candidate "${candidate}" \
    --json-output "${output_json}" >/dev/null
}

compare_reasoning_dataset() {
  local stage="$1"
  local dataset="$2"
  local budget="$3"
  local num_samples="$4"
  compare_pair \
    "$(run_root_for "${dataset}" fullkv "${RUN_TAG_PREFIX}_${dataset}_fullkv" 0 "${num_samples}")" \
    "$(run_root_for "${dataset}" horizonkv "${RUN_TAG_PREFIX}_${dataset}_horizonkv" "${budget}" "${num_samples}")" \
    "${COMPARE_ROOT}/${stage}_${dataset}_fullkv_vs_horizonkv.json"
  compare_pair \
    "$(run_root_for "${dataset}" triattention "${RUN_TAG_PREFIX}_${dataset}_triattention" "${budget}" "${num_samples}")" \
    "$(run_root_for "${dataset}" horizonkv "${RUN_TAG_PREFIX}_${dataset}_horizonkv" "${budget}" "${num_samples}")" \
    "${COMPARE_ROOT}/${stage}_${dataset}_triattention_vs_horizonkv.json"
}

write_summary() {
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

root = Path("/home/elicer/HorizonKV/experiments/paper_support") / os.environ["RUN_TAG_PREFIX"]
compare_root = root / "compare"
summary_root = root / "summaries"
summary_root.mkdir(parents=True, exist_ok=True)

def read(name: str):
    path = compare_root / name
    if not path.exists():
        return None
    return json.loads(path.read_text())

payload = {"headline": {}}

for dataset in ("aime25", "math500"):
    full = read(f"headline_{dataset}_fullkv_vs_horizonkv.json")
    tri = read(f"headline_{dataset}_triattention_vs_horizonkv.json")
    if full and tri:
        payload["headline"][dataset] = {
            "fullkv_acc": full["baseline"]["accuracy"],
            "triattention_acc": tri["baseline"]["accuracy"],
            "horizonkv_acc": full["candidate"]["accuracy"],
            "horizonkv_vs_fullkv": full["acc_delta"],
            "horizonkv_vs_triattention": tri["acc_delta"],
            "horizonkv_vs_fullkv_output_tps": full["output_tps_speedup"],
            "horizonkv_vs_triattention_output_tps": tri["output_tps_speedup"],
        }

headline_rows = list(payload["headline"].values())
if headline_rows:
    payload["headline_mean_delta"] = {
        "horizonkv_vs_fullkv": sum(r["horizonkv_vs_fullkv"] for r in headline_rows) / len(headline_rows),
        "horizonkv_vs_triattention": sum(r["horizonkv_vs_triattention"] for r in headline_rows) / len(headline_rows),
    }

out = summary_root / "claim_snapshot.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
}

run_longbench() {
  local method="$1"
  local output_root="${SUITE_ROOT}/nonreg/${method}/longbench"
  write_status "running" "longbench ${method}"
  log "start longbench_e method=${method}"
  local -a cmd=(
    "${PYTHON_BIN}" "${LONG_BENCH_SCRIPT}"
    --model-path "${MODEL_PATH}"
    --output-root "${output_root}"
    --longbench-e
    --tasks "${LONG_BENCH_TASKS[@]}"
    --max-examples "${LONG_BENCH_MAX_EXAMPLES}"
    --num-samples 1
    --python "${PYTHON_BIN}"
    --method "${method}"
    --attn-implementation sdpa
    --load-dtype bfloat16
  )
  if [[ "${method}" != "fullkv" ]]; then
    cmd+=(--kv-budget "${LONGCTX_BUDGET}" --triattention_stats_file "${STATS_V2_PATH}")
  fi
  "${cmd[@]}"
  log "done longbench_e method=${method}"
}

run_dfs() {
  local method="$1"
  local output_root="${SUITE_ROOT}/nonreg/${method}/dfs"
  write_status "running" "dfs ${method}"
  log "start dfs method=${method}"
  local -a cmd=(
    "${PYTHON_BIN}" "${DFS_SCRIPT}"
    --model-path "${MODEL_PATH}"
    --output-root "${output_root}"
    --max-examples "${DFS_MAX_EXAMPLES}"
    --max-new-tokens "${DFS_MAX_NEW_TOKENS}"
    --python "${PYTHON_BIN}"
    --method "${method}"
    --attn-implementation sdpa
    --load-dtype bfloat16
  )
  if [[ "${method}" != "fullkv" ]]; then
    cmd+=(--kv-budget "${LONGCTX_BUDGET}" --triattention_stats_file "${STATS_V2_PATH}")
  fi
  "${cmd[@]}"
  log "done dfs method=${method}"
}

main() {
  log "start claim-core harness"
  build_stats_v2
  if [[ "${STOP_AFTER_STAGE}" == "stats" ]]; then
    write_status "stopped" "after stats"
    log "stop after stats"
    return 0
  fi

  write_status "running" "headline"
  run_reasoning_variant aime25 fullkv fullkv 0 "${HEADLINE_AIME_SAMPLES}" "${HEADLINE_AIME_MAX_NEW_TOKENS}" "${HEADLINE_AIME_MAX_EXAMPLES}"
  run_reasoning_variant aime25 triattention triattention "${HEADLINE_AIME_BUDGET}" "${HEADLINE_AIME_SAMPLES}" "${HEADLINE_AIME_MAX_NEW_TOKENS}" "${HEADLINE_AIME_MAX_EXAMPLES}"
  run_reasoning_variant aime25 horizonkv horizonkv "${HEADLINE_AIME_BUDGET}" "${HEADLINE_AIME_SAMPLES}" "${HEADLINE_AIME_MAX_NEW_TOKENS}" "${HEADLINE_AIME_MAX_EXAMPLES}"
  compare_reasoning_dataset headline aime25 "${HEADLINE_AIME_BUDGET}" "${HEADLINE_AIME_SAMPLES}"
  write_summary >/dev/null

  run_reasoning_variant math500 fullkv fullkv 0 "${HEADLINE_MATH_SAMPLES}" "${HEADLINE_MATH_MAX_NEW_TOKENS}" "${HEADLINE_MATH_MAX_EXAMPLES}"
  run_reasoning_variant math500 triattention triattention "${HEADLINE_MATH_BUDGET}" "${HEADLINE_MATH_SAMPLES}" "${HEADLINE_MATH_MAX_NEW_TOKENS}" "${HEADLINE_MATH_MAX_EXAMPLES}"
  run_reasoning_variant math500 horizonkv horizonkv "${HEADLINE_MATH_BUDGET}" "${HEADLINE_MATH_SAMPLES}" "${HEADLINE_MATH_MAX_NEW_TOKENS}" "${HEADLINE_MATH_MAX_EXAMPLES}"
  compare_reasoning_dataset headline math500 "${HEADLINE_MATH_BUDGET}" "${HEADLINE_MATH_SAMPLES}"
  write_summary >/dev/null
  if [[ "${STOP_AFTER_STAGE}" == "headline" ]]; then
    write_status "stopped" "after headline"
    log "stop after headline"
    return 0
  fi

  write_status "running" "nonreg"
  local method
  for method in "${NONREG_METHODS[@]}"; do
    run_longbench "${method}"
    run_dfs "${method}"
  done
  if [[ "${STOP_AFTER_STAGE}" == "nonreg" ]]; then
    write_status "stopped" "after nonreg"
    log "stop after nonreg"
    return 0
  fi

  write_status "done" "claim-core harness complete"
  log "claim-core harness complete"
}

main "$@"
