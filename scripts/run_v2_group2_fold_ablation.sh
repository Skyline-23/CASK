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
DATASET="${DATASET:-aime24}"
MAX_EXAMPLES="${MAX_EXAMPLES:-6}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_RECORDS="${MAX_RECORDS:-6}"
BUDGET="${BUDGET:-384}"
TAG_PREFIX="${TAG_PREFIX:-v2_group2}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-experiments/analysis/v2/group2}"

REF_TAG="${TAG_PREFIX}_ref_${DATASET}"
REF_MERGED="experiments/outputs/${DATASET}/${MODEL_ALIAS}/sample1/fullkv/full_${REF_TAG}/merged/merged.jsonl"

mkdir -p "$ANALYSIS_ROOT"

"$PYTHON_BIN" scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset "$DATASET" \
  --method fullkv \
  --run-tag "$REF_TAG" \
  --max-examples "$MAX_EXAMPLES" \
  --num-samples "$NUM_SAMPLES" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE"

"$PYTHON_BIN" scripts/replay_reference_fidelity.py \
  --reference "$REF_MERGED" \
  --model-path "$MODEL_PATH" \
  --method triattention \
  --budget "$BUDGET" \
  --triattention-stats-file "$STATS_PATH" \
  --max-records "$MAX_RECORDS" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --json-output "${ANALYSIS_ROOT}/${DATASET}_triattention${BUDGET}.json" \
  --csv-output "${ANALYSIS_ROOT}/${DATASET}_triattention${BUDGET}.csv"

"$PYTHON_BIN" scripts/replay_reference_fidelity.py \
  --reference "$REF_MERGED" \
  --model-path "$MODEL_PATH" \
  --method snapkv \
  --budget "$BUDGET" \
  --triattention-stats-file "$STATS_PATH" \
  --max-records "$MAX_RECORDS" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --json-output "${ANALYSIS_ROOT}/${DATASET}_snapkv${BUDGET}.json" \
  --csv-output "${ANALYSIS_ROOT}/${DATASET}_snapkv${BUDGET}.csv"

"$PYTHON_BIN" scripts/replay_reference_fidelity.py \
  --reference "$REF_MERGED" \
  --model-path "$MODEL_PATH" \
  --method cask \
  --budget "$BUDGET" \
  --triattention-stats-file "$STATS_PATH" \
  --max-records "$MAX_RECORDS" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --cask-decode-merge-enabled false \
  --json-output "${ANALYSIS_ROOT}/${DATASET}_cask_preserve_only${BUDGET}.json" \
  --csv-output "${ANALYSIS_ROOT}/${DATASET}_cask_preserve_only${BUDGET}.csv"

"$PYTHON_BIN" scripts/replay_reference_fidelity.py \
  --reference "$REF_MERGED" \
  --model-path "$MODEL_PATH" \
  --method cask \
  --budget "$BUDGET" \
  --triattention-stats-file "$STATS_PATH" \
  --max-records "$MAX_RECORDS" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --cask-merge-operator mean \
  --cask-representative-mode weighted_latest \
  --cask-use-phase-markers false \
  --json-output "${ANALYSIS_ROOT}/${DATASET}_cask_fold_weakened${BUDGET}.json" \
  --csv-output "${ANALYSIS_ROOT}/${DATASET}_cask_fold_weakened${BUDGET}.csv"

"$PYTHON_BIN" scripts/replay_reference_fidelity.py \
  --reference "$REF_MERGED" \
  --model-path "$MODEL_PATH" \
  --method cask \
  --budget "$BUDGET" \
  --triattention-stats-file "$STATS_PATH" \
  --max-records "$MAX_RECORDS" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --json-output "${ANALYSIS_ROOT}/${DATASET}_cask_full${BUDGET}.json" \
  --csv-output "${ANALYSIS_ROOT}/${DATASET}_cask_full${BUDGET}.csv"
