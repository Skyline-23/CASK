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
BUDGET="${BUDGET:-384}"
TAG_PREFIX="${TAG_PREFIX:-v2_group1}"
DUMP_ROOT="${DUMP_ROOT:-experiments/analysis/v2/group1}"

REF_TAG="${TAG_PREFIX}_ref_${DATASET}"
TRI_TAG="${TAG_PREFIX}_tri${BUDGET}_${DATASET}"
HORIZON_TAG="${TAG_PREFIX}_horizon${BUDGET}_${DATASET}"
CASK_TAG="${TAG_PREFIX}_cask${BUDGET}_${DATASET}"

TRI_DUMP_DIR="${DUMP_ROOT}/${DATASET}_triattention"
HORIZON_DUMP_DIR="${DUMP_ROOT}/${DATASET}_horizonkv"
CASK_DUMP_DIR="${DUMP_ROOT}/${DATASET}_cask"

mkdir -p "$TRI_DUMP_DIR" "$HORIZON_DUMP_DIR" "$CASK_DUMP_DIR"

"$PYTHON_BIN" scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset "$DATASET" \
  --method fullkv \
  --run-tag "$REF_TAG" \
  --max-examples "$MAX_EXAMPLES" \
  --num-samples "$NUM_SAMPLES" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE"

"$PYTHON_BIN" scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset "$DATASET" \
  --method triattention \
  --budget "$BUDGET" \
  --stats-path "$STATS_PATH" \
  --run-tag "$TRI_TAG" \
  --max-examples "$MAX_EXAMPLES" \
  --num-samples "$NUM_SAMPLES" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --score-dump-dir "$TRI_DUMP_DIR" \
  --score-dump-max-events 16

"$PYTHON_BIN" scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset "$DATASET" \
  --method horizonkv \
  --budget "$BUDGET" \
  --stats-path "$STATS_PATH" \
  --run-tag "$HORIZON_TAG" \
  --max-examples "$MAX_EXAMPLES" \
  --num-samples "$NUM_SAMPLES" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --triattention-horizon-mode adaptive \
  --triattention-norm-mode rms2 \
  --score-dump-dir "$HORIZON_DUMP_DIR" \
  --score-dump-max-events 16

"$PYTHON_BIN" scripts/cli.py run-one \
  --model "$MODEL_ALIAS" \
  --dataset "$DATASET" \
  --method cask \
  --budget "$BUDGET" \
  --stats-path "$STATS_PATH" \
  --run-tag "$CASK_TAG" \
  --max-examples "$MAX_EXAMPLES" \
  --num-samples "$NUM_SAMPLES" \
  --attn-implementation "$ATTN_IMPL" \
  --load-dtype "$DTYPE" \
  --score-dump-dir "$CASK_DUMP_DIR" \
  --score-dump-max-events 16

"$PYTHON_BIN" scripts/diff_score_dumps.py \
  --baseline-dir "$TRI_DUMP_DIR" \
  --candidate-dir "$HORIZON_DUMP_DIR" \
  --json-output "${DUMP_ROOT}/${DATASET}_tri_vs_horizonkv.json"

"$PYTHON_BIN" scripts/summarize_selection_dumps.py \
  --dump-dir "$CASK_DUMP_DIR" \
  --json-output "${DUMP_ROOT}/${DATASET}_cask_selection_summary.json"
