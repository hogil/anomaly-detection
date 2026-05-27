#!/usr/bin/env bash
# Fine-tune a trained binary model from a labeled field image run.
#
# Usage:
#   bash scripts/run_field_finetune.sh field_runs/<timestamp>_images logs/<model_run>
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/.."

if [[ "$#" -ne 2 ]]; then
  echo "Usage: bash scripts/run_field_finetune.sh <field-images-dir> <model-run-dir>" >&2
  exit 2
fi

FIELD_IMAGES="$1"
MODEL_RUN="$2"
IMAGE_ROOT="${FIELD_IMAGES}/dev_binary_model_inputs"
PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/check_torch_runtime.py

if [[ ! -d "$IMAGE_ROOT/normal" || ! -d "$IMAGE_ROOT/abnormal" ]]; then
  echo "missing labeled binary folders under: $IMAGE_ROOT" >&2
  echo "Run field prediction with a label column first, for example:" >&2
  echo "  bash scripts/run_field_predict.sh fab_export/timeseries_labeled.csv $MODEL_RUN 판정" >&2
  exit 1
fi

exec "$PYTHON" scripts/add_training_from_folders.py \
  --model-run "$MODEL_RUN" \
  --image-root "$IMAGE_ROOT" \
  --epochs 3 \
  --lr 1e-5 \
  --scheduler cosine
