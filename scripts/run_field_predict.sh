#!/usr/bin/env bash
# Render a field timeseries CSV and predict it with a trained model.
#
# Usage:
#   bash scripts/run_field_predict.sh fab_export/timeseries.csv logs/<model_run>
#   bash scripts/run_field_predict.sh fab_export/timeseries_labeled.csv logs/<model_run> 판정
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/.."

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: bash scripts/run_field_predict.sh <timeseries.csv> <model-run-or-best_model.pth> [label-column]" >&2
  exit 2
fi

TIMESERIES="$1"
MODEL_RUN="$2"
LABEL_COL="${3:-}"
PYTHON="${PYTHON:-python}"
STAMP="$(date +%y%m%d_%H%M%S)"
OUT_ROOT="field_runs"
IMAGE_DIR="${OUT_ROOT}/${STAMP}_images"
PRED_DIR="${OUT_ROOT}/${STAMP}_predictions"

mkdir -p "$OUT_ROOT"

render_args=(
  scripts/generate_field_images.py
  --timeseries "$TIMESERIES"
  --out-dir "$IMAGE_DIR"
  --no-timestamp
  --model-run "$MODEL_RUN"
)
if [[ -n "$LABEL_COL" ]]; then
  render_args+=(--label-col "$LABEL_COL")
fi

"$PYTHON" "${render_args[@]}"

"$PYTHON" scripts/predict_images.py \
  --model "$MODEL_RUN" \
  --manifest "$IMAGE_DIR/manifest.csv" \
  --output-dir "$PRED_DIR" \
  --normal-threshold 0.9

if [[ -n "$LABEL_COL" || -d "$IMAGE_DIR/dev_binary_model_inputs" ]]; then
  "$PYTHON" scripts/binary_threshold_report.py \
    --predictions "$PRED_DIR/predictions.csv" || true
fi

echo "images:      $IMAGE_DIR"
echo "predictions: $PRED_DIR"
