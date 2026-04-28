#!/usr/bin/env bash
# Run the one-off sample-skip safety experiment separately from the main sweep.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dataset.yaml}"
NUM_WORKERS="${NUM_WORKERS:-24}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
FORCE=0
MAX_LAUNCHED=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/06_sample_skip.sh [options]

Options:
  --python PATH        Python executable (default: python or $PYTHON)
  --config PATH        Dataset config used on this server (default: dataset.yaml)
  --num-workers N      train.py DataLoader workers (default: 24)
  --prefetch-factor N  train.py prefetch factor (default: 4)
  --force              Re-run completed tag
  --max-launched N     Stop controller after launching N new runs
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --max-launched) MAX_LAUNCHED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

echo "== paper stage: sample_skip =="

run_cmd "$PYTHON" scripts/prepare_server_queue.py \
  --src validations/paper_nonfinite_loss_filter_queue.json \
  --dst validations/server_paper_nonfinite_loss_filter_queue.json \
  --config "$CONFIG" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR"

cmd=(
  "$PYTHON" scripts/adaptive_experiment_controller.py
  --queue validations/server_paper_nonfinite_loss_filter_queue.json
  --summary validations/server_paper_nonfinite_loss_filter_summary.json
  --markdown validations/server_paper_nonfinite_loss_filter_summary.md
  --target-min 5
  --target-max 15
  --stop-mode never
  --candidate-min-runs-before-skip 0
  --completion-exit-grace 15
)
if [[ "$FORCE" -eq 1 ]]; then
  cmd+=(--force)
fi
if [[ "$MAX_LAUNCHED" -gt 0 ]]; then
  cmd+=(--max-launched "$MAX_LAUNCHED")
fi

run_cmd "${cmd[@]}"
