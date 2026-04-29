#!/usr/bin/env bash
# Sample-skip safety experiment. Re-runs the baseline with
# `--filter_nonfinite_loss=true` so per-sample NaN/Inf losses are dropped at
# the step level. Compared against the same-seed baseline result. One-off.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dataset.yaml}"
detect_profile
NUM_WORKERS="${NUM_WORKERS:-$PROFILE_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PROFILE_PREFETCH}"
MAX_LAUNCHED="${MAX_LAUNCHED:-$PROFILE_MAX_LAUNCHED}"
LOG_DIR_GROUP="${LOG_DIR_GROUP:-$(date +%Y%m%d_%H%M%S)_sample_skip}"
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/sample_skip.sh [options]

Options:
  --python PATH        python executable
  --config PATH        dataset config (default dataset.yaml)
  --num-workers N      train.py DataLoader workers
  --prefetch-factor N  train.py prefetch factor
  --force              re-run completed tag
  --max-launched N     stop controller after launching N runs
  --log-dir-group NAME group runs under logs/<NAME>/
  -h, --help           show this help
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
    --log-dir-group) LOG_DIR_GROUP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

run_cmd() { echo; echo "+ $*"; "$@"; }

VAL_DIR="validations/${LOG_DIR_GROUP}"
mkdir -p "$VAL_DIR"

echo "== paper stage: sample_skip (output: $VAL_DIR) =="

run_cmd "$PYTHON" scripts/prepare_server_queue.py \
  --src validations/03_sample_skip_queue.json \
  --dst "${VAL_DIR}/03_sample_skip_active.json" \
  --config "$CONFIG" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR"

cmd=(
  "$PYTHON" scripts/adaptive_experiment_controller.py
  --queue "${VAL_DIR}/03_sample_skip_active.json"
  --summary "${VAL_DIR}/03_sample_skip_results.json"
  --markdown "${VAL_DIR}/03_sample_skip_results.md"
  --target-min 5
  --target-max 15
  --stop-mode never
  --candidate-min-runs-before-skip 0
  --completion-exit-grace 15
  --update-live-summary
)
[[ "$FORCE" -eq 1 ]] && cmd+=(--force)
[[ "$MAX_LAUNCHED" -gt 0 ]] && cmd+=(--max-launched "$MAX_LAUNCHED")
[[ -n "$LOG_DIR_GROUP" ]] && cmd+=(--log-dir-group "$LOG_DIR_GROUP")

run_cmd "${cmd[@]}"

run_cmd "$PYTHON" scripts/generate_stage_comparison.py \
  --results "${VAL_DIR}/03_sample_skip_results.json" \
  --baseline "${VAL_DIR}/01_baseline_results.json" \
  --out-md "${VAL_DIR}/03_sample_skip_results.md" \
  --out-plot "${VAL_DIR}/03_sample_skip_plot.png" \
  --title "Sample-skip probe vs baseline"
