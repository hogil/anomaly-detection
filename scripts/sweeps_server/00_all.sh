#!/usr/bin/env bash
# Server default: run all paper stages in order.
# Auto-detects the runtime profile (server/pc/minimal) for sane defaults.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/00_all.sh [run_paper_server_all options]

Stages, in order:
  baseline recheck -> core axes -> color -> sample_skip -> backbone ->
  logical_train -> gc -> bkm_combined -> postprocess

Individual stages (run alone):
  bash scripts/sweeps_server/axis.sh <axis>     # one axis (lr, gc, color, ...)
  bash scripts/sweeps_server/axis.sh baseline   # baseline 5-seed recheck only
  bash scripts/sweeps_server/sample_skip.sh     # nonfinite-loss sample-skip probe
  bash scripts/sweeps_server/backbone.sh        # rotate through weights/*.pth
  bash scripts/sweeps_server/logical_train.sh   # per-member logical dataset train
  bash scripts/sweeps_server/bkm_combined.sh    # apply every BKM value at once
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Forward shared options to the separate stage scripts.
SAMPLE_SKIP_ARGS=()
BACKBONE_ARGS=()
LOGICAL_TRAIN_ARGS=()
BKM_COMBINED_ARGS=()
HAS_LOG_DIR_GROUP=0
for raw_arg in "$@"; do
  if [[ "$raw_arg" == "--log-dir-group" ]]; then
    HAS_LOG_DIR_GROUP=1
    break
  fi
done
if [[ "$HAS_LOG_DIR_GROUP" -eq 0 ]]; then
  SHARED_LOG_DIR_GROUP="${LOG_DIR_GROUP:-run_$(date +%Y%m%d_%H%M%S)}"
  ALL_ARGS=("$@" "--log-dir-group" "$SHARED_LOG_DIR_GROUP")
  echo "[00_all] log_dir_group=$SHARED_LOG_DIR_GROUP"
else
  ALL_ARGS=("$@")
fi
idx=0
while [[ "$idx" -lt "${#ALL_ARGS[@]}" ]]; do
  arg="${ALL_ARGS[$idx]}"
  case "$arg" in
    --python|--config|--num-workers|--prefetch-factor|--max-launched|--log-dir-group)
      if [[ "$((idx + 1))" -ge "${#ALL_ARGS[@]}" ]]; then
        echo "Missing value for $arg" >&2
        exit 2
      fi
      SAMPLE_SKIP_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      BACKBONE_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      LOGICAL_TRAIN_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      BKM_COMBINED_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      idx=$((idx + 2))
      ;;
    --workers)
      if [[ "$((idx + 1))" -ge "${#ALL_ARGS[@]}" ]]; then
        echo "Missing value for $arg" >&2
        exit 2
      fi
      LOGICAL_TRAIN_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      idx=$((idx + 2))
      ;;
    --force)
      SAMPLE_SKIP_ARGS+=("$arg")
      BACKBONE_ARGS+=("$arg")
      LOGICAL_TRAIN_ARGS+=("$arg")
      BKM_COMBINED_ARGS+=("$arg")
      idx=$((idx + 1))
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done

PRE_COLOR_AXES="lr,warmup,normal_ratio,per_class,label_smoothing,stochastic_depth,focal_gamma,abnormal_weight,ema,allow_tie_save"

run_round1_axes "needed_pre_color" "$PRE_COLOR_AXES" "${ALL_ARGS[@]}"
run_round1_axes "color" "color" "${ALL_ARGS[@]}"

bash "$D/sample_skip.sh" "${SAMPLE_SKIP_ARGS[@]}"
bash "$D/backbone.sh" "${BACKBONE_ARGS[@]}"
bash "$D/logical_train.sh" "${LOGICAL_TRAIN_ARGS[@]}"

run_round1_axes "gc_last" "gc" "${ALL_ARGS[@]}"

bash "$D/bkm_combined.sh" "${BKM_COMBINED_ARGS[@]}"

run_paper_stage "postprocess" \
  --skip-weights \
  --skip-dataset \
  --skip-refcheck \
  --skip-round1 \
  "${ALL_ARGS[@]}"
