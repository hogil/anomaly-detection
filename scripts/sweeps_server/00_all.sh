#!/usr/bin/env bash
# Server default: run needed rawbase stages in paper follow-up order.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/00_all.sh [run_paper_server_all options]

Runs stages in order:
  core axes -> color -> sample_skip -> logical_train -> gc

For individual follow-up stages:
  bash scripts/sweeps_server/06_sample_skip.sh
  bash scripts/sweeps_server/50_logical_train.sh
  bash scripts/sweeps_server/90_gc.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SAMPLE_SKIP_ARGS=()
LOGICAL_TRAIN_ARGS=()
ALL_ARGS=("$@")
idx=0
while [[ "$idx" -lt "${#ALL_ARGS[@]}" ]]; do
  arg="${ALL_ARGS[$idx]}"
  case "$arg" in
    --python|--config|--num-workers|--prefetch-factor|--max-launched)
      if [[ "$((idx + 1))" -ge "${#ALL_ARGS[@]}" ]]; then
        echo "Missing value for $arg" >&2
        exit 2
      fi
      SAMPLE_SKIP_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      LOGICAL_TRAIN_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
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
      LOGICAL_TRAIN_ARGS+=("$arg")
      idx=$((idx + 1))
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done

PRE_COLOR_AXES="lr,warmup,normal_ratio,per_class,label_smoothing,stochastic_depth,focal_gamma,abnormal_weight,ema,allow_tie_save"

run_round1_axes "needed_pre_color" "$PRE_COLOR_AXES" "$@"
run_round1_axes "color" "color" "$@"

bash "$D/06_sample_skip.sh" "${SAMPLE_SKIP_ARGS[@]}"
bash "$D/50_logical_train.sh" "${LOGICAL_TRAIN_ARGS[@]}"

run_round1_axes "gc_last" "gc" "$@"
