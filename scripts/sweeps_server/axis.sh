#!/usr/bin/env bash
# Run one paper sweep axis. Replaces the old per-axis wrappers
# (10_lr.sh, 11_warmup.sh, ..., 41_allow_tie_save.sh, 90_gc.sh, 01_refcheck.sh).
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

VALID_AXES="lr warmup normal_ratio per_class label_smoothing stochastic_depth focal_gamma abnormal_weight ema color allow_tie_save gc baseline"

usage() {
  cat <<EOF
Usage:
  bash scripts/sweeps_server/axis.sh <axis> [run_paper_server_all options]

Axes (single-factor): $VALID_AXES

Special:
  baseline     baseline 5-seed recheck only (no sweep)

Examples:
  bash scripts/sweeps_server/axis.sh lr
  bash scripts/sweeps_server/axis.sh gc --max-launched 1
  bash scripts/sweeps_server/axis.sh baseline
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ $# -lt 1 ]] && exit 2
  exit 0
fi

axis="$1"
shift

case " $VALID_AXES " in
  *" $axis "*) ;;
  *) echo "Unknown axis: $axis" >&2; usage; exit 2 ;;
esac

if [[ "$axis" == "baseline" ]]; then
  run_paper_stage "baseline_recheck" \
    --skip-weights --skip-dataset --skip-round1 --skip-post "$@"
else
  run_round1_axes "axis_$axis" "$axis" "$@"
fi
