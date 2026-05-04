#!/usr/bin/env bash
# Stage 06: label_smoothing axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_label_smoothing" "label_smoothing" "$@"
