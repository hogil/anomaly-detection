#!/usr/bin/env bash
# Stage 09: abnormal_weight axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_abnormal_weight" "abnormal_weight" "$@"
