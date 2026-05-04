#!/usr/bin/env bash
# Stage 04: normal_ratio axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_normal_ratio" "normal_ratio" "$@"
