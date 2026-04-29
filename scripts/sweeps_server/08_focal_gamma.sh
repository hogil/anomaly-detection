#!/usr/bin/env bash
# Stage 08: focal_gamma axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_focal_gamma" "focal_gamma" "$@"
