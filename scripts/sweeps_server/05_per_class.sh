#!/usr/bin/env bash
# Stage 05: per_class axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_per_class" "per_class" "$@"
