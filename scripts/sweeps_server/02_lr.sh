#!/usr/bin/env bash
# Stage 02: lr axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_lr" "lr" "$@"
