#!/usr/bin/env bash
# Stage 16: gc axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_gc" "gc" "$@"
