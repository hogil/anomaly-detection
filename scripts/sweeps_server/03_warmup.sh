#!/usr/bin/env bash
# Stage 03: warmup axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_warmup" "warmup" "$@"
