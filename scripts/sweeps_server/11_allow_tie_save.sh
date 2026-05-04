#!/usr/bin/env bash
# Stage 11: allow_tie_save axis sweep.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_round1_axes "axis_allow_tie_save" "allow_tie_save" "$@"
