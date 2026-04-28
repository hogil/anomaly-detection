#!/usr/bin/env bash
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_round1_axes "round1_stochastic_depth" "stochastic_depth" "$@"
