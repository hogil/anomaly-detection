#!/usr/bin/env bash
# Stage 01: baseline 5-seed recheck.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
run_paper_stage "baseline_recheck" \
  --skip-weights --skip-dataset --skip-round1 --skip-post "$@"
