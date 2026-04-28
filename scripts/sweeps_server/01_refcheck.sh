#!/usr/bin/env bash
# Run only the current raw reference check.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

run_paper_stage "refcheck_raw" \
  --skip-weights \
  --skip-dataset \
  --skip-round1 \
  --skip-round2 \
  --skip-post \
  "$@"
