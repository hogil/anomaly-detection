#!/usr/bin/env bash
# Resume only the rawbase round1 work still needed for the main server evidence.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

NEEDED_AXES="gc,stochastic_depth,focal_gamma,abnormal_weight,ema,color,allow_tie_save"

run_paper_stage "round1_needed_only" \
  --skip-weights \
  --skip-dataset \
  --skip-refcheck \
  --round1-skip-completed \
  --round1-include-axes "$NEEDED_AXES" \
  --skip-round2 \
  --skip-post \
  "$@"
