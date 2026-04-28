#!/usr/bin/env bash
# Server default: run the current rawbase round1 axes needed for summary.md.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

NEEDED_AXES="normal_ratio,per_class,lr,warmup,gc,label_smoothing,stochastic_depth,focal_gamma,abnormal_weight,ema,color,allow_tie_save"

run_paper_stage "needed_only" \
  --skip-weights \
  --skip-dataset \
  --skip-refcheck \
  --round1-skip-completed \
  --round1-include-axes "$NEEDED_AXES" \
  --skip-round2 \
  --skip-post \
  "$@"
