#!/usr/bin/env bash
# Server default: run the current rawbase round1 axes needed for summary.md.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

NEEDED_AXES="lr,warmup,normal_ratio,per_class,label_smoothing,stochastic_depth,focal_gamma,abnormal_weight,ema,color,allow_tie_save,gc"

run_round1_axes "needed_only" "$NEEDED_AXES" "$@"
