#!/usr/bin/env bash
# Normal threshold (post-hoc decision threshold): 5 levels x 5 seeds = 25 runs
# CLAUDE memory: NO extreme NT (test-peeking). Keep in moderate [0.3, 0.7].
# Note: --normal_threshold changes train.py's default NT test; it ALSO reports
# the full sweep in best_info.json["normal_threshold_results"], so this group
# mostly lets us see whether picking a different primary NT changes logging.
source "$(dirname "$0")/_common.sh"

PAIRS=(
  "0p3    0.3"
  "0p4    0.4"
  "0p5    0.5"
  "0p6    0.6"
  "0p7    0.7"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_nt${tag}_n700_s${s}" \
      --normal_threshold "$val" --seed "$s"
  done
done
