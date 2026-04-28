#!/usr/bin/env bash
# Dense LR sweep around optimum (4.5e-5 ~ 5.5e-5):
# 5 new points x 5 seeds = 25 runs
# 02_lr.sh covered coarse 1.5e-5~1e-4. Close in on the 5e-5 peak.
source "$(dirname "$0")/_common.sh"

# tag      backbone  head
PAIRS=(
  "4p5e5  4.5e-5  4.5e-4"
  "4p8e5  4.8e-5  4.8e-4"
  "5p0e5  5.0e-5  5.0e-4"
  "5p2e5  5.2e-5  5.2e-4"
  "5p5e5  5.5e-5  5.5e-4"
)
SEEDS=(42 1 2 3 4)

for p in "${PAIRS[@]}"; do
  read tag lrb lrh <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_lrd${tag}_n700_s${s}" \
      --lr_backbone "$lrb" --lr_head "$lrh" --seed "$s"
  done
done
