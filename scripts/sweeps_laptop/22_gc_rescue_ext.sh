#!/usr/bin/env bash
# Extended GC rescue: gc × more-extreme LR (lr=3e-4) AND gc × large N unstable (n=2100)
# 03_gc.sh did gc × lr=1e-4. Here we test:
#   (a) gc × lr=3e-4 (even more unstable) : 5 gc levels x 5 seed = 25
#   (b) gc × n=2100 (broken-data unstable) : 5 gc levels x 5 seed = 25
# Total = 50 runs.
source "$(dirname "$0")/_common.sh"

GCS=(
  "0p25  0.25"
  "0p5   0.5"
  "1p0   1.0"
  "2p0   2.0"
  "5p0   5.0"
)
SEEDS=(42 1 2 3 4)

# (a) gc rescue @ lr=3e-4
for p in "${GCS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_gc${tag}_lr3em4_n700_s${s}" \
      --lr_backbone 3e-4 --lr_head 3e-3 \
      --grad_clip "$val" --seed "$s"
  done
done

# (b) gc rescue @ unstable n=2100
for p in "${GCS[@]}"; do
  read tag val <<<"$p"
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_gc${tag}_n2100_s${s}" \
      --normal_ratio 2100 \
      --grad_clip "$val" --seed "$s"
  done
done
