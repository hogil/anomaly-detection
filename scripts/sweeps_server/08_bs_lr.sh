#!/usr/bin/env bash
# Batch size × LR linear-scaling rule: 5 bs × 3 lr × 5 seeds = 75 runs
# Linear scaling rule: lr ∝ bs. If bs=256 optimum lr=5e-5, then:
#   bs=64  → lr=1.25e-5     bs=128 → lr=2.5e-5
#   bs=256 → lr=5e-5        bs=512 → lr=1e-4
# We test the rule across bs.
source "$(dirname "$0")/_common.sh"

# tag    bs
BSS=(
  "64    64"
  "128   128"
  "256   256"
  "384   384"
  "512   512"
)
# tag      backbone   head
LRS=(
  "s1    1.25e-5   1.25e-4"   # bs=256 baseline ÷4
  "s2    5e-5      5e-4"      # bs=256 baseline
  "s4    2e-4      2e-3"      # bs=256 baseline ×4
)
SEEDS=(42 1 2 3 4)

for b in "${BSS[@]}"; do
  read btag bs <<<"$b"
  for l in "${LRS[@]}"; do
    read ltag lrb lrh <<<"$l"
    for s in "${SEEDS[@]}"; do
      run_one "srv0416_bs${btag}_lr${ltag}_n700_s${s}" \
        --batch_size "$bs" \
        --lr_backbone "$lrb" --lr_head "$lrh" \
        --seed "$s"
    done
  done
done
