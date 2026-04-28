#!/usr/bin/env bash
# Gold combos: 8 variants x 5 seeds = 40 runs
# Start from top-1 of each axis (pc900 + sw5med + dp0.2 + wd0.02 + lr5e-5)
# and perturb one knob at a time.
source "$(dirname "$0")/_common.sh"

SEEDS=(42 1 2 3 4)

gold_base() {
  run_one "srv0416_gold_base_s$1" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 \
    --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --seed "$1"
}
gold_dp01() {
  run_one "srv0416_gold_dp01_s$1" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.1 \
    --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --seed "$1"
}
gold_wd005() {
  run_one "srv0416_gold_wd005_s$1" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 \
    --weight_decay 0.005 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --seed "$1"
}
gold_wd005_dp01() {
  run_one "srv0416_gold_wd005_dp01_s$1" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.1 \
    --weight_decay 0.005 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --seed "$1"
}
gold_lr3e5() {
  run_one "srv0416_gold_lr3e5_s$1" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 \
    --weight_decay 0.02 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --seed "$1"
}
gold_n1400() {
  run_one "srv0416_gold_n1400_s$1" \
    --max_per_class 900 --normal_ratio 1400 \
    --stochastic_depth_rate 0.2 \
    --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --seed "$1"
}
gold_long() {
  run_one "srv0416_gold_long_s$1" \
    --max_per_class 900 \
    --epochs 40 --patience 12 \
    --stochastic_depth_rate 0.2 \
    --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --seed "$1"
}
gold_warm8() {
  run_one "srv0416_gold_warm8_s$1" \
    --max_per_class 900 --warmup_epochs 8 \
    --stochastic_depth_rate 0.2 \
    --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --seed "$1"
}

for s in "${SEEDS[@]}"; do gold_base "$s"; done
for s in "${SEEDS[@]}"; do gold_dp01 "$s"; done
for s in "${SEEDS[@]}"; do gold_wd005 "$s"; done
for s in "${SEEDS[@]}"; do gold_wd005_dp01 "$s"; done
for s in "${SEEDS[@]}"; do gold_lr3e5 "$s"; done
for s in "${SEEDS[@]}"; do gold_n1400 "$s"; done
for s in "${SEEDS[@]}"; do gold_long "$s"; done
for s in "${SEEDS[@]}"; do gold_warm8 "$s"; done
