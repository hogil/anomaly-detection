#!/usr/bin/env bash
# Gold combos v2: layer on best-of-each-axis findings beyond 04_gold.sh
# 8 combos x 5 seeds = 40 runs
source "$(dirname "$0")/_common.sh"

SEEDS=(42 1 2 3 4)

# g2_fg2abw12: add focal gamma=2 + abnormal_weight=1.2 to base gold
for s in "${SEEDS[@]}"; do
  run_one "srv0416_g2_fg2abw12_s${s}" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --focal_gamma 2.0 --abnormal_weight 1.2 \
    --seed "$s"
done

# g2_mixup02: add mixup 0.2
for s in "${SEEDS[@]}"; do
  run_one "srv0416_g2_mixup02_s${s}" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --use_mixup --mixup_alpha 0.2 \
    --seed "$s"
done

# g2_ema999_ep40: ema 0.999 + long
for s in "${SEEDS[@]}"; do
  run_one "srv0416_g2_ema999_ep40_s${s}" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --ema_decay 0.999 --epochs 40 --patience 12 \
    --seed "$s"
done

# g2_ohem05: OHEM ratio 0.5
for s in "${SEEDS[@]}"; do
  run_one "srv0416_g2_ohem05_s${s}" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --ohem_ratio 0.5 \
    --seed "$s"
done

# g2_sampler_binary: balanced binary sampler
for s in "${SEEDS[@]}"; do
  run_one "srv0416_g2_samp_bin_s${s}" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --train_sampler balanced_binary \
    --seed "$s"
done

# g2_frz2: freeze backbone 2 epochs (head warm start)
for s in "${SEEDS[@]}"; do
  run_one "srv0416_g2_frz2_s${s}" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --freeze_backbone_epochs 2 \
    --seed "$s"
done

# g2_allin: kitchen sink of best picks
for s in "${SEEDS[@]}"; do
  run_one "srv0416_g2_allin_s${s}" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.2 --weight_decay 0.02 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --focal_gamma 2.0 --abnormal_weight 1.2 \
    --ema_decay 0.999 --epochs 40 --patience 12 \
    --seed "$s"
done

# g2_minimal: drop dropout+wd to test if over-regularized
for s in "${SEEDS[@]}"; do
  run_one "srv0416_g2_minimal_s${s}" \
    --max_per_class 900 \
    --stochastic_depth_rate 0.0 --weight_decay 0.0 \
    --lr_backbone 5e-5 --lr_head 5e-4 \
    --seed "$s"
done
