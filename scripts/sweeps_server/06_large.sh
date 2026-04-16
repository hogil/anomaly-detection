#!/usr/bin/env bash
# ConvNeXtV2-Base backbone ablation: 2 dataset sizes x 5 seeds = 10 runs
# Requires weights/convnextv2_base.fcmae_ft_in22k_in1k.pth  (download with download.py)
source "$(dirname "$0")/_common.sh"

NS=(700 2800)
SEEDS=(42 1 2 3 4)

for n in "${NS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "srv0416_large_n${n}_s${s}" \
      --backbone convnextv2_base.fcmae_ft_in22k_in1k \
      --normal_ratio "$n" \
      --batch_size 128 \
      --seed "$s"
  done
done
