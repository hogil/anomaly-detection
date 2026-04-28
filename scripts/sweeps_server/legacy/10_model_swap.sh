#!/usr/bin/env bash
# Backbone swap ablation: 5 models x 5 seeds = 25 runs
# Requires timm-compatible weights in weights/. All 5 are present locally.
# NOTE: this overrides --model_name; --backbone is NOT a train.py arg.
source "$(dirname "$0")/_common.sh"

# tag              model_name
MODELS=(
  "cnxv2t    convnextv2_tiny.fcmae_ft_in22k_in1k"
  "cnxv2b    convnextv2_base.fcmae_ft_in22k_in1k"
  "swint     swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
  "mxvit     maxvit_tiny_tf_224.in1k"
  "effv2s    tf_efficientnetv2_s.in21k_ft_in1k"
)
SEEDS=(42 1 2 3 4)

for m in "${MODELS[@]}"; do
  read tag name <<<"$m"
  for s in "${SEEDS[@]}"; do
    run_one "srv0416_model_${tag}_n700_s${s}" \
      --model_name "$name" \
      --batch_size 128 \
      --seed "$s"
  done
done
