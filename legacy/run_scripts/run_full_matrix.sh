#!/bin/bash
# Full experiment matrix — after Exp 1 (EMA 0.9999) verification
# Phase A: n × seeds with EMA (winning config)
# Phase B: Backbones × 3 seeds
# Phase C: 6-class multiclass fine-grained
# Phase D: Fine-tuning strategies

set -e
LOG=logs/orchestrator_full.log

echo "=== ORCHESTRATOR FULL START $(date +%H:%M:%S) ===" > $LOG

# =============================================================================
# Phase A: n × seeds (ConvNeXtV2-Tiny + EMA 0.9999)
# 4 × 5 = 20 runs
# =============================================================================
echo "=== Phase A: n × seeds (EMA 0.9999) $(date +%H:%M:%S) ===" >> $LOG

for n in 700 1400 2100 2800; do
  for s in 42 1 2 3 4; do
    DIR="logs/v9_emaN_n${n}_s${s}"
    if [ -f "${DIR}/best_info.json" ]; then
      echo "[SKIP] ${DIR}" >> $LOG
      continue
    fi
    echo "=== A ${DIR} START $(date +%H:%M:%S) ===" >> $LOG
    python train.py \
      --normal_ratio $n --seed $s \
      --ema_decay 0.999 \
      --log_dir $DIR >> $LOG 2>&1 || { echo "FAILED: ${DIR}" >> $LOG; }
    echo "=== A ${DIR} DONE $(date +%H:%M:%S) ===" >> $LOG
  done
done
echo "=== Phase A DONE $(date +%H:%M:%S) ===" >> $LOG

# =============================================================================
# Phase B: Backbone comparison (n=700, EMA 0.9999, 3 seeds each)
# 6 × 3 = 18 runs
# Per-backbone LR per feedback_lr_spike_backbone_tuning.md
# =============================================================================
echo "=== Phase B: Backbones (n=700, 3 seeds) $(date +%H:%M:%S) ===" >> $LOG

# (key, model_name, lr_backbone, lr_head, warmup)
declare -A MODELS=(
  ["convnextv2_tiny"]="convnextv2_tiny.fcmae_ft_in22k_in1k|2e-5|2e-4|5"
  ["efficientnetv2_s"]="tf_efficientnetv2_s.in21k_ft_in1k|2e-5|2e-4|5"
  ["swin_tiny"]="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k|1.5e-5|1.5e-4|8"
  ["maxvit_tiny"]="maxvit_tiny_tf_224.in1k|1e-5|1e-4|8"
  ["clip_vit_b16"]="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k|5e-6|5e-5|8"
  ["convnextv2_base"]="convnextv2_base.fcmae_ft_in22k_in1k|1.5e-5|1.5e-4|5"
)

for key in "convnextv2_tiny" "efficientnetv2_s" "swin_tiny" "maxvit_tiny" "clip_vit_b16" "convnextv2_base"; do
  IFS='|' read -r model lr_bb lr_head warm <<< "${MODELS[$key]}"
  for s in 42 1 2; do
    DIR="logs/v9_bbN_${key}_s${s}"
    if [ -f "${DIR}/best_info.json" ]; then
      echo "[SKIP] ${DIR}" >> $LOG
      continue
    fi
    echo "=== B ${DIR} START $(date +%H:%M:%S) ===" >> $LOG
    python train.py \
      --normal_ratio 700 --seed $s \
      --model_name "$model" \
      --lr_backbone $lr_bb --lr_head $lr_head \
      --warmup_epochs $warm \
      --ema_decay 0.999 \
      --log_dir $DIR >> $LOG 2>&1 || { echo "FAILED: ${DIR}" >> $LOG; }
    echo "=== B ${DIR} DONE $(date +%H:%M:%S) ===" >> $LOG
  done
done
echo "=== Phase B DONE $(date +%H:%M:%S) ===" >> $LOG

# =============================================================================
# Phase C: 6-class multiclass fine-grained (ConvNeXtV2-Tiny, n=2800)
# 3 seeds
# =============================================================================
echo "=== Phase C: 6-class multiclass (n=2800) $(date +%H:%M:%S) ===" >> $LOG

for s in 42 1 2; do
  DIR="logs/v9_mcN_n2800_s${s}"
  if [ -f "${DIR}/best_info.json" ]; then
    echo "[SKIP] ${DIR}" >> $LOG
    continue
  fi
  echo "=== C ${DIR} START $(date +%H:%M:%S) ===" >> $LOG
  python train.py \
    --normal_ratio 2800 --seed $s \
    --mode multiclass \
    --ema_decay 0.999 \
    --log_dir $DIR >> $LOG 2>&1 || { echo "FAILED: ${DIR}" >> $LOG; }
  echo "=== C ${DIR} DONE $(date +%H:%M:%S) ===" >> $LOG
done
echo "=== Phase C DONE $(date +%H:%M:%S) ===" >> $LOG

# =============================================================================
# Phase D: Fine-tuning strategies (ConvNeXtV2-Tiny, n=2800)
# - freeze backbone 5 ep first
# - weight decay 0.05
# - label smoothing 0.05
# - mixup 0.2
# =============================================================================
echo "=== Phase D: Fine-tuning strategies $(date +%H:%M:%S) ===" >> $LOG

for v in "wd05|--weight_decay 0.05" "ls05|--label_smoothing 0.05" "mix02|--use_mixup --mixup_alpha 0.2" "frz5|--freeze_backbone_epochs 5"; do
  IFS='|' read -r tag opts <<< "$v"
  for s in 42 1 2; do
    DIR="logs/v9_ftN_${tag}_n2800_s${s}"
    if [ -f "${DIR}/best_info.json" ]; then
      echo "[SKIP] ${DIR}" >> $LOG
      continue
    fi
    echo "=== D ${DIR} START $(date +%H:%M:%S) ===" >> $LOG
    python train.py \
      --normal_ratio 2800 --seed $s \
      --ema_decay 0.999 \
      $opts \
      --log_dir $DIR >> $LOG 2>&1 || { echo "FAILED: ${DIR}" >> $LOG; }
    echo "=== D ${DIR} DONE $(date +%H:%M:%S) ===" >> $LOG
  done
done
echo "=== Phase D DONE $(date +%H:%M:%S) ===" >> $LOG

echo "=== ALL DONE $(date +%H:%M:%S) ===" >> $LOG
