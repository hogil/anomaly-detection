#!/usr/bin/env bash
# ============================================================================
# Phase 4 — Regularization variants (label smoothing, mixup)
# ============================================================================
# Phase 1 base + 각 regularization 개별/조합 테스트
# 6 runs = 3 config × 2 seeds
# ============================================================================
set -e
cd "$(dirname "$0")/.."

PHASE=phase4_reg
N=2800
LR_BB=3e-5
LR_HEAD=3e-4
EPOCHS=20
WARMUP=5
PATIENCE=5
EMA=0.999

SEEDS="1 42"

# config: (tag, label_smoothing, mixup_flag, mixup_alpha)
CONFIGS=(
    "ls005:0.05:0:0.0"
    "mix01:0:1:0.1"
    "ls005_mix01:0.05:1:0.1"
)

echo "============================================================"
echo " Phase 4 — regularization (3 config × 2 seed = 6 runs)"
echo "============================================================"

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r TAG LS MIX_FLAG MIX_ALPHA <<< "$CFG"
    for SEED in $SEEDS; do
        LOG_DIR="logs/v9_${PHASE}_${TAG}_n${N}_s${SEED}"
        if [ -f "$LOG_DIR/best_info.json" ]; then
            echo "[SKIP] $LOG_DIR"
            continue
        fi
        echo ""
        echo "[RUN] $TAG seed=$SEED → $LOG_DIR"
        echo "  ls=$LS mix=$MIX_FLAG α=$MIX_ALPHA"
        echo "------------------------------------------------------------"

        MIX_ARG=""
        if [ "$MIX_FLAG" = "1" ]; then
            MIX_ARG="--use_mixup --mixup_alpha $MIX_ALPHA"
        fi

        python train_tie.py \
            --epochs $EPOCHS \
            --warmup_epochs $WARMUP \
            --patience $PATIENCE \
            --lr_backbone $LR_BB --lr_head $LR_HEAD \
            --normal_ratio $N --seed $SEED \
            --focal_gamma 0.0 --dropout 0.0 --mode binary \
            --smooth_window 3 --smooth_method median \
            --label_smoothing $LS $MIX_ARG \
            --ema_decay $EMA \
            --precision fp16 \
            --log_dir "$LOG_DIR" \
            2>&1 | tee "$LOG_DIR.stdout.log"
    done
done

echo ""
echo "============================================================"
echo " Phase 4 complete"
echo "============================================================"
python experiment_plan_fpfn/analyze.py --phase 4 || true
