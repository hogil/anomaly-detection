#!/usr/bin/env bash
# ============================================================================
# Phase 1d — Full 20-epoch training with per-epoch checkpoint
# ============================================================================
# 목적: 4 batch size 에서 1 seed 학습. 각 epoch 의 weights 저장 → post-hoc simulation.
# Simulation: metric smoothing × weight aggregation × tie handling = 18 strategies
#
# 조건: spike-prone config (n=700, lr 3e-5, seed 1)
# Early stop 없음 (--no_early_stop), 매 epoch checkpoint 저장 (--save_every_epoch)
# EMA off (raw weights 저장 → post-hoc 가공)
# ============================================================================
set -e
cd "$(dirname "$0")/.."

N=700
SEED=1
LR_BB=3e-5
LR_HEAD=3e-4
EPOCHS=20
WARMUP=5

BATCH_SIZES=(16 32 64 128)

echo "============================================================"
echo " Phase 1d — BS sweep with per-epoch checkpoint (seed 1)"
echo "============================================================"
echo ""

for BS in "${BATCH_SIZES[@]}"; do
    LOG_DIR="logs/v9_phase1d_bs${BS}_n${N}_s${SEED}"

    if [ -f "$LOG_DIR/best_info.json" ]; then
        echo "[SKIP] $LOG_DIR"
        continue
    fi

    echo ""
    echo "[BS $BS] → $LOG_DIR"
    echo "------------------------------------------------------------"

    python train_tie.py \
        --epochs $EPOCHS \
        --warmup_epochs $WARMUP \
        --patience 999 \
        --no_early_stop \
        --save_every_epoch \
        --eval_test_every_epoch \
        --lr_backbone $LR_BB --lr_head $LR_HEAD \
        --batch_size $BS \
        --normal_ratio $N --seed $SEED \
        --focal_gamma 0.0 --dropout 0.0 --mode binary \
        --weight_decay 0.01 \
        --smooth_window 3 --smooth_method median \
        --ema_decay 0.0 \
        --precision fp16 \
        --log_dir "$LOG_DIR" \
        2>&1 | tee "$LOG_DIR.stdout.log" || {
            echo "[FAIL bs=$BS] — possible OOM, continuing to next"
            continue
        }

    # 크기 보고
    SIZE=$(du -sh "$LOG_DIR/checkpoints" 2>/dev/null | cut -f1)
    echo "[DONE bs=$BS] checkpoints size: $SIZE"
done

echo ""
echo "============================================================"
echo " Phase 1d complete — ready for simulation"
echo "============================================================"
echo "Run: python experiment_plan_fpfn/simulate_strategies.py"
