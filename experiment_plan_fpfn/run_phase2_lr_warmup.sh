#!/usr/bin/env bash
# ============================================================================
# Phase 2 — LR / warmup 변주 (spike 구간 회피)
# ============================================================================
# 가설: warmup 늘리고 peak LR 낮추면 spike 빈도 ↓
# 9 runs = 3 config × 3 seeds (1/2/42)
# 예상 시간: ~20분/run × 9 = 180분
# ============================================================================
set -e
cd "$(dirname "$0")/.."

PHASE=phase2_lrwarmup
N=2800
EPOCHS=20
PATIENCE=5
EMA=0.999

SEEDS="1 2 42"

# config: (tag, lr_bb, lr_head, warmup)
CONFIGS=(
    "lr2e5_wu5:2e-5:2e-4:5"
    "lr3e5_wu10:3e-5:3e-4:10"
    "lr2e5_wu10:2e-5:2e-4:10"
)

echo "============================================================"
echo " Phase 2 — LR/warmup sweep (3 config × 3 seed = 9 runs)"
echo "============================================================"

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r TAG LR_BB LR_HEAD WARMUP <<< "$CFG"
    for SEED in $SEEDS; do
        LOG_DIR="logs/v9_${PHASE}_${TAG}_n${N}_s${SEED}"
        if [ -f "$LOG_DIR/best_info.json" ]; then
            echo "[SKIP] $LOG_DIR"
            continue
        fi
        echo ""
        echo "[RUN] $TAG seed=$SEED → $LOG_DIR"
        echo "  lr_bb=$LR_BB lr_head=$LR_HEAD warmup=$WARMUP"
        echo "------------------------------------------------------------"

        python train_tie.py \
            --epochs $EPOCHS \
            --warmup_epochs $WARMUP \
            --patience $PATIENCE \
            --lr_backbone $LR_BB --lr_head $LR_HEAD \
            --normal_ratio $N --seed $SEED \
            --focal_gamma 0.0 --dropout 0.0 --mode binary \
            --smooth_window 3 --smooth_method median \
            --ema_decay $EMA \
            --precision fp16 \
            --log_dir "$LOG_DIR" \
            2>&1 | tee "$LOG_DIR.stdout.log"
    done
done

echo ""
echo "============================================================"
echo " Phase 2 complete"
echo "============================================================"
python experiment_plan_fpfn/analyze.py --phase 2 || true
