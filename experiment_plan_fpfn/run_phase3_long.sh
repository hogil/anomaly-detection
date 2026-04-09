#!/usr/bin/env bash
# ============================================================================
# Phase 3 — Longer training (late convergence exploration)
# ============================================================================
# 가설: ep 10~15 에서 고정되는 대신 더 오래 돌리면 더 좋은 local minimum 발견 가능
# 6 runs = 2 config × 3 seeds
# ============================================================================
set -e
cd "$(dirname "$0")/.."

PHASE=phase3_long
N=2800
LR_BB=3e-5      # Phase 2 결과에 따라 덮어씀
LR_HEAD=3e-4
WARMUP=5
EMA=0.999

SEEDS="1 2 42"

# config: (tag, epochs, patience)  → min_stop = 10 + patience
CONFIGS=(
    "ep30_p10:30:10"
    "ep40_p15:40:15"
)

echo "============================================================"
echo " Phase 3 — long training (2 config × 3 seed = 6 runs)"
echo "============================================================"

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r TAG EPOCHS PATIENCE <<< "$CFG"
    for SEED in $SEEDS; do
        LOG_DIR="logs/v9_${PHASE}_${TAG}_n${N}_s${SEED}"
        if [ -f "$LOG_DIR/best_info.json" ]; then
            echo "[SKIP] $LOG_DIR"
            continue
        fi
        echo ""
        echo "[RUN] $TAG seed=$SEED → $LOG_DIR"
        echo "  epochs=$EPOCHS patience=$PATIENCE (min_stop=$((10+PATIENCE)))"
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
echo " Phase 3 complete"
echo "============================================================"
python experiment_plan_fpfn/analyze.py --phase 3 || true
