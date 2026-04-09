#!/usr/bin/env bash
# ============================================================================
# Phase 1 — Spike-proof baseline (5 seeds)
# ============================================================================
# 목표: EMA 0.999 + strict save + val_loss guard 의 기본 효과 검증
# 예상 시간: ~20분/run × 5 = 100분 (RTX 4060 Ti)
# 예상 결과: 5 seed 중 4 이상에서 FN≤2 AND FP≤2
# ============================================================================
set -e
cd "$(dirname "$0")/.."

PHASE=phase1_baseline
LR_BB=3e-5
LR_HEAD=3e-4
N=2800
EPOCHS=20
WARMUP=5
PATIENCE=5         # 최소 종료 ep 15 (= 10 + patience)
EMA=0.999

SEEDS="1 2 3 4 42"

echo "============================================================"
echo " Phase 1 baseline — EMA $EMA + strict save + val_loss guard"
echo "============================================================"
echo "  lr_bb=$LR_BB  lr_head=$LR_HEAD  n=$N"
echo "  epochs=$EPOCHS  warmup=$WARMUP  patience=$PATIENCE"
echo "  seeds: $SEEDS"
echo ""

for SEED in $SEEDS; do
    LOG_DIR="logs/v9_${PHASE}_n${N}_s${SEED}"

    if [ -f "$LOG_DIR/best_info.json" ]; then
        echo "[SKIP] $LOG_DIR (already done)"
        continue
    fi

    echo ""
    echo "[RUN] seed=$SEED → $LOG_DIR"
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

    # 학습 끝나고 간단 요약 출력
    python - <<PYEND
import json
bi = json.load(open("$LOG_DIR/best_info.json"))
tm = bi["test_metrics"]
print(f"  DONE seed=$SEED  best_ep={bi['epoch']}  F1={bi['test_f1']:.4f}  abn_R={tm['abnormal']['recall']:.4f}  nor_R={tm['normal']['recall']:.4f}")
PYEND
done

echo ""
echo "============================================================"
echo " Phase 1 complete — run analyze.py for summary"
echo "============================================================"
python experiment_plan_fpfn/analyze.py --phase 1 || true
