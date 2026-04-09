#!/usr/bin/env bash
# ============================================================================
# Phase 1 (n=700) — Spike-proof 검증 (spike-prone regime)
# ============================================================================
# 목적: EMA 0.999 + strict + val_loss guard 가 실제로 spike 를 막는지 검증
# n=700 은 n=2800 대비 4.6× 더 spike 많음 (0.55 vs 0.12 big spike / run)
# 그래서 spike-proof 로직은 n=700 에서 테스트해야 진짜 검증
#
# 예상: ep 4~8 에 spike 가 발생할 것이고, EMA 가 이를 흡수해서 saved model 은 깨끗
# ============================================================================
set -e
cd "$(dirname "$0")/.."

PHASE=phase1b     # v9_phase1b — val_loss guard 버그 수정 후 재실행
LR_BB=3e-5
LR_HEAD=3e-4
N=700
EPOCHS=20
WARMUP=5
PATIENCE=5         # min stop ep 15
EMA=0.999

SEEDS="1 2 3 4 42"

echo "============================================================"
echo " Phase 1 (n=700) — spike regime 에서 spike-proof 검증"
echo "============================================================"
echo "  lr_bb=$LR_BB  lr_head=$LR_HEAD  n=$N (spike-prone)"
echo "  ema_decay=$EMA  strict_save=True  val_loss_guard=2.0"
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

    python - <<PYEND
import json
bi = json.load(open("$LOG_DIR/best_info.json"))
tm = bi["test_metrics"]
th = bi.get("test_history", [])
fn = th[-1].get("fn", "-") if th else "-"
fp = th[-1].get("fp", "-") if th else "-"
print(f"  DONE seed=$SEED best_ep={bi['epoch']} f1={bi['test_f1']:.4f} abn={tm['abnormal']['recall']:.4f} nor={tm['normal']['recall']:.4f} fn={fn} fp={fp}")
PYEND
done

echo ""
echo "============================================================"
echo " Phase 1 n=700 complete"
echo "============================================================"
python experiment_plan_fpfn/analyze.py --phase 1 || true
