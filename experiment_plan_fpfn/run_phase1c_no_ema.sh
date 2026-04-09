#!/usr/bin/env bash
# ============================================================================
# Phase 1c — no EMA + clip + weight_decay + batch_size sweep
# ============================================================================
# EMA 제거. 짧은 학습 (~2640 step) 에선 EMA 가 93% 만 수렴해서 raw 보다 나쁨.
# 대신 clip_grad_norm_(1.0) + fp16 GradScaler + weight_decay + bs 효과 검증.
#
# 8 runs 축소안:
#   A: bs 32, wd 0.01 × seed 1,2,42    (3 runs) ← no-EMA baseline
#   B: bs 64, wd 0.01 × seed 1,2,42    (3 runs) ← batch size effect
#   D: bs 32, wd 0.05 × seed 1,42      (2 runs) ← weight_decay effect
#
# bs 64 LR 조정: √2 scaling → 3e-5 × √2 ≈ 4.2e-5 (round to 4e-5)
# 예상 시간: 8 runs × ~15 분 = ~2 시간
# ============================================================================
set -e
cd "$(dirname "$0")/.."

N=700
EPOCHS=20
WARMUP=5
PATIENCE=5

echo "============================================================"
echo " Phase 1c — no EMA + clip + wd + bs sweep (8 runs)"
echo "============================================================"
echo ""

# ============================================================================
# Group A: bs 32, wd 0.01, lr 3e-5 (baseline without EMA)
# ============================================================================
for SEED in 1 2 42; do
    LOG_DIR="logs/v9_phase1c_A_bs32_wd01_n${N}_s${SEED}"
    if [ -f "$LOG_DIR/best_info.json" ]; then
        echo "[SKIP] $LOG_DIR"; continue
    fi
    echo ""
    echo "[A bs32 wd01] seed=$SEED → $LOG_DIR"
    echo "------------------------------------------------------------"
    python train_tie.py \
        --epochs $EPOCHS --warmup_epochs $WARMUP --patience $PATIENCE \
        --lr_backbone 3e-5 --lr_head 3e-4 \
        --batch_size 32 --weight_decay 0.01 \
        --normal_ratio $N --seed $SEED \
        --focal_gamma 0.0 --dropout 0.0 --mode binary \
        --smooth_window 3 --smooth_method median \
        --ema_decay 0.0 \
        --precision fp16 \
        --log_dir "$LOG_DIR" \
        2>&1 | tee "$LOG_DIR.stdout.log"
done

# ============================================================================
# Group B: bs 64, wd 0.01, lr 4e-5 (batch size ↑ → lr √2 scaling)
# ============================================================================
for SEED in 1 2 42; do
    LOG_DIR="logs/v9_phase1c_B_bs64_wd01_n${N}_s${SEED}"
    if [ -f "$LOG_DIR/best_info.json" ]; then
        echo "[SKIP] $LOG_DIR"; continue
    fi
    echo ""
    echo "[B bs64 wd01] seed=$SEED → $LOG_DIR"
    echo "------------------------------------------------------------"
    python train_tie.py \
        --epochs $EPOCHS --warmup_epochs $WARMUP --patience $PATIENCE \
        --lr_backbone 4e-5 --lr_head 4e-4 \
        --batch_size 64 --weight_decay 0.01 \
        --normal_ratio $N --seed $SEED \
        --focal_gamma 0.0 --dropout 0.0 --mode binary \
        --smooth_window 3 --smooth_method median \
        --ema_decay 0.0 \
        --precision fp16 \
        --log_dir "$LOG_DIR" \
        2>&1 | tee "$LOG_DIR.stdout.log"
done

# ============================================================================
# Group D: bs 32, wd 0.05 (높은 weight decay, regularization 강화)
# ============================================================================
for SEED in 1 42; do
    LOG_DIR="logs/v9_phase1c_D_bs32_wd05_n${N}_s${SEED}"
    if [ -f "$LOG_DIR/best_info.json" ]; then
        echo "[SKIP] $LOG_DIR"; continue
    fi
    echo ""
    echo "[D bs32 wd05] seed=$SEED → $LOG_DIR"
    echo "------------------------------------------------------------"
    python train_tie.py \
        --epochs $EPOCHS --warmup_epochs $WARMUP --patience $PATIENCE \
        --lr_backbone 3e-5 --lr_head 3e-4 \
        --batch_size 32 --weight_decay 0.05 \
        --normal_ratio $N --seed $SEED \
        --focal_gamma 0.0 --dropout 0.0 --mode binary \
        --smooth_window 3 --smooth_method median \
        --ema_decay 0.0 \
        --precision fp16 \
        --log_dir "$LOG_DIR" \
        2>&1 | tee "$LOG_DIR.stdout.log"
done

echo ""
echo "============================================================"
echo " Phase 1c complete"
echo "============================================================"
python experiment_plan_fpfn/analyze.py --phase 1 || true
