#!/usr/bin/env bash
# Run all server sweeps sequentially (H200 bf16+compile+bs=256).
# Total: ~770 runs at ~25-40s/run effective = ~6-8h wall on single GPU.
set -u
D="$(cd "$(dirname "$0")" && pwd)"

# --- Tier 1: hyperparameter 2D grids ---
bash "$D/01_lrwd.sh"            # lr × wd (125)
bash "$D/09_focal_grid.sh"      # gamma × abnormal_weight (100)
bash "$D/11_dp_wd.sh"           # stoch_depth × wd (80)
bash "$D/12_pc_lr.sh"           # per-class × lr (80)

# --- Tier 2: scale & data ---
bash "$D/02_pcxn.sh"            # pc × N (80)
bash "$D/03_nscale.sh"          # N saturation (50)
bash "$D/13_nscale_ext.sh"      # N extreme (20)
bash "$D/08_bs_lr.sh"           # bs × lr linear scaling (75)

# --- Tier 3: optimization schedule ---
bash "$D/05_tuning.sh"          # head:backbone ratio (25)
bash "$D/07_long.sh"            # extended epochs + EMA (15)
bash "$D/14_ema_long.sh"        # EMA × long (40)

# --- Tier 4: combos (use Tier 1-3 findings) ---
bash "$D/04_gold.sh"            # gold v1 (40)
bash "$D/15_gold2.sh"           # gold v2 with more picks (40)

# --- Tier 5: model ablations ---
bash "$D/10_model_swap.sh"      # backbone swap (25)
bash "$D/06_large.sh"           # ConvNeXtV2-Base (10)

echo "=== all server sweeps done ==="
