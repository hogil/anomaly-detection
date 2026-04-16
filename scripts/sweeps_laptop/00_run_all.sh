#!/usr/bin/env bash
# Run all laptop sweeps sequentially.
# Groups are ordered by expected signal strength (highest-value first).
# Total: ~700 runs at ~2.5 min/run = ~29h if none skipped.
set -u
D="$(cd "$(dirname "$0")" && pwd)"

# --- Tier 1: core hyperparameters (highest leverage) ---
bash "$D/01_wd.sh"              # wd fine (40)
bash "$D/02_lr.sh"              # lr fine (45)
bash "$D/09_focal.sh"           # focal_gamma (35) — memory: "gamma is key"
bash "$D/10_abw.sh"             # abnormal_weight (35)

# --- Tier 2: regularization ---
bash "$D/05_dp.sh"              # stochastic_depth (30)
bash "$D/11_dropout.sh"         # classifier dropout (30)
bash "$D/06_ls.sh"              # label_smoothing (25)
bash "$D/14_ema.sh"             # EMA decay (25)
bash "$D/20_wd_ext.sh"          # wd extreme tails (30)

# --- Tier 3: stability & bad-case rescue ---
bash "$D/03_gc.sh"              # gc @ lr=1e-4 (35)
bash "$D/04_sw.sh"              # sw @ n=2100 (30)
bash "$D/22_gc_rescue_ext.sh"   # gc @ lr=3e-4 and n=2100 (50)
bash "$D/23_sw_rescue_ext.sh"   # sw @ n=3500 and lr=1e-4 (60)
bash "$D/16_patience.sh"        # early-stop patience (25)

# --- Tier 4: optimization schedule ---
bash "$D/07_warm.sh"            # warmup (25)
bash "$D/18_sched.sh"           # scheduler cosine vs step (25)
bash "$D/21_lr_dense.sh"        # lr dense around 5e-5 (25)
bash "$D/13_freeze.sh"          # freeze backbone (25)

# --- Tier 5: loss / sampling ---
bash "$D/15_ohem.sh"            # OHEM (25)
bash "$D/19_sampler.sh"         # sampler type (15)
bash "$D/12_mixup.sh"           # mixup (25)

# --- Tier 6: dataset knobs ---
bash "$D/08_persz.sh"           # per-class 50~950 (57)
bash "$D/24_bs.sh"              # batch size (20)
bash "$D/17_nt.sh"              # normal threshold moderate (25)

echo "=== all laptop sweeps done ==="
