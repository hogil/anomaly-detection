#!/usr/bin/env bash
# Extreme N saturation probe: 4 very-large N x 5 seeds = 20 runs
# 03_nscale.sh covered up to N=7000. Go past to see if error keeps rising
# (true overfit) or plateaus (saturation).
source "$(dirname "$0")/_common.sh"

NS=(10000 15000 20000 30000)
SEEDS=(42 1 2 3 4)

for n in "${NS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "srv0416_nscalext_n${n}_s${s}" \
      --normal_ratio "$n" --seed "$s"
  done
done
