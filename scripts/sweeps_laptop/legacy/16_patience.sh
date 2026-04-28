#!/usr/bin/env bash
# Patience (early stop counter): 5 levels x 5 seeds = 25 runs
# Patience counter starts from ep 10 (per project rule #15).
# min_epochs = 10 + patience
source "$(dirname "$0")/_common.sh"

PATS=(3 5 8 12 20)
SEEDS=(42 1 2 3 4)

for p in "${PATS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "ex0416_pat${p}_n700_s${s}" \
      --patience "$p" --seed "$s"
  done
done
