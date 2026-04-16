#!/usr/bin/env bash
# EMA decay × long training: 4 decays × 2 epoch configs × 5 seed = 40 runs
# Goal: does EMA actually stabilize late-epoch overfit?
source "$(dirname "$0")/_common.sh"

# tag     decay
EMAS=(
  "0p99    0.99"
  "0p995   0.995"
  "0p999   0.999"
  "0p9995  0.9995"
)
# tag       extra
EPS=(
  "ep40  --epochs 40 --patience 12"
  "ep60  --epochs 60 --patience 15"
)
SEEDS=(42 1 2 3 4)

for e in "${EMAS[@]}"; do
  read etag eval <<<"$e"
  for ep in "${EPS[@]}"; do
    read eptag epextra <<<"$ep"
    for s in "${SEEDS[@]}"; do
      run_one "srv0416_ema${etag}_${eptag}_n700_s${s}" \
        --ema_decay "$eval" $epextra --seed "$s"
    done
  done
done
