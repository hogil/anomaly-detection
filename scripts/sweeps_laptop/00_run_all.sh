#!/usr/bin/env bash
# Run all laptop groups sequentially (287 runs, ~12h at 2.5min/run).
# Order: most informative first (wd/lr/gc) so early quit still gives value.
set -u
D="$(cd "$(dirname "$0")" && pwd)"

bash "$D/01_wd.sh"
bash "$D/02_lr.sh"
bash "$D/03_gc.sh"
bash "$D/04_sw.sh"
bash "$D/05_dp.sh"
bash "$D/06_ls.sh"
bash "$D/07_warm.sh"
bash "$D/08_persz.sh"

echo "=== all laptop sweeps done ==="
