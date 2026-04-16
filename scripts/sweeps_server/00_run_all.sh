#!/usr/bin/env bash
# Run all server sweeps sequentially (345 runs, ~3-4h wall on H200 bf16+compile).
set -u
D="$(cd "$(dirname "$0")" && pwd)"

bash "$D/01_lrwd.sh"
bash "$D/02_pcxn.sh"
bash "$D/03_nscale.sh"
bash "$D/04_gold.sh"
bash "$D/05_tuning.sh"
bash "$D/06_large.sh"
bash "$D/07_long.sh"

echo "=== all server sweeps done ==="
