#!/usr/bin/env bash
# Runs extension sweep (seeds 5/6/7 for top-3 cells) + model soup AFTER bc=100.
# Insertion point: between bc finish and axis start.
# Priority: extension > soup > multi-severity > axis ablation (which then proceeds)
set -u
cd "$(dirname "$0")/../.."

echo "[ext-kickoff] started at $(date)"
# Wait for 100 bc completion
while true; do
  bc=$(ls -d logs/*vd080_bc_*_F* 2>/dev/null | grep -v vd080_bc_tie | wc -l)
  if [ "$bc" -ge 100 ]; then break; fi
  sleep 180
done
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done

echo "[ext-kickoff] 100 bc done, running seeds 5/6/7 extension at $(date)"
bash scripts/sweeps_laptop/legacy/70_extend_seeds567.sh

# After seed extension, run Model Soup on gc=0.5 sw=1raw (8 seeds)
echo "[ext-kickoff] running Model Soup at $(date)"
while ps -ef 2>/dev/null | grep -q "[t]rain.py"; do sleep 30; done
python scripts/model_soup.py --pattern "vd080_bc_gc0p5_sw1raw" \
  --out logs/soup_gc0p5_sw1raw_v1 2>&1 | tee -a validations/model_soup.log
python scripts/model_soup.py --pattern "vd080_bc_gc0p5_sw3mean" \
  --out logs/soup_gc0p5_sw3mean_v1 2>&1 | tee -a validations/model_soup.log
python scripts/model_soup.py --pattern "vd080_bc_gc1p0_sw3med" \
  --out logs/soup_gc1p0_sw3med_v1 2>&1 | tee -a validations/model_soup.log

# McNemar per-item comparison
echo "[ext-kickoff] McNemar tests at $(date)"
python scripts/mcnemar_compare.py --a-pattern "vd080_bc_gc0p5_sw1raw" \
  --b-pattern "vd080_bc_gc1p0_sw1raw" 2>&1 | tee -a validations/mcnemar.log
python scripts/mcnemar_compare.py --a-pattern "vd080_bc_gc0p5_sw1raw" \
  --b-pattern "vd080_bc_gc0p5_sw3mean" 2>&1 | tee -a validations/mcnemar.log
python scripts/mcnemar_compare.py --a-pattern "vd080_bc_gc0p5_sw1raw" \
  --b-pattern "vd080_bc_gc1p0_sw3med" 2>&1 | tee -a validations/mcnemar.log

echo "[ext-kickoff] done at $(date)"
