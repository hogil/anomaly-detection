#!/usr/bin/env bash
# Multi-severity eval kickoff — waits until ALL vd080 training queues finish,
# then runs eval_multitest.py in the gap before axis ablation.
# Provides natural GPU window for inference-only eval.
set -u
cd "$(dirname "$0")/../.."

echo "[multitest-kickoff] started at $(date)"
# Wait for 100 vd080_bc (pilot + ext) complete
while true; do
  bc=$(ls -d logs/*vd080_bc_*_F* 2>/dev/null | grep -v vd080_bc_tie | wc -l)
  if [ "$bc" -ge 100 ]; then break; fi
  sleep 180
done
# Ensure no training active right now
while ps -ef 2>/dev/null | awk '$0 ~ /[P]ython313.*train\.py/' | grep -q .; do sleep 30; done

echo "[multitest-kickoff] GPU idle, running multi-severity eval at $(date)"
python scripts/eval_multitest.py --prefix vd080_bc --run-filter s42 2>&1 | tee -a validations/multitest_eval.log
python scripts/eval_multitest.py --prefix vd080_bc --run-filter s1 2>&1 | tee -a validations/multitest_eval.log
# (add more seeds as needed)

echo "[multitest-kickoff] done at $(date)"
