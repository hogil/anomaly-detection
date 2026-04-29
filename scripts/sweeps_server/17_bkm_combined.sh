#!/usr/bin/env bash
# BKM (Best Known Method) combined sweep. Applies the per-axis BKM values
# documented in docs/summary.md all at once, in a single candidate, at 5 seeds.
# This is the "kitchen sink" comparison against the rawbase baseline.
#
# Source of BKM values: docs/summary.md "Best Known Method" table.
#   normal_ratio     700  -> 3300
#   gc               0.0  -> 0.5
#   label_smoothing  0.00 -> 0.02
#   stochastic_depth 0.00 -> 0.05
#   focal_gamma      0.0  -> 2.0
#   abnormal_weight  1.0  -> 1.5
#   ema              0.0  -> 0.95
#   allow_tie_save   off  -> on
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dataset.yaml}"
detect_profile
NUM_WORKERS="${NUM_WORKERS:-$PROFILE_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PROFILE_PREFETCH}"
MAX_LAUNCHED="${MAX_LAUNCHED:-$PROFILE_MAX_LAUNCHED}"
LOG_DIR_GROUP="${LOG_DIR_GROUP:-run_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-42,1,2,3,4}"
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/bkm_combined.sh [options]

Runs ONE candidate with every BKM value from docs/summary.md applied at once,
at the default 5 seeds. Compared against the rawbase baseline.

Options:
  --python PATH        python executable
  --config PATH        dataset config (default dataset.yaml)
  --num-workers N      train.py DataLoader workers
  --prefetch-factor N  train.py prefetch factor
  --seeds CSV          seeds (default: 42,1,2,3,4)
  --force              re-run completed tags
  --max-launched N     stop controller after launching N runs
  --log-dir-group NAME group runs under logs/<NAME>/
  -h, --help           show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --max-launched) MAX_LAUNCHED="$2"; shift 2 ;;
    --log-dir-group) LOG_DIR_GROUP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

run_cmd() { echo; echo "+ $*"; "$@"; }

QUEUE=validations/05_bkm_combined_queue.json
ACTIVE=validations/05_bkm_combined_active.json
SUMMARY=validations/05_bkm_combined_results.json
MARKDOWN=validations/05_bkm_combined_results.md

echo "== paper stage: bkm_combined =="

run_cmd "$PYTHON" - "$QUEUE" "$SEEDS" "$NUM_WORKERS" "$PREFETCH_FACTOR" <<'PY'
import json
import sys
from pathlib import Path

queue_path, seeds_raw, num_workers, prefetch = sys.argv[1:5]
seeds = [int(s.strip()) for s in seeds_raw.split(",") if s.strip()]
candidate = "fresh0412_v11_bkm_combined_n3300"

base_args = {
    "--mode": "binary",
    "--config": "dataset.yaml",
    "--epochs": 20,
    "--patience": 5,
    "--batch_size": 32,
    "--dropout": 0.0,
    "--precision": "fp16",
    "--num_workers": int(num_workers),
    "--prefetch_factor": int(prefetch),
    "--smooth_window": 1,
    "--smooth_method": "median",
    "--lr_backbone": "2e-5",
    "--lr_head": "2e-4",
    "--warmup_epochs": 5,
    "--weight_decay": 0.01,
    # BKM values applied together:
    "--normal_ratio": 3300,
    "--grad_clip": 0.5,
    "--label_smoothing": 0.02,
    "--stochastic_depth_rate": 0.05,
    "--focal_gamma": 2.0,
    "--abnormal_weight": 1.5,
    "--ema_decay": 0.95,
    "--allow_tie_save": True,
}

runs = []
for seed in seeds:
    runs.append({
        "tag": f"{candidate}_s{seed}",
        "candidate": candidate,
        "seed": seed,
        "args": dict(base_args),
        "reason": "BKM combined: every per-axis BKM value applied together",
    })

payload = {
    "created_by": "scripts/sweeps_server/bkm_combined.sh",
    "note": "All BKM values applied together; compare against rawbase baseline.",
    "runs": runs,
}
p = Path(queue_path)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[bkm] wrote {p} runs={len(runs)}")
PY

run_cmd "$PYTHON" scripts/prepare_server_queue.py \
  --src "$QUEUE" \
  --dst "$ACTIVE" \
  --config "$CONFIG" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR"

cmd=(
  "$PYTHON" scripts/adaptive_experiment_controller.py
  --queue "$ACTIVE"
  --summary "$SUMMARY"
  --markdown "$MARKDOWN"
  --target-min 5
  --target-max 15
  --stop-mode never
  --candidate-min-runs-before-skip 0
  --completion-exit-grace 15
  --update-live-summary
)
[[ "$FORCE" -eq 1 ]] && cmd+=(--force)
[[ "$MAX_LAUNCHED" -gt 0 ]] && cmd+=(--max-launched "$MAX_LAUNCHED")
[[ -n "$LOG_DIR_GROUP" ]] && cmd+=(--log-dir-group "$LOG_DIR_GROUP")

run_cmd "${cmd[@]}"

run_cmd "$PYTHON" scripts/generate_stage_comparison.py \
  --results "$SUMMARY" \
  --out-md validations/05_bkm_combined_results.md \
  --out-plot validations/05_bkm_combined_plot.png \
  --title "BKM combined vs baseline"
