#!/usr/bin/env bash
# Build per-member logical data/images and launch the logical baseline train.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dataset.yaml}"
SUFFIX="${SUFFIX:-logical_member_v11}"
detect_profile
WORKERS="${WORKERS:-$PROFILE_NUM_WORKERS}"
NUM_WORKERS="${NUM_WORKERS:-$PROFILE_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PROFILE_PREFETCH}"
LOGICAL_SEEDS="${LOGICAL_SEEDS:-42}"
LOG_DIR_GROUP="${LOG_DIR_GROUP:-run_$(date +%Y%m%d_%H%M%S)}"
FORCE=0
MAX_LAUNCHED="${MAX_LAUNCHED:-$PROFILE_MAX_LAUNCHED}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/50_logical_train.sh [options]

Options:
  --python PATH        Python executable (default: python or $PYTHON)
  --config PATH        Base dataset config (default: dataset.yaml)
  --suffix NAME        Dataset suffix (default: logical_member_v11)
  --workers N          Data/image generation workers (default: 24)
  --num-workers N      train.py DataLoader workers (default: 24)
  --prefetch-factor N  train.py prefetch factor (default: 4)
  --seeds CSV          Logical train seeds (default: 42)
  --force              Re-run completed tag and regenerate logical artifacts
  --max-launched N     Stop controller after launching N new runs
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --seeds) LOGICAL_SEEDS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --max-launched) MAX_LAUNCHED="$2"; shift 2 ;;
    --log-dir-group) LOG_DIR_GROUP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

SOURCE_CONFIG="configs/datasets/${SUFFIX}_source.yaml"
TRAIN_CONFIG="configs/datasets/${SUFFIX}_train.yaml"
SOURCE_DATA_DIR="data_${SUFFIX}"
PM_DATA_DIR="data_per_member_${SUFFIX}"
PM_IMAGE_DIR="images_per_member_${SUFFIX}"
PM_DISPLAY_DIR="display_per_member_${SUFFIX}"
PM_SCENARIOS="${PM_DATA_DIR}/scenarios_per_member.csv"
QUEUE="validations/server_paper_${SUFFIX}_train_queue.json"
SUMMARY="validations/server_paper_${SUFFIX}_train_summary.json"
MARKDOWN="validations/server_paper_${SUFFIX}_train_summary.md"

echo "== paper stage: logical_train =="

run_cmd "$PYTHON" - "$CONFIG" "$SOURCE_CONFIG" "$TRAIN_CONFIG" "$SUFFIX" <<'PY'
import sys
from pathlib import Path

import yaml

base_path, source_path, train_path, suffix = sys.argv[1:5]
cfg = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))

source = dict(cfg)
source["dataset"] = dict(cfg.get("dataset", {}))
source["dataset"]["all_legend_axes_per_group"] = True
source["output"] = dict(cfg.get("output", {}))
source["output"]["data_dir"] = f"data_{suffix}"
source["output"]["image_dir"] = f"images_{suffix}"
source["output"]["display_dir"] = f"display_{suffix}"

train = dict(source)
train["output"] = dict(source["output"])
train["output"]["data_dir"] = f"data_per_member_{suffix}"
train["output"]["image_dir"] = f"images_per_member_{suffix}"
train["output"]["display_dir"] = f"display_per_member_{suffix}"

for path, payload in ((Path(source_path), source), (Path(train_path), train)):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[logical] wrote {path}")
PY

if [[ "$FORCE" -eq 1 || ! -f "$SOURCE_DATA_DIR/scenarios.csv" || ! -f "$SOURCE_DATA_DIR/timeseries.csv" ]]; then
  run_cmd "$PYTHON" generate_data.py \
    --config "$SOURCE_CONFIG" \
    --workers "$WORKERS" \
    --all_legend_axes_per_group \
    --no_snapshot
else
  echo "[skip] logical source data already exists: $SOURCE_DATA_DIR"
fi

if [[ "$FORCE" -eq 1 || ! -f "$PM_SCENARIOS" || ! -d "$PM_IMAGE_DIR" ]]; then
  run_cmd "$PYTHON" scripts/generate_per_member_images.py \
    --config "$SOURCE_CONFIG" \
    --suffix "$SUFFIX" \
    --workers "$WORKERS"
else
  echo "[skip] logical per-member images already exist: $PM_IMAGE_DIR"
fi

run_cmd "$PYTHON" - "$QUEUE" "$TRAIN_CONFIG" "$PM_SCENARIOS" "$LOGICAL_SEEDS" "$NUM_WORKERS" "$PREFETCH_FACTOR" <<'PY'
import json
import sys
from pathlib import Path

queue_path, train_config, scenarios_csv, seeds_raw, num_workers, prefetch = sys.argv[1:7]
seeds = [int(item.strip()) for item in seeds_raw.split(",") if item.strip()]
if not seeds:
    raise SystemExit("no logical train seeds provided")

base_args = {
    "--mode": "binary",
    "--config": train_config,
    "--scenarios_csv": scenarios_csv,
    "--epochs": 20,
    "--patience": 5,
    "--batch_size": 32,
    "--dropout": 0.0,
    "--precision": "fp16",
    "--num_workers": int(num_workers),
    "--ema_decay": 0.0,
    "--normal_ratio": 700,
    "--smooth_window": 1,
    "--smooth_method": "median",
    "--lr_backbone": "2e-5",
    "--lr_head": "2e-4",
    "--warmup_epochs": 5,
    "--grad_clip": 0.0,
    "--weight_decay": 0.01,
    "--prefetch_factor": int(prefetch),
}
runs = []
for seed in seeds:
    runs.append({
        "tag": f"fresh0412_v11_logical_member_baseline_s{seed}",
        "candidate": "fresh0412_v11_logical_member_baseline",
        "seed": seed,
        "args": dict(base_args),
        "reason": "logical per-member baseline: highlighted anomaly member is abnormal, other highlighted members are normal",
    })

payload = {
    "created_by": "scripts/sweeps_server/50_logical_train.sh",
    "note": "Per-member logical attribution train, kept separate from strict one-factor sweep.",
    "runs": runs,
}
path = Path(queue_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[logical] wrote {path} runs={len(runs)}")
PY

cmd=(
  "$PYTHON" scripts/adaptive_experiment_controller.py
  --queue "$QUEUE"
  --summary "$SUMMARY"
  --markdown "$MARKDOWN"
  --target-min 5
  --target-max 15
  --stop-mode never
  --candidate-min-runs-before-skip 0
  --completion-exit-grace 15
  --update-live-summary
)
if [[ "$FORCE" -eq 1 ]]; then
  cmd+=(--force)
fi
if [[ "$MAX_LAUNCHED" -gt 0 ]]; then
  cmd+=(--max-launched "$MAX_LAUNCHED")
fi
if [[ -n "$LOG_DIR_GROUP" ]]; then
  cmd+=(--log-dir-group "$LOG_DIR_GROUP")
fi

run_cmd "${cmd[@]}"
