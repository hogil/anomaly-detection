#!/usr/bin/env bash
# Backbone sweep. Rotate through the preferred `download.py::MODELS` backbone
# order, plus any extra non-deprecated `weights/<name>.pth` files, and train
# each as a single-axis change while keeping the rest of the baseline fixed.
#
# Each candidate keeps the operating baseline (--lr_backbone 2e-5, etc.).
# Note: per-backbone LR re-tuning (memory: "LR Spike per Backbone") is a
# follow-up; this stage just establishes the ranking under the same baseline.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dataset.yaml}"
detect_profile
NUM_WORKERS="${NUM_WORKERS:-$PROFILE_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PROFILE_PREFETCH}"
BATCH_SIZE="${BATCH_SIZE:-$PROFILE_BATCH_SIZE}"
MAX_LAUNCHED="${MAX_LAUNCHED:-$PROFILE_MAX_LAUNCHED}"
LOG_DIR_GROUP="${LOG_DIR_GROUP:-$(date +%Y%m%d_%H%M%S)_backbone}"
CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION:-all}"
CHECKPOINT_RETENTION_SCOPE="${CHECKPOINT_RETENTION_SCOPE:-summary}"
SEEDS="${SEEDS:-42,1,2,3,4}"
FORCE=0
PREPARE_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/backbone.sh [options]

Rotates through the preferred download.py::MODELS backbone order, plus any
extra non-deprecated .pth file in weights/, and runs one candidate per backbone
with the operating baseline. Default seeds: 42,1,2,3,4.

Options:
  --python PATH        python executable
  --config PATH        dataset config (default dataset.yaml)
  --num-workers N      train.py DataLoader workers
  --prefetch-factor N  train.py prefetch factor
  --batch-size N       train.py --batch_size override (auto from profile)
  --seeds CSV          seeds (default: 42,1,2,3,4)
  --force              re-run completed tags
  --max-launched N     stop controller after launching N runs
  --log-dir-group NAME group runs under logs/<NAME>/
  --checkpoint-retention MODE      all | dataset-backbone-best (default: dataset-backbone-best)
  --checkpoint-retention-scope S   summary | log-group | logs (default: logs)
  --prepare-only       write 04_backbone_queue/active only; do not train
  -h, --help           show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --max-launched) MAX_LAUNCHED="$2"; shift 2 ;;
    --log-dir-group) LOG_DIR_GROUP="$2"; shift 2 ;;
    --checkpoint-retention) CHECKPOINT_RETENTION="$2"; shift 2 ;;
    --checkpoint-retention-scope) CHECKPOINT_RETENTION_SCOPE="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

run_cmd() { echo; echo "+ $*"; "$@"; }

VAL_DIR="validations/${LOG_DIR_GROUP}"
mkdir -p "$VAL_DIR"
QUEUE="${VAL_DIR}/04_backbone_queue.json"
ACTIVE="${VAL_DIR}/04_backbone_active.json"
SUMMARY="${VAL_DIR}/04_backbone_results.json"
MARKDOWN="${VAL_DIR}/04_backbone_results.md"

echo "== paper stage: backbone (output: $VAL_DIR) =="

run_cmd "$PYTHON" - "$QUEUE" "$SEEDS" "$NUM_WORKERS" "$PREFETCH_FACTOR" "$BATCH_SIZE" <<'PY'
import json
import re
import sys
from pathlib import Path

queue_path, seeds_raw, num_workers, prefetch, batch_size = sys.argv[1:6]
seeds = [int(s.strip()) for s in seeds_raw.split(",") if s.strip()]
weights_dir = Path("weights")
# Keep this order in sync with download.py::MODELS.
preferred_order = [
    "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnext_tiny.dinov3_lvd1689m",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swinv2_cr_tiny_ns_224.sw_in1k",
    "maxvit_tiny_tf_224.in1k",
]
available = []
for path in weights_dir.glob("*.pth"):
    name = path.stem
    if name == "best_model" or name.endswith(".fp16") or name.startswith("vit_") or name.startswith("swin_"):
        continue
    available.append(name)
available_set = set(available)
candidates = [name for name in preferred_order if name in available_set]
candidates.extend(sorted(name for name in available if name not in set(preferred_order)))

if not candidates:
    raise SystemExit("[backbone] no usable .pth in weights/")

base_args = {
    "--mode": "binary",
    "--config": "dataset.yaml",
    "--epochs": 20,
    "--patience": 5,
    "--batch_size": int(batch_size),
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
    "--label_smoothing": 0.0,
    "--prefetch_factor": int(prefetch),
}

def short(name: str) -> str:
    short = re.sub(r"\..+$", "", name)              # drop ".fcmae_ft_in22k_in1k"
    short = short.replace("_", "")                  # convnextv2tiny
    return short

runs = []
for backbone in candidates:
    cand = f"fresh0412_v11_bb_{short(backbone)}_n700"
    for seed in seeds:
        run_args = dict(base_args)
        run_args["--model_name"] = backbone
        runs.append({
            "tag": f"{cand}_s{seed}",
            "candidate": cand,
            "seed": seed,
            "args": run_args,
            "reason": f"backbone sweep: --model_name={backbone} under fixed baseline",
        })

payload = {
    "created_by": "scripts/sweeps_server/backbone.sh",
    "note": "Backbone sweep auto-generated from weights/*.pth; one candidate per backbone, baseline otherwise.",
    "runs": runs,
}

p = Path(queue_path)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[backbone] wrote {p} backbones={len(candidates)} runs={len(runs)}")
PY

# Same prep step as other axes (rawbase tag, worker injection, skip-completed).
run_cmd "$PYTHON" scripts/prepare_server_queue.py \
  --src "$QUEUE" \
  --dst "$ACTIVE" \
  --config "$CONFIG" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --batch-size "$BATCH_SIZE"

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
  echo "[backbone] prepare-only complete: $QUEUE, $ACTIVE"
  exit 0
fi

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
  --stage-comparison-baseline "${VAL_DIR}/01_baseline_results.json"
  --stage-comparison-md "$MARKDOWN"
  --stage-comparison-plot "${VAL_DIR}/04_backbone_plot.png"
  --stage-comparison-title "Backbone sweep"
  --checkpoint-retention "$CHECKPOINT_RETENTION"
  --checkpoint-retention-scope "$CHECKPOINT_RETENTION_SCOPE"
)
[[ "$FORCE" -eq 1 ]] && cmd+=(--force)
[[ "$MAX_LAUNCHED" -gt 0 ]] && cmd+=(--max-launched "$MAX_LAUNCHED")
[[ -n "$LOG_DIR_GROUP" ]] && cmd+=(--log-dir-group "$LOG_DIR_GROUP")

run_cmd "${cmd[@]}"

# Build comparison .md + bar plot vs baseline.
run_cmd "$PYTHON" scripts/generate_stage_comparison.py \
  --results "$SUMMARY" \
  --baseline "${VAL_DIR}/01_baseline_results.json" \
  --out-md "${VAL_DIR}/04_backbone_results.md" \
  --out-plot "${VAL_DIR}/04_backbone_plot.png" \
  --title "Backbone sweep"
