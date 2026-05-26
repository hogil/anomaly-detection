#!/usr/bin/env bash
# Paper experiment runner. Auto-detects the runtime profile (server vs PC vs
# minimal) from GPU memory + CPU count, then runs:
#   weights -> dataset/images -> baseline recheck -> single-axis sweep -> post.
#
# Inputs (validations/):
#   01_baseline_queue.json       baseline 5-seed recheck input
#   02_sweep_queue.json          per-axis sweep input (template)
#   03_sample_skip_queue.json    one-off "filter nonfinite-loss samples" probe
# Outputs (validations/):
#   02_sweep_active.json         server-prep of 02_sweep_queue.json
#   02_sweep_results.json/.md    live results updated each run
#   run.log                      combined stdout
set -euo pipefail

# Force Python utf-8 output so Korean prints render on Windows cp949 consoles.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/sweeps_server/_common.sh"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dataset.yaml}"
detect_profile
WORKERS="${WORKERS:-$PROFILE_NUM_WORKERS}"
NUM_WORKERS="${NUM_WORKERS:-$PROFILE_NUM_WORKERS}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-$PROFILE_PREFETCH}"
MAX_LAUNCHED="${MAX_LAUNCHED:-$PROFILE_MAX_LAUNCHED}"
BATCH_SIZE="${BATCH_SIZE:-$PROFILE_BATCH_SIZE}"
CANDIDATE_PREFIX="fresh0412_v11"
MIN_F1="0.99"
SKIP_WEIGHTS=0
SKIP_DATASET=0
SKIP_REFCHECK=0
SKIP_ROUND1=0
SKIP_ROUND2=1   # round2 deprecated; kept as inert flag
SKIP_POST=0
FORCE=0
ROUND1_START_AFTER_AXIS=""
ROUND1_START_AFTER_CANDIDATE=""
ROUND1_SKIP_COMPLETED=1
DEFAULT_ROUND1_AXES="normal_ratio,per_class,lr,warmup,label_smoothing,stochastic_depth,focal_gamma,abnormal_weight,ema,allow_tie_save,color,gc"
ROUND1_INCLUDE_AXES="$DEFAULT_ROUND1_AXES"
LOG_DIR_GROUP="${LOG_DIR_GROUP:-}"
CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION:-all}"
CHECKPOINT_RETENTION_SCOPE="${CHECKPOINT_RETENTION_SCOPE:-summary}"
MODEL_NAME="${MODEL_NAME:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_paper_server_all.sh [options]

Stages: weights -> dataset/images -> baseline recheck -> single-axis sweep -> post.
Auto-detects runtime profile (server/pc/minimal) from GPU memory + CPU count.

Options:
  --python PATH          python executable
  --config PATH          dataset config (default dataset.yaml)
  --workers N            data/image generation workers
  --num-workers N        train.py DataLoader workers
  --prefetch-factor N    train.py prefetch factor
  --batch-size N         train.py --batch_size override (auto: server=256, pc=32, minimal=8)
  --max-launched N       stop controller after launching N runs (0 = unlimited)
  --candidate-prefix STR prefix for prediction trend analysis
  --min-f1 FLOAT         min F1 for strong-run trend analysis
  --force                re-run completed tags
  --round1-include-axes CSV
  --round1-skip-completed / --round1-keep-completed
  --round1-start-after-candidate STR
  --skip-weights / --skip-dataset / --skip-refcheck / --skip-round1 / --skip-post
  --log-dir-group NAME   group all train.py runs under logs/<NAME>/ (default: run_<timestamp>)
  --checkpoint-retention MODE      all | dataset-backbone-best (default: all — never delete)
  --checkpoint-retention-scope S   summary | log-group | logs (default: summary — minimal scan)
  -h, --help             show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --candidate-prefix) CANDIDATE_PREFIX="$2"; shift 2 ;;
    --min-f1) MIN_F1="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --max-launched) MAX_LAUNCHED="$2"; shift 2 ;;
    --round1-after-gc) ROUND1_START_AFTER_AXIS="gc"; shift ;;
    --round1-include-gc) ROUND1_START_AFTER_AXIS=""; shift ;;
    --round1-start-after-candidate) ROUND1_START_AFTER_CANDIDATE="$2"; shift 2 ;;
    --round1-include-axes) ROUND1_INCLUDE_AXES="$2"; shift 2 ;;
    --round1-skip-completed) ROUND1_SKIP_COMPLETED=1; shift ;;
    --round1-keep-completed) ROUND1_SKIP_COMPLETED=0; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --skip-refcheck) SKIP_REFCHECK=1; shift ;;
    --skip-round1) SKIP_ROUND1=1; shift ;;
    --skip-round2) SKIP_ROUND2=1; shift ;;
    --skip-post) SKIP_POST=1; shift ;;
    --log-dir-group) LOG_DIR_GROUP="$2"; shift 2 ;;
    --checkpoint-retention) CHECKPOINT_RETENTION="$2"; shift 2 ;;
    --checkpoint-retention-scope) CHECKPOINT_RETENTION_SCOPE="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# Default group includes the config basename so parallel runs with different
# yaml files (e.g. --config dataset.yaml vs --config dataset1.yaml) land in
# distinct validations/<group>/ folders even if launched in the same second.
if [[ -z "$LOG_DIR_GROUP" ]]; then
  config_stem="$(basename "$CONFIG")"
  config_stem="${config_stem%.yaml}"
  config_stem="${config_stem%.yml}"
  LOG_DIR_GROUP="$(date +%Y%m%d_%H%M%S)_run_paper_${config_stem}"
fi
VAL_DIR="validations/${LOG_DIR_GROUP}"
mkdir -p "$VAL_DIR" logs docs
LOG="${VAL_DIR}/run.log"

run_cmd() { echo; echo "+ $*"; "$@"; }

config_path() {
  local key="$1"
  "$PYTHON" - "$CONFIG" "$key" <<'PY'
import sys
from pathlib import Path
import yaml
cfg, dotted = sys.argv[1:]
v = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
for part in dotted.split("."):
    v = v[part]
print(v)
PY
}

prepare_queue() {
  local src="$1" dst="$2"
  local start_after_axis="${3:-}" start_after_candidate="${4:-}"
  local skip_completed_summary="${5:-}" include_axes="${6:-}"
  local args=(
    scripts/prepare_server_queue.py
    --src "$src" --dst "$dst"
    --config "$CONFIG"
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --batch-size "$BATCH_SIZE"
  )
  [[ -n "$start_after_axis" ]] && args+=(--start-after-axis "$start_after_axis")
  [[ -n "$start_after_candidate" ]] && args+=(--start-after-candidate "$start_after_candidate")
  [[ -n "$skip_completed_summary" ]] && args+=(--skip-completed-summary "$skip_completed_summary")
  [[ -n "$include_axes" ]] && args+=(--include-axes "$include_axes")
  [[ -n "$MODEL_NAME" ]] && args+=(--model-name "$MODEL_NAME")
  run_cmd "$PYTHON" "${args[@]}"
}

run_controller() {
  local queue="$1" summary="$2" markdown="$3" stage="$4"
  local args=(
    scripts/adaptive_experiment_controller.py
    --queue "$queue" --summary "$summary" --markdown "$markdown"
    --target-min 5 --target-max 15 --stop-mode never
    --candidate-min-runs-before-skip 0 --completion-exit-grace 15
    --update-live-summary
    --checkpoint-retention "$CHECKPOINT_RETENTION"
    --checkpoint-retention-scope "$CHECKPOINT_RETENTION_SCOPE"
  )
  [[ "$FORCE" -eq 1 ]] && args+=(--force)
  [[ "$MAX_LAUNCHED" -gt 0 ]] && args+=(--max-launched "$MAX_LAUNCHED")
  [[ -n "$LOG_DIR_GROUP" ]] && args+=(--log-dir-group "$LOG_DIR_GROUP")
  echo "[stage] $stage"
  run_cmd "$PYTHON" "${args[@]}"
}

queue_run_count() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(0); raise SystemExit
d = json.loads(p.read_text(encoding="utf-8"))
print(len(d) if isinstance(d, list) else len(d.get("runs", [])))
PY
}

dataset_ready() {
  "$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import pandas as pd
import yaml
from src.data.schema import highlighted_member as read_highlighted_member

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
out = cfg["output"]
data_dir = Path(out["data_dir"])
image_dir = Path(out["image_dir"])
scenarios = data_dir / "scenarios.csv"
timeseries = data_dir / "timeseries.csv"

issues = []

if not scenarios.exists():
    issues.append(f"missing {scenarios}")
else:
    try:
        sc_cols = pd.read_csv(scenarios, nrows=1).columns.tolist()
    except Exception as exc:
        sc_cols = []
        issues.append(f"cannot read {scenarios}: {exc}")
    missing = [c for c in ["chart_id", "class", "split"] if c not in sc_cols]
    if missing:
        issues.append(f"{scenarios} missing columns: {','.join(missing)}; columns={sc_cols}")

if not timeseries.exists():
    issues.append(f"missing {timeseries}")
else:
    try:
        ts_cols = pd.read_csv(timeseries, nrows=1).columns.tolist()
    except Exception as exc:
        ts_cols = []
        issues.append(f"cannot read {timeseries}: {exc}")
    missing = [c for c in ["chart_id", "value"] if c not in ts_cols]
    if missing:
        issues.append(f"{timeseries} missing columns: {','.join(missing)}; columns={ts_cols}")

if not image_dir.exists():
    issues.append(f"missing image dir {image_dir}")
else:
    try:
        sc_df = pd.read_csv(scenarios) if scenarios.exists() else pd.DataFrame()
    except Exception as exc:
        sc_df = pd.DataFrame()
        issues.append(f"cannot read {scenarios} for image completeness check: {exc}")
    missing_images = []
    for row in sc_df.to_dict(orient="records"):
        split = str(row.get("split", ""))
        cls = str(row.get("class", ""))
        chart_id = str(row.get("chart_id", ""))
        candidates = []
        image_name = row.get("image_name")
        if image_name is not None and str(image_name).strip() and str(image_name) != "nan":
            candidates.append(str(image_name))
        highlighted = read_highlighted_member(row)
        if highlighted:
            candidates.append(f"{chart_id}_{highlighted}.png")
        candidates.append(f"{chart_id}.png")
        if not any((image_dir / split / cls / name).exists() for name in candidates):
            missing_images.append(f"{split}/{cls}/{chart_id}.png")
    if missing_images:
        preview = ", ".join(missing_images[:10])
        suffix = "" if len(missing_images) <= 10 else f", ... {len(missing_images) - 10} more"
        issues.append(f"missing rendered images for {len(missing_images)} scenarios: {preview}{suffix}")

if issues:
    for issue in issues:
        print(f"[dataset-check] {issue}")
    raise SystemExit(1)

print(f"[dataset-check] ready: {data_dir} / {image_dir}")
PY
}

main() {
  exec > >(tee -a "$LOG") 2>&1
  echo "== run started: $(date -Is) =="
  echo "config=$CONFIG profile=$PROFILE_NAME workers=$WORKERS num_workers=$NUM_WORKERS prefetch=$PREFETCH_FACTOR batch_size=$BATCH_SIZE max_launched=$MAX_LAUNCHED log_dir_group=$LOG_DIR_GROUP"
  echo "round1_axes=$ROUND1_INCLUDE_AXES"

  DATA_DIR="$(config_path output.data_dir)"
  IMAGE_DIR="$(config_path output.image_dir)"
  DISPLAY_DIR="$(config_path output.display_dir)"
  echo "dataset=$DATA_DIR images=$IMAGE_DIR display=$DISPLAY_DIR"

  if [[ "$SKIP_WEIGHTS" -eq 0 ]]; then
    run_cmd "$PYTHON" download.py
  fi

  if [[ "$SKIP_DATASET" -eq 0 ]]; then
    if dataset_ready; then
      echo "[skip] dataset/images already exist and schema is valid"
    else
      echo "[regen] dataset/images missing or invalid; regenerating"
      run_cmd "$PYTHON" generate_data.py --config "$CONFIG" --workers "$WORKERS"
      run_cmd "$PYTHON" generate_images.py --config "$CONFIG" --workers "$WORKERS"
      dataset_ready
      run_cmd "$PYTHON" scripts/validate_dataset.py \
        --config "$CONFIG" \
        --scenarios "$DATA_DIR/scenarios.csv" \
        --timeseries "$DATA_DIR/timeseries.csv" \
        --display-dir "$DISPLAY_DIR"
    fi
  fi

  if [[ "$SKIP_REFCHECK" -eq 0 ]]; then
    prepare_queue \
      validations/01_baseline_queue.json \
      "${VAL_DIR}/01_baseline_active.json"
    run_controller \
      "${VAL_DIR}/01_baseline_active.json" \
      "${VAL_DIR}/01_baseline_results.json" \
      "${VAL_DIR}/01_baseline_results.md" \
      "baseline_recheck"
  fi

  if [[ "$SKIP_ROUND1" -eq 0 ]]; then
    skip_summary=""
    if [[ "$ROUND1_SKIP_COMPLETED" -eq 1 && "$FORCE" -eq 0 ]]; then
      skip_summary="${VAL_DIR}/02_sweep_results.json"
    fi
    prepare_queue \
      validations/02_sweep_queue.json \
      "${VAL_DIR}/02_sweep_active.json" \
      "$ROUND1_START_AFTER_AXIS" \
      "$ROUND1_START_AFTER_CANDIDATE" \
      "$skip_summary" \
      "$ROUND1_INCLUDE_AXES"
    ROUND1_COUNT="$(queue_run_count "${VAL_DIR}/02_sweep_active.json")"
    if [[ "$ROUND1_COUNT" -gt 0 ]]; then
      run_controller \
        "${VAL_DIR}/02_sweep_active.json" \
        "${VAL_DIR}/02_sweep_results.json" \
        "${VAL_DIR}/02_sweep_results.md" \
        "axis_sweep"
    else
      echo "[skip] axis_sweep queue is empty"
    fi
  fi

  if [[ "$SKIP_POST" -eq 0 ]]; then
    run_cmd "$PYTHON" scripts/collect_instability_cases.py \
      --out-md "${VAL_DIR}/instability_cases_report.md" \
      --out-csv "${VAL_DIR}/instability_cases.csv" \
      --out-json "${VAL_DIR}/instability_cases.json"
    run_cmd "$PYTHON" scripts/analyze_prediction_trends.py \
      --config "$CONFIG" \
      --candidate-prefix "$CANDIDATE_PREFIX" \
      --min-f1 "$MIN_F1" \
      --out-prefix "${VAL_DIR}/prediction_trend_latest" \
      --report-label "$CANDIDATE_PREFIX"
    run_cmd "$PYTHON" scripts/generate_strict_one_factor_report.py \
      --strict-summary "${VAL_DIR}/02_sweep_results.json" \
      --markdown-out "${VAL_DIR}/02_sweep_results.md" \
      --report-out "${VAL_DIR}/02_sweep_report.md" \
      --plots-dir "${VAL_DIR}/02_sweep_plots"
  fi

  echo "== run completed: $(date -Is) =="
}

main
