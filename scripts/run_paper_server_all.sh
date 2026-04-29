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
LOG_DIR_GROUP="${LOG_DIR_GROUP:-$(date +%Y%m%d_%H%M%S)_run_paper}"

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
  --max-launched N       stop controller after launching N runs (0 = unlimited)
  --candidate-prefix STR prefix for prediction trend analysis
  --min-f1 FLOAT         min F1 for strong-run trend analysis
  --force                re-run completed tags
  --round1-include-axes CSV
  --round1-skip-completed / --round1-keep-completed
  --round1-start-after-candidate STR
  --skip-weights / --skip-dataset / --skip-refcheck / --skip-round1 / --skip-post
  --log-dir-group NAME   group all train.py runs under logs/<NAME>/ (default: run_<timestamp>)
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
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p validations logs docs
LOG="validations/run.log"

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
  )
  [[ -n "$start_after_axis" ]] && args+=(--start-after-axis "$start_after_axis")
  [[ -n "$start_after_candidate" ]] && args+=(--start-after-candidate "$start_after_candidate")
  [[ -n "$skip_completed_summary" ]] && args+=(--skip-completed-summary "$skip_completed_summary")
  [[ -n "$include_axes" ]] && args+=(--include-axes "$include_axes")
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

main() {
  exec > >(tee -a "$LOG") 2>&1
  echo "== run started: $(date -Is) =="
  echo "config=$CONFIG profile=$PROFILE_NAME workers=$WORKERS num_workers=$NUM_WORKERS prefetch=$PREFETCH_FACTOR max_launched=$MAX_LAUNCHED log_dir_group=$LOG_DIR_GROUP"
  echo "round1_axes=$ROUND1_INCLUDE_AXES"

  DATA_DIR="$(config_path output.data_dir)"
  IMAGE_DIR="$(config_path output.image_dir)"
  DISPLAY_DIR="$(config_path output.display_dir)"
  echo "dataset=$DATA_DIR images=$IMAGE_DIR display=$DISPLAY_DIR"

  if [[ "$SKIP_WEIGHTS" -eq 0 && ! -f weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth ]]; then
    run_cmd "$PYTHON" download.py
  fi

  if [[ "$SKIP_DATASET" -eq 0 ]]; then
    if [[ ! -f "$DATA_DIR/scenarios.csv" || ! -f "$DATA_DIR/timeseries.csv" || ! -d "$IMAGE_DIR" ]]; then
      run_cmd "$PYTHON" generate_data.py --config "$CONFIG" --workers "$WORKERS"
      run_cmd "$PYTHON" generate_images.py --config "$CONFIG" --workers "$WORKERS"
      run_cmd "$PYTHON" scripts/validate_dataset.py \
        --config "$CONFIG" \
        --scenarios "$DATA_DIR/scenarios.csv" \
        --timeseries "$DATA_DIR/timeseries.csv" \
        --display-dir "$DISPLAY_DIR"
    else
      echo "[skip] dataset/images already exist"
    fi
  fi

  if [[ "$SKIP_REFCHECK" -eq 0 ]]; then
    prepare_queue \
      validations/01_baseline_queue.json \
      validations/01_baseline_active.json
    run_controller \
      validations/01_baseline_active.json \
      validations/01_baseline_results.json \
      validations/01_baseline_results.md \
      "baseline_recheck"
  fi

  if [[ "$SKIP_ROUND1" -eq 0 ]]; then
    skip_summary=""
    if [[ "$ROUND1_SKIP_COMPLETED" -eq 1 && "$FORCE" -eq 0 ]]; then
      skip_summary="validations/02_sweep_results.json"
    fi
    prepare_queue \
      validations/02_sweep_queue.json \
      validations/02_sweep_active.json \
      "$ROUND1_START_AFTER_AXIS" \
      "$ROUND1_START_AFTER_CANDIDATE" \
      "$skip_summary" \
      "$ROUND1_INCLUDE_AXES"
    ROUND1_COUNT="$(queue_run_count validations/02_sweep_active.json)"
    if [[ "$ROUND1_COUNT" -gt 0 ]]; then
      run_controller \
        validations/02_sweep_active.json \
        validations/02_sweep_results.json \
        validations/02_sweep_results.md \
        "axis_sweep"
    else
      echo "[skip] axis_sweep queue is empty"
    fi
  fi

  if [[ "$SKIP_POST" -eq 0 ]]; then
    run_cmd "$PYTHON" scripts/collect_instability_cases.py
    run_cmd "$PYTHON" scripts/analyze_prediction_trends.py \
      --config "$CONFIG" \
      --candidate-prefix "$CANDIDATE_PREFIX" \
      --min-f1 "$MIN_F1" \
      --out-prefix validations/prediction_trend_latest \
      --report-label "$CANDIDATE_PREFIX"
    run_cmd "$PYTHON" scripts/generate_strict_one_factor_report.py \
      --strict-summary validations/02_sweep_results.json \
      --markdown-out validations/02_sweep_results.md \
      --report-out validations/02_sweep_report.md \
      --plots-dir validations/02_sweep_plots
    run_cmd "$PYTHON" scripts/generate_strict_one_factor_report.py \
      --strict-summary validations/02_sweep_results.json \
      --markdown-out docs/summary.md \
      --report-out docs/summary.md \
      --plots-dir docs/plots
  fi

  echo "== run completed: $(date -Is) =="
}

main
