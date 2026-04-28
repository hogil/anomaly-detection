#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
CONFIG="dataset.yaml"
WORKERS=24
NUM_WORKERS=24
PREFETCH_FACTOR=4
CANDIDATE_PREFIX="fresh0412_v11"
MIN_F1="0.99"
SKIP_WEIGHTS=0
SKIP_DATASET=0
SKIP_REFCHECK=0
SKIP_ROUND1=0
SKIP_ROUND2=0
SKIP_POST=0
FORCE=0
MAX_LAUNCHED=0
ROUND1_START_AFTER_AXIS=""
ROUND1_START_AFTER_CANDIDATE=""
ROUND1_SKIP_COMPLETED=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_paper_server_all.sh [options]

Runs the paper experiment pipeline in one command:
  weights check/download -> dataset/image generation if needed -> refcheck ->
  strict one-factor round1 -> select/run round2 -> instability/trend/report.

Options:
  --python PATH           Python executable (default: python or $PYTHON)
  --config PATH           Dataset config used on this server (default: dataset.yaml)
  --workers N             Data/image generation workers (default: 24)
  --num-workers N         train.py DataLoader workers in all queues (default: 24)
  --prefetch-factor N     train.py prefetch factor in all queues (default: 4)
  --candidate-prefix STR  Prefix for prediction trend analysis (default: fresh0412_v11)
  --min-f1 FLOAT          Minimum F1 for strong-run trend analysis (default: 0.99)
  --force                 Re-run completed tags
  --max-launched N        Stop controller after launching N new runs (debug/resume)
  --round1-after-gc       Start strict round1 after the GC block
  --round1-include-gc     Include the 5-condition GC block (default)
  --round1-start-after-candidate STR
                          Start strict round1 after this candidate's last queued seed
  --round1-skip-completed Omit completed/skipped tags from the existing round1 summary (default)
  --round1-keep-completed Keep completed/skipped tags in the prepared round1 queue
  --skip-weights          Do not run download.py when weights are missing
  --skip-dataset          Do not auto-generate data/images
  --skip-refcheck         Skip previous-state reference rerun
  --skip-round1           Skip strict round1 queue
  --skip-round2           Skip round2 selection/queue
  --skip-post             Skip instability/trend/report generation
  -h, --help              Show this help
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
    --round1-skip-completed) ROUND1_SKIP_COMPLETED=1; shift ;;
    --round1-keep-completed) ROUND1_SKIP_COMPLETED=0; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --skip-refcheck) SKIP_REFCHECK=1; shift ;;
    --skip-round1) SKIP_ROUND1=1; shift ;;
    --skip-round2) SKIP_ROUND2=1; shift ;;
    --skip-post) SKIP_POST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p validations logs docs

LOG="validations/paper_server_all.log"
STATE="validations/paper_server_all_state.json"

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

write_state() {
  local status="$1"
  local stage="$2"
  local message="${3:-}"
  "$PYTHON" - "$STATE" "$status" "$stage" "$message" "$CONFIG" "$NUM_WORKERS" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path, status, stage, message, config, num_workers = sys.argv[1:]
payload = {
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "status": status,
    "stage": stage,
    "message": message,
    "config": config,
    "num_workers": int(num_workers),
}
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

config_path() {
  local key="$1"
  "$PYTHON" - "$CONFIG" "$key" <<'PY'
import sys
from pathlib import Path

import yaml

config_path, dotted_key = sys.argv[1:]
payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
value = payload
for part in dotted_key.split("."):
    value = value[part]
print(value)
PY
}

prepare_queue() {
  local src="$1"
  local dst="$2"
  local start_after_axis="${3:-}"
  local start_after_candidate="${4:-}"
  local skip_completed_summary="${5:-}"
  local args=(
    scripts/prepare_server_queue.py
    --src "$src"
    --dst "$dst"
    --config "$CONFIG"
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
  )
  if [[ -n "$start_after_axis" ]]; then
    args+=(--start-after-axis "$start_after_axis")
  fi
  if [[ -n "$start_after_candidate" ]]; then
    args+=(--start-after-candidate "$start_after_candidate")
  fi
  if [[ -n "$skip_completed_summary" ]]; then
    args+=(--skip-completed-summary "$skip_completed_summary")
  fi
  run_cmd "$PYTHON" "${args[@]}"
}

run_controller() {
  local queue="$1"
  local summary="$2"
  local markdown="$3"
  local stage="$4"
  local args=(
    scripts/adaptive_experiment_controller.py
    --queue "$queue"
    --summary "$summary"
    --markdown "$markdown"
    --target-min 5
    --target-max 15
    --stop-mode never
    --candidate-min-runs-before-skip 0
    --completion-exit-grace 15
    --update-live-summary
  )
  if [[ "$FORCE" -eq 1 ]]; then
    args+=(--force)
  fi
  if [[ "$MAX_LAUNCHED" -gt 0 ]]; then
    args+=(--max-launched "$MAX_LAUNCHED")
  fi
  write_state "running" "$stage" "$queue"
  run_cmd "$PYTHON" "${args[@]}"
}

main() {
  exec > >(tee -a "$LOG") 2>&1
  echo "== paper server run started: $(date -Is) =="
  echo "config=$CONFIG workers=$WORKERS num_workers=$NUM_WORKERS prefetch=$PREFETCH_FACTOR"

  DATA_DIR="$(config_path output.data_dir)"
  IMAGE_DIR="$(config_path output.image_dir)"
  DISPLAY_DIR="$(config_path output.display_dir)"
  echo "dataset=$DATA_DIR images=$IMAGE_DIR display=$DISPLAY_DIR"

  write_state "running" "start" "server paper pipeline started"

  if [[ "$SKIP_WEIGHTS" -eq 0 && ! -f weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth ]]; then
    write_state "running" "weights" "downloading pretrained weights"
    run_cmd "$PYTHON" download.py
  fi

  if [[ "$SKIP_DATASET" -eq 0 ]]; then
    if [[ ! -f "$DATA_DIR/scenarios.csv" || ! -f "$DATA_DIR/timeseries.csv" || ! -d "$IMAGE_DIR" ]]; then
      write_state "running" "dataset" "generating dataset and images"
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
      validations/paper_refcheck_raw_queue.json \
      validations/server_paper_refcheck_raw_queue.json
    run_controller \
      validations/server_paper_refcheck_raw_queue.json \
      validations/server_paper_refcheck_raw_summary.json \
      validations/server_paper_refcheck_raw_summary.md \
      "refcheck_raw"
  fi

  if [[ "$SKIP_ROUND1" -eq 0 ]]; then
    round1_skip_completed_summary=""
    if [[ "$ROUND1_SKIP_COMPLETED" -eq 1 && "$FORCE" -eq 0 ]]; then
      round1_skip_completed_summary="validations/server_paper_rawbase_strict_single_factor_summary.json"
    fi
    prepare_queue \
      validations/paper_strict_single_factor_queue.json \
      validations/server_paper_rawbase_strict_single_factor_queue.json \
      "$ROUND1_START_AFTER_AXIS" \
      "$ROUND1_START_AFTER_CANDIDATE" \
      "$round1_skip_completed_summary"
    run_controller \
      validations/server_paper_rawbase_strict_single_factor_queue.json \
      validations/server_paper_rawbase_strict_single_factor_summary.json \
      validations/server_paper_rawbase_strict_single_factor_summary.md \
      "strict_round1"
  fi

  if [[ "$SKIP_ROUND2" -eq 0 ]]; then
    write_state "running" "select_round2" "selecting round2 from server round1 summary"
    run_cmd "$PYTHON" scripts/select_strict_single_factor_refinements.py \
      --summary validations/server_paper_rawbase_strict_single_factor_summary.json \
      --out-queue validations/server_paper_rawbase_strict_single_factor_round2_queue.json \
      --decision-md validations/server_paper_rawbase_strict_single_factor_round2_decision.md

    if [[ -f validations/server_paper_rawbase_strict_single_factor_round2_queue.json ]]; then
      ROUND2_COUNT="$("$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("validations/server_paper_rawbase_strict_single_factor_round2_queue.json")
print(len(json.loads(p.read_text(encoding="utf-8")).get("runs", [])))
PY
)"
      if [[ "$ROUND2_COUNT" -gt 0 ]]; then
        prepare_queue \
          validations/server_paper_rawbase_strict_single_factor_round2_queue.json \
          validations/server_paper_rawbase_strict_single_factor_round2_queue.prepared.json
        run_controller \
          validations/server_paper_rawbase_strict_single_factor_round2_queue.prepared.json \
          validations/server_paper_rawbase_strict_single_factor_round2_summary.json \
          validations/server_paper_rawbase_strict_single_factor_round2_summary.md \
          "strict_round2"
      else
        echo "[skip] round2 queue is empty"
      fi
    fi
  fi

  if [[ "$SKIP_POST" -eq 0 ]]; then
    write_state "running" "postprocess" "collecting instability, trends, and report"
    run_cmd "$PYTHON" scripts/collect_instability_cases.py
    run_cmd "$PYTHON" scripts/analyze_prediction_trends.py \
      --config "$CONFIG" \
      --candidate-prefix "$CANDIDATE_PREFIX" \
      --min-f1 "$MIN_F1" \
      --out-prefix validations/prediction_trend_latest \
      --report-label "$CANDIDATE_PREFIX"
    run_cmd "$PYTHON" scripts/generate_strict_one_factor_report.py \
      --strict-summary validations/server_paper_rawbase_strict_single_factor_summary.json \
      --round2-summary validations/server_paper_rawbase_strict_single_factor_round2_summary.json \
      --round2-queue validations/server_paper_rawbase_strict_single_factor_round2_queue.prepared.json \
      --state validations/paper_server_all_state.json \
      --markdown-out validations/server_paper_rawbase_strict_single_factor_summary.md \
      --report-out validations/server_paper_rawbase_strict_single_factor_report.md \
      --plots-dir validations/server_paper_rawbase_strict_single_factor_plots
    run_cmd "$PYTHON" scripts/publish_strict_report.py \
      --source-report validations/server_paper_rawbase_strict_single_factor_report.md \
      --source-round2 validations/server_paper_rawbase_strict_single_factor_round2_summary.md \
      --source-plots validations/server_paper_rawbase_strict_single_factor_plots
  fi

  write_state "complete" "done" "server paper pipeline completed"
  echo "== paper server run completed: $(date -Is) =="
}

main
