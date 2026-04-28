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
  --round1-start-after-candidate STR
                          Start strict round1 after this candidate's last queued seed
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
    --round1-start-after-candidate) ROUND1_START_AFTER_CANDIDATE="$2"; shift 2 ;;
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

prepare_queue() {
  local src="$1"
  local dst="$2"
  local start_after_axis="${3:-}"
  local start_after_candidate="${4:-}"
  "$PYTHON" - "$src" "$dst" "$CONFIG" "$NUM_WORKERS" "$PREFETCH_FACTOR" "$start_after_axis" "$start_after_candidate" <<'PY'
import json
import re
import sys
from pathlib import Path

src, dst, config, num_workers, prefetch_factor, start_after_axis, start_after_candidate = sys.argv[1:]
payload = json.loads(Path(src).read_text(encoding="utf-8"))
payload["server_rewritten_from"] = src
payload["server_config"] = config
payload["server_num_workers"] = int(num_workers)
if start_after_axis:
    payload["server_start_after_axis"] = start_after_axis
if start_after_candidate:
    payload["server_start_after_candidate"] = start_after_candidate


def candidate_name(run):
    candidate = run.get("candidate")
    if candidate:
        return str(candidate)
    return re.sub(r"_s\d+$", "", str(run.get("tag", "")))


def infer_axis(candidate):
    if re.search(r"_gc(?:\d|$)", candidate):
        return "gc"
    if re.search(r"_n\d+$", candidate):
        return "normal_ratio"
    if "_regls" in candidate:
        return "label_smoothing"
    if "_regdp" in candidate:
        return "stochastic_depth"
    if "_fg" in candidate:
        return "focal_gamma"
    if "_aw" in candidate:
        return "abnormal_weight"
    if "_ema" in candidate:
        return "ema"
    if "_color_" in candidate:
        return "color"
    if "_tie_" in candidate:
        return "allow_tie_save"
    return "other"


def trim_runs(runs):
    if not start_after_axis and not start_after_candidate:
        return runs
    last_idx = -1
    for idx, run in enumerate(runs):
        candidate = candidate_name(run)
        if start_after_candidate and candidate == start_after_candidate:
            last_idx = idx
        if start_after_axis and infer_axis(candidate) == start_after_axis:
            last_idx = idx
    if last_idx < 0:
        target = start_after_candidate or f"axis:{start_after_axis}"
        raise SystemExit(f"start-after target not found in queue {src}: {target}")
    kept = runs[last_idx + 1:]
    if not kept:
        target = start_after_candidate or f"axis:{start_after_axis}"
        raise SystemExit(f"queue has no runs after start-after target: {target}")
    print(f"[queue] trimmed {len(runs) - len(kept)} runs before start-after target; kept {len(kept)}")
    return kept


if isinstance(payload.get("runs"), list):
    if start_after_axis or start_after_candidate:
        payload["runs"] = trim_runs(payload["runs"])
elif start_after_axis or start_after_candidate:
    raise SystemExit(f"queue has no explicit runs list to trim: {src}")

for run in payload.get("runs", []):
    args = run.setdefault("args", {})
    args["--config"] = config
    args["--num_workers"] = int(num_workers)
    args["--prefetch_factor"] = int(prefetch_factor)

Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(dst)
PY
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

  write_state "running" "start" "server paper pipeline started"

  if [[ "$SKIP_WEIGHTS" -eq 0 && ! -f weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth ]]; then
    write_state "running" "weights" "downloading pretrained weights"
    run_cmd "$PYTHON" download.py
  fi

  if [[ "$SKIP_DATASET" -eq 0 ]]; then
    if [[ ! -f data/scenarios.csv || ! -f data/timeseries.csv || ! -d images ]]; then
      write_state "running" "dataset" "generating dataset and images"
      run_cmd "$PYTHON" generate_data.py --config "$CONFIG" --workers "$WORKERS"
      run_cmd "$PYTHON" generate_images.py --config "$CONFIG" --workers "$WORKERS"
      run_cmd "$PYTHON" scripts/validate_dataset.py --config "$CONFIG"
    else
      echo "[skip] dataset/images already exist"
    fi
  fi

  if [[ "$SKIP_REFCHECK" -eq 0 ]]; then
    prepare_queue \
      validations/paper_refcheck_gcsmooth_queue.json \
      validations/server_paper_refcheck_gcsmooth_queue.json
    run_controller \
      validations/server_paper_refcheck_gcsmooth_queue.json \
      validations/server_paper_refcheck_gcsmooth_summary.json \
      validations/server_paper_refcheck_gcsmooth_summary.md \
      "refcheck_gcsmooth"
  fi

  if [[ "$SKIP_ROUND1" -eq 0 ]]; then
    prepare_queue \
      validations/paper_strict_single_factor_queue.json \
      validations/server_paper_strict_single_factor_queue.json \
      "$ROUND1_START_AFTER_AXIS" \
      "$ROUND1_START_AFTER_CANDIDATE"
    run_controller \
      validations/server_paper_strict_single_factor_queue.json \
      validations/server_paper_strict_single_factor_summary.json \
      validations/server_paper_strict_single_factor_summary.md \
      "strict_round1"
  fi

  if [[ "$SKIP_ROUND2" -eq 0 ]]; then
    write_state "running" "select_round2" "selecting round2 from server round1 summary"
    run_cmd "$PYTHON" scripts/select_strict_single_factor_refinements.py \
      --summary validations/server_paper_strict_single_factor_summary.json \
      --out-queue validations/server_paper_strict_single_factor_round2_queue.json \
      --decision-md validations/server_paper_strict_single_factor_round2_decision.md

    if [[ -f validations/server_paper_strict_single_factor_round2_queue.json ]]; then
      ROUND2_COUNT="$("$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("validations/server_paper_strict_single_factor_round2_queue.json")
print(len(json.loads(p.read_text(encoding="utf-8")).get("runs", [])))
PY
)"
      if [[ "$ROUND2_COUNT" -gt 0 ]]; then
        prepare_queue \
          validations/server_paper_strict_single_factor_round2_queue.json \
          validations/server_paper_strict_single_factor_round2_queue.prepared.json
        run_controller \
          validations/server_paper_strict_single_factor_round2_queue.prepared.json \
          validations/server_paper_strict_single_factor_round2_summary.json \
          validations/server_paper_strict_single_factor_round2_summary.md \
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
      --strict-summary validations/server_paper_strict_single_factor_summary.json \
      --round2-summary validations/server_paper_strict_single_factor_round2_summary.json \
      --round2-queue validations/server_paper_strict_single_factor_round2_queue.prepared.json \
      --state validations/paper_server_all_state.json \
      --markdown-out validations/server_paper_strict_single_factor_summary.md \
      --report-out validations/server_paper_strict_single_factor_report.md \
      --plots-dir validations/server_paper_strict_single_factor_plots
    run_cmd "$PYTHON" scripts/publish_strict_report.py \
      --source-report validations/server_paper_strict_single_factor_report.md \
      --source-round2 validations/server_paper_strict_single_factor_round2_summary.md \
      --source-plots validations/server_paper_strict_single_factor_plots
  fi

  write_state "complete" "done" "server paper pipeline completed"
  echo "== paper server run completed: $(date -Is) =="
}

main
