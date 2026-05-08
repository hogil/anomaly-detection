#!/usr/bin/env bash
# One-epoch smoke test for every (backbone × dataset) combination.
#
# For each dataset yaml (default: 7 standard yamls):
#   1) generate scenarios.csv / timeseries.csv / images/ if missing
#   2) for each backbone .pth in weights/ (excluding best_model + *.fp16),
#      run train.py --epochs 1 --patience 0 --seed 42 once
#
# Failures of an individual run are recorded but do NOT abort the whole sweep.
# Results land in validations/smoke_1ep_<ts>/ with per-run logs and a single
# Markdown report (report.md).
#
# Usage:
#   bash scripts/smoke_one_epoch_all_backbones.sh
#   bash scripts/smoke_one_epoch_all_backbones.sh --datasets dataset.yaml,dataset1_noise_15.yaml
#   PYTHON=/path/to/python bash scripts/smoke_one_epoch_all_backbones.sh

set -uo pipefail

# Force Python utf-8 output so Korean prints render on Windows cp949 consoles.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GEN_WORKERS="${GEN_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"   # train.py --max_samples_per_split (per train/val/test)
SKIP_MISSING_DATA="${SKIP_MISSING_DATA:-1}"  # 1 = skip yamls without generated data; 0 = generate first
DATASETS_CSV=""

DEFAULT_DATASETS=(
  dataset.yaml
  dataset1_noise_15.yaml
  dataset2_noise_30.yaml
  dataset3_anomaly_15.yaml
  dataset4_anomaly_30.yaml
  dataset5_all_15.yaml
  dataset6_all_30.yaml
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --python)   PYTHON="$2"; shift 2 ;;
    --gen-workers) GEN_WORKERS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --skip-missing-data) SKIP_MISSING_DATA=1; shift ;;
    --gen-missing-data)  SKIP_MISSING_DATA=0; shift ;;
    -h|--help)
      sed -n '1,30p' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$DATASETS_CSV" ]]; then
  IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

TS="$(date +%Y%m%d_%H%M%S)"
GROUP="smoke_1ep_${TS}"
OUT_DIR="validations/${GROUP}"
mkdir -p "$OUT_DIR"
REPORT="${OUT_DIR}/report.md"
SWEEP_LOG="${OUT_DIR}/sweep.log"

# discover backbones (sorted, exclude best_model + *.fp16)
BACKBONES=()
while IFS= read -r p; do
  name="$(basename "$p" .pth)"
  case "$name" in
    best_model|*.fp16) continue ;;
  esac
  BACKBONES+=("$name")
done < <(ls weights/*.pth 2>/dev/null | sort)

if [[ "${#BACKBONES[@]}" -eq 0 ]]; then
  echo "[fatal] no usable .pth in weights/" >&2
  exit 1
fi

# Pre-flight: warn (not block) if another non-self python is already on GPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  other_train=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
    | grep -i 'python' | wc -l)
  if [[ "$other_train" -gt 0 ]]; then
    echo "[preflight] warning: $other_train python process(es) currently on GPU — contention may slow this sweep:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null >&2
  fi
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null
fi

{
  echo "# One-Epoch Smoke Test ${TS}"
  echo
  echo "- datasets: ${DATASETS[*]}"
  echo "- backbones (${#BACKBONES[@]}): ${BACKBONES[*]}"
  echo "- python: ${PYTHON}"
  echo "- max_samples_per_split: ${MAX_SAMPLES}"
  echo "- skip_missing_data: ${SKIP_MISSING_DATA}"
  echo
  echo "| # | Dataset | Backbone | Status | Elapsed | Last log line |"
  echo "|---|---|---|---|---|---|"
} > "$REPORT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SWEEP_LOG"; }

cfg_get() {
  local cfg="$1" key="$2"
  "$PYTHON" -c "import yaml,sys; v=yaml.safe_load(open('$cfg',encoding='utf-8')); [v:=v[k] for k in '$key'.split('.')]; print(v)"
}

ensure_data() {
  local cfg="$1"
  local data_dir image_dir display_dir
  data_dir="$(cfg_get "$cfg" output.data_dir)"
  image_dir="$(cfg_get "$cfg" output.image_dir)"
  display_dir="$(cfg_get "$cfg" output.display_dir)"
  if [[ -f "$data_dir/scenarios.csv" && -f "$data_dir/timeseries.csv" && -d "$image_dir" ]]; then
    log "[skip-gen] $cfg ($data_dir / $image_dir already exists)"
    return 0
  fi
  if [[ "$SKIP_MISSING_DATA" -eq 1 ]]; then
    log "[skip-missing] $cfg (no data; SKIP_MISSING_DATA=1)"
    return 2
  fi
  log "[gen] $cfg → $data_dir / $image_dir (workers=$GEN_WORKERS)"
  set +e
  "$PYTHON" generate_data.py --config "$cfg" --workers "$GEN_WORKERS" >>"$SWEEP_LOG" 2>&1
  local rc1=$?
  if [[ $rc1 -ne 0 ]]; then log "  [FAIL] generate_data.py rc=$rc1"; return 1; fi
  "$PYTHON" generate_images.py --config "$cfg" --workers "$GEN_WORKERS" >>"$SWEEP_LOG" 2>&1
  local rc2=$?
  if [[ $rc2 -ne 0 ]]; then log "  [FAIL] generate_images.py rc=$rc2"; return 1; fi
  "$PYTHON" scripts/validate_dataset.py \
    --config "$cfg" \
    --scenarios "$data_dir/scenarios.csv" \
    --timeseries "$data_dir/timeseries.csv" \
    --display-dir "$display_dir" >>"$SWEEP_LOG" 2>&1
  local rc3=$?
  set -e
  if [[ $rc3 -ne 0 ]]; then log "  [warn] validate_dataset rc=$rc3 (continuing)"; fi
  return 0
}

ROW_INDEX=0
for cfg in "${DATASETS[@]}"; do
  cfg="${cfg// /}"
  if [[ -z "$cfg" ]]; then continue; fi
  if [[ ! -f "$cfg" ]]; then
    log "[skip] missing yaml: $cfg"
    continue
  fi
  cfg_stem="$(basename "${cfg%.yaml}")"
  cfg_stem="${cfg_stem%.yml}"

  ensure_data "$cfg"; ed_rc=$?
  if [[ $ed_rc -eq 2 ]]; then
    log "[skip-bbsweep] $cfg (no data, marked NO_DATA)"
    for bb in "${BACKBONES[@]}"; do
      ROW_INDEX=$((ROW_INDEX + 1))
      echo "| $ROW_INDEX | $cfg | $bb | NO_DATA | - | (data not generated; rerun with --gen-missing-data) |" >> "$REPORT"
    done
    continue
  elif [[ $ed_rc -ne 0 ]]; then
    log "[skip-bbsweep] $cfg (data gen failed; recording row)"
    for bb in "${BACKBONES[@]}"; do
      ROW_INDEX=$((ROW_INDEX + 1))
      echo "| $ROW_INDEX | $cfg | $bb | DATA_GEN_FAIL | - | (data unavailable) |" >> "$REPORT"
    done
    continue
  fi

  for bb in "${BACKBONES[@]}"; do
    ROW_INDEX=$((ROW_INDEX + 1))
    bb_short="$(echo "$bb" | sed -E 's/\..+$//; s/_//g')"
    tag="smoke_${cfg_stem}_${bb_short}_s42"
    run_log="${OUT_DIR}/${tag}.log"
    log "[$ROW_INDEX/$((${#DATASETS[@]} * ${#BACKBONES[@]}))] $cfg × $bb → $tag"
    start=$(date +%s)
    set +e
    "$PYTHON" -u train.py \
      --mode binary \
      --config "$cfg" \
      --epochs 1 \
      --patience 0 \
      --batch_size 32 \
      --dropout 0.0 \
      --precision fp16 \
      --num_workers 0 \
      --prefetch_factor 1 \
      --normal_ratio 700 \
      --smooth_window 1 \
      --smooth_method median \
      --lr_backbone 2e-5 \
      --lr_head 2e-4 \
      --warmup_epochs 0 \
      --grad_clip 0.0 \
      --weight_decay 0.01 \
      --label_smoothing 0.0 \
      --ema_decay 0.0 \
      --max_samples_per_split "$MAX_SAMPLES" \
      --seed 42 \
      --model_name "$bb" \
      --log_dir "$tag" \
      --log_dir_group "$GROUP" \
      >"$run_log" 2>&1
    rc=$?
    set -e
    elapsed=$(( $(date +%s) - start ))
    if [[ $rc -eq 0 ]]; then
      status="OK"
    else
      status="FAIL($rc)"
    fi
    last_line="$(tail -n 1 "$run_log" 2>/dev/null | tr -d '\r' | tr '|' '/' | head -c 90)"
    echo "| $ROW_INDEX | $cfg | $bb | $status | ${elapsed}s | ${last_line} |" >> "$REPORT"
    log "  → $status elapsed=${elapsed}s"
  done
done

# summary footer
ok=$(grep -c '| OK |' "$REPORT" || true)
fail=$(grep -c '| FAIL' "$REPORT" || true)
gen_fail=$(grep -c '| DATA_GEN_FAIL |' "$REPORT" || true)
no_data=$(grep -c '| NO_DATA |' "$REPORT" || true)
{
  echo
  echo "## Summary"
  echo
  echo "- OK: $ok"
  echo "- FAIL: $fail"
  echo "- DATA_GEN_FAIL: $gen_fail"
  echo "- NO_DATA (skipped): $no_data"
  echo "- Total: $ROW_INDEX"
} >> "$REPORT"

log "[done] OK=$ok FAIL=$fail GEN_FAIL=$gen_fail NO_DATA=$no_data report=$REPORT"
echo
echo "Report: $REPORT"
