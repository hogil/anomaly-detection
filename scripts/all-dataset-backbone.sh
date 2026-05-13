#!/usr/bin/env bash
# Run the full sweep (all axes incl. gc-last + sample_skip + backbone +
# logical_train + bkm_combined + postprocess) across every dataset yaml
# listed, then build a cross-dataset comparison report.
#
# Per dataset the flow is:
#   1) run_paper_server_all.sh --skip-round1 --skip-post   # weights + data + baseline
#   2) sweeps_server/00_all.sh                             # everything else
# Both calls share the same --log-dir-group so all logs and validations land
# under logs/<group>/ and validations/<group>/.
#
# Default datasets: dataset.yaml + 6 variants (matches HOW_TO_RUN weekend).
# Override with --datasets dataset.yaml,dataset1_noise_15.yaml,...
#
# By default this wrapper lets run_paper_server_all.sh run download.py so
# weights/{model_name}.pth exists before training. Use --skip-weights only when
# weights/*.pth was already copied into place on a closed-network machine.
set -euo pipefail

# Force Python utf-8 output so Korean prints render on Windows cp949 consoles
# without users having to set env vars themselves.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

D="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$D/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DATASETS_CSV=""
SKIP_PREP=0
SKIP_FULL=0
SKIP_REPORT=0
RESET_DATA=0
PREP_DATA_ONLY=0
SKIP_WEIGHTS=0
PASS_ARGS=()

DEFAULT_DATASETS=(
  dataset.yaml
  dataset1_noise_15.yaml
  dataset2_noise_30.yaml
  dataset3_anomaly_15.yaml
  dataset4_anomaly_30.yaml
  dataset5_all_15.yaml
  dataset6_all_30.yaml
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/all-dataset-backbone.sh [options] [-- <pass-through args for 00_all.sh>]

For every listed dataset yaml: prepares weights/data/baseline, then runs the
full sweep (all axes incl. gc-last, color, sample_skip, all backbones in
download.py::MODELS order plus extra non-deprecated weights, logical_train,
bkm_combined, postprocess). After the loop,
generates a cross-dataset comparison report under
validations/cross_dataset_report_<timestamp>/.

Options:
  --datasets CSV         Comma-separated yamls (default: 7 standard yamls)
  --python PATH          Python executable forwarded to inner scripts
  --skip-prep            Skip weights/data/baseline prep (assume all present)
  --skip-full            Skip 00_all.sh (only do prep + report)
  --skip-report          Skip cross-dataset report at the end
  --reset-data           Delete the config's data/image/display outputs before prep
  --prep-data-only       Prep only data/images/validation; skip baseline refcheck
  --skip-weights         Do not run download.py during prep; require weights/*.pth already present
  -h, --help             Show this help

Anything after `--` is forwarded verbatim to 00_all.sh, e.g.:
  bash scripts/all-dataset-backbone.sh -- --max-launched 1 --force

Long-running on a server? Wrap with nohup:
  nohup bash scripts/all-dataset-backbone.sh > /tmp/all_dsbk.log 2>&1 &
  disown
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --skip-prep) SKIP_PREP=1; shift ;;
    --skip-full) SKIP_FULL=1; shift ;;
    --skip-report) SKIP_REPORT=1; shift ;;
    --reset-data) RESET_DATA=1; shift ;;
    --prep-data-only) PREP_DATA_ONLY=1; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --) shift; PASS_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -n "$DATASETS_CSV" ]]; then
  IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Filter to existing yamls; warn for missing.
RESOLVED=()
for cfg in "${DATASETS[@]}"; do
  cfg="${cfg// /}"
  [[ -z "$cfg" ]] && continue
  if [[ ! -f "$cfg" ]]; then
    echo "[skip] missing dataset yaml: $cfg" >&2
    continue
  fi
  RESOLVED+=("$cfg")
done

if [[ "${#RESOLVED[@]}" -eq 0 ]]; then
  echo "no dataset yaml resolved; nothing to do" >&2
  exit 1
fi

if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
  usable_weights=0
  shopt -s nullglob
  for p in weights/*.pth; do
    name="$(basename "$p")"
    if [[ "$name" == "best_model.pth" || "$name" == *.fp16.pth || "$name" == vit_*.pth || "$name" == swin_*.pth ]]; then
      continue
    fi
    usable_weights=$((usable_weights + 1))
  done
  shopt -u nullglob
  if [[ "$usable_weights" -eq 0 ]]; then
    echo "[fatal] --skip-weights was set, but no usable weights/*.pth files exist" >&2
    echo "        run python download.py, or copy weights/*.pth into this repo first" >&2
    exit 1
  fi
fi

WRAPPER_TS="$(date +%Y%m%d_%H%M%S)"
echo "== all-dataset-backbone start: $(date -Is) =="
echo "datasets: ${RESOLVED[*]}"
echo "skip_prep=$SKIP_PREP skip_full=$SKIP_FULL skip_report=$SKIP_REPORT reset_data=$RESET_DATA prep_data_only=$PREP_DATA_ONLY skip_weights=$SKIP_WEIGHTS"

GROUP_NAMES=()
GROUP_CONFIGS=()

for cfg in "${RESOLVED[@]}"; do
  config_stem="$(basename "$cfg")"
  config_stem="${config_stem%.yaml}"
  config_stem="${config_stem%.yml}"
  ts="$(date +%Y%m%d_%H%M%S)"
  group="${ts}_run_paper_${config_stem}"
  GROUP_NAMES+=("$group")
  GROUP_CONFIGS+=("$cfg")

  echo
  echo "============================================================"
  echo "[dataset] $cfg"
  echo "[group]   $group"
  echo "============================================================"

  if [[ "$SKIP_PREP" -eq 0 ]]; then
    if [[ "$RESET_DATA" -eq 1 ]]; then
      "$PYTHON" - "$cfg" "$REPO_ROOT" <<'PY'
import shutil
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2]).resolve()
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

for key in ("data_dir", "image_dir", "display_dir"):
    raw = Path(cfg["output"][key])
    target = raw if raw.is_absolute() else repo_root / raw
    target = target.resolve()
    try:
        target.relative_to(repo_root)
    except ValueError:
        raise SystemExit(f"refusing to delete outside repo: {target}")
    if target == repo_root:
        raise SystemExit("refusing to delete repo root")
    if target.exists():
        print(f"[reset-data] delete {target}")
        shutil.rmtree(target)
PY
    fi

    echo "[step 1/2] prep (weights + data + baseline)"
    PREP_EXTRA=()
    if [[ "$PREP_DATA_ONLY" -eq 1 ]]; then
      PREP_EXTRA+=(--skip-refcheck)
    fi
    if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
      PREP_EXTRA+=(--skip-weights)
    fi
    bash scripts/run_paper_server_all.sh \
      --config "$cfg" \
      --python "$PYTHON" \
      --log-dir-group "$group" \
      --skip-round1 \
      --skip-post \
      "${PREP_EXTRA[@]}"
  else
    echo "[step 1/2] prep skipped"
  fi

  if [[ "$SKIP_FULL" -eq 0 ]]; then
    echo "[step 2/2] full sweep (00_all.sh)"
    bash scripts/sweeps_server/00_all.sh \
      --config "$cfg" \
      --python "$PYTHON" \
      --log-dir-group "$group" \
      "${PASS_ARGS[@]}"
  else
    echo "[step 2/2] full sweep skipped"
  fi
done

if [[ "$SKIP_REPORT" -eq 1 ]]; then
  echo "[skip] cross-dataset report"
  echo "== all-dataset-backbone done: $(date -Is) =="
  exit 0
fi

REPORT_DIR="validations/cross_dataset_report_${WRAPPER_TS}"
mkdir -p "$REPORT_DIR"

# Build groups CSV: name=config pairs joined with ;
GROUPS_ARG=""
for i in "${!GROUP_NAMES[@]}"; do
  pair="${GROUP_NAMES[$i]}=${GROUP_CONFIGS[$i]}"
  if [[ -z "$GROUPS_ARG" ]]; then
    GROUPS_ARG="$pair"
  else
    GROUPS_ARG="${GROUPS_ARG};${pair}"
  fi
done

echo
echo "== cross-dataset report =="
"$PYTHON" scripts/generate_cross_dataset_report.py \
  --groups "$GROUPS_ARG" \
  --validations-root validations \
  --out-dir "$REPORT_DIR"

echo "== all-dataset-backbone done: $(date -Is) =="
echo "report: $REPORT_DIR"
