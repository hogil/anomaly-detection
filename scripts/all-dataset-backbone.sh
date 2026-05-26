#!/usr/bin/env bash
# Run the full sweep across (backbone x dataset) cells, then build a
# cross-dataset comparison report.
#
# Two modes:
#
#   1) Legacy (default when --backbones is NOT set): one pass per dataset, with
#      stage 14 (backbone rotation) running every weights/*.pth. Existing
#      behavior; produces validations/cross_dataset_report_<ts>/.
#
#   2) Cross-product (when --backbones CSV is set): outer loop is
#      dataset x backbone. Per cell: data prep (once per dataset), then
#      00_all.sh with --model-name $bb and --skip-{sample-skip,backbone-sweep,
#      logical-train} so each cell trains exactly one backbone. Required for
#      paper main result: per-(backbone, dataset) BKM.
#
# Per cell the flow is:
#   1) run_paper_server_all.sh prep (weights + data + images)            [once per dataset]
#   2) sweeps_server/00_all.sh with cell-specific --model-name           [per cell]
#
# Default datasets: 4 (baseline + anomaly_10 + noise_15 + all_a10n15).
# Override with --datasets dataset.yaml,dataset1_noise_15.yaml,...
#
# Default backbones (cross-product only): top-3 + OpenAI CLIP.
# Override with --backbones convnext_tiny.dinov3_lvd1689m,...
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
BACKBONES_CSV=""
SKIP_PREP=0
SKIP_FULL=0
SKIP_REPORT=0
RESET_DATA=0
PREP_DATA_ONLY=0
SKIP_WEIGHTS=0
BATCH_SIZE="${BATCH_SIZE:-}"
PASS_ARGS=()

DEFAULT_DATASETS=(
  dataset.yaml
  dataset1_noise_15.yaml
  dataset3_anomaly_10.yaml
  dataset5_all_a10n15.yaml
)

# Default 4 backbones for the paper matrix: top-3 from prior backbone sweep
# plus OpenAI ViT-B/16 CLIP. Used only when --backbones is set (or empty
# string passed explicitly to enable cross-product with defaults).
DEFAULT_BACKBONES=(
  "convnext_tiny.dinov3_lvd1689m"
  "convnextv2_base.fcmae_ft_in22k_in1k"
  "convnextv2_tiny.fcmae_ft_in22k_in1k"
  "vit_base_patch16_clip_224.openai_ft_in1k"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/all-dataset-backbone.sh [options] [-- <pass-through args for 00_all.sh>]

Modes:
  Legacy (default):     one 00_all run per dataset; stage 14 rotates backbones.
  Cross-product (-x):   outer loop is dataset x backbone; stages 13/14/15 skipped.

Options:
  --datasets CSV         Comma-separated yamls (default: 4 paper datasets)
  --backbones CSV        Comma-separated backbones to enable cross-product mode.
                         Use the literal string "default" to fill from the
                         4-backbone DEFAULT_BACKBONES list.
  -x | --cross-product   Shorthand for --backbones default.
  --python PATH          Python executable forwarded to inner scripts
  --skip-prep            Skip weights/data/baseline prep (assume all present)
  --skip-full            Skip 00_all.sh (only do prep + report)
  --skip-report          Skip cross-dataset report at the end
  --reset-data           Delete the config's data/image/display outputs before prep
  --prep-data-only       Prep only data/images/validation; skip baseline refcheck.
                         Auto-enabled in cross-product mode (per-cell baseline
                         is produced by 00_all.sh stage 01 with the right
                         --model-name).
  --skip-weights         Do not run download.py during prep; require weights/*.pth already present
  --batch-size N         train.py --batch_size override (auto: H100/H200 server=256)
  -h, --help             Show this help

Anything after `--` is forwarded verbatim to 00_all.sh, e.g.:
  bash scripts/all-dataset-backbone.sh -x -- --max-launched 1 --force

Long-running on a server? Wrap with nohup:
  nohup bash scripts/all-dataset-backbone.sh -x > /tmp/all_dsbk.log 2>&1 &
  disown
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --backbones) BACKBONES_CSV="$2"; shift 2 ;;
    -x|--cross-product) BACKBONES_CSV="default"; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    --skip-prep) SKIP_PREP=1; shift ;;
    --skip-full) SKIP_FULL=1; shift ;;
    --skip-report) SKIP_REPORT=1; shift ;;
    --reset-data) RESET_DATA=1; shift ;;
    --prep-data-only) PREP_DATA_ONLY=1; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
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

CROSS_PRODUCT=0
BACKBONES=()
if [[ -n "$BACKBONES_CSV" ]]; then
  CROSS_PRODUCT=1
  if [[ "$BACKBONES_CSV" == "default" ]]; then
    BACKBONES=("${DEFAULT_BACKBONES[@]}")
  else
    IFS=',' read -r -a BACKBONES <<< "$BACKBONES_CSV"
  fi
  # Per-cell baseline is produced by stage 01 with --model-name, so prep step
  # should only render data/images (not run baseline).
  PREP_DATA_ONLY=1
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

# In cross-product mode, refuse to start if requested backbone weights are missing.
if [[ "$CROSS_PRODUCT" -eq 1 ]]; then
  missing_bb=()
  for bb in "${BACKBONES[@]}"; do
    if [[ ! -f "weights/${bb}.pth" ]]; then
      missing_bb+=("$bb")
    fi
  done
  if [[ "${#missing_bb[@]}" -gt 0 ]]; then
    echo "[fatal] cross-product mode: weights/*.pth missing for:" >&2
    for bb in "${missing_bb[@]}"; do echo "  - weights/${bb}.pth" >&2; done
    echo "        run python download.py, or copy weights/*.pth into this repo first" >&2
    exit 1
  fi
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
REPORT_DIR="validations/cross_dataset_report_${WRAPPER_TS}"
echo "== all-dataset-backbone start: $(date -Is) =="
echo "datasets: ${RESOLVED[*]}"
if [[ "$CROSS_PRODUCT" -eq 1 ]]; then
  echo "backbones: ${BACKBONES[*]}"
  echo "mode: cross-product (4 stages: 01 + 02 + 16 + 17; 13/14/15 skipped)"
else
  echo "mode: legacy (single backbone-rotation pass per dataset)"
fi
echo "skip_prep=$SKIP_PREP skip_full=$SKIP_FULL skip_report=$SKIP_REPORT reset_data=$RESET_DATA prep_data_only=$PREP_DATA_ONLY skip_weights=$SKIP_WEIGHTS"

GROUP_NAMES=()
GROUP_CONFIGS=()
GROUP_MODELS=()

# 14_backbone.sh::short — mirror the same shortening used elsewhere.
short_bb() {
  local s="$1"
  s="${s%%.*}"
  echo "${s//_/}"
}

build_groups_arg() {
  local groups_arg=""
  local pair
  for i in "${!GROUP_NAMES[@]}"; do
    if [[ -n "${GROUP_MODELS[$i]}" ]]; then
      pair="${GROUP_NAMES[$i]}=${GROUP_CONFIGS[$i]}=${GROUP_MODELS[$i]}"
    else
      pair="${GROUP_NAMES[$i]}=${GROUP_CONFIGS[$i]}"
    fi
    if [[ -z "$groups_arg" ]]; then
      groups_arg="$pair"
    else
      groups_arg="${groups_arg};${pair}"
    fi
  done
  printf '%s' "$groups_arg"
}

write_cross_dataset_report() {
  local label="$1"
  [[ "$SKIP_REPORT" -eq 1 ]] && return 0
  if [[ "${#GROUP_NAMES[@]}" -eq 0 ]]; then
    return 0
  fi
  mkdir -p "$REPORT_DIR"
  echo
  echo "== cross-dataset report: $label =="
  "$PYTHON" scripts/generate_cross_dataset_report.py \
    --groups "$(build_groups_arg)" \
    --validations-root validations \
    --out-dir "$REPORT_DIR"
  echo "report: $REPORT_DIR"
}

do_prep() {
  local cfg="$1" prep_group="$2"
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

  echo "[prep] weights + data + images (cfg=$cfg)"
  local prep_extra=()
  if [[ "$PREP_DATA_ONLY" -eq 1 ]]; then
    prep_extra+=(--skip-refcheck)
  fi
  if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
    prep_extra+=(--skip-weights)
  fi
  if [[ -n "$BATCH_SIZE" ]]; then
    prep_extra+=(--batch-size "$BATCH_SIZE")
  fi
  bash scripts/run_paper_server_all.sh \
    --config "$cfg" \
    --python "$PYTHON" \
    --log-dir-group "$prep_group" \
    --skip-round1 \
    --skip-post \
    "${prep_extra[@]}"
}

run_cell() {
  local cfg="$1" bb="$2" group="$3"
  echo "[cell] cfg=$cfg backbone=$bb group=$group"
  local full_extra=()
  if [[ -n "$BATCH_SIZE" ]]; then
    full_extra+=(--batch-size "$BATCH_SIZE")
  fi
  bash scripts/sweeps_server/00_all.sh \
    --config "$cfg" \
    --python "$PYTHON" \
    --log-dir-group "$group" \
    --model-name "$bb" \
    --skip-sample-skip \
    --skip-backbone-sweep \
    --skip-logical-train \
    "${full_extra[@]}" \
    "${PASS_ARGS[@]}"
}

run_legacy() {
  local cfg="$1" group="$2"
  echo "[legacy] cfg=$cfg group=$group (stage 14 rotates backbones)"
  local full_extra=()
  if [[ -n "$BATCH_SIZE" ]]; then
    full_extra+=(--batch-size "$BATCH_SIZE")
  fi
  bash scripts/sweeps_server/00_all.sh \
    --config "$cfg" \
    --python "$PYTHON" \
    --log-dir-group "$group" \
    "${full_extra[@]}" \
    "${PASS_ARGS[@]}"
}

for cfg in "${RESOLVED[@]}"; do
  config_stem="$(basename "$cfg")"
  config_stem="${config_stem%.yaml}"
  config_stem="${config_stem%.yml}"

  echo
  echo "============================================================"
  echo "[dataset] $cfg"
  echo "============================================================"

  cfg_ts="$(date +%Y%m%d_%H%M%S)"
  if [[ "$CROSS_PRODUCT" -eq 1 ]]; then
    prep_group="${WRAPPER_TS}_prep_${config_stem}"
  else
    prep_group="${cfg_ts}_run_paper_${config_stem}"
  fi

  if [[ "$SKIP_PREP" -eq 0 ]]; then
    do_prep "$cfg" "$prep_group"
  else
    echo "[prep] skipped"
  fi

  if [[ "$CROSS_PRODUCT" -eq 1 ]]; then
    for bb in "${BACKBONES[@]}"; do
      bb_short="$(short_bb "$bb")"
      cell_ts="$(date +%Y%m%d_%H%M%S)"
      group="${cell_ts}_${config_stem}_${bb_short}"
      GROUP_NAMES+=("$group")
      GROUP_CONFIGS+=("$cfg")
      GROUP_MODELS+=("$bb")

      if [[ "$SKIP_FULL" -eq 0 ]]; then
        run_cell "$cfg" "$bb" "$group"
      else
        echo "[cell] $group skipped (--skip-full)"
      fi

      write_cross_dataset_report "$cfg / $bb_short  cells=${#GROUP_NAMES[@]}"
    done
  else
    GROUP_NAMES+=("$prep_group")
    GROUP_CONFIGS+=("$cfg")
    GROUP_MODELS+=("")

    if [[ "$SKIP_FULL" -eq 0 ]]; then
      run_legacy "$cfg" "$prep_group"
    else
      echo "[legacy] $prep_group skipped (--skip-full)"
    fi

    write_cross_dataset_report "completed ${#GROUP_NAMES[@]}/${#RESOLVED[@]} datasets"
  fi
done

if [[ "$SKIP_REPORT" -eq 1 ]]; then
  echo "[skip] cross-dataset report"
  echo "== all-dataset-backbone done: $(date -Is) =="
  exit 0
fi

write_cross_dataset_report "final cells=${#GROUP_NAMES[@]}"

echo "== all-dataset-backbone done: $(date -Is) =="
echo "report: $REPORT_DIR"
