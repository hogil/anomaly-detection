#!/usr/bin/env bash
# Server default: run all paper stages in order.
# Auto-detects the runtime profile (server/pc/minimal) for sane defaults.
set -euo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
source "$D/_common.sh"
auto_enable_ddp

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweeps_server/00_all.sh [run_paper_server_all options]

Stages, in order:
  baseline recheck -> core axes -> color -> sample_skip -> backbone ->
  logical_train -> bkm_combined -> postprocess

Additional 00_all.sh options:
  --model-name NAME           train.py --model_name override forwarded to every
                              stage (baseline / axes / sample_skip / backbone /
                              logical_train / bkm_combined). Used by
                              all-dataset-backbone.sh for backbone x dataset
                              cross-product, where each cell trains one specific
                              backbone instead of letting stage 14 rotate.
  --skip-sample-skip          skip stage 13 (sample_skip safety probe)
  --skip-backbone-sweep       skip stage 14 (backbone rotation). Required for
                              cross-product mode so the outer loop is the sole
                              backbone driver.
  --skip-logical-train        skip stage 15 (per-member logical training)

Individual stages (run alone):
  bash scripts/sweeps_server/01_baseline.sh
  bash scripts/sweeps_server/02_lr.sh ... 12_color.sh         # one axis each
  bash scripts/sweeps_server/13_sample_skip.sh                # nonfinite-loss probe
  bash scripts/sweeps_server/14_backbone.sh                   # rotate weights/*.pth
  bash scripts/sweeps_server/15_logical_train.sh              # per-member logical
  bash scripts/sweeps_server/17_bkm_combined.sh               # apply all BKMs
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Extract --skip-* convenience flags before the shared forwarding loop so the
# inner stage scripts (which do not recognize them) never see them.
SKIP_SAMPLE_SKIP=0
SKIP_BACKBONE_SWEEP=0
SKIP_LOGICAL_TRAIN=0
FILTERED_ARGS=()
for raw_arg in "$@"; do
  case "$raw_arg" in
    --skip-sample-skip)    SKIP_SAMPLE_SKIP=1 ;;
    --skip-backbone-sweep) SKIP_BACKBONE_SWEEP=1 ;;
    --skip-logical-train)  SKIP_LOGICAL_TRAIN=1 ;;
    *) FILTERED_ARGS+=("$raw_arg") ;;
  esac
done
set -- "${FILTERED_ARGS[@]}"

# Forward shared options to the separate stage scripts.
SAMPLE_SKIP_ARGS=()
BACKBONE_ARGS=()
LOGICAL_TRAIN_ARGS=()
BKM_COMBINED_ARGS=()
HAS_LOG_DIR_GROUP=0
for raw_arg in "$@"; do
  if [[ "$raw_arg" == "--log-dir-group" ]]; then
    HAS_LOG_DIR_GROUP=1
    break
  fi
done
if [[ "$HAS_LOG_DIR_GROUP" -eq 0 ]]; then
  SHARED_LOG_DIR_GROUP="${LOG_DIR_GROUP:-$(date +%Y%m%d_%H%M%S)_run_paper}"
  ALL_ARGS=("$@" "--log-dir-group" "$SHARED_LOG_DIR_GROUP")
  echo "[00_all] log_dir_group=$SHARED_LOG_DIR_GROUP"
else
  ALL_ARGS=("$@")
fi
idx=0
while [[ "$idx" -lt "${#ALL_ARGS[@]}" ]]; do
  arg="${ALL_ARGS[$idx]}"
  case "$arg" in
    --python|--config|--num-workers|--prefetch-factor|--batch-size|--max-launched|--log-dir-group|--checkpoint-retention|--checkpoint-retention-scope|--model-name)
      if [[ "$((idx + 1))" -ge "${#ALL_ARGS[@]}" ]]; then
        echo "Missing value for $arg" >&2
        exit 2
      fi
      SAMPLE_SKIP_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      BACKBONE_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      LOGICAL_TRAIN_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      BKM_COMBINED_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      idx=$((idx + 2))
      ;;
    --workers)
      if [[ "$((idx + 1))" -ge "${#ALL_ARGS[@]}" ]]; then
        echo "Missing value for $arg" >&2
        exit 2
      fi
      LOGICAL_TRAIN_ARGS+=("$arg" "${ALL_ARGS[$((idx + 1))]}")
      idx=$((idx + 2))
      ;;
    --force)
      SAMPLE_SKIP_ARGS+=("$arg")
      BACKBONE_ARGS+=("$arg")
      LOGICAL_TRAIN_ARGS+=("$arg")
      BKM_COMBINED_ARGS+=("$arg")
      idx=$((idx + 1))
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done

# Stage 14 does not accept --model-name (it rotates backbones itself). Drop
# any --model-name flag from BACKBONE_ARGS so 14_backbone.sh does not reject it.
PRUNED_BACKBONE_ARGS=()
i=0
while [[ "$i" -lt "${#BACKBONE_ARGS[@]}" ]]; do
  if [[ "${BACKBONE_ARGS[$i]}" == "--model-name" ]]; then
    i=$((i + 2))
    continue
  fi
  PRUNED_BACKBONE_ARGS+=("${BACKBONE_ARGS[$i]}")
  i=$((i + 1))
done
BACKBONE_ARGS=("${PRUNED_BACKBONE_ARGS[@]}")

# Stage 13 / 15 currently do not accept --model-name either; strip it too.
prune_model_name() {
  local arr_name="$1"
  local -n src="$arr_name"
  local out=()
  local k=0
  while [[ "$k" -lt "${#src[@]}" ]]; do
    if [[ "${src[$k]}" == "--model-name" ]]; then
      k=$((k + 2))
      continue
    fi
    out+=("${src[$k]}")
    k=$((k + 1))
  done
  src=("${out[@]}")
}
prune_model_name SAMPLE_SKIP_ARGS
prune_model_name LOGICAL_TRAIN_ARGS

PRE_COLOR_AXES="lr,warmup,normal_ratio,per_class,weight_decay,smoothing,label_smoothing,asl,stochastic_depth,focal_gamma,abnormal_weight,ema,allow_tie_save"

run_round1_axes "needed_pre_color" "$PRE_COLOR_AXES" "${ALL_ARGS[@]}"
run_round1_axes "color" "color" "${ALL_ARGS[@]}"

if [[ "$SKIP_SAMPLE_SKIP" -eq 0 ]]; then
  bash "$D/13_sample_skip.sh" "${SAMPLE_SKIP_ARGS[@]}"
else
  echo "[00_all] skip stage 13: sample_skip"
fi
if [[ "$SKIP_BACKBONE_SWEEP" -eq 0 ]]; then
  bash "$D/14_backbone.sh" "${BACKBONE_ARGS[@]}"
else
  echo "[00_all] skip stage 14: backbone_sweep"
fi
if [[ "$SKIP_LOGICAL_TRAIN" -eq 0 ]]; then
  bash "$D/15_logical_train.sh" "${LOGICAL_TRAIN_ARGS[@]}"
else
  echo "[00_all] skip stage 15: logical_train"
fi

bash "$D/17_bkm_combined.sh" "${BKM_COMBINED_ARGS[@]}"

run_paper_stage "postprocess" \
  --skip-weights \
  --skip-dataset \
  --skip-refcheck \
  --skip-round1 \
  "${ALL_ARGS[@]}"
