#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-dataset}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON="python"
CONFIG="dataset.yaml"
WORKERS=1
SKIP_VALIDATE=0
NORMAL_RATIO=700
MAX_PER_CLASS=0
SEED=42
LOG_NAME="company_ref"
NAME_PREFIX="company_run"
BASE_N=700
NUM_WORKERS=1
PREFETCH_FACTOR=4
EXTRA_ARGS=()

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --skip-validate)
      SKIP_VALIDATE=1
      shift
      ;;
    --normal-ratio)
      NORMAL_RATIO="$2"
      shift 2
      ;;
    --max-per-class)
      MAX_PER_CLASS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --log-name)
      LOG_NAME="$2"
      shift 2
      ;;
    --name-prefix)
      NAME_PREFIX="$2"
      shift 2
      ;;
    --base-n)
      BASE_N="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --prefetch-factor)
      PREFETCH_FACTOR="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

case "$STAGE" in
  weights)
    run_cmd "$PYTHON" download.py "${EXTRA_ARGS[@]}"
    ;;

  dataset)
    run_cmd "$PYTHON" generate_data.py --config "$CONFIG" --workers "$WORKERS"
    run_cmd "$PYTHON" generate_images.py --config "$CONFIG" --workers "$WORKERS"
    if [[ "$SKIP_VALIDATE" -eq 0 ]]; then
      run_cmd "$PYTHON" scripts/validate_dataset.py --config "$CONFIG"
    fi
    ;;

  train)
    TRAIN_ARGS=(
      train.py
      --config "$CONFIG"
      --seed "$SEED"
      --num_workers "$NUM_WORKERS"
      --prefetch_factor "$PREFETCH_FACTOR"
      --log_dir "$LOG_NAME"
    )
    if [[ "$NORMAL_RATIO" -gt 0 ]]; then
      TRAIN_ARGS+=(--normal_ratio "$NORMAL_RATIO")
    fi
    if [[ "$MAX_PER_CLASS" -gt 0 ]]; then
      TRAIN_ARGS+=(--max_per_class "$MAX_PER_CLASS")
    fi
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
    run_cmd "$PYTHON" "${TRAIN_ARGS[@]}"
    ;;

  sweep)
    run_cmd "$PYTHON" run_experiments_v11.py \
      --groups sweep \
      --num_workers "$NUM_WORKERS" \
      --name-prefix "$NAME_PREFIX" \
      "${EXTRA_ARGS[@]}"
    ;;

  perclass)
    run_cmd "$PYTHON" run_experiments_v11.py \
      --groups perclass \
      --num_workers "$NUM_WORKERS" \
      --name-prefix "$NAME_PREFIX" \
      "${EXTRA_ARGS[@]}"
    ;;

  ablation)
    run_cmd "$PYTHON" run_experiments_v11.py \
      --groups lr gc smooth reg \
      --base_n "$BASE_N" \
      --num_workers "$NUM_WORKERS" \
      --name-prefix "$NAME_PREFIX" \
      "${EXTRA_ARGS[@]}"
    ;;

  summary)
    run_cmd "$PYTHON" run_experiments_v11.py \
      --only-summary \
      --base_n "$BASE_N" \
      --name-prefix "$NAME_PREFIX" \
      "${EXTRA_ARGS[@]}"
    ;;

  paper)
    run_cmd "$PYTHON" scripts/paper_followup_v11.py \
      --prefix "$NAME_PREFIX" \
      --base-n "$BASE_N" \
      --num-workers "$NUM_WORKERS" \
      "${EXTRA_ARGS[@]}"
    ;;

  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Valid stages: weights, dataset, train, sweep, perclass, ablation, summary, paper" >&2
    exit 1
    ;;
esac
