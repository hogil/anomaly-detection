#!/usr/bin/env bash
# 서버 wrapper — 실제 작업(데이터 생성 -> 이미지 렌더 -> 학습)은
# scripts/train_new_dataset.py 가 전부 한다. 이 파일은 환경변수 편의용이다.
#
#   ./scripts/run_new_dataset_server.sh validations/<TS>_all_dataset_backbone
#   ./scripts/run_new_dataset_server.sh validations/<TS>_all_dataset_backbone \
#        configs/datasets/dataset_v24.yaml
#
# python 으로 직접 돌려도 완전히 같다:
#   python scripts/train_new_dataset.py --validation validations/<TS>_all_dataset_backbone
#
# 인자
#   $1  예전 sweep 폴더 (validations/ 또는 logs/ 어느 쪽이든)  [필수]
#   $2  데이터셋 yaml   (기본 configs/datasets/dataset_v24.yaml)
#
# 환경변수
#   BACKBONE=convnext_tiny.dinov3_lvd1689m   학습 backbone
#   WORKERS=0                                생성 병렬 worker (0=auto)
#   SEED=42                                  학습 seed
#   EXTRA="--batch_size 96"                  train.py 로 그대로 넘길 옵션
#   DRY_RUN=1                                명령만 출력
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="${1:-}"
PY="${PYTHON:-python}"

if [[ -z "$SOURCE" ]]; then
  sed -n '2,23p' "$0"
  echo "ERROR: sweep 폴더를 인자로 주세요" >&2
  exit 2
fi

ARGS=(--validation "$SOURCE")
[[ -n "${2:-}" ]]            && ARGS+=(--config "$2")
[[ -n "${BACKBONE:-}" ]]     && ARGS+=(--backbone "$BACKBONE")
[[ -n "${WORKERS:-}" ]]      && ARGS+=(--workers "$WORKERS")
[[ -n "${SEED:-}" ]]         && ARGS+=(--seed "$SEED")
[[ -n "${DRY_RUN:-}" ]]      && ARGS+=(--dry-run)

# shellcheck disable=SC2086
exec "$PY" scripts/train_new_dataset.py "${ARGS[@]}" ${EXTRA:+-- $EXTRA}
