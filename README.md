# anomaly-detection

이 레포는 시계열 anomaly 데이터 생성, 이미지 렌더링, 학습, 추론, 실험 오케스트레이션, 논문용 분석 스크립트를 함께 담고 있습니다.

현재 파일 수가 많아서, **처음 열 때는 아래 순서로 보는 것**을 기준으로 잡습니다.

## Start Here

### Core entrypoints

- [train.py](train.py)
  - 메인 학습 엔트리포인트
- [generate_data.py](generate_data.py)
  - 시계열/시나리오 CSV 생성
- [generate_images.py](generate_images.py)
  - `timeseries.csv` + `scenarios.csv`를 학습/표시 이미지로 렌더링
- [inference.py](inference.py)
  - 단일 데이터셋 추론
- [scripts/server_batch_predict.py](scripts/server_batch_predict.py)
  - 여러 제품 폴더를 재귀 스캔해서 일괄 추론, FP/FN/TP/TN 저장

### Main configs

- [dataset.yaml](dataset.yaml)
  - 기본 데이터셋/렌더링 설정
- [config.yaml](config.yaml)
  - 기본 학습 설정
- [configs/](configs)
  - 파생 데이터셋/실험 설정 모음

### Main runbooks

- [HOW_TO_RUN.md](HOW_TO_RUN.md)
- [HOW_TO_RUN_V11.md](HOW_TO_RUN_V11.md)
- [SERVER_SETUP.md](SERVER_SETUP.md)
- [SERVER_RUNBOOK_V11.md](SERVER_RUNBOOK_V11.md)
- [SERVER_BATCH_PREDICT.md](SERVER_BATCH_PREDICT.md)
- [docs/reports/strict_one_factor_latest/README.md](docs/reports/strict_one_factor_latest/README.md)
  - 최신 strict one-factor 결과 snapshot

## Repo map

- [REPO_MAP.md](REPO_MAP.md)
  - 레포 전체 구조, active 영역, generated 영역, legacy clutter 후보
- [scripts/README.md](scripts/README.md)
  - `scripts/` 내부 분류
- [legacy/README.md](legacy/README.md)
  - 루트에서 치운 historical 파일 모음

## Current rule of thumb

- **코드 읽기 시작점**: `train.py`, `generate_data.py`, `generate_images.py`, `inference.py`
- **실험 자동화**: `run_experiments_v11.py`, `scripts/`
- **generated artifacts**: `data_*`, `images_*`, `display_*`, `logs/`, `validations/`
- **historical files**: root 에서 치운 구버전 파일은 `legacy/` 아래에 모아둠

## Immediate cleanup stance

이 레포는 이미 사용자 변경이 많은 dirty worktree 상태이므로, 대규모 이동/삭제보다:

1. 안내 문서 추가
2. generated artifact ignore 강화
3. active path와 legacy path 구분
4. low-risk historical root 파일을 `legacy/`로 이동

을 먼저 적용합니다.
