# Repo Map

이 문서는 현재 레포를 **active**, **generated**, **historical**로 나눠서 보는 기준입니다.

## 1. Active core

이 영역이 지금 실제 개발과 운영의 중심입니다.

### Root core files

- `train.py`
- `generate_data.py`
- `generate_images.py`
- `inference.py`
- `run_experiments_v11.py`
- `dataset.yaml`
- `config.yaml`

### Source tree

- `src/`
  - 데이터 생성/렌더링/모델 보조 코드

### Active script hub

- `scripts/`
  - 배치 추론
  - 분석 리포트
  - 실험 큐/오케스트레이션
  - dataset helper

### Config hub

- `configs/`
  - 파생 데이터셋/실험용 설정

## 2. Generated / output-heavy

이 영역은 대부분 재생성 가능 산출물입니다. GitHub에서 주 탐색 대상으로 보면 안 됩니다.

- `data/`
- `data_*`
- `data_per_member_*`
- `images/`
- `images_*`
- `images_per_member_*`
- `display/`
- `display_*`
- `display_per_member_*`
- `logs/`
- `weights/`
- `validations/`
- `preview/`
- `experiments/`
- `inference_output/`
- `server_inference*/`

원칙:
- 코드 리뷰 대상이 아님
- 필요 시 대표 산출물만 문서에서 링크
- 대부분 `.gitignore` 대상

## 3. Historical / clutter candidates

현재 메인 경로가 아닌 파일들입니다. 당장 이동/삭제는 안 했지만, GitHub에서 혼란을 만드는 영역입니다.

### Top-level historical experiments

아직 로컬에 남아 있을 수 있지만, GitHub 정리 기준에서는 historical 후보로 봅니다.

### Old train variants

- `legacy/train_variants/train_tie.py`
- `legacy/train_variants/train_v8era.py`

### Old chain/orchestrator files

- `legacy/run_scripts/run_pipeline.sh`
- `legacy/run_scripts/run_full_matrix.sh`
- `legacy/run_scripts/run_sweep_2gpu.sh`
- `legacy/runners/run_experiments.py`

### Old or backup configs

- `config_2k.yaml`
- `config_v2.yaml`
- `legacy/configs/config_backup_v9mid7h.yaml`

### Other legacy files already moved

- `legacy/docs/EXPERIMENTS.md`
- `legacy/docs/ema_design.md`
- `legacy/inference/nt_inference.py`

## 4. Practical reading order

GitHub에서 처음 볼 때:

1. `README.md`
2. `train.py`
3. `generate_data.py`
4. `generate_images.py`
5. `inference.py`
6. `scripts/README.md`
7. `configs/`
8. `src/`

그 다음에만 개별 분석 스크립트와 과거 실험 파일을 봅니다.

## 5. Recommended next cleanup moves

이건 아직 적용 안 했고, 다음 정리 단계 후보입니다.

1. root 에 남아 있는 추가 historical 파일을 `legacy/`로 더 이동
2. `config_2k.yaml`, `config_v2.yaml`까지 `legacy/configs/`로 이동
3. `download.py` 같은 비핵심 루트 유틸 정리
4. `docs/` 아래에 paper/report 문서, runbook 문서, design note를 분리

지금은 dirty worktree를 깨지 않기 위해 **low-risk historical root 파일만 먼저 이동**했고, 나머지 큰 이동은 보류한 상태입니다.
