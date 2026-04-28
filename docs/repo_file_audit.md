# GitHub 파일 감사

_기준: sweep launcher 정리 후 tracked file tree._

## 결론

- 현재 GitHub에 반드시 보여야 하는 핵심 파일은 root entrypoint, `src/`, active config, server/predict script, `docs/summary.md`입니다.
- `docs/reports/` 중첩 리포트는 제거했고, 최신 결과는 `docs/summary.md`와 `docs/plots/`로 바로 접근합니다.
- `legacy/` 아래 파일은 지금 삭제하지 않고 historical archive로 유지합니다.
- `configs/datasets/dataset_v11_202604*.yaml`처럼 timestamp가 붙은 과거 dataset snapshot은 실행에는 필요하지 않은 것이 많습니다. 다만 재현성 확인 가능성이 있어서 이번 커밋에서는 삭제하지 않고 “정리 후보”로 표시합니다.
- `scripts/sweeps_server/`는 current paper pipeline entrypoint만 남겼고, 과거 server/laptop sweep launcher는 각 `legacy/`로 이동했습니다.
- 로컬에는 untracked/modified 실험 파일이 많습니다. 이번 audit은 GitHub `main/master`에 tracked 된 파일만 대상으로 했습니다.

## 유지: 핵심 진입점

- `README.md`: GitHub 첫 화면 안내.
- `REPO_MAP.md`: active/generated/legacy 구분.
- `.gitignore`: generated dataset, image, logs, weights, validations 제외 규칙.
- `requirements.txt`: 실행 의존성.
- `train.py`: 학습 entrypoint.
- `generate_data.py`: tabular 시계열/시나리오 생성.
- `generate_images.py`: 학습/display image 생성.
- `inference.py`: 단일 추론.
- `run_experiments_v11.py`: v11 실험 실행 entrypoint.
- `download.py`: 보조 다운로드 유틸. 핵심은 아니지만 작고 독립적이라 유지.

## 유지: 실행 문서

- `HOW_TO_RUN.md`
- `HOW_TO_RUN_V11.md`
- `SERVER_SETUP.md`
- `SERVER_RUNBOOK_V11.md`
- `SERVER_BATCH_PREDICT.md`
- `CLI.md`
- `CLAUDE.md`

## 유지: 현재 GitHub 문서

- `docs/README.md`: docs index.
- `docs/summary.md`: 최신 strict one-factor 결과 해석, summary, 전체 표, plot link.
- `docs/round2_summary.md`: round-2 진행 현황.
- `docs/plots/*.png`: GitHub에서 바로 볼 수 있는 축별 plot.
- `docs/repo_file_audit.md`: 이 파일.

## 유지: active config

- `dataset.yaml`
- `config.yaml`
- `configs/datasets/latest.yaml`
- `configs/datasets/v11.yaml`
- `configs/datasets/v9mid3.yaml`
- `configs/datasets/v9mid4.yaml`
- `configs/datasets/v9mid5.yaml`
- `configs/datasets/v9mid6.yaml`
- `configs/datasets/v9mid7_harder.yaml`
- `configs/datasets/dataset_v9mid7_harder_20260410_112102.yaml`
- `configs/train/winning.yaml`

## 정리 후보: dataset snapshot config

아래 파일들은 tracked 되어 있지만 대부분 timestamp snapshot입니다. 현재 코드의 기본 진입점에서는 직접 참조되지 않았습니다. 재현성용 archive로 `legacy/configs/dataset_snapshots/` 이동을 검토할 수 있습니다.

- `configs/datasets/dataset_v11_20260410_223333.yaml`
- `configs/datasets/dataset_v11_20260410_232104.yaml`
- `configs/datasets/dataset_v11_20260410_232240.yaml`
- `configs/datasets/dataset_v11_20260410_232345.yaml`
- `configs/datasets/dataset_v11_20260411_020404.yaml`
- `configs/datasets/dataset_v11_20260411_060501.yaml`
- `configs/datasets/dataset_v11_20260411_072621.yaml`
- `configs/datasets/dataset_v11_20260411_081247.yaml`
- `configs/datasets/dataset_v11_20260411_090542.yaml`
- `configs/datasets/dataset_v11_20260411_091433.yaml`
- `configs/datasets/dataset_v11_20260411_093902.yaml`

## 유지: source tree

- `src/__init__.py`
- `src/data/__init__.py`
- `src/data/baseline_generator.py`
- `src/data/defect_synthesizer.py`
- `src/data/image_renderer.py`
- `src/data/scenario_generator.py`
- `src/models/__init__.py`
- `src/models/focal_loss.py`
- `src/utils/__init__.py`

## 유지: active scripts

- `scripts/README.md`
- `scripts/server_batch_predict.py`
- `scripts/run_server_batch_predict.sh`
- `scripts/validate_dataset.py`
- `scripts/generate_strict_one_factor_report.py`
- `scripts/publish_strict_report.py`
- `scripts/build_warmup_lr_followup_queue_20260428.py`
- `scripts/paper_followup_v11.py`
- `scripts/run_v11_pipeline.ps1`
- `scripts/run_v11_pipeline.sh`
- `scripts/run_v11d_matrix.py`

## 정리 후보: old analysis scripts

아래 파일들은 0408 계열 또는 특정 과거 분석용입니다. 삭제보다는 `legacy/scripts/` 이동 후보입니다.

- `scripts/aggregate_results_0408.py`
- `scripts/compare_conditions_0408.py`
- `scripts/stability_and_worst_0408.py`

## Sweep launchers

- `scripts/sweeps_server/*.sh`: current paper server pipeline wrappers.
- `scripts/sweeps_server/legacy/*.sh`: old broad server grid sweeps.
- `scripts/sweeps_laptop/legacy/*.sh`: old laptop exploration sweeps.

현재 서버 실험 진입점은 `scripts/sweeps_server/`입니다. `legacy/` 아래 파일은 실행 기록 보존용입니다.

## 유지: historical archive

- `legacy/README.md`
- `legacy/configs/config_backup_v9mid7h.yaml`
- `legacy/docs/EXPERIMENTS.md`
- `legacy/docs/ema_design.md`
- `legacy/inference/nt_inference.py`
- `legacy/run_scripts/run_full_matrix.sh`
- `legacy/run_scripts/run_pipeline.sh`
- `legacy/run_scripts/run_sweep_2gpu.sh`
- `legacy/runners/run_experiments.py`
- `legacy/train_variants/train_tie.py`
- `legacy/train_variants/train_v8era.py`

## 다음 정리 제안

1. `configs/datasets/dataset_v11_202604*.yaml`를 `legacy/configs/dataset_snapshots/`로 이동.
2. `scripts/aggregate_results_0408.py`, `compare_conditions_0408.py`, `stability_and_worst_0408.py`를 `legacy/scripts/`로 이동.
3. sweep launcher 43개는 현재 유지하되, 실제 사용 중인 것만 active로 남기고 나머지는 archive.
4. 로컬 untracked 실험 스크립트는 한 번에 올리지 말고, 목적별로 검토 후 필요한 것만 추가.
