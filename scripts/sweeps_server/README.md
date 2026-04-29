# `sweeps_server/`

논문 실험 sweep을 stage 단위로 실행하는 wrapper 모음. `_common.sh`가 GPU 메모리·CPU 수를 보고 `server` / `pc` / `minimal` 프로필을 자동 선택합니다. 서버용·PC용 스크립트를 따로 두지 않습니다.

## 한 번에 다 돌리기

```bash
bash scripts/sweeps_server/00_all.sh
```

순서: baseline 재확인 → core 축들 → `color` → `sample_skip` → `backbone` → `logical_train` → `gc` → `bkm_combined` → postprocess.

## 파일 7개

| 파일 | 역할 |
|---|---|
| `00_all.sh` | 위 순서대로 모든 stage 호출 |
| `axis.sh <axis>` | 한 축만 실행. `axis.sh baseline`은 baseline 5-seed 재확인만. |
| `sample_skip.sh` | `--filter_nonfinite_loss=true` 안전 실험 1-seed |
| `backbone.sh` | `weights/`에 있는 모든 `*.pth` (단, `best_model.pth`와 `*.fp16.pth` 제외)를 자동 검출해서 한 번씩 학습. 개수·이름 hardcoded 아님. |
| `logical_train.sh` | member별 logical 데이터셋 생성 + 학습 |
| `bkm_combined.sh` | BKM 값을 한 번에 쌓은 조합 candidate 실행 |
| `_common.sh` | env auto-detect + 공통 헬퍼 (다른 sh가 source) |

## `axis.sh` 사용 가능한 축

`lr, warmup, normal_ratio, per_class, label_smoothing, stochastic_depth, focal_gamma, abnormal_weight, ema, color, allow_tie_save, gc, baseline`

```bash
bash scripts/sweeps_server/axis.sh lr                  # lr 축 sweep
bash scripts/sweeps_server/axis.sh gc --max-launched 1 # gc 축 1 run만
bash scripts/sweeps_server/axis.sh baseline            # baseline 5-seed
```

## 환경 자동 감지 (`_common.sh::detect_profile`)

| 프로필 | 조건 | num_workers | prefetch | max_launched |
|---|---|---:|---:|---:|
| `server` | GPU ≥ 40 GB | 24 | 4 | 0 (무제한) |
| `pc` | GPU ≥ 12 GB | 2 | 2 | 1 |
| `minimal` | 그 외 / GPU 없음 | 0 | 1 | 1 |

`num_workers`는 `cpus-2`로 cap. CLI 플래그(`--num-workers`, `--prefetch-factor`, `--max-launched`)로 항상 덮어쓸 수 있습니다.

## 입출력 (validations/)

| 파일 | 역할 |
|---|---|
| `01_baseline_queue.json` | baseline 5-seed 재확인 명세 (template) |
| `01_baseline_active.json` / `01_baseline_results.{json,md}` | 위 stage 실행 큐 + 결과 |
| `02_sweep_queue.json` | 축별 sweep 명세 (template, 313 run) |
| `02_sweep_active.json` / `02_sweep_results.{json,md}` | 서버 실행 큐 + live 결과 |
| `02_sweep_report.md` / `02_sweep_plots/` | postprocess 종합 리포트 + 축별 plot |
| `03_sample_skip_queue.json` | sample-skip 1-run 명세 |
| `03_sample_skip_active.json` / `03_sample_skip_results.{json,md}` | 위 실행 큐 + 결과 |
| `04_backbone_queue.json` | backbone sweep 명세 (`backbone.sh`가 자동 생성) |
| `04_backbone_active.json` / `04_backbone_results.{json,md}` | 위 실행 큐 + 결과 |
| `05_bkm_combined_queue.json` | BKM combined 명세 (`bkm_combined.sh`가 자동 생성) |
| `05_bkm_combined_active.json` / `05_bkm_combined_results.{json,md}` | 위 실행 큐 + 결과 |
| `run.log` | 통합 실행 로그 |

## logs/ 그룹화

batch 실행은 `--log-dir-group <name>`으로 모든 train.py 출력을 `logs/<name>/<run>/` 아래에 모읍니다 (기본값 `run_<timestamp>`). 단일 학습은 그대로 `logs/<run>/` 아래.

`prepare_server_queue.py`가 template(`*_queue.json`)을 읽고 → tag에 `rawbase_` 붙이고 → 활성 축 외엔 baseline 강제 → 이미 끝난 tag 제외 → `*_active.json`을 만듭니다.

## logs에서 표·plot

```bash
python scripts/generate_log_history_report.py \
  --logs-dir logs \
  --out-prefix validations/log_history_report_rawbase \
  --contains fresh0412_v11
```
