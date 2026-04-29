# `sweeps_server/`

논문 실험 sweep wrapper. `_common.sh::detect_profile`이 GPU 메모리·시스템 RAM·CPU 수로 `server`/`pc`/`minimal` 프로필을 자동 선택합니다. 서버용·PC용 별도 스크립트는 두지 않습니다.

## 한 번에 다 돌리기

```bash
bash scripts/sweeps_server/00_all.sh
```

## 파일 — 실행 순서대로 번호 prefix

| 번호 | 파일 | 역할 |
|---|---|---|
| 00 | `00_all.sh` | 아래 모든 stage를 순서대로 호출 |
| 01 | `01_baseline.sh` | baseline 5-seed 재확인 |
| 02 | `02_lr.sh` | lr 축 sweep |
| 03 | `03_warmup.sh` | warmup epoch 축 |
| 04 | `04_normal_ratio.sh` | normal 샘플 수 축 |
| 05 | `05_per_class.sh` | class당 cap 축 |
| 06 | `06_label_smoothing.sh` | label smoothing 축 |
| 07 | `07_stochastic_depth.sh` | stochastic depth 축 |
| 08 | `08_focal_gamma.sh` | focal loss γ 축 |
| 09 | `09_abnormal_weight.sh` | anomaly class 가중치 축 |
| 10 | `10_ema.sh` | EMA decay 축 |
| 11 | `11_allow_tie_save.sh` | val_f1 tie save 허용 축 |
| 12 | `12_color.sh` | trend·fleet 색·alpha 축 |
| 13 | `13_sample_skip.sh` | `--filter_nonfinite_loss=true` 1-seed 안전 실험 |
| 14 | `14_backbone.sh` | `weights/*.pth`(`best_model.pth`·`*.fp16.pth` 제외) 자동 검출해서 한 번씩 학습 |
| 15 | `15_logical_train.sh` | member별 logical 데이터셋 생성 + 학습 |
| 16 | `16_gc.sh` | grad_clip 축 (학습 불안정 위험으로 마지막) |
| 17 | `17_bkm_combined.sh` | 모든 BKM 값 한 번에 적용 + baseline 비교 plot |
| – | `_common.sh` | env auto-detect + 헬퍼 (다른 sh가 source) |
| – | `README.md` | 이 문서 |

## 환경 자동 감지 (`_common.sh::detect_profile`)

| 프로필 | 조건 | num_workers | prefetch | max_launched |
|---|---|---:|---:|---:|
| `server` | GPU ≥ 40 GB **and** RAM ≥ 64 GB | 24 | 4 | 0 (무제한) |
| `pc` | GPU ≥ 12 GB **and** RAM ≥ 16 GB | 2 | 2 | 1 |
| `minimal` | 그 외 / GPU 없음 | 0 | 1 | 1 |

`num_workers`는 `cpus-2`로 cap. CLI 플래그(`--num-workers`, `--prefetch-factor`, `--max-launched`, `--log-dir-group`)로 항상 덮어쓸 수 있습니다.

## 입출력 (validations/)

| 파일 | 역할 |
|---|---|
| `01_baseline_queue.json` / `01_baseline_active.json` / `01_baseline_results.{json,md}` | baseline stage |
| `02_sweep_queue.json` / `02_sweep_active.json` / `02_sweep_results.{json,md}` | 02~12 + 16 axes 통합 결과 (controller live) |
| `02_sweep_report.md` / `02_sweep_plots/` | postprocess 종합 리포트 + 축별 plot |
| `03_sample_skip_queue.json` / `03_sample_skip_results.{json,md}` / `03_sample_skip_plot.png` | 13단계 결과 + 비교 plot |
| `04_backbone_queue.json` / `04_backbone_results.{json,md}` / `04_backbone_plot.png` | 14단계 결과 + 비교 plot |
| `05_bkm_combined_queue.json` / `05_bkm_combined_results.{json,md}` / `05_bkm_combined_plot.png` | 17단계 결과 + 비교 plot |
| `run.log` | 통합 실행 로그 |

## logs/ 그룹화

batch 실행은 자동으로 `--log-dir-group <YYYYMMDD_HHMMSS>_run_paper`을 붙여서 모든 train.py 출력을 `logs/<YYYYMMDD_HHMMSS>_run_paper/<YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/` 형태로 모읍니다 (시각 prefix가 앞에 와서 시간순 sort).

단독 stage 실행은 stage 이름을 group 명에 씁니다: `<timestamp>_backbone`, `<timestamp>_sample_skip`, `<timestamp>_bkm_combined`, `<timestamp>_logical_train`.

`--log_dir`로 받는 값은 `topic`(조건명)이고, 실제 폴더명에는 `train.py`가 자동으로 시작 시각과 best F1/Recall을 붙입니다.

```bash
python scripts/generate_group_report.py --group-dir logs/20260430_120000_run_paper
```

## 참고

- `prepare_server_queue.py`가 template(`*_queue.json`)을 읽고 → tag에 `rawbase_` 붙이고 → 활성 축 외엔 baseline 강제 → 이미 끝난 tag 제외 → `*_active.json`을 만듭니다.
- 단일 학습은 `python train.py --log_dir my_run` — 폴더는 `logs/<YYMMDD_HHMMSS>_my_run_F<f1>_R<recall>/`.
