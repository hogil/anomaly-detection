# How To Run

이 프로젝트는 **trend 이미지(시계열 차트)** 를 보고 normal/anomaly를 분류합니다. 흐름:

```
dataset.yaml  →  데이터/이미지 생성  →  학습  →  추론
                                       ↑
                  현업 CSV는 generate_inference_images.py 로 끼워넣음
```

목차:
1. 데이터셋 정의 (`dataset.yaml`)
2. 데이터/이미지 생성
3. 단일 학습 (`train.py`)
4. 논문 sweep — 한 번에 전부 돌리기
5. **stage 부분만 돌리기 — 구체 예시**
6. 추론 (`inference.py`) + abnormal/normal 텍스트 리스트
7. 현업 CSV 가져왔을 때
8. 추가 학습 (folder)
9. logs에서 표·plot 만들기
10. Grad-CAM
11. FP/FN 치우칠 때

---

## 1. 데이터셋 정의는 `dataset.yaml` 한 군데

**바꿀 것은 이 파일뿐**입니다 — episode 길이, 노이즈, anomaly 종류·세기, fleet 수, 색·alpha 등 데이터셋 정의가 전부 여기.

각 학습 run은 자기가 본 yaml 스냅샷(`logs/<run>/data_config_used.yaml`)을 따로 저장하므로, `dataset.yaml`을 나중에 바꿔도 옛 학습은 자기 스냅샷으로 재현됩니다.

---

## 2. 데이터/이미지 생성

```bash
python generate_data.py   --config dataset.yaml --workers 24
python generate_images.py --config dataset.yaml --workers 24
```

출력:

```
data/scenarios.csv, data/timeseries.csv      # 시나리오·시계열
images/   <- 모델 입력용
display/  <- 사람이 보는 확인용
```

`run_paper_server_all.sh`가 이미지가 없으면 자동으로 위 두 명령을 호출합니다. 별도로 미리 만들어둘 필요 없음.

---

## 3. 단일 학습 (`train.py`)

가장 단순한 호출:

```bash
python train.py \
  --config dataset.yaml \
  --mode binary \
  --epochs 20 \
  --batch_size 32 \
  --precision fp16 \
  --normal_ratio 700 \
  --seed 42 \
  --log_dir my_run
```

자주 쓰는 옵션:

| 옵션 | 의미 | 예시 |
|---|---|---|
| `--model_name` | timm 백본 이름. `weights/<이름>.pth` 가 있어야 함 | `--model_name swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` |
| `--lr_backbone` / `--lr_head` | LR. ConvNeXtV2-Tiny 기본 `2e-5 / 2e-4` | `--lr_backbone 3e-5 --lr_head 3e-4` |
| `--warmup_epochs` | warmup epoch 수 | `--warmup_epochs 3` |
| `--grad_clip` | gradient clip max_norm. `0` 이면 끔 | `--grad_clip 0.5` |
| `--label_smoothing` | label smoothing | `--label_smoothing 0.02` |
| `--stochastic_depth_rate` | stochastic depth | `--stochastic_depth_rate 0.05` |
| `--focal_gamma` | focal loss γ | `--focal_gamma 2.0` |
| `--abnormal_weight` | anomaly class 가중치 | `--abnormal_weight 1.5` |
| `--ema_decay` | EMA. `0` 이면 끔 | `--ema_decay 0.95` |
| `--allow_tie_save` | val_f1 tie 시도 저장 허용 | `--allow_tie_save` |
| `--filter_nonfinite_loss` | NaN/Inf loss 샘플 step skip | `--filter_nonfinite_loss` |
| `--log_dir_group` | `logs/<group>/<run>/` 형식으로 묶음 (group 명에 시각이 앞에 와야 sort가 깨지지 않음) | `--log_dir_group 20260430_120000_run_paper` |

출력 폴더 (자동으로 시작 시각과 best F1/Recall이 붙음):

```
logs/                                                       # 단일 학습 (그룹 없음)
└── <YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/
       best_model.pth, best_info.json, history.json
       confusion_matrix.png, training_curves.png
       data_config_used.yaml   <- 이 run이 본 데이터셋 정의
       train_config_used.yaml  <- 이 run의 학습 인자 전부
       predictions/

logs/                                                       # 00_all.sh / batch 실행
└── <YYYYMMDD_HHMMSS>_run_paper/                            #   group 폴더 (시각 prefix가 먼저)
        ├── <YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/
        ├── <YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/
        └── ...
```

`--log_dir`로 넣는 값은 **topic(조건명)** 이고, 폴더명에 들어가는 시각·F1/Recall은 train.py가 자동으로 붙입니다 (best 갱신 시마다 F/R 숫자 갱신).

BKM combined 한 번 돌리는 단일 명령 예시 (`05_bkm_combined`와 같은 조합):

```bash
python train.py \
  --config dataset.yaml --mode binary --epochs 20 --batch_size 32 --precision fp16 \
  --normal_ratio 3300 \
  --grad_clip 0.5 \
  --label_smoothing 0.02 \
  --stochastic_depth_rate 0.05 \
  --focal_gamma 2.0 \
  --abnormal_weight 1.5 \
  --ema_decay 0.95 \
  --allow_tie_save \
  --seed 42 --log_dir bkm_combined_s42
```

---

## 4. 논문 sweep — 한 번에 전부 돌리기

```bash
bash scripts/sweeps_server/00_all.sh
```

내부 순서:

| # | stage | 한 줄 설명 |
|---|---|---|
| 0 | weights/dataset 준비 | 없으면 `download.py`, `generate_data.py`, `generate_images.py` 자동 |
| 1 | baseline | 같은 baseline 5-seed 재확인 |
| 2 | axis sweep (core) | lr/warmup/normal_ratio/per_class/label_smoothing/stochastic_depth/focal_gamma/abnormal_weight/ema/allow_tie_save 한 축씩만 |
| 3 | color | trend·fleet 색·alpha 축 |
| 4 | sample_skip | nonfinite-loss 샘플 step-skip 안전 실험 1-seed |
| 5 | backbone | `weights/`에 들어있는 모든 `*.pth`(`best_model.pth`·`*.fp16.pth` 제외)를 자동 검출해서 한 번씩 학습 |
| 6 | logical_train | member별 logical 데이터셋 + 학습 |
| 7 | gc (last) | grad_clip 축 (불안정 위험으로 마지막) |
| 8 | bkm_combined | 8개 BKM 값 다 적용한 candidate 1개 × 5-seed |
| 9 | postprocess | 종합 리포트 + 축별 plot + instability/trend |

GPU 메모리·CPU 수로 자동 프로필 결정:

| 프로필 | 조건 | num_workers | prefetch | 동시 launch |
|---|---|---:|---:|---:|
| `server` | GPU ≥ 40 GB | 24 | 4 | 무제한 |
| `pc` | GPU ≥ 12 GB | 2 | 2 | 1 |
| `minimal` | 그 외 | 0 | 1 | 1 |

`--num-workers`, `--prefetch-factor`, `--max-launched`, `--log-dir-group` 직접 지정하면 위 자동값 덮어씀.

`00_all.sh` 시작할 때 `LOG_DIR_GROUP=<YYYYMMDD_HHMMSS>_run_paper` 한 번 만들어서 모든 stage가 `logs/<YYYYMMDD_HHMMSS>_run_paper/<YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/` 아래로 모입니다 (시각이 앞에 와야 `ls logs/` 시간순 정렬). 단독 stage 실행도 같은 패턴: `13_sample_skip.sh` → `<timestamp>_sample_skip`, `14_backbone.sh` → `<timestamp>_backbone`, `15_logical_train.sh` → `<timestamp>_logical_train`, `17_bkm_combined.sh` → `<timestamp>_bkm_combined`.

---

## 5. stage 부분만 돌리기 — 구체 예시

### 한 축만

```bash
bash scripts/sweeps_server/axis.sh lr            # stage 2의 lr만
bash scripts/sweeps_server/axis.sh gc            # stage 7
bash scripts/sweeps_server/axis.sh baseline      # stage 1
```

`axis.sh`가 받는 이름: `lr, warmup, normal_ratio, per_class, label_smoothing, stochastic_depth, focal_gamma, abnormal_weight, ema, color, allow_tie_save, gc, baseline`.

### stage 1~3만 (baseline + 모든 core 축 + color)

축마다 호출하거나:

```bash
for axis in baseline lr warmup normal_ratio per_class label_smoothing \
            stochastic_depth focal_gamma abnormal_weight ema allow_tie_save color; do
  bash scripts/sweeps_server/axis.sh "$axis"
done
```

또는 한 줄:

```bash
bash scripts/run_paper_server_all.sh \
  --skip-weights --skip-dataset \
  --round1-include-axes lr,warmup,normal_ratio,per_class,label_smoothing,stochastic_depth,focal_gamma,abnormal_weight,ema,allow_tie_save,color \
  --skip-post
```

### stage 0,4만 (데이터 준비 + sample_skip)

```bash
bash scripts/run_paper_server_all.sh --skip-refcheck --skip-round1 --skip-post
bash scripts/sweeps_server/sample_skip.sh
```

### stage 5,8만 (backbone + bkm_combined)

```bash
bash scripts/sweeps_server/backbone.sh
bash scripts/sweeps_server/bkm_combined.sh
```

### stage 9만 (이미 결과 있을 때 표/plot만 갱신)

```bash
bash scripts/run_paper_server_all.sh \
  --skip-weights --skip-dataset --skip-refcheck --skip-round1
```

### 데이터셋 변형 yaml 7종

base + 6개 변형 (×1.15와 ×1.30 두 강도 × 3 종류 변형). 각 yaml의 `output.*_dir` 이 분리돼서 동시 생성도 안 부딪힘:

| yaml | 변형 | 출력 폴더 |
|---|---|---|
| `dataset.yaml` | base | `data/` |
| `dataset1_noise_15.yaml` | noise ×1.15 | `data_noise_15/` |
| `dataset2_noise_30.yaml` | noise ×1.30 | `data_noise_30/` |
| `dataset3_anomaly_15.yaml` | anomaly ×1.15 | `data_anomaly_15/` |
| `dataset4_anomaly_30.yaml` | anomaly ×1.30 | `data_anomaly_30/` |
| `dataset5_all_15.yaml` | noise ×1.15 **and** anomaly ×1.15 | `data_all_15/` |
| `dataset6_all_30.yaml` | noise ×1.30 **and** anomaly ×1.30 | `data_all_30/` |

scaling 적용 항목:
- noise: `gaussian.sigma_range`, `laplacian.b_range`, `correlated.sigma_range` 전부에 factor 곱함
- anomaly: `mean_shift.shift_sigma_range`, `standard_deviation.scale_range`, `spike.magnitude_sigma_range` (+ `min_magnitude_sigma`), `drift.slope_sigma_range` (+ `min_max_drift_sigma`, `visual_floor_sigma`), 그리고 `defect.enforcement.*_floor_sigma` 도 같이 올려서 약한 case 가 floor 에 잘리지 않게 함

### 주말 한 GPU 순차 실행 (7개 yaml 자동, GUI/터미널 끊어도 계속 돔)

```bash
ssh user@server     # GUI 터미널이든 PuTTY든 상관없음

nohup bash -c '
for cfg in dataset.yaml \
           dataset1_noise_15.yaml dataset2_noise_30.yaml \
           dataset3_anomaly_15.yaml dataset4_anomaly_30.yaml \
           dataset5_all_15.yaml dataset6_all_30.yaml; do
  bash scripts/run_paper_server_all.sh --config "$cfg"
done
' > /tmp/weekend.log 2>&1 &
disown

# 이제 SSH 끊거나 GUI 끄거나 노트북 덮어도 학습은 계속 돔
```

진행 확인 / 종료 / 살아있나 체크:

```bash
ssh user@server
tail -f /tmp/weekend.log               # 실시간 진행 (Ctrl+C 로 빠져나옴, 학습은 안 죽음)
ps -ef | grep run_paper                # 살아있는 프로세스 확인
nvidia-smi                             # GPU 사용 중인지 확인
pkill -f run_paper_server_all.sh       # 강제 종료
```

각 yaml 결과는 자동으로 분리된 폴더에 (시간순 정렬):

```
validations/
├── 20260502_120000_run_paper_dataset/
├── 20260502_180000_run_paper_dataset1_noise_15/
├── 20260503_000000_run_paper_dataset2_noise_30/
├── 20260503_060000_run_paper_dataset3_anomaly_15/
├── 20260503_120000_run_paper_dataset4_anomaly_30/
├── 20260503_180000_run_paper_dataset5_all_15/
└── 20260504_000000_run_paper_dataset6_all_30/
logs/
└── (같은 group prefix 로 분리됨)
```

데이터셋별 group 폴더만 따로 정리해서 보고 싶으면:

```bash
python scripts/generate_group_report.py --group-dir logs/20260502_120000_run_paper_dataset
python scripts/generate_group_report.py --group-dir logs/20260502_180000_run_paper_dataset1_noise_15
# ...
```

### 데이터셋 여러 개 **순차** 실행 (한 GPU 자동 자율 실행)

가장 단순:

```bash
bash scripts/run_paper_server_all.sh --config dataset.yaml && \
bash scripts/run_paper_server_all.sh --config dataset1.yaml && \
bash scripts/run_paper_server_all.sh --config dataset2.yaml
```

`&&` = 앞이 성공해야 다음 실행. `;` 로 바꾸면 실패해도 계속.

⚠️ `cmd1 & cmd2` 는 **순차 아님** — `&`는 백그라운드 실행이라 둘이 동시에 시작합니다. 한 GPU면 OOM.

SSH 끊어도 끝까지 자동으로 돌리고 싶으면 `nohup` + 로그:

```bash
nohup bash -c '
for cfg in dataset.yaml dataset1.yaml dataset2.yaml; do
  bash scripts/run_paper_server_all.sh --config "$cfg"
done
' > /tmp/run_seq.log 2>&1 &
disown
```

확인 / 종료:

```bash
tail -f /tmp/run_seq.log               # 실시간 진행
ps -ef | grep run_paper                # 살아있나
pkill -f run_paper_server_all.sh       # 종료
```

각 yaml 결과는 자동으로 `validations/<timestamp>_run_paper_<yaml-stem>/`에 분리되어 쌓입니다 (예: `validations/20260430_120000_run_paper_dataset/`, `validations/20260430_133000_run_paper_dataset1/`).

### 데이터셋 여러 개 **병렬** (다른 yaml + 다른 GPU)

`--config` 로 yaml을 따로 지정하면 됩니다. 기본 `LOG_DIR_GROUP`이 yaml 파일명을 자동으로 포함하므로 같은 초에 launch해도 group 폴더가 충돌하지 않습니다 (`<timestamp>_run_paper_dataset`, `<timestamp>_run_paper_dataset1`처럼).

⚠️ 각 yaml의 `output.data_dir`/`image_dir`/`display_dir`이 **서로 달라야** 데이터 생성도 충돌하지 않습니다 (예: `dataset.yaml` → `data/`, `dataset1.yaml` → `data1/`).

⚠️ 같은 GPU에 두 학습이 들어가면 OOM. GPU 카드별로 분리:

```bash
# GPU 0 → dataset.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_server_all.sh --config dataset.yaml &

# GPU 1 → dataset1.yaml
CUDA_VISIBLE_DEVICES=1 bash scripts/run_paper_server_all.sh --config dataset1.yaml &

wait    # 둘 다 끝날 때까지 대기
```

각각 `validations/<timestamp>_run_paper_dataset/`, `validations/<timestamp>_run_paper_dataset1/` 에 결과가 분리되어 쌓입니다.

GPU 카드가 2개 이상이면 nohup으로 백그라운드 + 로그 분리:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_server_all.sh --config dataset.yaml  > /tmp/run_ds0.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 bash scripts/run_paper_server_all.sh --config dataset1.yaml > /tmp/run_ds1.log 2>&1 &
disown
```

### 같은 group 으로 묶어 돌리기

여러 명령을 한 묶음으로 보고 싶을 때 group 명을 직접 지정:

```bash
GROUP=run_$(date +%Y%m%d_%H%M%S)
bash scripts/sweeps_server/axis.sh baseline --log-dir-group "$GROUP"
bash scripts/sweeps_server/axis.sh lr        --log-dir-group "$GROUP"
bash scripts/sweeps_server/backbone.sh       --log-dir-group "$GROUP"
```

전 stage 결과가 `logs/$GROUP/<run>/` 아래로 모입니다.

### 1 run만 검증 (학습 launch 안 함)

```bash
bash scripts/sweeps_server/backbone.sh --prepare-only          # queue/active만 만듦
bash scripts/sweeps_server/axis.sh lr --max-launched 1         # lr 축에서 1 run만
```

---

## 6. 추론 (`inference.py`) + 텍스트 리스트

```bash
python inference.py \
  --model logs/<group>/<run>/best_model.pth \
  --output_dir my_inference
```

출력 폴더는 시각 prefix 자동 부여: `<YYMMDD_HHMMSS>_my_inference/` (덮어쓰기 방지, ls 시간순 정렬):

```
<YYMMDD_HHMMSS>_my_inference/
├── abnormal/             <- 불량 판정 display 이미지
├── normal/               <- 정상 판정 display 이미지
├── predictions.csv       <- 모든 chart, p_abnormal, predicted, p_normal
├── predictions.txt       <- 통합 텍스트 (ABNORMAL 위, NORMAL 아래)
├── abnormal_list.txt     <- 불량 chart_id 목록만
└── normal_list.txt       <- 정상 chart_id 목록만
```

`abnormal_list.txt` / `normal_list.txt` 형식:

```
ch_09100   p_abn=0.9991   ch_09100.png
ch_09572   p_abn=0.0017   ch_09572.png
```

특정 split만 보고 싶으면:

```bash
python inference.py --model logs/<run>/best_model.pth --split test
```

서버 batch 추론 (모델 폴더 통째로):

```bash
python scripts/server_batch_predict.py --model-run logs/<group>/<run>
```

---

## 7. 현업 CSV 가져왔을 때

현업 `timeseries.csv` + `scenarios.csv` 두 개가 있다고 가정하고, 그걸로 추론까지 가는 **전체 흐름**:

```bash
# (1) 현업 CSV → 모델 입력 이미지 생성
python scripts/generate_inference_images.py \
  --timeseries fab_export/timeseries.csv \
  --scenarios  fab_export/scenarios.csv \
  --out-dir    inference_inputs

# 출력:
#   inference_inputs/model_inputs/   <- 모델 입력용 이미지
#   inference_inputs/display/        <- 사람 확인용
#   inference_inputs/manifest.csv    <- chart_id ↔ 파일 매핑

# (2) 학습된 모델로 분류 + 텍스트 리스트
python inference.py \
  --model logs/<group>/<run>/best_model.pth \
  --data_dir inference_inputs \
  --output_dir fab_results

# 출력 <YYMMDD_HHMMSS>_fab_results/:
#   abnormal/  normal/  predictions.csv  predictions.txt
#   abnormal_list.txt  normal_list.txt
```

`generate_inference_images.py` 는 yaml에서 렌더 스타일만 읽고, 데이터 위치는 `--timeseries`, `--scenarios`, `--out-dir` 로 직접 받습니다. 즉 `dataset.yaml` 안 건드리고 현업 데이터 그대로 처리.

---

## 8. normal/abnormal 폴더로 추가 학습

이미 분류된 폴더가 있을 때 fine-tune:

```
extra_images/
  normal/
  abnormal/
```

```bash
python scripts/add_training_from_folders.py \
  --model-run logs/<group>/<run> \
  --image-root extra_images \
  --epochs 3 --lr 1e-5 --scheduler cosine
```

`best_model.pth`에서 weight만 불러와서 fine-tune. 출력은 `logs/addtrain_*/`. 추가 학습 입력은 **모델 입력용 이미지** (display 아님).

---

## 9. logs에서 표·plot 만들기

```bash
python scripts/generate_log_history_report.py \
  --logs-dir logs \
  --out-prefix validations/log_history_report_rawbase \
  --contains rawbase \
  --top-k 30
```

`logs/<group>/<run>/history.json` 까지 자동으로 추적합니다. 특정 group만 보고 싶으면 `--contains 20260430_120000_run_paper` 처럼 지정.

출력: markdown / CSV / PNG (candidate F1 막대, val_f1 곡선, grad p99 곡선, FN/FP 산점).

---

## 10. Grad-CAM

```bash
python scripts/gradcam_report.py \
  --model-run logs/<group>/<run> \
  --image-root images/test \
  --out-dir validations/gradcam_probe \
  --include-classes normal,mean_shift,standard_deviation,spike,drift,context \
  --limit-per-class 6 \
  --save-heat-only \
  --heat-threshold 0.0 --heat-min-alpha 0.18
```

CAM 은 모델 근거 위치를 보여주는 것이지 실제 anomaly 위치가 아닙니다. 후처리 룰로는 쓰지 않음.

FP/FN만 따로 보고 싶으면 `gradcam_error_report.py`.

---

## 11. FP / FN 한쪽으로 치우칠 때

| 현상 | 우선 시도 |
|---|---|
| FP가 많다 (정상을 anomaly로 잡음) | `normal_ratio`↑ 또는 `--max_per_class`↑, `abnormal_weight`↓ 또는 `focal_gamma`↓ |
| FN이 많다 (불량을 정상으로 놓침) | `abnormal_weight`↑, `focal_gamma`↑, `label_smoothing` 주변값 |

한 번에 여러 축 동시에 바꾸지 말고 한 축씩만 비교 (이게 `02_sweep_queue.json` 의 핵심 원칙).

---

## 결과 파일 (`validations/`)

template 큐는 batch끼리 공유하므로 root에:

| 파일 | 뜻 |
|---|---|
| `validations/01_baseline_queue.json` | baseline 5-seed 입력 |
| `validations/02_sweep_queue.json` | 축별 sweep 입력 |
| `validations/03_sample_skip_queue.json` | sample-skip 1-run 입력 |

**출력은 batch마다 분리**돼서 `validations/<group>/` (group = `<YYYYMMDD_HHMMSS>_run_paper`):

| 파일 (`<group>/` 안) | 뜻 |
|---|---|
| `01_baseline_active.json` / `_results.{json,md}` | baseline 실행 큐 + 결과 |
| `02_sweep_active.json` / `_results.{json,md}` | sweep 실행 큐 + live 결과 |
| `02_sweep_report.md` / `02_sweep_plots/` | postprocess 종합 리포트 + 축별 plot |
| `03_sample_skip_active.json` / `_results.{json,md}` / `_plot.png` | sample-skip + 비교 plot |
| `04_backbone_queue.json` / `_active.json` / `_results.{json,md}` / `_plot.png` | backbone + 비교 plot |
| `05_bkm_combined_queue.json` / `_active.json` / `_results.{json,md}` / `_plot.png` | BKM combined + 비교 plot |
| `15_logical_train_queue.json` / `_results.{json,md}` | logical train |
| `instability_cases.{csv,json,md}` / `prediction_trend_latest.*` | postprocess 분석 |
| `run.log` | 이 batch 의 통합 실행 로그 |

여러 번 `00_all.sh` 돌리면 group 폴더가 시간순으로 정렬되고 서로 안 덮어씁니다.
