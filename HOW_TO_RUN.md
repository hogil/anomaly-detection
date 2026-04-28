# How To Run

## 1. 데이터 생성

```bash
python generate_data.py --config dataset.yaml --workers 24
```

## 2. 이미지 생성

```bash
python generate_images.py --config dataset.yaml --workers 24
```

`images/`는 모델 입력이고 `display/`는 사람이 보는 확인용 이미지입니다.

## 3. 학습

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

출력:

```text
logs/<run>/
  best_model.pth
  best_info.json
  history.json
  confusion_matrix.png
  confusion_matrix_nt.png
  training_curves.png
  predictions/
```

## 4. best_model 추론

실전처럼 이미 만들어진 trend CSV에서 추론용 이미지만 먼저 만들려면:

```bash
python scripts/generate_inference_images.py \
  --timeseries data/timeseries.csv \
  --scenarios data/scenarios.csv \
  --out-dir inference_inputs
```

출력:

```text
inference_inputs/
  model_inputs/
  display/
  manifest.csv
```

이 스크립트에서 YAML은 렌더링 스타일만 읽습니다. 데이터 위치는 `--timeseries`, `--scenarios`, `--out-dir`로 직접 지정합니다.

```bash
python inference.py \
  --model logs/<run>/best_model.pth
```

## 5. 서버 batch 추론

```bash
python scripts/server_batch_predict.py \
  --model-run logs/<run>
```

## 6. normal/abnormal 폴더로 추가학습

```text
extra_images/
  normal/
  abnormal/
```

```bash
python scripts/add_training_from_folders.py \
  --model-run logs/<run> \
  --image-root extra_images \
  --epochs 3 \
  --lr 1e-5 \
  --scheduler cosine
```

`best_model.pth`에서는 weight만 불러옵니다. 추가학습용 optimizer LR과 scheduler는 위 옵션으로 새로 정합니다. 출력은 `logs/addtrain_*/best_model.pth`, `best_info.json`, `history.json`, `confusion_matrix.png`입니다.

추가학습 폴더에는 display 이미지가 아니라 모델 입력용 이미지를 넣습니다. display 이미지는 사람이 확인하는 용도입니다.

## 7. logs에서 summary table/plot 생성

```bash
python scripts/generate_log_history_report.py \
  --logs-dir logs \
  --out-prefix validations/log_history_report_rawbase \
  --contains rawbase \
  --top-k 30
```

출력:

```text
validations/log_history_report_rawbase.md
validations/log_history_report_rawbase_candidates.csv
validations/log_history_report_rawbase_runs.csv
validations/log_history_report_rawbase_candidate_f1.png
validations/log_history_report_rawbase_fn_fp.png
validations/log_history_report_rawbase_val_f1_curves.png
validations/log_history_report_rawbase_grad_p99_curves.png
```

성능 plot 이미지는 이 스크립트가 직접 생성합니다. `generate_images.py`는 학습/display 이미지 생성용입니다.

## 8. 현재 서버 실험 재개

```bash
bash scripts/sweeps_server/00_all.sh
```

현재 서버 queue는 `lr`와 `warmup`을 먼저 실행합니다. ref 자체의 LR/warmup을 바꾸려면 `validations/paper_refcheck_raw_queue.json`에서 각 seed의 아래 값을 바꿉니다.

```json
"--lr_backbone": "3e-5",
"--lr_head": "3e-4",
"--warmup_epochs": 3
```

단일 run으로 먼저 확인하려면:

```bash
python train.py \
  --config dataset.yaml \
  --mode binary \
  --lr_backbone 3e-5 \
  --lr_head 3e-4 \
  --warmup_epochs 3 \
  --grad_clip 0.0 \
  --smooth_window 1 \
  --normal_ratio 700 \
  --log_dir ref_lrwarm3_probe
```

## 9. FP/FN이 치우칠 때

FP가 너무 많으면 정상 이미지를 anomaly로 많이 잡는 상태입니다. 먼저 `normal_ratio`나 `max_per_class`를 올려 normal 쪽 근거를 늘리고, `abnormal_weight`를 낮추거나 `focal_gamma`를 낮춰 anomaly 쪽 압박을 줄입니다. `label_smoothing`이 크면 결정 경계가 흐려질 수 있으니 낮은 값도 같이 봅니다.

FN이 너무 많으면 불량을 normal로 놓치는 상태입니다. `abnormal_weight`를 올리고, `focal_gamma`나 `label_smoothing` 주변값을 봅니다. `stochastic_depth`는 과적합을 줄여 FN/FP 균형이 좋아지는지 확인하는 축이고, `EMA`는 seed별 진동을 줄이는지 확인하는 축입니다.

한 번에 여러 값을 섞지 말고 한 축만 바꿔서 봅니다. 서버 main sweep은 각 주요 축을 5조건 기준으로 비교합니다.
