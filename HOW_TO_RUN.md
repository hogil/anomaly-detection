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

```bash
python inference.py \
  --model logs/<run>/best_model.pth
```

## 5. 서버 batch 추론

```bash
python scripts/server_batch_predict.py \
  --model-run logs/<run>
```

## 6. logs에서 summary table/plot 생성

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

## 7. 현재 서버 실험 재개

```bash
bash scripts/sweeps_server/00_all.sh
```
