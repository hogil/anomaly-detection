# Server Batch Predict

## Purpose

여러 제품 폴더 아래에 있는 `timeseries.csv` + `scenarios*.csv` 데이터셋을 재귀 스캔해서:

1. best model 로 예측
2. 예측 이미지 저장
3. `FP/FN/TP/TN` 버킷 저장
4. `predictions.csv`, `fp_list.csv`, `fn_list.csv`, `summary.json` 생성

## Inputs

- 모델 run 디렉터리
  - `best_model.pth`
  - `best_info.json`
- 데이터셋 디렉터리
  - `timeseries.csv`
  - `scenarios.csv` 또는 `scenarios_per_member.csv`

## Main Script

```bash
python scripts/server_batch_predict.py \
  --model-run logs/<run_dir> \
  --input-root /data/products \
  --output-root server_inference \
  --overwrite
```

`scenarios_per_member.csv`를 우선으로 보려면:

```bash
python scripts/server_batch_predict.py \
  --model-run logs/<run_dir> \
  --input-root /data/products \
  --prefer-per-member \
  --output-root server_inference \
  --overwrite
```

모델 입력용 overlay 이미지까지 저장하려면:

```bash
python scripts/server_batch_predict.py \
  --model-run logs/<run_dir> \
  --input-root /data/products \
  --output-root server_inference \
  --save-model-inputs \
  --overwrite
```

## Shell Wrapper

```bash
bash scripts/run_server_batch_predict.sh \
  --model-run logs/<run_dir> \
  --input-root /data/products \
  --output-root server_inference \
  --overwrite
```

## Output Structure

각 데이터셋마다:

```text
server_inference/<dataset_name>/
  predictions/
    normal/
    abnormal/
    fp_normal/
    fn_abnormal/
    tp_abnormal/
    tn_normal/
  model_inputs/                # --save-model-inputs 사용 시
  predictions.csv
  fp_list.csv
  fn_list.csv
  tp_list.csv
  tn_list.csv
  summary.json
```

루트에는 전체 요약:

```text
server_inference/
  summary.csv
  summary.json
```

## Notes

- `true_class` 또는 `class` 컬럼이 있으면 `normal` 대 `abnormal`로 자동 이진화해서 `FP/FN/TP/TN`을 계산합니다.
- `class != normal`은 모두 `abnormal`로 처리합니다.
- `--dry-run`으로 스캔되는 데이터셋 목록만 먼저 확인할 수 있습니다.
