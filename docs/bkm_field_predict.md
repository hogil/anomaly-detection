# BKM 현업 CSV Predict

서버에서 `all-dataset-backbone.sh -x`를 모두 돌린 뒤, Best Known Method(BKM) 모델로 현업 CSV를 판정하는 방법만 정리한다.

현업 predict CSV에는 보통 `양호/불량` 정답 컬럼이 없다. 없어도 정상이다. 이 경우 `predicted`, `p_normal`, `p_abnormal`만 나오고, `FN/FP/F1` 같은 성능 지표는 계산하지 않는다.

## 1. BKM Model Run 고르기

global BKM 모델을 운영 기본 후보로 쓴다. 서버에서 먼저 후보를 확인한다.

```bash
python - <<'PY'
from pathlib import Path
import json

rows = []
for ckpt in Path("logs").rglob("best_model.pth"):
    text = ckpt.as_posix()
    if "all_dataset_backbone" not in text or "bkm_global" not in text:
        continue
    info_path = ckpt.parent / "best_info.json"
    if not info_path.exists():
        continue
    info = json.loads(info_path.read_text(encoding="utf-8"))
    f1 = info.get("test_f1", -1)
    rows.append((float(f1), ckpt.parent.as_posix()))

for f1, run_dir in sorted(rows, reverse=True):
    print(f"{f1:.4f}\t{run_dir}")
PY
```

쓸 run 폴더를 하나 정해서 변수로 잡는다.

```bash
MODEL_RUN="logs/<all_dataset_backbone>/<...>_bkm_global/<run_folder>"
```

반드시 이 파일들이 있어야 한다.

```text
$MODEL_RUN/best_model.pth
$MODEL_RUN/best_info.json
$MODEL_RUN/data_config_used.yaml
```

특정 dataset에 맞춘 BKM을 쓰려면 `bkm_global` 대신 `bkm_dataset_...` 후보를 고른다. 특정 dataset-backbone cell 전용 모델을 쓰려면 각 cell의 `05_bkm_combined` 결과 run을 쓴다.

## 2. 현업 CSV 준비

정답 컬럼은 필요 없다. 최소 컬럼은 다음이다.

| 역할 | 컬럼 예시 |
|---|---|
| chart id | `chart_id` |
| x축 | `time_index`, `timestamp`, `datetime`, `date`, `time` 중 하나 |
| 측정값 | `value` |
| member/fleet | `eqp_id`, `chamber`, `recipe`, `member`, `member_id`, `tool_id` 중 하나 |

`chart_id`가 없으면 `device,step,item` 조합으로 만들 수 있다. 그 경우 CSV에 `device`, `step`, `item`이 있어야 한다.

예시:

```csv
chart_id,timestamp,eqp_id,value
chart_001,2026-06-06 10:00:00,EQP_A,12.3
chart_001,2026-06-06 10:01:00,EQP_A,12.5
chart_001,2026-06-06 10:00:00,EQP_B,11.9
chart_001,2026-06-06 10:01:00,EQP_B,12.1
```

## 3. 한 번에 실행

라벨 없는 현업 CSV:

```bash
bash scripts/run_field_predict.sh fab_export/timeseries.csv "$MODEL_RUN"
```

이 wrapper가 내부에서 다음을 순서대로 한다.

```text
1. scripts/generate_field_images.py
   현업 CSV -> model input image + manifest.csv

2. scripts/predict_images.py
   best_model.pth -> normal/abnormal 판정
```

## 4. 결과 위치

실행하면 `field_runs/` 밑에 이미지 폴더와 예측 폴더가 생긴다.

```text
field_runs/<YYMMDD_HHMMSS>_images/
├── model_inputs/
├── display/
├── manifest.csv
├── field_scenarios.csv
├── timeseries.csv
└── summary.json

field_runs/<YYMMDD_HHMMSS>_predictions/
├── predictions.csv
├── summary.json
└── predictions/
    ├── normal/
    └── abnormal/
```

최종 판정은 이 파일을 본다.

```text
field_runs/<YYMMDD_HHMMSS>_predictions/predictions.csv
```

중요 컬럼:

```text
chart_id
highlighted_member
predicted
p_normal
p_abnormal
normal_threshold
image_path
copied_image
```

판정 기준:

```text
p_normal > normal_threshold  -> normal
그 외                         -> abnormal
```

현재 wrapper는 `normal_threshold=0.9`로 predict한다. 즉 `p_normal > 0.9`일 때만 양호로 통과시키고, 나머지는 불량 후보로 보낸다.

## 5. 수동 2단계 실행

wrapper 대신 직접 나눠서 돌리고 싶을 때만 쓴다.

```bash
python scripts/generate_field_images.py \
  --timeseries fab_export/timeseries.csv \
  --out-dir field_runs/manual_images \
  --model-run "$MODEL_RUN" \
  --no-timestamp
```

```bash
python scripts/predict_images.py \
  --model "$MODEL_RUN" \
  --manifest field_runs/manual_images/manifest.csv \
  --output-dir field_runs/manual_predictions \
  --normal-threshold 0.9 \
  --overwrite
```

## 6. 라벨이 있는 CSV인 경우

현업 검증용으로 `판정` 같은 정답 컬럼이 있으면 세 번째 인자로 준다.

```bash
bash scripts/run_field_predict.sh fab_export/timeseries_labeled.csv "$MODEL_RUN" 판정
```

라벨이 있으면 threshold report까지 자동으로 만들어진다. 그래도 운영 predict와 구분해서 해석한다. 라벨 없는 production CSV에서는 `FN/FP/F1`을 계산하지 않는다.
