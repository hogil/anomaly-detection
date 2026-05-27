# Server Full Run And Field Data Runbook

이 문서는 서버에서 repo를 받은 뒤 처음부터 전체 실험을 다시 돌리는 명령과,
현업 CSV를 모델에 태우는 절차만 짧게 모은 runbook이다.

## 1. 서버에서 처음부터 한 번에 돌리기

### 1.1 코드 받기

처음 받는 서버:

```bash
git clone https://github.com/hogil/anomaly-detection.git
cd anomaly-detection
```

이미 받은 서버:

```bash
cd anomaly-detection
git pull --ff-only
```

환경:

```bash
conda env create -f environment-py311.yml
conda activate anomaly-py311
```

이미 env가 있으면:

```bash
conda activate anomaly-py311
```

## 2. 권장 한방 명령

### 2.1 단일 기본 dataset 전체 sweep

clean clone에서 바로 돌릴 때는 `00_all.sh`를 직접 치지 말고 wrapper를 쓴다.
이 명령은 weights, data, images 준비 후 `00_all.sh` 전체 stage를 실행한다.

```bash
nohup bash scripts/all-dataset-backbone.sh \
  --datasets dataset.yaml \
  --reset-data \
  > /tmp/ad_full_dataset_$(date +%Y%m%d_%H%M%S).log 2>&1 &
disown
```

실시간 확인:

```bash
tail -f /tmp/ad_full_dataset_*.log
nvidia-smi
```

결과 위치:

```text
logs/<YYYYMMDD_HHMMSS>_run_paper_dataset/
validations/<YYYYMMDD_HHMMSS>_run_paper_dataset/
```

`--reset-data`는 해당 yaml의 `output.data_dir`, `output.image_dir`,
`output.display_dir`만 repo 안에서 삭제하고 다시 만든다. 기존 data/images를
살리고 싶으면 `--reset-data`를 빼면 된다.

### 2.2 paper용 dataset x backbone matrix

paper main matrix처럼 dataset과 backbone cross-product로 돌릴 때:

```bash
python download.py

nohup bash scripts/all-dataset-backbone.sh -x \
  --reset-data \
  > /tmp/ad_cross_product_$(date +%Y%m%d_%H%M%S).log 2>&1 &
disown
```

`-x`는 기본 4 dataset x 기본 4 backbone cross-product mode다.
cross-product mode는 시작 전에 `weights/*.pth`가 있는지 먼저 확인하므로,
인터넷 되는 서버에서는 `python download.py`를 먼저 실행한다.
폐쇄망 서버에서는 인터넷 머신에서 받은 `weights/` 폴더를 복사한 뒤 실행한다.

특정 dataset만:

```bash
nohup bash scripts/all-dataset-backbone.sh -x \
  --datasets dataset.yaml,dataset1_noise_15.yaml \
  > /tmp/ad_cross_subset.log 2>&1 &
disown
```

특정 GPU만:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash scripts/all-dataset-backbone.sh -x \
  > /tmp/ad_cross_gpu0123.log 2>&1 &
disown
```

## 3. `00_all.sh`를 직접 쓰는 경우

`00_all.sh`는 stage loop용이다. fresh clone에서 weights/data/images 준비까지
보장하려면 위의 `all-dataset-backbone.sh`를 쓰는 것이 안전하다.

prep을 이미 끝낸 같은 group에서 stage만 직접 돌리고 싶을 때:

```bash
GROUP=20260527_120000_run_paper_dataset

bash scripts/sweeps_server/00_all.sh \
  --config dataset.yaml \
  --log-dir-group "$GROUP"
```

처음부터 새 결과로 다시 돌릴 때는 새 group을 쓰면 된다. 직접 group을 주지
않으면 timestamp가 붙은 새 group이 자동 생성된다.

```bash
bash scripts/sweeps_server/00_all.sh --config dataset.yaml
```

단, clean clone이면 `00_all.sh` 직접 실행보다 2.1의 wrapper 명령을 권장한다.

## 4. 이번 `02_sweep_results` 이후 중단 원인

문제는 학습 결과가 아니라 shell stage 전환부였다.

`00_all.sh`가 `02_sweep_results`까지 만든 뒤 stage 13 skip 분기로 넘어가면서
아래 메시지의 괄호가 서버에서 `syntax error near unexpected token '('`로
터졌다.

```bash
echo "[00_all] skip stage 13 (sample_skip)"
```

현재는 괄호 없는 형식으로 고쳤다.

```bash
echo "[00_all] skip stage 13: sample_skip"
echo "[00_all] skip stage 14: backbone_sweep"
echo "[00_all] skip stage 15: logical_train"
```

검증:

```bash
bash -n scripts/sweeps_server/*.sh
bash scripts/sweeps_server/00_all.sh --help
git diff --check
```

## 5. 현업 CSV 추론

현업 데이터는 합성하지 않는다. 이미 있는 시계열을 이미지로 렌더링한 뒤
학습된 모델로 분류한다.

### 5.1 입력 CSV 최소 컬럼

권장 컬럼:

| 역할 | 컬럼 예시 | 비고 |
| --- | --- | --- |
| chart id | `chart_id` | 없으면 `device,step,item` 조합으로 생성 가능 |
| x축 | `time_index`, `timestamp`, `datetime`, `date`, `time` | 자동 감지, 다르면 `--x-col` 지정 |
| 측정값 | `value` | 다르면 `--value-col` 지정 |
| fleet/member | `eqp_id`, `chamber`, `recipe`, `member`, `member_id`, `tool_id` | 자동 감지, 다르면 `--legend-axis` 지정 |
| metadata | `device`, `step`, `item` | chart_id가 없을 때 기본 조합 |
| label | `class`, `판정` 등 | 선택. 있으면 검증/추가학습에 사용 |

### 5.2 라벨 없는 현업 데이터

이미지 생성:

```bash
python scripts/generate_field_images.py \
  --timeseries fab_export/timeseries.csv \
  --out-dir fab_images \
  --model-run logs/<group>/<run>
```

출력 예:

```text
<YYMMDD_HHMMSS>_fab_images/
  model_inputs/
  display/
  manifest.csv
  field_scenarios.csv
  timeseries.csv
  summary.json
```

분류:

```bash
python scripts/predict_images.py \
  --model logs/<group>/<run> \
  --manifest <YYMMDD_HHMMSS>_fab_images/manifest.csv \
  --output-dir fab_predictions \
  --normal-threshold 0.9
```

결과:

```text
fab_predictions/
  predictions.csv
  summary.json
  predictions/normal/
  predictions/abnormal/
```

운영 판정 기준은 `normal_threshold=0.9`다. 즉 `p_normal > 0.9`일 때만
normal로 통과시키고, 나머지는 abnormal로 보낸다.

### 5.3 라벨 있는 현업 데이터 검증

라벨 컬럼이 `판정`이고 값이 `양호`, `정상`, `normal`, `불량`, `abnormal`,
`drift`, `spike` 같은 형태라면:

```bash
python scripts/generate_field_images.py \
  --timeseries fab_export/timeseries_labeled.csv \
  --label-col 판정 \
  --out-dir fab_dev_images \
  --model-run logs/<group>/<run>

python scripts/predict_images.py \
  --model logs/<group>/<run> \
  --manifest <YYMMDD_HHMMSS>_fab_dev_images/manifest.csv \
  --output-dir fab_dev_predictions \
  --normal-threshold 0.9

python scripts/binary_threshold_report.py \
  --predictions fab_dev_predictions/predictions.csv
```

`binary_threshold_report`는 binary `TN/FN/FP/TP`, abnormal recall,
normal recall, F1을 threshold별로 만든다. 현업 labeled 검증도 먼저
binary `FN`, `FP`, abnormal recall부터 본다.

### 5.4 현업 라벨로 추가 학습

라벨 있는 현업 데이터로 이미지를 만들면 binary 추가학습용 폴더가 같이 생긴다.

```text
<YYMMDD_HHMMSS>_fab_dev_images/dev_binary_model_inputs/
  normal/
  abnormal/
```

라벨을 사람이 확인한 뒤 fine-tune:

```bash
python scripts/add_training_from_folders.py \
  --model-run logs/<group>/<run> \
  --image-root <YYMMDD_HHMMSS>_fab_dev_images/dev_binary_model_inputs \
  --epochs 3 \
  --lr 1e-5 \
  --scheduler cosine
```

출력은 `logs/addtrain_*/` 아래에 생긴다. 추가학습 입력은 display 이미지가
아니라 `model_inputs` 계열 이미지여야 한다.

## 6. 현업 데이터에서 2-stage를 쓰는 경우

2-stage는 binary gate 모델과 anomaly type 모델이 둘 다 있어야 한다.

```bash
python scripts/two_stage_predict.py \
  --binary-model-run logs/<binary_group>/<binary_run> \
  --type-model-run logs/<type_group>/<type_run> \
  --dataset-dir <YYMMDD_HHMMSS>_fab_images \
  --scenarios-file <YYMMDD_HHMMSS>_fab_images/field_scenarios.csv \
  --normal-threshold 0.9 \
  --output-dir fab_two_stage \
  --save-display
```

주의:

- Stage 1에서 `p_normal > normal_threshold`이면 normal로 종료한다.
- Stage 2는 Stage 1이 abnormal로 보낸 predicted positive만 본다.
- Stage 2는 Stage 1 binary `FN`을 rescue하지 못한다.
- labeled 검증에서는 Stage 2 type accuracy를 `stage2_ran=true`이고
  true abnormal type이 Stage 2 class list에 있는 subset에서만 해석한다.

현업 CSV에 `chart_id`가 없으면 `generate_field_images.py`가
`field_scenarios.csv`에는 chart_id를 만들지만, 원본 `timeseries.csv`에는
없을 수 있다. 2-stage까지 쓸 계획이면 입력 CSV에 `chart_id`를 명시해 두는
것이 가장 안전하다.

## 7. 중단, 재시작, 확인

프로세스 확인:

```bash
ps -ef | grep -E 'train.py|adaptive_experiment_controller|00_all|all-dataset'
nvidia-smi
```

로그 확인:

```bash
tail -f /tmp/ad_full_dataset_*.log
tail -f validations/<group>/run.log
```

강제 종료:

```bash
pkill -f adaptive_experiment_controller.py
pkill -f train.py
pkill -f scripts/sweeps_server/00_all.sh
```

같은 group에서 이어 돌릴 때는 기존 group을 지정한다.

```bash
bash scripts/sweeps_server/00_all.sh \
  --config dataset.yaml \
  --log-dir-group <existing_group>
```

처음부터 다시 돌릴 때는 group을 새로 쓰면 된다. `--force`는 같은 group의
완료 run까지 다시 돌릴 때만 사용한다.
