# Server Full Run And Field Data Runbook

서버에서 사람이 직접 칠 기본 명령은 옵션 없이 간다. 필요한 옵션은 wrapper
script 안에 고정한다.

## 1. 코드 받기

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
conda activate anomaly-py311
```

환경이 아직 없으면 한 번만 만든다.

```bash
conda env create -f environment-py311.yml
conda activate anomaly-py311
```

## 2. 전체 재실행

기본 dataset 하나로 전체 stage를 처음부터 다시 돌린다.

```bash
bash scripts/run_full.sh
```

이 wrapper가 내부에서 하는 일:

- `dataset.yaml` 기준 실행
- 기존 `data/`, `images/`, `display/` 삭제 후 재생성
- 필요한 backbone weight 다운로드
- baseline prep 후 `00_all.sh` 전체 stage 실행
- 결과를 `logs/<group>/`, `validations/<group>/`에 저장

결과 위치:

```text
logs/<YYYYMMDD_HHMMSS>_run_paper_dataset/
validations/<YYYYMMDD_HHMMSS>_run_paper_dataset/
```

paper용 dataset x backbone matrix를 돌릴 때:

```bash
bash scripts/run_paper_matrix.sh
```

이 wrapper는 `python download.py`를 먼저 실행하고, 기본 paper dataset x
기본 backbone cross-product를 돌린다.

## 3. 이번 중단 원인

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

## 4. 진행 확인

터미널을 계속 열어둘 수 있으면 그냥 실행 화면을 보면 된다.

다른 터미널에서 확인:

```bash
nvidia-smi
ps -ef | grep -E 'train.py|adaptive_experiment_controller|run_full|00_all'
```

현재 batch 로그:

```bash
tail -f validations/<group>/run.log
```

강제 종료:

```bash
pkill -f adaptive_experiment_controller.py
pkill -f train.py
pkill -f scripts/sweeps_server/00_all.sh
```

SSH를 끊어도 계속 돌리고 싶을 때만 사용:

```bash
nohup bash scripts/run_full.sh > run_full.log 2>&1 &
```

## 5. 현업 CSV 추론

현업 데이터는 합성하지 않는다. 이미 있는 시계열 CSV를 이미지로 렌더링한 뒤
학습된 모델로 분류한다.

입력 CSV 권장 컬럼:

| 역할 | 컬럼 예시 | 비고 |
| --- | --- | --- |
| chart id | `chart_id` | 없으면 `device,step,item` 조합으로 생성 가능 |
| x축 | `time_index`, `timestamp`, `datetime`, `date`, `time` | 자동 감지 |
| 측정값 | `value` | 기본 측정값 컬럼 |
| fleet/member | `eqp_id`, `chamber`, `recipe`, `member`, `member_id`, `tool_id` | 자동 감지 |
| metadata | `device`, `step`, `item` | chart_id가 없을 때 기본 조합 |
| label | `class`, `판정` 등 | 선택. 있으면 검증/추가학습에 사용 |

라벨 없는 현업 CSV:

```bash
bash scripts/run_field_predict.sh fab_export/timeseries.csv logs/<group>/<run>
```

라벨 있는 현업 CSV:

```bash
bash scripts/run_field_predict.sh fab_export/timeseries_labeled.csv logs/<group>/<run> 판정
```

입력 CSV에 이미 `class` 컬럼이 있으면 label 컬럼 인자를 생략해도 된다.

출력:

```text
field_runs/<YYMMDD_HHMMSS>_images/
  model_inputs/
  display/
  manifest.csv
  field_scenarios.csv
  timeseries.csv
  summary.json

field_runs/<YYMMDD_HHMMSS>_predictions/
  predictions.csv
  summary.json
  predictions/normal/
  predictions/abnormal/
```

운영 판정 기준은 wrapper 안에서 `normal_threshold=0.9`로 고정한다.
즉 `p_normal > 0.9`일 때만 normal로 통과시키고, 나머지는 abnormal로 보낸다.

라벨이 있으면 `binary_threshold_report`까지 자동으로 만든다. 현업 labeled
검증도 먼저 binary `FN`, `FP`, abnormal recall부터 본다.

## 6. 현업 라벨로 추가 학습

먼저 라벨 있는 현업 CSV를 추론 wrapper로 렌더링한다.

```bash
bash scripts/run_field_predict.sh fab_export/timeseries_labeled.csv logs/<group>/<run> 판정
```

결과 이미지를 사람이 확인한 뒤 fine-tune:

```bash
bash scripts/run_field_finetune.sh field_runs/<YYMMDD_HHMMSS>_images logs/<group>/<run>
```

이 wrapper는 내부에서 다음 폴더를 사용한다.

```text
field_runs/<YYMMDD_HHMMSS>_images/dev_binary_model_inputs/
  normal/
  abnormal/
```

출력은 `logs/addtrain_*/` 아래에 생긴다. 추가학습 입력은 display 이미지가
아니라 `model_inputs` 계열 이미지다.

## 7. 2-stage를 쓰는 경우

2-stage는 binary gate 모델과 anomaly type 모델이 둘 다 있어야 한다.
이 단계는 모델 경로가 두 개 필요해서 wrapper로 숨기기 어렵다.

```bash
python scripts/two_stage_predict.py \
  --binary-model-run logs/<binary_group>/<binary_run> \
  --type-model-run logs/<type_group>/<type_run> \
  --dataset-dir field_runs/<YYMMDD_HHMMSS>_images \
  --scenarios-file field_runs/<YYMMDD_HHMMSS>_images/field_scenarios.csv \
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
