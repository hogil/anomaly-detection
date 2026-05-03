# Two-Stage Binary Gate + Defect Type Workflow

## 목적

이 프로젝트의 운영 구조는 **1차 binary pass/fail gate**와 **2차 defect type diagnosis**를 분리한다.

- 1차 모델: `normal` / `abnormal` 판정
- 2차 모델: `context`, `drift`, `mean_shift`, `spike`, `standard_deviation` 같은 abnormal type 분류
- 핵심 운영 지표: 1차 binary의 `FN`, `FP`, abnormal recall, threshold sweep
- 핵심 분석 지표: binary error를 `true_class`별로 분해하고, 2차 type classifier가 abnormal sample을 어떤 family로 보는지 확인
- 2차 실행 대상: 1차 binary가 `abnormal`로 예측한 **predicted positive**만. 즉 `TP_abnormal + FP_normal`이며, true label로 route하지 않는다.

2차 모델은 1차에서 `abnormal`로 분기된 chart만 본다. 따라서 runtime에서 1차 `FN`을 직접 구제하지는 못한다. 1차가 `normal`로 통과시킨 실제 abnormal sample은 2차까지 가지 않는다.

## 구조

```text
timeseries.csv + scenarios.csv
        |
        v
trend chart image rendering
        |
        v
Stage 1: binary model
normal vs abnormal
        |
        |-- p_normal > normal_threshold
        |       -> final_pred = normal
        |
        |-- p_normal <= normal_threshold
                -> binary_pred = abnormal
                -> Stage 2: anomaly_type model
                -> final_pred = abnormal/<defect_type>
```

`normal_threshold`는 `p_normal` 기준이다. 예를 들어 `normal_threshold=0.9`이면 `p_normal > 0.9`일 때만 normal로 통과시킨다. 더 높은 threshold는 더 보수적인 gate라서 보통 `FN`은 줄고 `FP`는 늘 수 있다.

Stage 2는 모델이 예측한 positive만 본다. Labeled test에서 true abnormal인지 여부는 routing에 쓰지 않고, 사후 평가에서만 `TP_abnormal`, `FP_normal`, `FN_abnormal`, `TN_normal` bucket을 계산한다. 따라서 Stage 2 평가 대상은 `stage1 predicted positive = binary_pred == abnormal`이다.

## 학습

### 1차 binary gate

```bash
python train.py \
  --config dataset.yaml \
  --mode binary \
  --epochs 20 \
  --batch_size 32 \
  --precision fp16 \
  --normal_ratio 700 \
  --seed 42 \
  --log_dir binary_gate_s42
```

1차 모델은 defect type을 맞히는 모델이 아니다. `normal`과 `abnormal` 사이의 operating boundary를 학습해서 불량 chart를 놓치지 않는 것이 목적이다.

### 2차 anomaly_type classifier

```bash
python train.py \
  --config dataset.yaml \
  --mode anomaly_type \
  --epochs 20 \
  --batch_size 32 \
  --precision fp16 \
  --seed 42 \
  --log_dir type_classifier_s42
```

`anomaly_type`은 normal을 제외한 abnormal class만 학습한다. PatchCore, EfficientAD 같은 normal-only one-class AD가 아니다. 1차 gate 뒤에서 abnormal sample의 defect family를 붙이는 모델이다.

### CPU smoke용 짧은 학습

CPU에서는 전체 학습이 느릴 수 있으므로, 동작 확인만 할 때는 sample cap과 worker를 줄인다.

```bash
python train.py \
  --config dataset.yaml \
  --mode anomaly_type \
  --epochs 1 \
  --batch_size 4 \
  --precision fp32 \
  --max_samples_per_split 20 \
  --num_workers 0 \
  --log_dir type_smoke_cpu
```

이 명령은 성능 검증용이 아니라 학습 경로 smoke용이다.

## 2-stage 추론

구현 파일:

```text
scripts/two_stage_predict.py
```

기본 실행:

```bash
python scripts/two_stage_predict.py \
  --binary-model-run logs/<binary_gate_run> \
  --type-model-run logs/<type_classifier_run> \
  --dataset-dir data \
  --split test \
  --normal-threshold 0.9 \
  --output-dir two_stage_test \
  --device cpu
```

소량 확인:

```bash
python scripts/two_stage_predict.py \
  --binary-model-run logs/<binary_gate_run> \
  --type-model-run logs/<type_classifier_run> \
  --dataset-dir data \
  --split test \
  --limit 20 \
  --normal-threshold 0.9 \
  --output-dir two_stage_test_small \
  --device cpu
```

이미지까지 저장:

```bash
python scripts/two_stage_predict.py \
  --binary-model-run logs/<binary_gate_run> \
  --type-model-run logs/<type_classifier_run> \
  --dataset-dir data \
  --split test \
  --limit 20 \
  --normal-threshold 0.9 \
  --output-dir two_stage_test_images \
  --device cpu \
  --save-display \
  --save-model-inputs
```

## 입력

`--dataset-dir` 아래에 다음 파일이 있어야 한다.

```text
data/
├── timeseries.csv
└── scenarios.csv
```

별도 scenarios 파일을 쓰려면:

```bash
python scripts/two_stage_predict.py \
  --binary-model-run logs/<binary_gate_run> \
  --type-model-run logs/<type_classifier_run> \
  --dataset-dir data \
  --scenarios-file data/scenarios.csv \
  --split test \
  --output-dir two_stage_test
```

`timeseries.csv`가 커도 selected `chart_id`만 chunk로 읽도록 구현되어 있다. `--chunksize`로 chunk 크기를 조정할 수 있다.

## 출력

```text
two_stage_test/
├── two_stage_predictions.csv
├── summary.json
├── display/          # --save-display를 준 경우
└── model_inputs/     # --save-model-inputs를 준 경우
```

주요 CSV 컬럼:

| column | meaning |
| --- | --- |
| `chart_id` | chart 식별자 |
| `true_class` | labeled data의 원래 class |
| `true_binary` | `normal` 또는 `abnormal` |
| `p_normal` | 1차 binary 모델의 normal 확률 |
| `p_abnormal` | 1차 binary 모델의 abnormal 확률 |
| `binary_pred` | 1차 결과 |
| `stage2_ran` | 2차 모델 실행 여부 |
| `stage2_pred` | 2차 defect type 예측 |
| `stage2_confidence` | 2차 예측 확률 |
| `final_pred` | `normal` 또는 `abnormal/<type>` |
| `bucket` | labeled data 기준 `tn_normal`, `fp_normal`, `fn_abnormal`, `tp_abnormal` |

`summary.json`에는 처리 건수, 1차 normal/abnormal 수, 2차 실행 수, `TN/FN/FP/TP`, binary accuracy/recall/F1, 2차 type accuracy가 들어간다.

2차 type accuracy는 `stage2_ran=true`이고 실제 class가 type class 안에 있는 binary true-positive subset에서만 의미가 있다.

## 결과 해석

### 1차 gate 해석

- `fn_abnormal`: 실제 abnormal인데 1차가 normal로 통과시킨 경우. 운영상 가장 위험하다.
- `fp_normal`: 실제 normal인데 1차가 abnormal로 잡은 경우. 검토 비용을 늘린다.
- `tp_abnormal`: 실제 abnormal이고 1차가 abnormal로 잡은 경우. 이 subset만 2차 type classifier로 넘어간다.
- `tn_normal`: 실제 normal이고 1차가 normal로 통과시킨 경우.

### 2차 classifier 해석

2차 classifier의 목적은 pass/fail gate를 대체하는 것이 아니라, 잡힌 abnormal sample을 defect family로 설명하는 것이다.

예시:

```text
binary_pred = abnormal
stage2_pred = mean_shift
final_pred = abnormal/mean_shift
```

이 경우 운영 판정은 이미 1차에서 abnormal로 끝났고, 2차 결과는 리포트, routing, 원인 분석, data-generation feedback에 쓰인다.

### FN/FP를 difficulty 조절로 연결하는 방법

Labeled validation/test에서 `bucket`과 `true_class`를 같이 본다.

| pattern | 해석 | 다음 조치 |
| --- | --- | --- |
| `mean_shift` FN 많음 | normal과 mean-shift gap이 약하거나 target/fleet 차이가 작음 | mean-shift minimum amplitude/floor 증가, target gap 강화, threshold sweep |
| `spike` FN 많음 | local spike가 짧거나 작거나 렌더링에서 묻힘 | spike height/width/floor 증가, local visibility 확인 |
| `drift` FN 많음 | slope change가 normal trend variation과 겹침 | drift slope/span floor 증가, normal augmentation overlap 점검 |
| `context` FN 많음 | target 단독 shape보다 fleet/context 비교가 필요함 | fleet visibility, highlighted member design, target/baseline 표현 점검 |
| normal FP 많음 | normal outlier를 abnormal로 봄 | normal variation/normal ratio 확대, hard normal Grad-CAM 확인, abnormal bias 완화 |

## Smoke 결과 기록

2026-05-03에 CPU smoke를 실행했다.

- `python -m py_compile scripts/two_stage_predict.py`
- 기존 binary checkpoint 1개와 임시 dummy `anomaly_type` checkpoint를 사용
- `--split test --limit 3 --normal-threshold 1.0 --device cpu`
- `two_stage_predictions.csv`와 `summary.json` 생성 확인
- 3건 모두 1차 abnormal 분기 후 2차 실행 확인
- `--save-display --save-model-inputs`도 1건 확인
- `--limit 0` 빈 결과 처리 확인

주의: dummy `anomaly_type` checkpoint는 stage routing과 파일 저장 확인용이다. 2차 type 성능 근거가 아니다. 실제 성능 평가는 `train.py --mode anomaly_type`으로 학습한 run을 넣어 다시 실행해야 한다.

## 실제 pilot 결과 기록

2026-05-03에 실제 학습된 2차 `anomaly_type` run으로 full test 2-stage pilot을 실행했다.

Stage 2 학습 run:

```text
logs/260503_154705_stage2_anomaly_type_pilot2_F0.7084_R0.7240
```

Stage 2 학습 명령:

```bash
python train.py \
  --config dataset.yaml \
  --mode anomaly_type \
  --epochs 6 \
  --batch_size 16 \
  --precision fp16 \
  --max_samples_per_split 1200 \
  --num_workers 0 \
  --freeze_backbone_epochs 6 \
  --best_update_start_single 1 \
  --best_update_start_smoothed 1 \
  --early_stop_start 10 \
  --smooth_window 1 \
  --lr_backbone 0 \
  --lr_head 0.0007 \
  --weight_decay 0 \
  --focal_gamma 0 \
  --log_dir stage2_anomaly_type_pilot2 \
  --no_progress
```

2-stage 추론 명령:

```bash
python scripts/two_stage_predict.py \
  --binary-model-run logs/20260430_135349_pcsafe_00all/260430_135401_fresh0412_v11_rawbase_lr3e5_n700_s42_F0.9967_R0.9967 \
  --type-model-run logs/260503_154705_stage2_anomaly_type_pilot2_F0.7084_R0.7240 \
  --dataset-dir data \
  --split test \
  --normal-threshold 0.9 \
  --output-dir validations/two_stage_pilot2_test \
  --device cuda \
  --chunksize 200000
```

출력:

```text
validations/two_stage_pilot2_test/two_stage_predictions.csv
validations/two_stage_pilot2_test/summary.json
```

1차 binary gate 결과:

| metric | value |
| --- | ---: |
| total | 1500 |
| stage1 predicted normal | 747 |
| stage1 predicted abnormal | 753 |
| Stage 2 evaluated | 753 |
| TN | 746 |
| FN | 1 |
| FP | 4 |
| TP | 749 |
| binary accuracy | 0.9967 |
| abnormal recall | 0.9987 |
| normal recall | 0.9947 |
| binary F1 | 0.9967 |

Type별 1차 error:

| true class | result |
| --- | ---: |
| normal | FP 4 / 750 |
| standard_deviation | FN 1 / 150 |
| context | FN 0 / 150 |
| drift | FN 0 / 150 |
| mean_shift | FN 0 / 150 |
| spike | FN 0 / 150 |

여기서 Stage 2 evaluated `753`은 true abnormal 전체가 아니라 **1차 predicted positive 전체**다. 구성은 `TP_abnormal 749 + FP_normal 4`이다. `standard_deviation` FN 1건은 1차에서 normal로 통과했으므로 Stage 2로 가지 않았다.

2차 type classifier 결과:

- Stage 2 type accuracy on binary TP subset: `0.7303`.
- `context`는 150/150으로 잘 맞았다.
- `drift`는 `mean_shift`로 많이 섞였다: drift true 150건 중 drift 42, mean_shift 97, standard_deviation 11.
- `spike`는 `standard_deviation`과 섞였다: spike true 150건 중 spike 99, standard_deviation 51.
- normal FP 4건도 predicted positive라 Stage 2를 탔고, `mean_shift` 2건, `spike` 2건으로 가짜 defect type이 붙었다. 이 값은 type diagnosis 오염으로 따로 보고해야 한다.

핵심 해석:

- Pass/fail 성능은 1차 binary gate가 결정했다.
- Stage 2는 1차가 잡은 abnormal의 type diagnosis에는 유용하지만, 1차 FN을 runtime에서 rescue하지 않는다.
- 이번 pilot에서 다음 개선 대상은 2차 `drift` vs `mean_shift`, `spike` vs `standard_deviation` confusion이다.
- 1차 binary 난이도 관점에서는 `standard_deviation` FN 1건과 normal FP 4건을 Grad-CAM 또는 원본 trend로 단순 분석하면 된다. 바로 generation rule을 고치기보다, 반복되는 hard sample인지 먼저 봐야 한다.

## 계속 기록 규칙

2-stage나 threshold 관련 run을 새로 실행하면 다음 위치를 함께 갱신한다.

- `docs/two_stage_workflow.md`: 실제 command, run path, output path, threshold, `TN/FN/FP/TP`, `stage2_evaluated`, type confusion headline.
- `docs/summary.md`: 프로젝트 전체 요약에 들어갈 핵심 숫자와 해석.
- `AGENTS.md`: 다음 팀 에이전트가 반드시 지켜야 하는 routing/reporting 규칙이 바뀐 경우.
- `~/.codex/skills/anomaly-paper-evidence/references/current-project-state.md`: 현재 프로젝트 상태와 evidence 기록.
- `~/.codex/skills/anomaly-result-analysis/SKILL.md`: 반복 분석 절차가 바뀐 경우.
- Memory: 중요한 run 결과와 운영 규칙을 짧은 observation으로 저장.

항상 `Stage 2 대상 = predicted positive = binary_pred == abnormal`이라고 명시한다. Type classifier 결과를 true abnormal 전체에 대해 해석하지 않는다.

## 자주 틀리는 해석

- `anomaly_type`은 one-class anomaly detection이 아니다. Normal을 제외한 supervised defect type classifier다.
- 2차 classifier는 1차 `FN`을 runtime에서 구제하지 않는다. 1차가 normal로 보낸 sample은 2차로 가지 않는다.
- `binary > multiclass`를 보편 법칙처럼 쓰면 안 된다. 이 프로젝트의 근거는 운영 gate와 진단 classifier의 역할 분리다.
- 단일 최고 F1만 보지 않는다. `FN`, `FP`, threshold별 confusion, `true_class`별 error breakdown을 같이 봐야 한다.
