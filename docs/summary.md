# 실험 요약

## 실험 방식

- 같은 baseline에서 한 번에 하나의 축만 바꾸는 strict one-factor 실험.
- 기본 seed 5개: `42, 1, 2, 3, 4`. 성능은 F1, FN, FP, 완료 seed 수로 봅니다.
- baseline은 `fresh0412_v11_refcheck_raw_n700` (raw smoothing, `grad_clip=0`, `label_smoothing=0`, `NT=0.9`).
- 실행: `bash scripts/sweeps_server/00_all.sh` 한 줄. GPU 메모리·CPU 수로 `server`/`pc`/`minimal` 프로필을 자동 선택합니다.

## 학습 이미지 예시

학습 데이터는 `normal`과 불량 class별 이미지로 구성합니다. 모델 입력은 training image이고, display image는 같은 sample을 사람이 확인하기 쉽게 축/legend/색을 붙인 렌더링입니다.

### Training Image
![training images by class](images/sample_overview_train.png)

### Display Image
![display images by class](images/sample_overview_display.png)

### Legend Axis Image

같은 `device/step/item` group 의 chart 라도 `legend_axis` 를 `eqp_id`, `chamber`, `recipe` 로 바꾸면 fleet member 와 legend 가 달라지므로 별도 이미지로 생성합니다. 각 column 은 다른 member 축으로 그린 같은 chart 의 변형 — 왼쪽 `eqp_id` (장비 단위), 가운데 `chamber` (챔버 단위), 오른쪽 `recipe` (레시피 단위). `highlighted_member` 는 빨강으로 강조되는 member 이고, 회색 가로 기준선이 `target` 값.

![display images by legend_axis](images/sample_overview_legend_axis.png)

불량 class는 `mean_shift`, `standard_deviation`, `spike`, `drift`, `context` 다섯 종류이고, 각 이미지는 해당 class label로 학습됩니다.

## Logical Member Attribution Example

같은 `legend_axis` chart를 member별 class 판단 이미지로 확장합니다. **불량인 EQP를 highlight 한 이미지만 anomaly class**, **나머지 EQP를 highlight 한 이미지는 normal class** 로 학습됩니다 (family 전체 이상 감지가 아니라 highlighted_member 단위 label).

같은 chart `ch_09100` 의 5개 EQP 이미지를 한 장에:

![logical member class examples](images/logical_member_targets_ch09100.png)

회색 점들은 같은 `legend_axis` 안의 비교 fleet, 컬러 점들이 highlighted_member 의 trend. class 텍스트는 normal=검정, anomaly=빨강. 위 예시에서는 `CH_C` 만 anomaly, 나머지 4 EQP 는 normal. 같은 chart 데이터에서도 어떤 EQP를 highlight 하느냐에 따라 label 이 달라지므로 한 chart 가 EQP 수만큼의 학습 샘플을 만듭니다. 5 EQP 면 5 sample, 4 EQP 면 4 sample.

## Grad-CAM / Postprocess Check

Grad-CAM의 heat는 실제 anomaly 위치가 아니라 `abnormal` logit에 기여한 모델 근거 위치입니다. 넓은 불량은 heat도 넓게 퍼질 수 있고, `spike` 같은 국소 패턴은 더 좁게 잡히는 경향이 있습니다. 좌측 불량과 우측 정상의 대비 때문에 우측에 heat가 생기기도 해서, CAM 위치만으로 left/right defect를 판정하지 않습니다.

class별 6행 sample 6열로 원본 trend 이미지 위에 CAM colormap을 반투명으로 얹은 예시입니다. 빨강은 큰 CAM 값, 파랑도 같이 표시해서 CAM이 넓게 퍼지는지 확인.

![class Grad-CAM overlay](images/gradcam_class_overlay.png)

Grad-CAM 은 설명/검토용으로 두고 후처리 룰로는 바로 적용하지 않습니다.

## Best Known Method

각 축에서 한 가지만 바꿨을 때 기준선을 가장 많이 개선한 값. 단일-축 evidence이고, 조합한 결과가 아님.

| axis | baseline | BKM value | F1 | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `normal_ratio` | `700` | `3300` | 0.9988 | 0.8 | 1.0 |
| `gc` | `1.0` | `0.5` | 0.9964 | 2.2 | 3.2 |
| `label_smoothing` | `0.00` | `0.02` | 0.9981 | 0.8 | 2.0 |
| `stochastic_depth` | `0.00` | `0.05` | 0.9985 | 0.8 | 1.4 |
| `focal_gamma` | `0.0` | `2.0` | 0.9964 | 2.4 | 3.0 |
| `abnormal_weight` | `1.0` | `1.5` | 0.9956 | 3.0 | 3.6 |
| `ema` | `0.0 / off` | `0.95` | 0.9964 | 2.2 | 3.2 |
| `allow_tie_save` | `off` | `on` | 0.9964 | 2.4 | 3.0 |

## 축별 성능 표

기준선(`fresh0412_v11_refcheck_raw_n700`): F1 0.9944, FN 4.6, FP 3.8 (5/5 seeds).

### normal_ratio
![normal_ratio](plots/normal_ratio.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 700 (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| 1400 | 5/5 | 0.9956 | +0.0012 | 3.6 | -1.0 | 3.0 | -0.8 |
| 2100 | 5/5 | 0.9959 | +0.0015 | 2.2 | -2.4 | 4.0 | +0.2 |
| 2800 | 5/5 | 0.9951 | +0.0007 | 4.8 | +0.2 | 2.6 | -1.2 |
| 3000 | 5/5 | 0.9980 | +0.0036 | 0.8 | -3.8 | 2.2 | -1.6 |
| 3150 | 5/5 | 0.9984 | +0.0040 | 1.6 | -3.0 | 0.8 | -3.0 |
| **3300 (BKM)** | 5/5 | **0.9988** | +0.0044 | 0.8 | -3.8 | 1.0 | -2.8 |
| 3500 | 5/5 | 0.9957 | +0.0013 | 3.4 | -1.2 | 3.0 | -0.8 |

### per_class
![per_class](plots/per_class.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / off (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| 100 | 5/5 | 0.9851 | -0.0093 | 6.4 | +1.8 | 16.0 | +12.2 |
| 200 | 5/5 | 0.9905 | -0.0039 | 6.0 | +1.4 | 8.2 | +4.4 |
| 300 | 5/5 | 0.9933 | -0.0011 | 3.0 | -1.6 | 7.0 | +3.2 |
| 400 | 5/5 | 0.9936 | -0.0008 | 3.2 | -1.4 | 6.4 | +2.6 |
| 500 | 5/5 | 0.9937 | -0.0007 | 3.0 | -1.6 | 6.4 | +2.6 |
| 600 | 5/5 | 0.9955 | +0.0011 | 3.2 | -1.4 | 3.6 | -0.2 |
| 700 | 5/5 | 0.9960 | +0.0016 | 2.4 | -2.2 | 3.6 | -0.2 |
| 800 | 5/5 | 0.9933 | -0.0011 | 3.4 | -1.2 | 6.6 | +2.8 |
| 900 | 5/5 | 0.9945 | +0.0001 | 4.4 | -0.2 | 3.8 | 0 |
| 1000 | 5/5 | 0.9945 | +0.0001 | 2.8 | -1.8 | 5.4 | +1.6 |

### LR
![lr](plots/lr.png) ![lr schedule](plots/lr_lr_schedule.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-5 / 1e-4 | 5/5 | 0.9940 | -0.0004 | 5.2 | +0.6 | 3.8 | 0 |
| **2e-5 / 2e-4 (기준)** | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| 3e-5 / 3e-4 | 5/5 | 0.9963 | +0.0019 | 2.8 | -1.8 | 2.8 | -1.0 |
| 5e-5 / 5e-4 | 5/5 | 0.9963 | +0.0018 | 2.4 | -2.2 | 3.2 | -0.6 |
| 1e-4 / 1e-3 | 5/5 | 0.9964 | +0.0020 | 2.0 | -2.6 | 3.4 | -0.4 |

### warmup
![warmup](plots/warmup.png) ![warmup schedule](plots/warmup_lr_schedule.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup=0 | 5/5 | 0.9868 | -0.0076 | 8.0 | +3.4 | 11.8 | +8.0 |
| warmup=3 | 5/5 | 0.9957 | +0.0013 | 2.4 | -2.2 | 4.0 | +0.2 |
| **warmup=5 (기준)** | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| warmup=8 | 5/5 | 0.9951 | +0.0006 | 4.0 | -0.6 | 3.4 | -0.4 |
| warmup=10 | 5/5 | 0.9939 | -0.0006 | 3.8 | -0.8 | 5.4 | +1.6 |

### GC (grad_clip)
![gc](plots/gc.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0 / off (기준)** | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| 0.1 | 5/5 | 0.9951 | +0.0006 | 3.4 | -1.2 | 4.0 | +0.2 |
| 0.25 | 5/5 | 0.9953 | +0.0009 | 3.4 | -1.2 | 3.6 | -0.2 |
| **0.5 (BKM)** | 5/5 | **0.9964** | +0.0020 | 2.2 | -2.4 | 3.2 | -0.6 |
| 1.5 | 5/5 | 0.9940 | -0.0004 | 3.2 | -1.4 | 5.8 | +2.0 |
| 5.0 | 5/5 | 0.9951 | +0.0006 | 2.2 | -2.4 | 5.2 | +1.4 |

### label_smoothing
![label_smoothing](plots/label_smoothing.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| **0.02 (BKM)** | 5/5 | **0.9981** | +0.0037 | 0.8 | -3.8 | 2.0 | -1.8 |
| 0.05 | 5/5 | 0.9959 | +0.0014 | 2.8 | -1.8 | 3.4 | -0.4 |
| 0.10 | 5/5 | 0.9949 | +0.0005 | 3.0 | -1.6 | 4.6 | +0.8 |
| 0.15 | 5/5 | 0.9971 | +0.0027 | 0.4 | -4.2 | 4.0 | +0.2 |
| 0.20 | 5/5 | 0.9972 | +0.0028 | 0.6 | -4.0 | 3.6 | -0.2 |

### stochastic_depth
![stochastic_depth](plots/stochastic_depth.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| **0.05 (BKM)** | 5/5 | **0.9985** | +0.0041 | 0.8 | -3.8 | 1.4 | -2.4 |
| 0.10 | 5/5 | 0.9975 | +0.0030 | 0.8 | -3.8 | 3.0 | -0.8 |
| 0.15 | 5/5 | 0.9949 | +0.0005 | 3.0 | -1.6 | 4.6 | +0.8 |
| 0.20 | 5/5 | 0.9953 | +0.0009 | 3.0 | -1.6 | 4.0 | +0.2 |
| 0.30 | 5/5 | 0.9977 | +0.0033 | 1.0 | -3.6 | 2.4 | -1.4 |

### focal_gamma
![focal_gamma](plots/focal_gamma.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| 0.5 | 5/5 | 0.9957 | +0.0013 | 2.6 | -2.0 | 3.8 | 0 |
| 1.0 | 5/5 | 0.9947 | +0.0003 | 2.8 | -1.8 | 5.2 | +1.4 |
| 1.5 | 5/5 | 0.9957 | +0.0013 | 2.2 | -2.4 | 4.2 | +0.4 |
| **2.0 (BKM)** | 5/5 | **0.9964** | +0.0020 | 2.4 | -2.2 | 3.0 | -0.8 |
| 2.5 | 5/5 | 0.9963 | +0.0019 | 1.8 | -2.8 | 3.8 | 0 |

### abnormal_weight
![abnormal_weight](plots/abnormal_weight.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 5/5 | 0.9952 | +0.0008 | 2.6 | -2.0 | 4.6 | +0.8 |
| 0.8 | 5/5 | 0.9956 | +0.0012 | 2.8 | -1.8 | 3.8 | 0 |
| 1.0 (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| 1.2 | 5/5 | 0.9951 | +0.0007 | 3.2 | -1.4 | 4.2 | +0.4 |
| **1.5 (BKM)** | 5/5 | **0.9956** | +0.0012 | 3.0 | -1.6 | 3.6 | -0.2 |
| 2.0 | 5/5 | 0.9951 | +0.0007 | 3.0 | -1.6 | 4.4 | +0.6 |
| 3.0 | 5/5 | 0.9949 | +0.0005 | 3.0 | -1.6 | 4.6 | +0.8 |

### EMA
![ema](plots/ema.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 / off (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| 0.90 | 5/5 | 0.9948 | +0.0004 | 3.8 | -0.8 | 4.0 | +0.2 |
| **0.95 (BKM)** | 5/5 | **0.9964** | +0.0020 | 2.2 | -2.4 | 3.2 | -0.6 |
| 0.99 | 5/5 | 0.9959 | +0.0014 | 2.4 | -2.2 | 3.8 | 0 |
| 0.995 | 5/5 | 0.9956 | +0.0012 | 2.2 | -2.4 | 4.4 | +0.6 |
| 0.999 | 5/5 | 0.9957 | +0.0013 | 2.6 | -2.0 | 3.8 | 0 |

### color
![color](plots/color.png)

조건: `baseline`=trend blue·fleet alpha 0.4, `c01`=trend red·alpha 0.4, `c02`=blue·alpha 0.15, `c03`=red·alpha 0.15.

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| c01 | 5/5 | 0.9952 | +0.0008 | 3.8 | -0.8 | 3.4 | -0.4 |

### allow_tie_save
![allow_tie_save](plots/allow_tie_save.png)

| condition | seeds | F1 | ΔF1 | FN | ΔFN | FP | ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off (기준) | 5/5 | 0.9944 | 0 | 4.6 | 0 | 3.8 | 0 |
| **on (BKM)** | 5/5 | **0.9964** | +0.0020 | 2.4 | -2.2 | 3.0 | -0.8 |

## NT (normal_threshold) effect

`normal_threshold` 는 "p_normal 이 이 값 이상이면 normal 로 판정" 하는 기준. 기본 reporting NT = 0.9. NT 올라갈수록 더 많은 케이스를 abnormal 로 보내 **FN ↓ / FP ↑** trade-off.

전체 `fresh0412_v11*` candidate 319 run paired 평균 (collapsed run 39개 제외, 모든 NT level 동시 보유한 run 만):

![nt effect](plots/nt_effect.png)

| condition | runs | F1 mean | FN mean | FP mean |
| --- | ---: | ---: | ---: | ---: |
| no NT (argmax) | 319 | 0.9950 | 4.96 | 2.48 |
| NT=0.9 | 319 | 0.9955 | 3.80 | 2.91 |
| **NT=0.99** | 319 | **0.9957** | 2.74 | 3.65 |
| NT=0.999 | 319 | 0.9954 | 1.75 | 5.12 |

- argmax → NT=0.9 : FN −1.16, FP +0.43, F1 +0.0005
- NT=0.9 → NT=0.99 : FN −1.06, FP +0.74, F1 +0.0002 (F1 정점)
- NT=0.99 → NT=0.999 : FN −0.99, FP +1.47, F1 −0.0003 (FP 가 FN 감소를 추월)

**해석**: F1 평균은 NT=0.99 에서 가장 높지만 F1 차이는 +0.0002 수준이라 미세함. 안정적 reporting 은 여전히 **NT=0.9** 권장 — FP/FN trade-off 가 가장 균형 잡혀있고 test-peeking 위험도 낮음. 0.999 는 FN 우선이 필요한 use case 에서만 (불량 놓치는 비용이 매우 클 때) 고려. 0.9999 는 모든 케이스를 abnormal 로 분류해 degenerate 라 사용 금지 (memory `feedback_normal_threshold_099`).

재생성:
```bash
python scripts/generate_nt_effect_report.py \
  --candidate-contains fresh0412_v11 \
  --out-md validations/nt_effect.md \
  --out-csv validations/nt_effect.csv \
  --out-plot validations/nt_effect.png
cp validations/nt_effect.png docs/plots/
```

## 운영 스크립트

| 작업 | 명령어 |
| --- | --- |
| 데이터/이미지 생성 | `python generate_data.py --config dataset.yaml --workers 24 && python generate_images.py --config dataset.yaml --workers 24` |
| 단일 학습 | `python train.py --config dataset.yaml --mode binary --epochs 20 --batch_size 32 --precision fp16 --normal_ratio 700 --seed 42 --log_dir my_run` |
| 전체 sweep | `bash scripts/sweeps_server/00_all.sh` |
| 한 축만 sweep | `bash scripts/sweeps_server/axis.sh <axis>` (lr, gc, color, …) |
| 추론용 이미지 | `python scripts/generate_inference_images.py --timeseries data/timeseries.csv --scenarios data/scenarios.csv --out-dir inference_inputs` |
| best 모델 추론 | `python inference.py --model logs/<run>/best_model.pth` |
| 폴더 추가학습 | `python scripts/add_training_from_folders.py --model-run logs/<run> --image-root extra_images --epochs 3 --lr 1e-5 --scheduler cosine` |
| logs 표·plot | `python scripts/generate_log_history_report.py --logs-dir logs --out-prefix validations/log_history_report_rawbase --contains rawbase` |
