# 실험 요약

## 실험 방식

- 같은 baseline에서 한 번에 하나의 축만 바꾸는 strict one-factor 실험.
- 기본 seed 5개: `42, 1, 2, 3, 4`. 성능은 F1, FN, FP, 완료 seed 수로 봅니다.
- baseline은 `fresh0412_v11_refcheck_raw_n700` (raw smoothing, `grad_clip=0`, `label_smoothing=0`, `NT=0.9`).
- 실행: `bash scripts/sweeps_server/00_all.sh` 한 줄. GPU 메모리·CPU 수로 `server`/`pc`/`minimal` 프로필을 자동 선택합니다.

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

## 학습 이미지 예시

학습 데이터는 `normal`과 불량 class별 이미지로 구성합니다. 모델 입력은 training image이고, display image는 같은 sample을 사람이 확인하기 쉽게 축/legend/색을 붙인 렌더링입니다.

### Training Image
![training images by class](images/sample_overview_train.png)

### Display Image
![display images by class](images/sample_overview_display.png)

### Legend Axis Image

같은 `device/step/item` group에서도 `legend_axis`를 `eqp_id`, `chamber`, `recipe`로 바꾸면 fleet member와 legend가 달라지므로 별도 이미지로 생성합니다. `highlighted_member`는 강조되는 member, `target`은 가운데 가로 기준선 값.

![display images by legend_axis](images/sample_overview_legend_axis.png)

불량 class는 `mean_shift`, `standard_deviation`, `spike`, `drift`, `context` 다섯 종류이고, 각 이미지는 해당 class label로 학습됩니다.

## Logical Member Attribution Example

같은 `legend_axis` chart를 member별 class 판단 이미지로 확장합니다. 불량 member 이미지만 anomaly class이고, 양호 member 이미지는 normal class입니다.

아래 예시는 같은 chart `ch_09100`을 `EQP A`, `EQP B`, `EQP C`, `EQP D`, `EQP E` class 이미지로 펼친 것입니다. 각 EQP의 highlighted trend만 서로 다른 색으로 표시하고, 회색 점들은 같은 `legend_axis` 안의 비교 fleet입니다. class 글자는 normal은 검정, anomaly는 빨강. 이 예시에서는 `EQP C`만 anomaly class.

![logical member class examples](images/logical_member_targets_ch09100.png)

즉 family 전체 이상 감지가 아니라, `highlighted_member` 단위로 label을 부여하는 학습 예시.

## Grad-CAM / Postprocess Check

Grad-CAM의 heat는 실제 anomaly 위치가 아니라 `abnormal` logit에 기여한 모델 근거 위치입니다. 넓은 불량은 heat도 넓게 퍼질 수 있고, `spike` 같은 국소 패턴은 더 좁게 잡히는 경향이 있습니다. 좌측 불량과 우측 정상의 대비 때문에 우측에 heat가 생기기도 해서, CAM 위치만으로 left/right defect를 판정하지 않습니다.

class별 6행 sample 6열로 원본 trend 이미지 위에 CAM colormap을 반투명으로 얹은 예시입니다. 빨강은 큰 CAM 값, 파랑도 같이 표시해서 CAM이 넓게 퍼지는지 확인.

![class Grad-CAM overlay](images/gradcam_class_overlay.png)

| check | F1 macro | FN | FP | result |
| --- | ---: | ---: | ---: | --- |
| full image baseline | 0.9960 | 5 | 1 | 기준 |
| right-crop rescue | 0.9378 | 0 | 93 | FN 5개는 모두 잡지만 FP가 너무 커짐 |
| Grad-CAM normal rescue, best F1 | 0.9753 | 5 | 32 | FP만 늘고 FN rescue 없음 |
| Grad-CAM normal rescue, best FN | 0.9344 | 0 | 98 | FN은 모두 잡지만 FP가 너무 커짐 |

결론: Grad-CAM은 설명/검토용으로 두고, 후처리 룰은 바로 적용하지 않음. `full=abnormal, right_crop=normal`은 좌측/과거 불량 가능성 검토, `full=abnormal, right_crop=abnormal`은 최근 우측 불량 가능성 검토 정도로만 사용.

### FP Grad-CAM Check

최신 model run의 test 실제 FP는 1개. `ch_09572`는 true normal인데 `p_abnormal=0.99995`, right-crop도 `p_abnormal=0.99999`로 abnormal 판정. CAM mass는 left 0.730, mid 0.227, right 0.043이라 우측 최근 불량이 아니라 좌측/초반의 국소 outlier와 cluster-edge를 강한 abnormal 근거로 본 케이스. FP 갤러리는 6장을 맞추기 위해 실제 FP 1개와 test normal 중 `p_abnormal` 상위 hard-normal 5개를 함께 표시. 현재 실제 FP만 보면 spike-like 정상 outlier를 spike성 불량으로 오인.

![FP Grad-CAM examples](images/gradcam_fp_examples.png)

## NT (normal_threshold) effect

`normal_threshold` 는 "p_normal 이 이 값 이상이면 normal 로 판정" 하는 기준. 기본 reporting NT = 0.9. NT 적용 → 더 많은 케이스를 abnormal 로 보내 **FN ↓ / FP ↑** trade-off.

전체 `fresh0412_v11*` candidate 350 run paired 평균 (collapsed run 8개 제외):

![nt effect](plots/nt_effect.png)

| condition | runs | F1 mean | FN mean | FP mean |
| --- | ---: | ---: | ---: | ---: |
| 그냥 (no NT) | 350 | 0.9951 | 4.82 | 2.53 |
| NT=0.9 | 350 | 0.9955 | 3.60 | 3.12 |

NT 적용으로 FN −1.22 (4.82 → 3.60), FP +0.59 (2.53 → 3.12), F1 +0.0004 (0.9951 → 0.9955). **순효과: 불량을 1.2 개 더 잡고 정상은 0.6 개 더 abnormal 로 오인** — 87 candidate × 5 seed 규모에서 NT=0.9 가 안정적인 reporting threshold임을 확인.

극단값(0.99 / 0.999 / 0.9999) 은 FP 가 빠르게 증가하고 0.9999 에서는 모든 케이스를 abnormal 로 분류 (FP=전체 normal 수) — test-peeking 위험 (memory `feedback_normal_threshold_099`).

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
