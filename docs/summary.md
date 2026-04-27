# 실험 요약

_자동 갱신 시각: `2026-04-28T07:58:36`._

## 결과 해석

- 이번 strict one-factor round에서는 기준을 고정한 채 `normal_ratio`, `per_class`, `lr`, `warmup`, `gc`, `weight_decay`, `smoothing`, `label_smoothing`, `stochastic_depth`, `focal_gamma`, `abnormal_weight`, `ema`, `color`, `allow_tie_save`를 개별 축으로 확인했습니다.
- 유의미한 최적값 후보가 보이는 축은 `label_smoothing`은 `0.15` 근처에서 가장 강한 개선이 보였고, 너무 낮거나 높으면 FP/FN 균형이 다시 나빠졌습니다; `abnormal_weight`는 `1.5` 근처에서 sweet spot이 보였고, 더 크게 주면 FN이 다시 증가했습니다; `stochastic_depth`는 `0.1` 인근에서 유의미한 개선이 나타났습니다.
- 넓은 양호 구간으로 해석하는 편이 맞는 축은 `gc`는 단일 sharp optimum보다는 넓은 양호 구간이 보였고, 헌팅 값 하나가 축 스케일을 왜곡하는 형태였습니다.
- 현재로서는 뚜렷한 최적값이 약하거나 추가 확인이 필요한 축은 `focal_gamma`는 여러 값이 비슷해서 뚜렷한 최적값보다는 broad-good 혹은 약한 효과 축에 가깝습니다; `normal_ratio`는 성능이 전반적으로 좋아지는 구간은 보이지만, 현재 점들만으로는 매끈한 단일 sweet spot이라고 단정하기 어렵습니다; `ema`는 기준 대비 개선은 있으나 강한 최적값 주장을 하기는 아직 어렵습니다.

## 요약

- 고정 기준 ref: `fresh0412_v11_n700_existing` -> `F1=0.9901`, `FN=9.8`, `FP=5` over `5/5` seeds.
- 메인 strict queue: `158` 완료 run, decision `queue_exhausted`.
- Round-2 refinement: `9/40` 완료 run, stage `strict_single_factor_round2`, status `running`.

- `label_smoothing` 현재 완료된 조건 중 최선은 `0.15` with `F1=0.9977`, `FN=0.8`, `FP=2.6`.
- `abnormal_weight` 현재 완료된 조건 중 최선은 `1.5` with `F1=0.9979`, `FN=1.2`, `FP=2`.
- `stochastic_depth` 현재 완료된 조건 중 최선은 `0.1` with `F1=0.9975`, `FN=1.2`, `FP=2.6`.
- `GC` 넓은 양호 구간이 유지되고 있으며 현재 총 오류가 가장 낮은 쪽은 대략 `5` with `F1=0.9967`, `FN=3`, `FP=2`. `1.25` is pending.
- `color`는 trend를 빨강으로 바꾸면 recall에 도움되고, fleet를 너무 연하게 하면 FP가 악화됩니다: `c01 0.9971 / FN 0.6 / FP 3.8` vs `c02 0.9944 / FN 1.8 / FP 6.6`.

## 임시 황금 레시피

_아직 one-factor evidence 단계입니다. round-2 종료 후 joint combo validation이 필요합니다._

- `normal_ratio = 3300`: `F1=0.9973`, `FN=2.4`, `FP=1.6`
- `gc = 5`: `F1=0.9967`, `FN=3`, `FP=2`
- `label_smoothing = 0.15`: `F1=0.9977`, `FN=0.8`, `FP=2.6`
- `stochastic_depth = 0.1`: `F1=0.9975`, `FN=1.2`, `FP=2.6`
- `focal_gamma = 0.5`: `F1=0.9969`, `FN=2.8`, `FP=1.8`
- `abnormal_weight = 1.5`: `F1=0.9979`, `FN=1.2`, `FP=2`
- `ema = 0.99`: `F1=0.9972`, `FN=1`, `FP=3.2`
- `allow_tie_save = on`: `F1=0.9974`, `FN=2.2`, `FP=1.8`

## 남은 Round-2 확인 항목

- `gc = 1.25`: `4/5` 완료
- `label_smoothing = 0.125`: `0/5` 완료
- `label_smoothing = 0.175`: `0/5` 완료
- `stochastic_depth = 0.15`: `0/5` 완료
- `focal_gamma = 1`: `0/5` 완료
- `abnormal_weight = 1.2`: `0/5` 완료
- `ema = 0.995`: `0/5` 완료

## 플롯 목록

- `normal_ratio`: [normal_ratio.png](plots/normal_ratio.png)
- `per_class`: [per_class.png](plots/per_class.png)
- `lr`: [lr.png](plots/lr.png)
- `lr` learning-rate schedule: [lr_lr_schedule.png](plots/lr_lr_schedule.png)
- `warmup`: [warmup.png](plots/warmup.png)
- `warmup` learning-rate schedule: [warmup_lr_schedule.png](plots/warmup_lr_schedule.png)
- `gc`: [gc.png](plots/gc.png)
- `weight_decay`: [weight_decay.png](plots/weight_decay.png)
- `smoothing`: [smoothing.png](plots/smoothing.png)
- `label_smoothing`: [label_smoothing.png](plots/label_smoothing.png)
- `stochastic_depth`: [stochastic_depth.png](plots/stochastic_depth.png)
- `focal_gamma`: [focal_gamma.png](plots/focal_gamma.png)
- `abnormal_weight`: [abnormal_weight.png](plots/abnormal_weight.png)
- `ema`: [ema.png](plots/ema.png)
- `color`: [color.png](plots/color.png)
- `allow_tie_save`: [allow_tie_save.png](plots/allow_tie_save.png)

## normal_ratio

![normal_ratio](plots/normal_ratio.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 700 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 1400 | 5/5 | 0.9899 | 12 | 3.2 | 완료 |
| 2100 | 5/5 | 0.9916 | 10 | 2.6 | 완료 |
| 2800 | 5/5 | 0.9923 | 9.8 | 1.8 | 완료 |
| 3000 | 5/5 | 0.9968 | 3.2 | 1.6 | 완료 |
| 3150 | 5/5 | 0.9939 | 7.2 | 2 | 완료 |
| 3300 | 5/5 | 0.9973 | 2.4 | 1.6 | 완료 |
| 3500 | 5/5 | 0.9960 | 4.2 | 1.8 | 완료 |

## per_class

![per_class](plots/per_class.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 100 | 5/5 | 0.9921 | 5.8 | 6 | 완료 |
| 200 | 5/5 | 0.9948 | 5.8 | 2 | 완료 |
| 300 | 5/5 | 0.9955 | 3.8 | 3 | 완료 |
| 400 | 5/5 | 0.9953 | 4.6 | 2.4 | 완료 |
| 500 | 5/5 | 0.9957 | 4.8 | 1.6 | 완료 |
| 600 | 5/5 | 0.9960 | 4 | 2 | 완료 |
| 700 | 5/5 | 0.9945 | 6.6 | 1.6 | 완료 |
| 800 | 5/5 | 0.9968 | 2.6 | 2.2 | 완료 |
| 900 | 5/5 | 0.9937 | 7.4 | 2 | 완료 |
| 1000 | 5/5 | 0.9957 | 4.8 | 1.6 | 완료 |

## LR

![lr](plots/lr.png)

![lr learning-rate schedule](plots/lr_lr_schedule.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 1e-5 / 1e-4<br>bb/head=1e-5 / 1e-4 | 3/3 | 0.9967 | 2.333 | 2.667 | 완료 |
| 2e-5 / 2e-4<br>bb/head=2e-5 / 2e-4 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 3e-5 / 3e-4<br>bb/head=3e-5 / 3e-4 | 3/3 | 0.9969 | 2.667 | 2 | 완료 |
| 5e-5 / 5e-4<br>bb/head=5e-5 / 5e-4 | 3/3 | 0.9951 | 4 | 3.333 | 완료 |
| 1e-4 / 1e-3<br>bb/head=1e-4 / 1e-3 | 3/3 | 0.7767 | 250 | 1.667 | 완료 |

## warmup

![warmup](plots/warmup.png)

![warmup learning-rate schedule](plots/warmup_lr_schedule.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| warmup=3<br>lr=2e-5/2e-4 | 3/3 | 0.9964 | 3.333 | 2 | 완료 |
| warmup=5<br>lr=2e-5/2e-4 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| warmup=8<br>lr=2e-5/2e-4 | 3/3 | 0.9940 | 6.667 | 2.333 | 완료 |

## GC

![gc](plots/gc.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 3/3 | 0.3626 | 478.667 | 253.333 | 완료 |
| 0.1 | 5/5 | 0.9961 | 4 | 1.8 | 완료 |
| 0.25 | 5/5 | 0.9964 | 2.8 | 2.6 | 완료 |
| 0.35 | 5/5 | 0.9940 | 6.6 | 2.4 | 완료 |
| 0.5 | 5/5 | 0.9964 | 4 | 1.4 | 완료 |
| 0.75 | 5/5 | 0.9951 | 5.6 | 1.8 | 완료 |
| 1.0 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 1.25 | 4/5 | 0.9978 | 0.25 | 3 | 부분완료 |
| 1.5 | 5/5 | 0.9957 | 4.2 | 2.2 | 완료 |
| 2 | 5/5 | 0.9965 | 3 | 2.2 | 완료 |
| 3 | 5/5 | 0.9959 | 3.8 | 2.4 | 완료 |
| 5 | 5/5 | 0.9967 | 3 | 2 | 완료 |

## weight_decay

![weight_decay](plots/weight_decay.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 3/3 | 0.9976 | 1.667 | 2 | 완료 |
| 0.01 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 0.02 | 3/3 | 0.9967 | 2.667 | 2.333 | 완료 |
| 0.05 | 3/3 | 0.9960 | 3.333 | 2.667 | 완료 |

## smoothing

![smoothing](plots/smoothing.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 1-raw | 3/3 | 0.9953 | 4.333 | 2.667 | 완료 |
| 3-mean | 3/3 | 0.9951 | 5.333 | 2 | 완료 |
| 3-median | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 5-median | 3/3 | 0.9960 | 3.333 | 2.667 | 완료 |

## label_smoothing

![label_smoothing](plots/label_smoothing.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.00 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 0.02 | 5/5 | 0.9923 | 4.8 | 6.8 | 완료 |
| 0.1 | 3/3 | 0.9984 | 1 | 1.333 | 완료 |
| 0.125 | 0/5 | - | - | - | queued |
| 0.15 | 5/5 | 0.9977 | 0.8 | 2.6 | 완료 |
| 0.175 | 0/5 | - | - | - | queued |
| 0.2 | 5/5 | 0.9969 | 2.4 | 2.2 | 완료 |

## stochastic_depth

![stochastic_depth](plots/stochastic_depth.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.00 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 0.05 | 5/5 | 0.9949 | 5.6 | 2 | 완료 |
| 0.1 | 5/5 | 0.9975 | 1.2 | 2.6 | 완료 |
| 0.15 | 0/5 | - | - | - | queued |
| 0.2 | 3/3 | 0.9967 | 3 | 2 | 완료 |
| 0.3 | 5/5 | 0.9973 | 1.8 | 2.2 | 완료 |

## focal_gamma

![focal_gamma](plots/focal_gamma.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.0 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 0.5 | 5/5 | 0.9969 | 2.8 | 1.8 | 완료 |
| 1 | 0/5 | - | - | - | queued |
| 1.5 | 5/5 | 0.9952 | 5.6 | 1.6 | 완료 |
| 2 | 5/5 | 0.9968 | 3.2 | 1.6 | 완료 |

## abnormal_weight

![abnormal_weight](plots/abnormal_weight.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.5 | 5/5 | 0.9949 | 4.8 | 2.8 | 완료 |
| 0.8 | 5/5 | 0.9953 | 3.8 | 3.2 | 완료 |
| 1.0 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 1.2 | 0/5 | - | - | - | queued |
| 1.5 | 5/5 | 0.9979 | 1.2 | 2 | 완료 |
| 2 | 5/5 | 0.9967 | 3.4 | 1.6 | 완료 |
| 3 | 5/5 | 0.9939 | 7.8 | 1.4 | 완료 |

## EMA

![ema](plots/ema.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.0 / off | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| 0.99 | 5/5 | 0.9972 | 1 | 3.2 | 완료 |
| 0.995 | 0/5 | - | - | - | queued |
| 0.999 | 5/5 | 0.9967 | 2.4 | 2.6 | 완료 |

## color

![color](plots/color.png)

조건 설명:

- `기준`: trend blue `#4878CF`, fleet alpha `0.4`
- `c01`: trend red `#E43320`, fleet alpha `0.4`
- `c02`: trend blue `#4878CF`, fleet alpha `0.15`
- `c03`: trend red `#E43320`, fleet alpha `0.15`

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 기준 | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| c01 | 5/5 | 0.9971 | 0.6 | 3.8 | 완료 |
| c02 | 5/5 | 0.9944 | 1.8 | 6.6 | 완료 |

## allow_tie_save

![allow_tie_save](plots/allow_tie_save.png)

| condition | seeds | F1 | FN | FP | status |
| --- | ---: | ---: | ---: | ---: | --- |
| off | 5/5 | 0.9901 | 9.8 | 5 | 기준 |
| on | 5/5 | 0.9974 | 2.2 | 1.8 | 완료 |
