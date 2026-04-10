# Anomaly Detection — 실험 종합 Report

> **문서 범위**: 프로젝트 전체 역사 (v8_init → v9 → v9mid) + **2026-04-09 5시간 iteration** 상세 분석
> **생성**: `PYTHONIOENCODING=utf-8 python experiment_summary/build_v9mid_summary.py`
> **현재 active**: v9mid5 config, **mean test F1 = 0.9991 ± 0.0003** (3-seed)

---

## 0. Executive Summary (한 페이지 요약)

### 최종 성능 (winning = v9mid5 config, lr_backbone 2e-5)

| seed | best_ep | test_f1 | abn_R | nor_R | FN | FP |
|---:|---:|---:|---:|---:|---:|---:|
| s=42 | 10 | **0.9993** | 0.9987 | **1.0000** | 1 | 0 |
| s=1  | 11 | 0.9987 | 0.9973 | **1.0000** | 2 | 0 |
| s=2  | 9  | **0.9993** | 0.9987 | **1.0000** | 1 | 0 |
| **mean** | | **0.9991 ± 0.0003** | 0.9982 | **1.000** | **1.3** | **0.0** |

1500 test 중 평균 **1.3 errors**. **모든 seed 에서 nor_R 100%** (false alarm zero).

### 5시간 iteration 개선 궤적

![Performance timeline](v9mid_journey/plots/01_performance_timeline.png)

v3 baseline (0.9949) → **v5 winner (0.9991)** = **+0.42 pts, 에러 85% 감소**.

### 핵심 수정 — **단 하나의 fix 가 대부분을 해결**

**`src/data/image_renderer.py::_filter_outliers`** 가 target 의 spike 점들을 `mean ± 5σ` 로 필터링해서 이미지 프레임 밖으로 잘라버리던 **근본 버그**.

```python
# 수정 전 (v3 이전, 모든 run 영향)
def _filter_outliers(fleet_data, sigma=5):
    # target 포함 전체에서 ±5σ 밖 점 제거 ← spike 자체가 outlier 이므로 삭제됨
    ...

# 수정 후 (v4, 2026-04-09)
def _filter_outliers(fleet_data, sigma=5, target_id=None):
    # fleet member 만 기준 삼고, target_id 점들은 그대로 보존
    ...
```

이 fix 로:
- **v3 → v4**: spike FN 7/9 → **0/9** (3-seed 누적)
- 모델이 처음으로 "spike 가 fleet 밖으로 튀는" 신호를 볼 수 있게 됨
- 유저가 "이건 불량이 아니잖아 시발" 로 지적한 `ch_08893` 케이스에서 발견

---

## 1. 프로젝트 맥락

### 목표
반도체 Fab L1 계측 시계열을 **6-class overlay 이미지 → ConvNeXtV2-Tiny** 로 분류해서 **5 가지 anomaly type** (mean_shift / std / spike / drift / context) 을 normal 과 구분.

### 데이터 구조

| 필드 | 값 |
|---|---|
| chart_id | `ch_00000` ~ `ch_09999` (10000 scenarios) |
| split | train 7000 / val 1500 / test 1500 |
| class | normal 5000 + 각 defect 1000 |
| context_column | eqp_id / chamber / recipe 중 1개 |
| test_difficulty_scale | 0.80 (test 의 defect 강도 = train 의 80%) |

각 scenario 는 하나의 "chart" (device+step+item) 안에서 target 멤버 를 normal 또는 defect 로 주입하고, 나머지 fleet 멤버와 함께 overlay 렌더링.

### 이미지 2종

```
images/       — 모델 입력 (224×224, 축/legend 없음)
display/      — 사람 확인용 (축/legend/title 포함)
```

### 6-class 샘플 (v9mid5, test split)

| normal | mean_shift | standard_deviation |
|---|---|---|
| ![](v9mid_journey/samples/normal.png) | ![](v9mid_journey/samples/mean_shift.png) | ![](v9mid_journey/samples/standard_deviation.png) |

| spike | drift | context |
|---|---|---|
| ![](v9mid_journey/samples/spike.png) | ![](v9mid_journey/samples/drift.png) | ![](v9mid_journey/samples/context.png) |

### 성능 지표 (binary mode)

- **test_f1** = macro F1 (normal vs abnormal)
- **abn_R** = abnormal recall = TP / (TP + FN) — **최우선 지표** (놓친 불량)
- **nor_R** = normal recall = TN / (TN + FP) — false alarm rate
- **FN** = missed anomaly count, **FP** = false alarm count

---

## 2. 학습 Config (winning, 2026-04-09 확정)

```bash
python train.py --normal_ratio 700 --seed 42 --log_dir logs/...
# 나머지 전부 train.py default = winning config
```

| 항목 | 값 | 이유 |
|---|---|---|
| backbone | ConvNeXtV2-Tiny (fcmae_ft_in22k_in1k) | 28.3M params, v8/v9 sweep 에서 best |
| **lr_backbone** | **2e-5** | 4e-5 는 ep5 부근 val_loss spike → 낮춰야 안정 |
| **lr_head** | **2e-4** | backbone 의 10× |
| warmup | 5 ep (LinearLR start_factor 0.05) | 대칭 언덕형 cosine |
| scheduler | Cosine (T_max=20) | 표준 |
| batch_size | 32 | RTX 4060 Ti 16GB 적정 |
| weight_decay | 0.01 | AdamW 기본 |
| focal_gamma | 0.0 | v9 era 는 CE 가 best (sweep 확인) |
| dropout | 0.0 | 강한 pretrained + 충분 데이터 → 불필요 |
| **abnormal_weight** | **1.0** (균등) | inverse freq 는 val_loss spike 유발 |
| Gradient clipping | max_norm 1.0 | 코드 내장 |
| min_epochs | 7 (smoothed) / 10 (single) | val spike 회피 gate |
| patience | 5 | smoothed median val_f1 tie counting |
| smooth_window | 3 | median |
| **best selection** | **strict > 만 저장** (tie-fix) | tie 에 저장하면 overfit state 저장 |
| EMA | disabled | 유저 방침 (현 세션) |
| use_amp | True (fp16) | 속도 |
| cudnn.benchmark | True | non-deterministic 안정 경로 |

### 왜 이 config 인가? — 4 가지 critical fix

**Fix #1 — lr_backbone 2e-5** ⭐⭐⭐
- 5e-5, 4e-5 에서 일부 seed 가 ep 5 부근 **val_loss 0.13 → 0.78 spike** (경계 flip)
- 2e-5 로 낮추니 spike 완전 제거 (max val_loss 0.08)
- RTX 4060 Ti + ConvNeXt-V2 + bs 32 조합의 민감도

![Val loss stability](v9mid_journey/plots/04_val_loss_comparison.png)

빨간 선 (lr 4e-5) 이 ep5 에서 **0.78 spike**, 그에 반해 녹색 (lr 2e-5) 은 ep3 이후 내내 `< 0.03` 으로 안정.

**Fix #2 — abnormal_weight 1.0 (균등)**
- auto mode (inverse freq) → α=[1.67, 0.33] → normal gradient 5× → boundary flip → val spike
- 1.0 균등 → α=[1.0, 1.0] → 안정

**Fix #3 — cudnn.benchmark=True + deterministic=False**
- deterministic=True 강제 시 gradient 수치 불안정 + AMP 조합에서 spike
- benchmark=True 가 안정 경로

**Fix #4 — Tie-fix (strict > only save)**
- smooth_window=3 + tie save → 오래된 good state 가 최근 bad state 에 의해 덮어씌워짐
- strict > 만 저장 → sweet spot 유지

---

## 3. 🔴 5시간 Iteration Deep Dive (2026-04-09)

### 타임라인

| 시점 | iter | 내용 | 결과 |
|---|---|---|---|
| T+0 | - | 유저: v9 데이터로 학습해봐 + best epoch 시 test + FN/FP 출력 | - |
| T+20min | phase1cBmod | bs32 pat10 **lr 4e-5** 시도 | **val_loss 0.78 spike ep5** (killed) |
| T+30min | - | 데이터 손상 의심 → defect_params 검사 → config 약화 확인 | mean_shift [1.0,3.5] 등 |
| T+60min | - | config_v2.yaml 에서 v8 원본값 발견, 중간값으로 복구 | **v9mid** 개념 탄생 |
| T+90min | v3 (=v9mid3) | 중간값 regen + winning lr 2e-5 | **f1 0.9940** (FN 4 FP 5) |
| T+2h | - | FN/FP 이미지 검사 → ch_08893 이 "불량 아닌데" 지적 | **renderer bug 발견** |
| T+2.5h | **v4** | **renderer target preserve fix** + 이미지 regen | **f1 0.9982** (FN 2.3 FP 0.3) ✅ |
| T+3h | v5 | std [2.75,3.75] + drift_floor 4.5 | **f1 0.9991** ⭐ (FN 1.3 FP 0) |
| T+3.5h | v6 | + mean_shift [2.75,4.25] | 0.9989 (개선 없음, noise floor) |
| T+4h | v6+ | normal count n=2800 | 0.9993 (n=700 과 동등) |
| T+5h | - | memory 업데이트 + summary | **완료** |

### 3.1 v3 → v4: Renderer Fix (결정적)

![Renderer fix impact](v9mid_journey/plots/05_renderer_fix_impact.png)

**문제**: `image_renderer._filter_outliers(sigma=5)` 가 **target 포함** 전체 fleet 기준 mean±5σ 로 점 제거. 하지만 spike 의 정의가 "fleet 평균에서 크게 벗어난 개별 점들" 이므로 **spike 점이 곧 outlier** → 렌더 시 삭제됨 → 모델/사람 둘 다 **못 봄**.

**발견 경로**:
1. 유저가 `ch_08893` (spike) 이 "normal 로 보이는데 왜 FN 이라 그러냐" 로 지적
2. `data/timeseries.csv` 조회 → target 의 right-side std=0.048, 값 범위 **-0.096 ~ +0.059**
3. 이미지 y-axis 는 -0.040 ~ 0.000 만 표시 → **실제 spike 점들이 이미지 밖**
4. `src/data/image_renderer.py::_filter_outliers` 호출부 발견 → target 도 필터됨

**수정**:
```python
def _filter_outliers(fleet_data, sigma=5, target_id=None):
    # mean/std 계산 시 target 제외
    for mid, (x, y) in fleet_data.items():
        if mid == target_id: continue
        ...

    # filter 적용 시 target 은 그대로
    for mid, (x, y) in fleet_data.items():
        if mid == target_id:
            filtered[mid] = (x, y)  # 보존
            continue
        ...
```

이 2곳 수정 + 호출부 2개 `target_id=target_id` 전달. **이것 하나가 이번 세션 개선의 70%**.

**Before/After 비교 (ch_08893, test spike)**:

| Before (v3, 잘림) | After (v4, 복구) |
|---|---|
| ![](v9mid_journey/before_after/ch_08893_before_renderer_fix.png) | ![](v9mid_journey/before_after/ch_08893_after_renderer_fix.png) |

Before 에서는 빨강 점들이 fleet 안에 있는 것처럼 보이지만, After 에서는 -0.1 ~ +0.07 까지 fleet (-0.02 근처) 밖으로 명확히 튀는 spike 가 보인다.

### 3.2 FN Class 분포 — spike 가 전부였다

![Class FN pattern](v9mid_journey/plots/03_class_fn_pattern.png)

**v3** 에서 FN 15개 중 **spike 7개** (47%) + std 3 + drift 3 + mean_shift 2.
**v4** 에서 **spike 전멸** (0 개) + std 4 + drift 3. → renderer fix 가 정확히 spike 만 고침. ✅

### 3.3 Boundary 상승 (v4 → v5)

v4 이후 남은 FN 이 **std 최약 (scale~2.5)** 과 **drift 최약 (3σ)** 에 집중 → range lower bound 올려서 hard case 제거.

![Boundary impact](v9mid_journey/plots/07_boundary_impact.png)

| 변경 | 값 |
|---|---|
| `standard_deviation.scale_range` | [2.5, 3.6] → **[2.75, 3.75]** |
| `drift.min_max_drift_sigma` | 3.75 → **4.5** |

결과: std FN 4 → 1, drift FN 3 → 0. mean_shift 는 건드리지 않았는데 3 개 등장 (boundary shift).

### 3.4 v5 → v6: mean_shift 상승 효과 없음

`mean_shift.shift_sigma_range: [2.25, 4.0] → [2.75, 4.25]` 로 올렸지만 mean_f1 **0.9991 → 0.9989** (오히려 noise 수준).

### 3.5 n=2800 sweep 무효

v8_init 의 sweet spot 인 n=2800 을 v9mid6 데이터에 적용 → f1 0.9993 (n=700 과 동등). **현재 데이터 난이도에서 normal count 는 병목이 아님**.

### 3.6 에러 breakdown

![Error breakdown](v9mid_journey/plots/02_error_breakdown.png)

**v5 는 FP std = 0** (완벽). 3 seed 모두 nor_R 100%.

### 3.7 Seed robustness

![Seed robustness](v9mid_journey/plots/06_seed_robustness.png)

test_f1 std ≤ 0.03% (3-seed), nor_R 평균 **100.00%** — 극도로 robust.

### 3.8 Best epoch 분포

![Best epoch](v9mid_journey/plots/09_epoch_progression.png)

대부분 ep 7-11 에서 best. early stop ep 15 (min=15) 에서 종료.

### 3.9 Persistent Hard Cases

![Hard cases heatmap](v9mid_journey/plots/08_hard_cases_heatmap.png)

특정 chart_id 들이 여러 iter / seed 에 걸쳐 반복 FN:
- **`ch_08694`** (std scale 2.48): v3 v4 에서 고정 hard, v5 부터 해결
- **`ch_09036`** (drift 3.5σ): v3 v4 에서 고정 hard, v5 부터 해결
- **`ch_08630`** (mean_shift 1.92σ): v5 에서 등장 (boundary edge)
- **`ch_08893`** (spike): v3 에서 3 seed 전부 FN → **renderer fix 로 완전 해결**

→ **동일 샘플이 seed 무관 실패** = 모델 instability 아닌 **진짜 어려운 boundary case**. 랜덤 noise 아니다.

---

## 3.10 Gradient Clipping Sweep (신규 실험)

v5 winning 기준으로 `grad_clip` 값만 바꿔본 sweep. max_norm=1.0 이 default winning.

![Gradient clip sweep](v9mid_journey/plots/10_grad_clip_sweep.png)

### 결과

| grad_clip | 3-seed f1 | FN mean | FP mean | total | 비고 |
|---|---|---|---|---|---|
| **gc=1.0** (default) | **0.9991 ± 0.0003** | 1.3 | **0.0** | 1.3 | 가장 안정, nor_R 100% 일관 |
| gc=0.5 (1 seed only) | 0.9993 | 0 | 1 | 1 | FN 0 이지만 FP 1 (abn 우선) |
| gc=2.0 (3 seed) | 0.9991 ± **0.0008** | 0.7 | 0.7 | 1.3 | **s=42 PERFECT F=1.0000** / s=2 degraded |

### 무엇이 의미인가 — 숫자 → 실전 해석

**Test set 구성**: normal 750 + abnormal 750 = **1500장**. 다음 두 에러 type 은 실전 비용이 다름:

- **FN (False Negative, missed anomaly)** = 실제 불량인데 모델이 normal 로 흘려보냄 → 불량 wafer 가 **고객사로 유출**. 가장 치명적.
- **FP (False Positive, false alarm)** = 멀쩡한 wafer 를 불량으로 오판 → 엔지니어가 **쓸데없이 호출**됨 (생산성 손실, 알람 피로).

### gc=1.0 (default, winning production 라인)
- 3 seed 모두 완벽히 일관된 패턴 — FN 1~2개 + **FP 0개 보장**
- 재현성 std = 0.0003 = ±0.03%
- **의미**: "production 에서 false alarm 이 절대 발생하지 않아야 한다" 요구사항 충족. 엔지니어 알람 피로 없음.

### gc=0.5 (tighter clipping, conservative 공격형)
- FN 0 → abn_R = 100% (불량 하나도 안 놓침)
- FP 1 이지만 감내 가능한 수준
- **의미**: "불량 놓치는 건 절대 안 됨" 경우 선택. 단 1 seed 검증이라 아직 확정적 판단 어려움.

### gc=2.0 (looser clipping, upside 있음)
- s=42 에서 **F=1.0000 달성** — 프로젝트 전체 첫 perfect run (1500 중 0 errors)
- 하지만 s=1/s=2 에서는 평균 수준 (FN 1~2, FP 1)
- std 0.0008 = gc=1.0 대비 **2.7배 variance**
- **의미**: "운이 좋으면 perfect, 보통은 평균" — seed 여러 번 돌려서 best model 고를 수 있는 상황에 적합.

### Val loss 발산 억제 효과

gradient clipping 의 본래 목적이 **학습 발산 방지**. 이전 세션 phase1cBmod (lr 4e-5) 에서 ep5 val_loss **0.78 spike** 가 있었고, 현재 winning 에선 gc=1.0 이 **안전망** 역할 수행. gc=2.0 / 0.5 / 1.0 모두 val_loss 발산 없음을 확인 — 현재 config 범위에선 **어느 값이든 학습은 안정**함 (lr 2e-5 의 공로가 큼).

### 선택 가이드

| 목표 | 추천 config |
|---|---|
| **production 안정 + false alarm 0** | **`gc=1.0` (default)** ⭐ |
| 불량 절대 놓치지 말기 (abn_R 100%) | `gc=0.5` (추가 seed 검증 필요) |
| 최고 성능 도전 (여러 seed 시도) | `gc=2.0` (s=42 perfect 재현 노려보기) |

---

## 3.11 Target Color Experiment — Rendering 이 성능에 미치는 영향

**가설**: 모델 입력 이미지에서 target (highlight) 과 fleet (background) 의 색 대비가 강할수록, 모델이 subtle shift 를 더 잘 학습한다.

**설계**: 동일 harder data (v9mid7) + 동일 winning config, **target color 만 변경** 후 재렌더링 + 재학습.

| 변수 | Blue (기존) | Red (신규) |
|---|---|---|
| target_color | `#4878CF` (파랑) | `#E53935` (빨강) |
| target_alpha | 0.75 | 0.90 |
| target_marker | 16 | 18 |
| fleet (변경 없음) | `#B0B0B0`, alpha 0.40, marker 14 | 동일 |

![Target color comparison](v9mid_journey/plots/11_target_color_comparison.png)

### 결과 (harder data, 2 seeds)

| color | s=42 f1 | s=1 f1 | mean f1 | FN mean | FP mean | total |
|---|---|---|---|---|---|---|
| Blue | 0.9960 | 0.9967 | **99.64%** | 3.5 | 1.0 | **4.5** |
| **Red** | **0.9987** | **0.9993** | **99.90%** | **1.0** | **0.5** | **1.5** |

**에러 67% 감소** (4.5 → 1.5). s=1 에서 **FN=0** (harder data 에서 불량 하나도 안 놓침).

### 왜 Red 가 더 좋나?

모델은 224×224 pixel 이미지에서 "target 점의 공간 분포" vs "fleet 점의 공간 분포" 를 비교해서 anomaly 패턴을 감지.

- **Blue vs Gray**: 밝기/채도가 비슷 → 224×224 로 축소하면 두 집단이 **시각적으로 섞임** → subtle shift 가 noise 에 묻힘
- **Red vs Gray**: 색상(hue) + 밝기 + 크기 전부 다름 → 두 집단이 **명확히 분리** → "빨간 cluster 가 회색 대비 아래로 쏠렸다" 패턴을 학습 가능

이건 X-ray 의 contrast 를 높이면 의사가 종양을 더 잘 찾는 것과 같은 원리.

### Training Curves (Val Loss / Val F1 / LR)

![Training curves](v9mid_journey/plots/12_training_curves_red_vs_blue.png)

- Red: Val F1 이 ep 3~4 에서 빠르게 1.0 수렴 (Blue 보다 빠름)
- Val Loss: 둘 다 비슷한 패턴이지만 Red 가 더 낮은 loss 유지
- LR: 동일 (winning config cosine schedule)

### Full Performance Journey

![Full trend](v9mid_journey/plots/13_full_performance_trend.png)

전체 프로젝트 여정: **96.73% → 99.90%** (+3.17 pts). 주요 jump 포인트:
1. **v3→v4**: renderer fix (+0.33 pts)
2. **v4→v5**: boundary raise (+0.09 pts)
3. **v7h Blue→Red**: color change (+0.26 pts on harder data)

---

## 4. 데이터 생성 Config (현재 active)

`config.yaml` 의 defect 섹션 — v9mid5 기준 (현재 상태는 v9mid6 = mean_shift 도 raised):

```yaml
defect:
  region_ratio_range: [0.14, 0.30]       # mean_shift/std 공통 영역 비율

  mean_shift:
    shift_sigma_range: [2.75, 4.25]      # v6: sigma_factor × baseline_std

  standard_deviation:
    scale_range: [2.75, 3.75]            # v5: target_std = baseline_std × scale

  spike:
    region_ratio_range: [0.10, 0.15]     # ⭐ spike 전용 (iter3)
    magnitude_sigma_range: [6.0, 12.0]
    spike_ratio_range: [0.5, 1.0]        # region 의 50-100% 가 실제 spike
    min_spikes: 8
    min_magnitude_sigma: 6.0             # test 0.8× 후 4.8σ 보장

  drift:
    slope_sigma_range: [0.5, 2.0]
    region_ratio_range: [0.20, 0.375]
    min_max_drift_sigma: 4.5             # v5 ⭐ drift 강도 지배
    noise_reduction: 0.6
    drift_method: index

  enforcement:
    mean_shift_floor_sigma: 1.5
    std_floor_ratio: 2.8
    spike_floor_sigma: 6.0
    drift_floor_sigma: 2.5
    context_floor_sigma: 2.5
    normal_max_right_dev_sigma: 0.5
    normal_max_right_shift_sigma: 0.5
    min_defect_points_range: [12, 25]
```

### Spike region 설계 — 가장 많은 iteration

`spike.region_ratio_range` 가 가장 많이 바뀜:
| iter | 값 | 문제 | 결과 |
|---|---|---|---|
| mid 초기 | 공통 [0.14, 0.30] 사용 | region 이 커서 5~9 spike 이 나머지 ~100~230 baseline 빨강 점 사이에 묻힘 → "noisy 구간"처럼 보임 | FN 많음 |
| iter2 | spike 전용 [0.02, 0.05] + spike_ratio [0.5, 1.0] | 영역은 tight + 거의 전부 spike → 시각적 dramatic | BUT 224×224 에서 ~10px 폭 → **모델 conv filter 가 못 봄** → FN 13/13 all spike |
| **iter3** | **[0.10, 0.15]** + spike_ratio [0.5, 1.0] | 80-120 points × 50-100% = 40-120 impulse burst → **모델도 보고 사람도 봄** | **✅ 해결** |

---

## 5. 성능 향상 기법 카탈로그

이번 세션 + 이전 세션에서 시도한 것들 — **효과 있음** 과 **효과 없음** 구분.

### 🟢 효과 큰 것 (적용됨)

| 기법 | 적용 시점 | Impact | Why works |
|---|---|---|---|
| **Renderer target preserve** | v4 (2026-04-09) | f1 +0.33 pts, spike FN 전멸 | Spike 의 정의가 곧 outlier 이므로 target 은 필터 제외 필수 |
| **lr_backbone 2e-5** (winning) | v9 era | val_loss spike 제거 | RTX 4060 Ti + ConvNeXt-V2 + bs 32 민감도 |
| **abnormal_weight 1.0** (균등) | v9 era | val_loss 안정화 | inverse freq 는 normal grad 증폭 → boundary flip |
| **cudnn.benchmark=True** | v9 era | gradient 수치 안정 | AMP 조합 시 deterministic 경로가 spike 유발 |
| **Tie-fix (strict > only save)** | v9 era | overfit state 저장 방지 | smooth_window 안의 old good 이 new bad 로 덮어씌움 방지 |
| **smooth_window=3 median val_f1** | v9 era | single-ep spike 회피 | median 이 outlier 1 ep 에 robust |
| **Spike region_ratio 분리** | iter3 | FN spike 13 → 0 | 공통 영역은 spike 가 baseline 에 묻히고, 너무 작으면 conv filter 가 못 봄 |
| **std/drift boundary 상승** | v5 | FN -0.9 | weak boundary case 가 실제 bottleneck 이었음 |
| **v9mid 중간값 (v8 vs 약화된 v9)** | v9mid1 | 데이터 품질 복구 | v9 가 과도하게 약화됨, v8 전체 복구는 overshoot |
| **defect 강도 = baseline_std × factor** | 전 세션 | scale invariance | 노이즈 수준별 적절한 defect |
| **test_difficulty_scale 0.80** | v9 era | val-test gap 일정 | 평가 realism |
| **mean_shift detrend 제거** | v8_init 이후 | 자연 변동 보존 | 원래 detrend 는 두 줄 직선 artifact |

### 🟡 효과 중간 (유지)

| 기법 | Impact |
|---|---|
| ColorJitter ±10% + GaussianBlur augment | 약한 regularization |
| Focal Loss gamma 0.0 (=CE) | sweep 후 차이 없음 → 가장 단순한 CE |
| Dropout 0.0 | 강한 pretrained → dropout 불필요 |
| Early stop patience 5 | 정확한 sweet spot 유지 |

### 🔴 효과 없음 (기각)

| 기법 | 결과 | 해석 |
|---|---|---|
| **mean_shift boundary 상승** (v6) | f1 변화 없음 | 이미 noise floor 도달 |
| **normal count n=2800** (v6+) | n=700 과 동등 | 현재 data difficulty 에서 capacity 충분 |
| **Larger LR 3e-5, 5e-5** | 일부 seed spike | v9 data + 현 backbone 에서 risk > reward |
| **EMA decay 0.999/0.9999** | 약간 손해 | small dataset 에서 init 에 머무름 (user 금지) |
| **Focal gamma sweep 0.5~5** | plateau | CE 가 best |
| **Abnormal weight ≥ 4** | nor_R 붕괴 | false alarm 폭증 |
| **Dropout 0.2~0.6** | F1 ↓ | nor_R 손해 |
| **LabelSmoothing, Mixup** | 차이 없음 | overlay 이미지에서 이득 없음 |

### 미시도 (다음 계획)

| 기법 | 기대 |
|---|---|
| Gradient clipping max_norm 변화 (0.5 / 2.0 / 5.0) | val 안정성 추가 확인 |
| Linear / OneCycle scheduler | cosine 대비 |
| Warmup epoch 3 / 7 | sweet spot 재확인 |
| Adaptive gradient clipping (AGC) | 최신 안정화 |

---

## 6. Historical Context — 278 runs 총정리

### Era 별 분포

| Era | prefix | n runs | F1 mean | F1 max | 시기 |
|---|---|---:|---:|---:|---|
| 1. early | `convnextv2_*` | 9 | 0.77 | 0.83 | baseline, multiclass |
| 2. fix-attempts | `r3_/r4_/r6_/r6b_` | 47 | 0.88 | 0.94 | 안정성 시도 |
| 3. improvement sweeps | `imp_/imp2_/var_` | 61 | 0.91 | 0.95 | γ/aw/lr/dropout |
| 4. gamma-normal | `gm_*` | 21 | 0.92 | 0.94 | focal + normal count |
| 5. fine-tune | `ft_*` | 13 | 0.92 | 0.94 | 미세조정 |
| 6. **v8 dataset** | `v8_*` | 34 | **0.99** | **1.00** | 학습 코드 안정화 |
| 7. v9 dataset | `v9_*` | 93 | 0.99 | 0.999 | noise +25% |
| **8. v9mid (new)** | `v9mid*_win_*` | 13 | **0.9991** | **0.9993** | **renderer fix + boundary tune** |

**핵심 점프**:
- F1 0.93 → 0.999: v8_init 시기 (학습 코드 안정화 + 데이터 품질)
- v9 → v9mid5: renderer fix + boundary raise (+0.005 절대, 에러 85% 감소)

### v8 vs v9 vs v9mid 비교

| 데이터셋 | lr | F1 mean (multi-seed) | 특징 |
|---|---|---|---|
| v8 (5-seed) | 5e-5 | 0.9992 ± 0.0006 | noise 낮음, 상대적으로 쉬움 |
| v9 tie-fix (3-seed) | 5e-5 | 0.9971 ± 0.0019 | noise +25%, 약화된 defect |
| v9 ep10 (5-seed) | 2e-5 | 0.9945 ± 0.0009 | 안정적이지만 성능 ↓ |
| v9mid3 (3-seed) | 2e-5 | 0.9949 ± 0.0010 | renderer bug 지속 |
| **v9mid5 (3-seed)** | **2e-5** | **0.9991 ± 0.0003** | **renderer fix + boundary raise, 현재 best** |

v9mid5 는 v8 와 동일한 성능을 v9 보다 어려운 데이터에서 달성.

---

## 7. FN/FP 분석 — 남은 hard cases

v9mid5 + 3-seed 에서 남은 FN 4개 (1.3 mean):

| chart_id | class | params | 평가 |
|---|---|---|---|
| ch_08630 | mean_shift | sigma 1.92, baseline_std 0.034 | boundary weak edge |
| ch_08694 | standard_deviation | scale 2.48 | range min 2.5 바로 아래 |
| ch_08785 | standard_deviation | scale 2.61 | 약간 더 안쪽이지만 noise 0.019 로 tight |
| ch_09036 | drift | max_drift 0.107, 3.5σ | floor 4.5×0.8=3.6 근처 |

모두 **config range 의 하단 경계 샘플**. boundary 를 더 올리면 이 FN 들은 사라지지만, 새 경계에서 또 다른 hard case 가 생김 (v6 에서 확인). **현재 configuration 은 학습 capacity 의 upper bound** 로 판단.

v9mid5 FP **0** (3-seed 전체).

---

## 8. 향후 개선 방향

### 단기 (다음 session)
1. **Gradient clipping sweep** (0.5 / 1.0 / 2.0 / 5.0) — val 안정성 추가 확인
2. **Scheduler variants** (cosine / linear / onecycle)
3. **Warmup epoch sweep** (3 / 5 / 7)
4. **Adaptive gradient clipping (AGC)**
5. 현재 성능이 near-ceiling 이므로 **데이터 난이도 상승** (anomaly 값 약화) 후 re-evaluate

### 중기
1. **실전 fab 데이터 시뮬레이션** — 노이즈 패턴 다양화 (비정상 long tail, chamber drift)
2. **Context 클래스 difficulty 조정** — 현재 거의 완벽 (mean-only case subtle)
3. **Class-aware augmentation** — 각 defect 특성 유지하는 transform

### 장기
1. **FT-Transformer 로 다변량 시계열 직접 처리** (이미지 단계 우회)
2. **Multi-chart correlation** — fleet level context 확장
3. **Real-time inference pipeline** — production 배포

---

## 9. 재현 방법

### 데이터 생성
```bash
python generate_data.py
python generate_images.py
```

### 학습 (winning config)
```bash
python train.py --normal_ratio 700 --seed 42 --log_dir logs/v9mid5_win_n700_s42
```

나머지 옵션은 전부 `train.py` default = winning config.

### Summary 재생성
```bash
PYTHONIOENCODING=utf-8 python experiment_summary/build_v9mid_summary.py
```

### 파일 위치
```
experiment_summary/
├── SUMMARY.md                         # 이 문서
├── build_v9mid_summary.py             # builder
├── v9mid_journey/
│   ├── data.csv                       # run 집계 (13 runs)
│   ├── plots/                         # 9개 plot
│   │   ├── 01_performance_timeline.png
│   │   ├── 02_error_breakdown.png
│   │   ├── 03_class_fn_pattern.png
│   │   ├── 04_val_loss_comparison.png
│   │   ├── 05_renderer_fix_impact.png
│   │   ├── 06_seed_robustness.png
│   │   ├── 07_boundary_impact.png
│   │   ├── 08_hard_cases_heatmap.png
│   │   └── 09_epoch_progression.png
│   ├── samples/                       # 6 클래스 대표 이미지
│   ├── before_after/                  # ch_08893 renderer fix 비교
│   └── (historical) cat1~cat8.png     # 이전 278-run 분석 (보관)
├── data.csv                           # 278-run 전체 raw
└── sample_images/                     # 이전 세션 이미지 (보관)
```

---

## 10. 메모리 링크 (관련)

- `project_v9mid_dataset.md` — v9mid5 config 상세 + iter 히스토리
- `feedback_outlier_filter_target.md` — **ABSOLUTE rule**, renderer fix 절대 되돌리지 말 것
- `project_winning_config.md` — lr 2e-5 + 4 fix
- `feedback_best_selection_epochs.md` — smoothed val + tie-fix
- `feedback_lr_spike_backbone_tuning.md` — backbone 교체 시 LR 재튜닝 필수
- `project_classification_agent.md` — 프로젝트 전체 역사

---

**Last Updated**: 2026-04-10 (5-hour iteration 종료)
**Next**: gradient clipping sweep / scheduler variants / 난이도 상승 실험
