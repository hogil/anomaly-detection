# 실험 결과 정리 (binary, ConvNeXtV2-Tiny)

생성일: 2026-04-09 · 데이터 출처: `logs/v8*` + `logs/v9*` 의 89 runs (binary mode, `best_info.json` 기준)

원본 raw data: [`data.csv`](data.csv)
재생성: `python experiment_summary/build_summary.py`

---

## 0. 공통 설정

| 항목 | 값 |
|---|---|
| Backbone | `convnextv2_tiny.fcmae_ft_in22k_in1k` (28.3M params, ImageNet-22k → 1k) |
| Mode | binary (normal vs abnormal) |
| Test set | normal 750 + abnormal 750 (5 type × 150) |
| Optimizer | AdamW + warmup 5ep + cosine |
| Loss | FocalLoss(γ=0) ≈ CrossEntropy |
| Metric 우선순위 | `abn_R` (불량 잡기) ≥ `F1` ≥ `nor_R` |

**데이터셋 변형**
- **v8**: 기본 noise. test_difficulty 0.85
- **v9**: noise +25%, sparse region 62%, test_difficulty 0.80 (더 어려움)

---

## Category 1 — 데이터셋 크기 sweep (v8_init, single seed)

데이터를 더 크게 만들어서 학습 (각 클래스 균등 비율 유지). single seed 라 noise 큼 — Cat 2 의 multi-seed 가 더 신뢰성 있음.

| normal | run | F1 % | abn_R % | nor_R % | best_ep |
|---:|---|---:|---:|---:|---:|
| 700  | v8_init    | 99.07 | 98.80 | 99.33 | 2 |
| 1400 | v8_init_n2 | 99.07 | 98.67 | 99.47 | 5 |
| 2100 | v8_init_n3 | 99.27 | 99.33 | 99.20 | 3 |
| 2800 | v8_init_n4 | **99.40** | 99.20 | 99.60 | 3 |
| 3500 | v8_init_n5 | 99.00 | 98.40 | 99.60 | 5 |

![Cat 1](cat1_dataset_scale.png)

**관찰**
- ∩-curve. n=2800 정점, n=3500 에서 dip.
- single seed 라 ±0.5 정도의 noise 가 있음 → Cat 2 의 multi-seed 결과로 검증 필요.

---

## Category 2 — Normal 갯수만 늘림 (multi-seed, v8 dataset)

`--normal_ratio` 만 바꾸고 abnormal 은 고정. 5 seeds (s1, s2, s3, s4, s42) 평균.

### Cat 2a — v8 dataset, 5 seeds × 5 normal counts = 25 runs

| normal | n_seeds | F1 % mean±std | abn_R % | nor_R % |
|---:|---:|---:|---:|---:|
| 700  | 5 | 99.88 ± 0.05 | 99.89 ± 0.10 | 99.87 ± 0.08 |
| 1400 | 5 | 99.83 ± 0.16 | 99.71 ± 0.33 | 99.95 ± 0.07 |
| 2100 | 5 | 99.88 ± 0.09 | 99.81 ± 0.20 | 99.95 ± 0.11 |
| **2800** | 5 | **99.92 ± 0.06** ⭐ | **99.87 ± 0.15** | **99.97 ± 0.05** |
| 3500 | 5 | 99.76 ± 0.16 | 99.60 ± 0.32 | 99.92 ± 0.07 |

![Cat 2](cat2_normal_count.png)

### Cat 2b — v9 dataset, lr3tie config, single seed s1

같은 sweep 을 새 v9 데이터 + lr3tie config 로 (개별 single seed).

| normal | run | F1 % | abn_R % | nor_R % |
|---:|---|---:|---:|---:|
| 700  | v9_lr3tie_n700_s1  | 99.40 | 99.87 | 98.93 |
| **1400** | **v9_lr3tie_n1400_s1** | **99.93** ⭐ | **99.87** | **100.00** |
| 2100 | v9_lr3tie_n2100_s1 | 99.67 | 99.33 | 100.00 |
| 2800 | v9_lr3tie_n2800_s1 | 96.13 ⚠️ | 92.27 ⚠️ | 100.00 |

**관찰**
1. **v8 sweet spot = n=2800** (multi-seed F1 99.92 ± 0.06)
2. **v8 n=3500 dip** — 데이터 너무 많으면 abn 식별력 희석 → abn_R 99.60 까지 하락
3. **v9 lr3tie 단일시드는 n=1400 이 best** — n=2800 에서 LR spike 로 collapse (96.13)
4. FN (불량 놓침) 은 normal 늘려도 거의 안 줄어들고, FP 만 줄어듦. → FN 개선은 다른 방향 필요

---

## Category 3 — Noise 증가 영향 (v8 → v9, n=2800)

같은 학습 config 에서 데이터 noise 만 +25% 증가시킨 영향.

| dataset | n_seeds | F1 % | abn_R % | nor_R % |
|---|---:|---:|---:|---:|
| **v8 baseline** (lr 5e-5)         | 5 | **99.92 ± 0.06** | **99.87 ± 0.15** | 99.97 ± 0.05 |
| v9 tie-fix (+25% noise, lr 5e-5)  | 3 | 99.71 ± 0.19 | 99.69 ± 0.13 | 99.73 ± 0.29 |
| v9 ep10 (+25% noise, lr 2e-5)     | 5 | 99.45 ± 0.09 | 98.93 ± 0.19 | 99.97 ± 0.05 |

![Cat 3](cat3_noise_impact.png)

**관찰**
1. **+25% noise → F1 -0.21pt ~ -0.47pt** (99.92 → 99.71/99.45)
2. **abn_R 가 F1 보다 더 떨어짐** — noise 추가 시 불량 식별이 가장 먼저 무너진다
3. v9 tie-fix (lr 5e-5 + tie-update best save) 가 v9 ep10 (lr 2e-5) 보다 성능은 좋으나 std (0.19) 가 큼 → spike 위험
4. **noise 증가 → 불량 강도/스케일이 baseline_std 비례라 결국 noise 도 같이 증가하는 셈** → fleet 대비 신호 비율은 비슷, 그러나 sparse region 62% + test_diff 0.80 이 같이 증가해서 종합 난이도 높아짐

---

## Category 4 — LR 변경으로 spike 감소 + 성능 개선

v8 dataset 에서 잘 되던 lr 5e-5 가 v9 (noise +25%) 로 가면서 깨졌다. LR 을 낮추거나 best 갱신 로직 (tie) 을 바꿔서 안정화 시도.

### Cat 4a — n=2800 apples-to-apples

| config | LR | n_seeds | F1 % | abn_R % | nor_R % |
|---|---:|---:|---:|---:|---:|
| **v8 baseline**       | 5e-5 | 5 | **99.92 ± 0.06** | 99.87 ± 0.15 | 99.97 ± 0.05 |
| v9 tie-fix            | 5e-5 | 3 | 99.71 ± 0.19 | 99.69 ± 0.13 | 99.73 ± 0.29 |
| v9 ep10               | 2e-5 | 5 | 99.45 ± 0.09 | 98.93 ± 0.19 | 99.97 ± 0.05 |
| v9 lr3tie ⚠️          | 3e-5 | 1 | 96.13 | 92.27 | 100.00 |

⚠️ v9 lr3tie 는 n=2800 에서 single seed 가 spike collapse. 다른 n 에서는 best 였음 (Cat 4b 참고).

### Cat 4b — v9 lr3tie 의 normal_count sweep (single seed s1)

| normal | F1 % | abn_R % |
|---:|---:|---:|
| 700  | 99.40 | 99.87 |
| **1400** | **99.93** ⭐ | **99.87** |
| 2100 | 99.67 | 99.33 |
| 2800 | 96.13 ⚠️ | 92.27 ⚠️ |

![Cat 4](cat4_lr_tuning.png)

**LR 튜닝 발견사항**
1. **LR 5e-5 (v8 default) 는 v9 에서 spike 위험** — 노이즈 증가가 loss landscape 거칠게 만듦
2. **LR 2e-5 (v9 ep10)** 는 spike 안전하지만 abn_R 0.5pt 손해
3. **LR 3e-5 (v9 lr3tie)** 는 n=1400 에서 single seed F1 99.93 (전체 중 best) 달성, 그러나 n=2800 에서 collapse — **LR/데이터양 trade-off** 존재
4. **tie-update best save** (`val_f1` 동률 시 최신으로 갱신) 가 LR 5e-5 의 spike 한 부분을 완화
5. backbone 교체 시 LR 재튜닝 필수 — ConvNeXtV2-Tiny 의 sweet LR ≠ 다른 backbone

---

## 종합

### Best models so far (v9 dataset 기준)

| Rank | Run | F1 | abn_R | n_seeds | 비고 |
|---:|---|---:|---:|---:|---|
| 🥇 | `v9_lr3tie_n1400_s1` | 99.93 | 99.87 | 1 | 단일시드 best, 재검증 필요 |
| 🥈 | `v9_tiefix_n2800_s1` | 99.93 | 99.87 | (s1 only) | 3-seed mean 99.71 |
| 🥉 | `v9_ep10_n2800_s42` | 99.60 | 99.20 | 5-seed mean 99.45 | 가장 안정적 |

### v8 → v9 → 다음 단계

1. **v8 시대**: lr 5e-5 + n=2800 + 5-seed = F1 99.92 ± 0.06 (안정적인 baseline)
2. **v9 (noise +25%)**: lr 5e-5 그대로 쓰면 spike. 해결 시도:
   - lr 2e-5 (보수적) → 안정 but 성능 -0.47pt
   - tie-update + lr 5e-5 → 성능 회복 but std 증가
   - lr 3e-5 + tie → n=1400 best, n=2800 collapse
3. **다음 할 일**:
   - lr3tie 를 multi-seed 로 검증 (현재 single seed)
   - v9 + n=2800 에서 lr 3e-5 가 collapse 한 원인 진단 (gradient norm 추적?)
   - **EMA/SWA 같은 weight averaging** 시도 — spike 완화에 가장 효과적이라는 연구결과
   - data quality 개선 (FN 감소 위해)

---

## 부록 — 카테고리에 안 들어간 실험

`logs/` 에는 위 4 카테고리 외에도 다음 실험이 있다 (이 요약에선 제외):

- **Backbone 비교** (`v9_bb_*`): efficientnetv2_s, swin_tiny, maxvit_tiny, clip_vit_b16
- **Regularization sweep** (`v9reg_*`): dropout, label smoothing, mixup, weight_decay
- **EMA / SWA** (`v9_ema*`): 가중치 평균
- **Augmentation** (`v9_avg5_*`, `v9_smooth*`): smooth window 변형
- **earlier era** (`gm_*`, `imp2_*`, `ft_*`): focal gamma / abnormal_weight 초기 sweep

필요하면 별도 카테고리로 정리 가능.
