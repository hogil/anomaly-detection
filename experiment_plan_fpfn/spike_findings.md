# Spike 분석 요약 — 139 spikes / 82 runs

전체 278 runs 중 82 runs (30%) 에서 val_loss spike 발생. 총 139 spike (일부 run 은 여러 번).
Spike 정의: `val_loss_t > 3 × val_loss_{t-1}` AND `val_loss_t > 0.02`.

## 주요 발견

### 1. Spike 는 **ep 4~8 에 압도적 집중**

```
ep 4:  17 spikes
ep 5:  19 spikes   ← warmup 끝 직후
ep 6:  22 spikes   ← cosine peak
ep 7:  14 spikes
ep 8:  11 spikes
ep 9:   8
ep 10: 11
ep 11: 10
ep 12:  4
ep 13+: 미미
```

**ep 4~8 = warmup 5 ep 직후 + cosine LR peak 구간**. step size 가 가장 크고 weight 이동이 빠른 시기에 loss landscape cliff 통과 확률 최대.

### 2. 같은 seed+데이터는 **spike 까지 정확히 재현**

6개 서로 다른 config runs 가 모두 **ep 6 에서 tr_loss=0.0299, val_loss=0.9228, ratio 65.9** 로 **바이트 수준 동일**:

```
v9_avg5_n700_s1
v9_med3_n700_s1
v9_med3p10_e25_n700_s1
v9_med3p5_n700_s1
v9_noise_sparse_n700_s1
v9_smooth3_n700_s1
```

이 6개는 smooth_window / ema / best_selection 등이 서로 다른 config 였지만 seed, 데이터, warmup, 초기 lr 스케줄이 같아서 **같은 weight trajectory 를 타고 같은 spike 를 겪음**.

**함의**: spike 는 loss landscape 의 **deterministic feature**. randomness 가 아니라 특정 weight 위치에서 일어나는 구조적 사건. 해결책은 trajectory/save logic 변경 — seed 만 바꿔서 피할 수 있는 게 아님.

### 3. Train loss 는 spike 의 50% 에서 완전 정상

```
spike 중 tr_loss < 0.01: 69 / 139 (50%)
spike 중 tr_loss ≥ 0.01: 70 / 139 (50%)
```

**의미**:
- gradient magnitude 가 원인이라면 → 해당 batch 의 loss 가 커서 epoch 평균에 반영됐어야 함
- 실제 epoch 평균 tr_loss 는 정상 → gradient 들 **크기** 자체는 평범
- **gradient 방향** 이 val distribution 과 mismatch 되는 weight 위치로 이동하는 것
- 즉 "우연한 거대 step" 이 아니라 **구조적 drift**

### 4. Best test_f1 이 saved final 과 불일치한 case 6건

```
v9_lr3tie_n2800_s1   ep10 f1=0.9973 (history) → saved ep12 f1=0.9613  gap 3.6pt ⚠️
v9_ls05_n700_s4      ep7  f1=0.9980 (history) → saved ep14 f1=0.9820  gap 1.6pt
v9_ep10_n2800_s4     ep6  f1=0.9993 (history) → saved ep10 f1=0.9933  gap 0.6pt
v9_aw15_n2800_s42    ep4  f1=0.9953 (history) → saved ep7  f1=0.9920  gap 0.33pt
v9_ep10_n2800_s1     ep7  f1=0.9973 (history) → saved ep10 f1=0.9940  gap 0.33pt
v9_n2800_s2          ep5  f1=0.9940 (history) → saved ep8  f1=0.9907  gap 0.33pt
```

**2가지 원인**:
1. **min_epochs=10 gate**: ep 4~7 이 실제론 더 좋았지만 best 후보 자격 없어서 강제 skip
2. **tie-update overwrite**: 좋은 ep 10 save 를 더 나쁜 ep 12 TIE 로 덮어쓰기

### 5. Weight averaging 이 spike 를 견딤 — 이미 증명됨

`v9_avg5_n700_s1` 은 **ep 6 에서 val_loss 0.9228 spike 정면으로 맞음**. 그런데 마지막 5 epoch weight 평균으로 final:
- test_f1 **99.80**
- abn_R **100.00%**
- nor_R **99.60%**

즉 **ep 6 spike 의 bad weights 가 5-epoch 평균에서 1/5 비중** 으로 희석되어 최종 성능이 구원됨.

**함의**: EMA (매 step 평균, 1000 step 기준 0.1% 비중) 는 더 강하게 spike 를 흡수할 수 있음.

### 6. 최악 spike 들

| run | ep | tr_loss | val_loss | val/tr ratio |
|---|---:|---:|---:|---:|
| v9_tiefix_n2800_s2 | 7 | 0.0242 | **2.2903** | 94.6 |
| v9_avg5_n700_s1 | 6 | 0.0299 | 0.9228 | 30.9 |
| v9_fix_n700_s42 | 3 | 0.0058 | 0.3748 | 64.6 |
| v9_lr3_n700_s2_p20 | 11 | 0.0026 | 0.2743 | **105.5** |
| v9_fix_n700_s42 | 9 | 0.0030 | 0.1874 | 62.5 |
| v9_cudnn_n700_s42 | 14 | 0.0003 | 0.1011 | **337** |

val_loss 2.29 는 binary cross entropy 기준으로 대다수 sample 을 **완전 반대 방향** 예측. 하지만 모두 **1-2 epoch 안에 회복**. self-correcting.

## 개선 방향 (증거 기반)

### 1. EMA 0.999 ★★★★★
- **avg5 의 성공 사례** 가 유력 증거
- 매 step 매우 약한 (0.1%) update 로 spike 영향 최소화
- AdamW 내부 moments (m_t, v_t) 와 orthogonal — 중복 아님

### 2. Warmup 10 + lr 2e-5 ★★★★
- spike 는 ep 4~8 = warmup 끝 + cosine peak 구간
- warmup 10 이면 peak 가 ep 10 으로 밀림 → 그 때는 model 이 이미 안정화
- lr 2e-5 는 step size 자체를 낮춤

### 3. min_epochs 분리 + strict save ★★★★
- **smoothed (smooth_window > 1) 이면 ep 7 부터 save 허용** — 정작 좋았던 ep 4~7 포착
- **strict `>` 비교** — tie-update 로 망한 weights 덮어쓰기 차단

### 4. val_loss guard ★★★
- `val_loss > best_val_loss * 2.0` 이면 save reject
- spike 순간의 weights 저장 물리적 차단

### 5. 매 epoch test 평가 (ep 10+) ★★★
- 사후 분석 시 "어느 epoch 이 진짜 best 였나" 복원 가능
- Debug + research 가치

### 6. gradient clip 강화 (1.0 → 0.5) ★★
- 절반의 spike 에서 tr_loss 정상이라 효과 미지수
- 적용 비용은 낮으니 phase 5+ 에서 시도 가능

### 7. β2 튜닝 ★
- 사용자 초기 제안이었으나 **데이터가 반증**
- β2 튜닝 효과는 late training 안정화에 있는데 우리 spike 는 early training
- skip
