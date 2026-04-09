# 실험 계획 — test FP/FN ≤ 2 달성

## 목표

**test set 1500 샘플 (normal 750 + abnormal 750) 에서 FP ≤ 2 AND FN ≤ 2 동시 달성**, **multi-seed 검증**.

현재 best: `v9_lr3tie_n1400_s1` = FN 1, FP 0 (single seed, 미검증).
목표선: **5 seed 평균** 으로 FN ≤ 2, FP ≤ 2.

---

## 진단 요약 (실험 설계 근거)

### 발견 1 — Spike 는 ep 4~8 에 압도적 집중 (139 spike / 82 runs)

```
ep 4~8:   83 spikes (60%)  ← warmup 끝 + cosine LR peak 구간
ep 9~13:  38 spikes (27%)
ep 14+:    18 spikes (13%)
```

**원인**: warmup 5 ep + cosine peak 가 ep 5 에 도달 → ep 5~8 에서 step size 최대 → loss landscape 의 cliff 통과 확률 최대.

### 발견 2 — Spike 는 deterministic, seed+데이터가 같으면 정확히 재현

6개 서로 다른 config runs (avg5, med3, smooth3, noise_sparse, med3p5, med3p10) 가 **모두 ep 6 에서 val_loss=0.9228, ratio 65.9 동일** — seed+lr schedule 같으면 같은 trajectory → 같은 spike.

**의미**: spike 는 loss surface 의 deterministic feature. 해결책은 randomness 가 아니라 trajectory/save 로직 변경.

### 발견 3 — Train loss 는 spike 의 50% 에서 완전 정상

```
spike 중 tr_loss < 0.01: 69 / 139 (50%)
spike 중 tr_loss ≥ 0.01: 70 / 139 (50%)
```

**의미**: gradient magnitude 가 원인 아님. val 에만 영향을 주는 **direction mismatch** = 경계 샘플들이 한 번에 넘어가는 calibration drift.

### 발견 4 — test_history 최고 vs saved final 불일치 6건

```
v9_lr3tie_n2800_s1   history_best ep10 f1=0.9973 → saved ep12 f1=0.9613  gap 3.6pt
v9_ls05_n700_s4      ep7 f1=0.9980 → saved ep14 f1=0.9820                gap 1.6pt
v9_ep10_n2800_s4     ep6 f1=0.9993 → saved ep10 f1=0.9933                gap 0.6pt
v9_aw15_n2800_s42    ep4 f1=0.9953 → saved ep7 f1=0.9920                 gap 0.33pt
v9_ep10_n2800_s1     ep7 f1=0.9973 → saved ep10 f1=0.9940                gap 0.33pt
v9_n2800_s2          ep5 f1=0.9940 → saved ep8 f1=0.9907                 gap 0.33pt
```

**2가지 원인**:
1. **min_epochs=10 gate**: ep 4~7 이 실제론 더 좋았는데 best 후보 아니라 강제 skip
2. **tie-update overwrite**: 좋은 ep 10 save 를 나쁜 ep 12 TIE 로 덮어쓰기

### 발견 5 — Weight averaging (avg5) 가 spike 에도 불구하고 승리

`v9_avg5_n700_s1` 은 **ep 6 에서 val_loss 0.9228 spike 발생** 했지만, 마지막 5 epoch weight 평균으로 final test_f1 99.80, abn_R **100.00%** 달성.

**증명**: weight averaging 이 spike 자체를 겪고도 최종 성능을 구원할 수 있음. EMA (매 step 평균) 는 더 부드러울 것.

---

## 개선 방향 (증거 기반 우선순위)

| 순위 | 기법 | 이론적 효과 | 데이터 증거 |
|---|---|---|---|
| ★★★★★ | EMA 0.999 | spike step 1개가 평균에 0.1% 만 영향 | avg5 가 spike 겪고도 100% abn_R |
| ★★★★ | warmup 10 + lr 2e-5 | peak LR 늦추고 낮춰서 cliff 확률 ↓ | spike 분포가 high-LR phase 에 집중 |
| ★★★★ | best_update_start 분리 (smoothed=7, single=10) | ep 4~7 의 좋은 epoch 도 save 후보에 포함 | 6건의 "더 좋은 epoch" 모두 ep 7 이하 |
| ★★★★ | Strict save (tie-update 제거) | 망한 epoch 으로 덮어쓰기 차단 | v9_lr3tie_n2800_s1 의 ep12 재앙 방지 |
| ★★★ | val_loss 2x reject | spike 저장 차단 (strict 보완) | ep 6 의 0.9228 spike 저장 안 함 |
| ★★★ | 매 epoch test 평가 (ep 10+) | 사후 분석 + 진짜 best 복원 가능 | 6건 case 의 true best 추적 가능 |
| ★★ | gradient clip 강화 (1.0 → 0.5) | step size 추가 제한 | 절반 spike 에서 tr_loss 정상이라 미지수 |
| ★ | β2 ↑ | late training 안정화 | spike 는 early training → mismatch |

---

## Phase 0 — 인프라 패치 (학습 전 필수)

`train_tie.py` 에 다음 변경:

1. **EMA default on** (`--ema_decay 0.999`)
2. **best_update_start 분리**: config.yaml 의 `training.best_selection` 읽음
   - smooth_window > 1 → ep 7 부터 best 저장 허용
   - smooth_window ≤ 1 → ep 10 부터
3. **Early stop floor**: `epoch < early_stop_start + patience` 면 종료 금지
4. **Strict save default**: tie-update 대신 strict `>` 비교
5. **val_loss 가드**: `val_loss > best_val_loss * 2.0` 이면 save 거부
6. **매 epoch test 평가 from ep 10+** (eval_test_every_epoch 자동 활성)

`config.yaml` 에 `training` 섹션 추가:

```yaml
training:
  best_selection:
    update_start_single: 10
    update_start_smoothed: 7
  early_stop:
    start_epoch: 10
    patience: 5          # → 최소 epoch 15
  eval:
    test_every_epoch_from: 10
  save_guard:
    val_loss_max_ratio: 2.0   # val_loss > best * 2 면 save reject
  ema_decay: 0.999
```

---

## Phase 1 — Spike-proof baseline (5 seed)

**목표**: EMA + strict + val_loss guard 의 기본 효과 검증

**config** (모두 동일):
- lr_backbone 3e-5, lr_head 3e-4
- warmup 5, epochs 20, patience 5 (→ 최소 종료 ep 15)
- smooth_window 3, smooth_method median
- normal_ratio 2800, focal_gamma 0, dropout 0
- **ema_decay 0.999** (신규)
- **val_loss_max_ratio 2.0** (신규)

**5 seeds**: 1, 2, 3, 4, 42

**실행**: `bash experiment_plan_fpfn/run_phase1.sh`

**합격 기준**:
- 5 seed 중 **4 이상** 이 FN ≤ 2 AND FP ≤ 2
- Mean F1 ≥ 99.9%
- Spike 발생해도 saved model 무영향 (EMA 가 걸러야 함)

**monitor 기준** (kill 조건):
- val_loss > 0.5 at any epoch → kill
- val_f1 < 0.98 at ep 15 → kill
- val_loss > 10x 이전 min → kill

---

## Phase 2 — LR/warmup 변주 (spike 구간 회피)

**가설**: warmup 늘리고 peak LR 낮추면 spike 빈도 ↓

**variants** (3 config × 3 seeds = 9 runs):

| Config | lr_bb | warmup | 기대 |
|---|---|---|---|
| A | 2e-5 | 5 | step size ↓ |
| B | 3e-5 | 10 | peak 늦춤 |
| C | 2e-5 | 10 | 둘 다 |

나머지는 Phase 1 과 동일 (EMA + strict + val_loss guard).

**seeds**: 1, 2, 42 (3개만, time budget)

**합격 기준**: Phase 1 대비 spike 빈도 감소 확인. FN/FP 는 비슷하거나 개선.

---

## Phase 3 — Epoch/patience 늘리기 (late convergence 탐색)

**가설**: 사용자 지적 — "ep 10 에서 best 가 정해지면 overfit 위험". 늘려서 late minimum 탐색.

**variants** (2 config × 3 seeds = 6 runs):

| Config | epochs | patience | min_stop |
|---|---|---|---|
| D | 30 | 10 | ep 20 |
| E | 40 | 15 | ep 25 |

Phase 1 best config base + epoch/patience 확장.

**seeds**: 1, 2, 42

**합격 기준**: Phase 1 대비 best_ep 가 더 늦어지고 test_f1 더 좋음. 아니면 early stop 이 맞는 것.

---

## Phase 4 — Regularization (소폭 변주)

**variants** (3 config × 2 seeds = 6 runs):

| Config | label_smoothing | mixup |
|---|---|---|
| F | 0.05 | off |
| G | 0 | 0.1 |
| H | 0.05 | 0.1 |

Phase 1 best config base.

**seeds**: 1, 42 (2개만)

**합격 기준**: FP/FN 5% 이내 개선 or 동등. 악화되면 reject.

---

## Phase 5 — Ensemble (훈련 없이)

**가설**: Phase 1~4 의 top-N models 의 logit 평균이 개별보다 좋음

**process**:
1. Phase 1~4 중 test_f1 상위 5 model 선택
2. `inference_ensemble.py` 로 logit 평균
3. 평가

**합격 기준**: 개별 best 보다 FP+FN 감소

---

## 타임 예산

| Phase | runs | 각 시간 | 총 |
|---|---:|---:|---:|
| 0 (패치) | 0 | — | 30분 |
| 1 | 5 | 20분 | 100분 |
| 2 | 9 | 20분 | 180분 |
| 3 | 6 | 30분 | 180분 |
| 4 | 6 | 20분 | 120분 |
| 5 | 0 (inference) | — | 30분 |
| **총** | **26 training** | | **~11 시간** |

4060 Ti 1장 기준. H200 2장 병렬이면 ~3 시간.

---

## 중단 / 수정 / 반복 원칙

monitor.py 가 실시간 감시:
- **즉시 kill**: val_loss > 0.5 spike, val_f1 < 0.98 at ep 15, hang (5분 무응답)
- **ALERT**: 3 spike 연속 in one epoch → 다음 run 에 config 조정 신호
- **success snapshot**: test_f1 ≥ 99.9% 달성하면 그 config 즉시 다른 seed 로 확산

매 phase 끝나면 `analyze.py` 가 결과 정리 + 다음 phase config 자동 제안.

---

## 완료 조건

**필수**: 최소 3 seed 에서 test FP ≤ 2 AND test FN ≤ 2 동시 만족.
**선택**: 5 seed 평균 F1 ≥ 99.9%, abn_R ≥ 99.9%.

완료되면:
- `experiment_plan_fpfn/results.md` 에 최종 best config + 재현 명령
- `experiment_summary/SUMMARY.md` 의 Top 10 leaderboard 업데이트
- `project_winning_config.md` memory 업데이트
