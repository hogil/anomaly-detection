# EXPERIMENTS — L1 Anomaly Detection

반도체 Fab 계측 이상감지(Binary 분류) 실험 기록.
winning config는 `train.py` default로 박혀 있음. 새 실험은 `run_experiments.py`로 실행.

- **평가 지표**: Binary(`abn_R`/`nor_R`/F1) + (필요 시) 6-class 개별 recall
- **데이터 세대**: v1 → v6 → v8 → v8_init → **v9 (현재)**
- **폴더명 규칙**: `{version}_{config_tag}_n{normal}_s{seed}` — 설정 변경 시 무조건 새 폴더

---

## 1. Winning Config (2026-04-08 기준)

`train.py` default — 추가 인자 없이 돌리면 이 설정.

| Arg | Value | 이유 |
|---|---|---|
| `--epochs` | 20 | cosine T_max = 15 |
| `--lr_backbone` | 5e-5 | 1e-4는 seed 운에 cliff fall |
| `--lr_head` | 5e-4 | backbone ×10 |
| `--warmup_epochs` | 5 | lr peak 점진 도달 |
| `--batch_size` | 32 | 16GB VRAM 여유 |
| `--weight_decay` | 0.01 | 표준 AdamW |
| `--scheduler` | cosine | StepLR보다 후반 학습 유리 |
| `--use_amp` | True | 속도 ↑ |
| `--dropout` | 0.0 | v8_init에서 불필요 확인 |
| `--focal_gamma` | 0.0 | CE가 best |
| `--abnormal_weight` | 1.0 | class balance 가설 기각 |
| `--min_epochs` | 10 | val spike 회피 |
| `--smooth_window` | 3 | 최근 3 epoch |
| `--smooth_method` | median | spike robust |
| `--patience` | 5 | 5 tie 연속 → stop |
| Gradient clipping | max_norm 1.0 | 코드 내장, collapse 방지 |
| `--mode` | binary | abn_R 최우선 |

### Best-selection logic (핵심 혁신)

```python
window = deque(maxlen=3)          # 최근 3 epoch
smoothed = median(window)          # spike 무시
if smoothed >= best:
    best_epoch = epoch
    save model
    if strict >: patience_counter = 0
    else:        patience_counter += 1   # tie
else:
    patience_counter += 1
if patience_counter >= 5: stop
```

---

## 2. 전체 실험 이력 (history compact)

### Round 1~6 — Early exploration

| Round | Best run | Abn_R | Nor_R | F1 | Key change |
|---|---|---:|---:|---:|---|
| R1 (200/cls) | `convnextv2_lr1e4` | 0.750 | – | – | 초기 baseline |
| R2 (500/cls) | `steplr_15_05_hiLR` | 0.827 | – | – | StepLR 도입 |
| R3 (500/cls) | `r3_cosine_mixup` | 0.842 | – | – | cosine + warmup + mixup |
| R4 (1000/cls) | `r4_binary` | 0.959 | 0.993 | – | Normal 산포 제어 + Drift 강화 |
| R5 (1000/cls) | `r5_at_ls` | 0.960 (at) | – | – | 데이터 수정 |
| R6 (1000/cls) | `r6_binary_full` | 0.976 | 0.993 | 0.941 | StepLR15, base binary |
| imp | `imp_cosine` | **0.996** | 0.813 | 0.933 | cosine + aw3 + min8 |
| imp2 | `imp2_cos_aw10` | 0.997 | 0.760 | 0.915 | cosine + aw10 |
| var (800/cls×5) | avg of 5 seeds | 0.973 ± 0.002 | – | – | 800 sweet spot 최안정 |

### v8 data (2026-04-07)

| Run | Abn_R | Nor_R | F1 | 비고 |
|---|---:|---:|---:|---|
| `v8_base` (n=700) | 0.999 | 0.960 | 0.988 | 새 데이터 첫 학습 |
| `v8_init_n4` (n=2800, single seed) | 0.9987 | 0.9933 | 0.9960 | normal 다양성 확장 |

### v8_init multi-seed (25 trials, 2026-04-08)

5 normal counts × 5 seeds. `lr 5e-5 + clip 1.0 + ep 20 + min_ep 10` (collapse-safe).

| normal | abn_R mean ± std | nor_R mean ± std | **F1 mean ± std** | best_ep |
|---:|---:|---:|---:|---:|
| 700 | 0.9989 ± 0.0011 | 0.9987 ± 0.0009 | 0.9988 ± 0.0005 | 17.6 |
| 1400 | 0.9971 ± 0.0037 | 0.9995 ± 0.0007 | 0.9983 ± 0.0018 | 16.8 |
| 2100 | 0.9981 ± 0.0022 | 0.9995 ± 0.0012 | 0.9988 ± 0.0010 | 17.2 |
| **2800** ⭐ | **0.9987 ± 0.0016** | **0.9997 ± 0.0006** | **0.9992 ± 0.0007** | 17.4 |
| 3500 | 0.9960 ± 0.0035 | 0.9992 ± 0.0007 | 0.9976 ± 0.0018 | 18.4 |

**발견**:
- ∩ 곡선 — n=2800 peak, n=3500 dip (over-normalized → abn 식별력 희석)
- Class imbalance 가설 기각 — CE + pretrained backbone은 1:5 ~ 4:5 모두 강건
- FN 거의 정체, FP가 주로 감소 → normal 다양성이 false alarm을 줄임
- n=2800 권장 (F1/nor_R 최고 + 안정), n=700은 경제성 대안 (0.0004 차이)

### v9 data (2026-04-08 현재)

**변경**: noise +25% (`gaussian.sigma [0.020,0.045]→[0.025,0.055]`), `test_difficulty_scale 0.85→0.80`, sparse region 45% → 62%.

**winning config 검증 (seed 1, worst case)**:

| Setting | best_ep | test_f1 | abn_R | nor_R | FN | FP |
|---|---:|---:|---:|---:|---:|---:|
| 단순 best (val_f1 단일 epoch) | 10 | 0.9927 | 0.9880 | 0.9973 | 9 | 2 |
| post-hoc avg5 (마지막 5 ep 평균) | – | 0.9980 | 1.0000 | 0.9960 | 0 | 3 |
| **smooth med3 patience 5** ⭐ | 15 | **0.9987** | **1.0000** | 0.9973 | **0** | **2** |

**82% error 감소** (11 → 2). 이 설정이 `train.py` default로 박혔음.

---

## 3. 핵심 발견 & 폐기된 가설

| 가설 | 검증 | 결론 |
|---|---|---|
| Class imbalance가 abn_R 낮춤 | v8_init 25 trial (aw 1.0/3.0/10.0) | **기각** — pretrained + CE가 imbalance에 강건 |
| Dropout 0.5가 필요 | v8_init | **기각** — 0.0이 best |
| Focal gamma > 0 필요 | v8_init | **기각** — CE(gamma 0)가 best |
| StepLR이 cosine보다 나음 | R3 | **기각** — cosine이 후반 학습에 유리 |
| Normal threshold 0.7이 효과 | predictions 4Q 분석 | **기각** — 모델이 이미 confident |
| lr_backbone 1e-4가 default | v8seed, seed 1 | **기각** — seed에 따라 cliff fall, 5e-5로 하향 |
| 단일 epoch val_f1 기준 best | v9 seed 1 | **기각** — spike 잠금, FN 9 발생 |
| FN은 normal 수로 개선 | v8_init | **기각** — FN 정체, FP만 감소 |

---

## 4. Research 기법 우선순위 (미적용, 적용 순)

출처: `memory/project_research_overfitting.md` (15+ 논문 조사)

| 우선 | 기법 | 구현 난이도 | 예상 효과 | 상태 |
|:---:|---|---|---|---|
| 1 | **EMA of weights** (decay 0.9999) | 쉬움 (30줄) | 큼 | 미적용 — train.py 수정 필요 |
| 2 | Smoothed median val selection | 쉬움 | 큼 | **적용 완료** |
| 3 | Label smoothing 0.1 | 1줄 | 중간 | `run_experiments.py` reg 그룹 |
| 4 | DropPath 0.2 | 1줄 | 중간 | ConvNeXt 내장 옵션 |
| 5 | Layer-wise LR decay | 중간 | 중간 | 미적용 |
| 6 | Mixup α=0.2 | 쉬움 | 작음~중간 | `run_experiments.py` reg 그룹 |
| 7 | SWA | 쉬움 | EMA와 유사 | 미적용 |

### 참고 논문
Izmailov 2018 (SWA), Tarvainen & Valpola 2017 (Mean Teacher EMA), Morales-Brotons 2024, Foret 2021 (SAM), Prechelt 1998, Cawley & Talbot 2010, Keskar 2017, Müller 2019 (LS), Wortsman 2022 (Model Soups), Woo 2023 (ConvNeXt-V2).

---

## 5. Pending Experiments — `run_experiments.py`

`python run_experiments.py` 하나로 모두 실행. skip-if-exists. 기존 결과 절대 삭제 안 함.

### Group `sweep` — v9 normal_ratio × multi-seed (15 runs)
v8에서 확인한 n=2800 sweet spot이 v9(noise 강화)에서도 유지되는지 재검증.

| n | seeds | 이름 |
|---:|---|---|
| 700 | 1, 2, 42 | `v9x_n700_s{1,2,42}` |
| 1400 | 1, 2, 42 | `v9x_n1400_s{1,2,42}` |
| 2100 | 1, 2, 42 | `v9x_n2100_s{1,2,42}` |
| 2800 | 1, 2, 42 | `v9x_n2800_s{1,2,42}` |
| 3500 | 1, 2, 42 | `v9x_n3500_s{1,2,42}` |

### Group `reg` — regularization ablation at n=2800 s42 (5 runs)

| 이름 | 변경 | 가설 |
|---|---|---|
| `v9reg_ls01_n2800_s42` | `--label_smoothing 0.1` | val saturation 완화 |
| `v9reg_mix02_n2800_s42` | `--use_mixup --mixup_alpha 0.2` | strong regularization (overlay 이미지 주의) |
| `v9reg_drop02_n2800_s42` | `--dropout 0.2` | v8_init에서는 0.0 best, v9 재확인 |
| `v9reg_fg20_n2800_s42` | `--focal_gamma 2.0` | v8_init에서는 CE best, 재확인 |
| `v9reg_wd05_n2800_s42` | `--weight_decay 0.05` | overfitting 완화 |

### Group `lr` — LR 민감도 ablation at n=2800 s42 (3 runs)

| 이름 | 변경 | 가설 |
|---|---|---|
| `v9lr_bb3e5_n2800_s42` | `--lr_backbone 3e-5 --lr_head 3e-4` | 더 보수적 |
| `v9lr_bb1e4_n2800_s42` | `--lr_backbone 1e-4 --lr_head 1e-3` | 이전 default, cliff fall 재확인 |
| `v9lr_warm8_n2800_s42` | `--warmup_epochs 8` | 느린 warmup |

### Group `mc` — multiclass 보조 (1 run)

| 이름 | 변경 | 목적 |
|---|---|---|
| `v9mc_n2800_s42` | `--mode multiclass` | 6-class 개별 recall 확인 (binary 주, mc 보조) |

**총 24 runs.** 예상 실행 시간: ~5 분/run × 24 = **~2 시간** (RTX 4060 Ti).

---

## 6. 실행 방법

### 노트북 (4060 Ti 16GB, fp16, sequential)

```bash
# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"

# 전부 실행 (skip-if-exists)
python run_experiments.py

# 그룹 선택
python run_experiments.py --groups sweep reg

# 명령만 확인 (dry-run)
python run_experiments.py --dry-run

# 기존 결과 요약만 재생성
python run_experiments.py --only-summary
```

### H200 폐쇄망 서버 (Ubuntu 24, 32 코어, 384GB RAM, H200 × 2)

자세한 셋업: [`SERVER_SETUP.md`](SERVER_SETUP.md)

```bash
# 한 방에 (data → images → 24 실험)
bash run_pipeline.sh

# 부분 실행 (data/이미지 이미 존재)
bash run_pipeline.sh skip-data skip-img

# 그룹만
bash run_pipeline.sh skip-data skip-img sweep
```

`--server h200` 프로파일이 자동 주입하는 학습 args:

| 옵션 | 값 | 효과 |
|---|---|---|
| `--precision` | bf16 | H200 네이티브, GradScaler 불필요 |
| `--compile` | (flag) | torch.compile max-autotune |
| `--batch_size` | 256 | bs 8× (4060 Ti 32 → 256) |
| `--num_workers` | 16 | 32 코어 절반 (GPU 2장 동시) |
| `--prefetch_factor` | 8 | 큰 배치 + 빠른 GPU |

**예상 시간 (전체 24 runs)**: 4060 Ti 순차 ~120 분 → H200 × 2 병렬 **~12 분 (10× 가속)**.

### 산출물

```
logs/{exp_name}/
    best_model.pth              # 최고 모델 가중치
    best_info.json              # 성능 + hparams + timing
    history.json                # epoch별 기록
    training_curves.png         # loss/f1/lr 곡선
    confusion_matrix.png        # NT=0.5
    confusion_matrix_nt.png     # NT=0.7
    predictions/
        tn_normal/              # cap 100
        fn_abnormal/            # 전부 (false negatives)
        fp_normal/              # 전부 (false positives)
        tp_abnormal/            # cap 100

logs/experiments_summary.json   # 통합 요약 (run_experiments.py 출력)
```

---

## 7. 절대 규칙 (이 저장소)

1. **⛔ 학습 결과 폴더 삭제 금지** — `logs/<run_dir>/`. 새 실험은 무조건 새 폴더명.
2. **Binary 학습 우선** — `abn_R`가 최우선 지표. mc는 보조.
3. **성능은 항상 2개 보고** — Binary(abn_R/nor_R/F1) + (mc 때만) 6-class 개별.
4. **Smoothed median val_f1 기준** — 단일 epoch val 금지 (spike 위험).
5. **GPU 필수** — CUDA 확인 후 사용.
6. **추론 입력은 tabular만** — images는 파이프라인 내부에서 생성.
7. **설정 변경 즉시 기록** — `CLAUDE.md`, `memory/`, 이 파일 업데이트.

---

## 8. 다음 단계

1. `run_experiments.py` 전체 실행 (Phase 1 pending 완료)
2. 결과 분석 → 최고 조합 확정
3. **Phase 2: EMA of weights 적용** — train.py 수정 필요 (`ema_design.md` 참조)
4. Phase 3: Inference 파이프라인 production 검증 (`inference.py`)
5. Phase 4: End-to-End Orchestration (FastAPI + React)
