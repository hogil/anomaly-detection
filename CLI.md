# train.py CLI 가이드

v9 dataset + ConvNeXtV2-Tiny 기준 — 2026-04-09 시점 winning config 포함.

---

## 1. 핵심 사용법

```bash
python train.py \
    --normal_ratio 700 --seed 1 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --min_epochs 10 --patience 20 \
    --normal_threshold 0.9 \
    --log_dir logs/v9_lr3_n700_s1_p20
```

위 명령어 하나로 전체 파이프라인 (data load → train → val → test → best 저장 → test_history 기록 → confusion matrix → predictions 4분면 저장) 완료.

---

## 2. 전체 CLI 옵션 레퍼런스

### 2.1 LR / Optimizer / Scheduler

| 옵션 | default | 설명 | 권장값 |
|---|---|---|---|
| `--lr_backbone` | 2e-5 | backbone peak LR | ConvNeXtV2-Tiny: 2~3e-5 (5e-5 는 spike 위험) |
| `--lr_head` | 2e-4 | head peak LR | backbone × 10 유지 |
| `--warmup_epochs` | 5 | linear warmup 구간 | 5 (작은 모델), 8 (vit 계열) |
| `--weight_decay` | 0.01 | AdamW weight decay | 0.01 표준 |
| `--scheduler` | cosine | cosine / step / plateau | cosine |
| `--step_size` | 10 | step scheduler only | |
| `--step_gamma` | 0.5 | step scheduler only | |
| `--epochs` | 20 | 최대 epoch | 20 (v9) |
| `--batch_size` | 32 | | RTX4060: 32, H200: 64~128 |

### 2.2 Best 선정 / Early Stopping

| 옵션 | default | 설명 |
|---|---|---|
| `--smooth_window` | 3 | val_f1 smoothing window (0 = 비활성 = raw) |
| `--smooth_method` | median | `median` (spike 에 robust) 또는 `mean` |
| `--save_strict_only` | True | `>` 만 best 업데이트. tie 덮어쓰기 방지 |
| `--best_update_start_smoothed` | 7 | smooth_window>1 일 때 best 저장 시작 epoch |
| `--best_update_start_single` | 10 | smooth_window≤1 일 때 best 저장 시작 epoch |
| `--early_stop_start` | 10 | patience counter 시작 epoch |
| `--patience` | 5 | 연속 non-improvement → early stop. **20 으로 주면 사실상 off** |
| `--val_loss_max_ratio` | 2.0 | save guard: val_loss > best_val_loss × 2 면 저장 거부 (degrade 방지) |
| `--val_loss_guard_min_abs` | 0.02 | val_loss 이 값 미만이면 guard off |
| `--avg_last_n` | 0 | 마지막 N epoch weight 평균. **5 권장** (val 포화 문제 회피) |
| `--eval_test_every_epoch` | off | 매 epoch test 평가 → history.json 에 기록 |
| `--min_epochs` | -1 | [DEPRECATED] best 갱신 최소 epoch. -1 = 자동 (smoothed=7, single=10). 직접 지정 가능 (예: `--min_epochs 10`) |

> **Gradient clipping** 은 `max_norm=1.0` 으로 **코드 내 hard-coded** — CLI 플래그 없음. 바꾸려면 `train.py` 직접 수정.

### 2.3 Regularization / Loss

| 옵션 | default | 설명 |
|---|---|---|
| `--dropout` | 0.0 | classification head dropout |
| `--label_smoothing` | 0.0 | label smoothing (0.05~0.1) |
| `--focal_gamma` | 0.0 | 0 = CE, 2.0 = FocalLoss gamma |
| `--abnormal_weight` | 1.0 | binary class weight (1.0=균등, 0=inverse freq 자동) |
| `--label_weights` | "" | multiclass 전용, 예: `"normal=0.5,spike=3.0"` |
| `--ema_decay` | 0.999 | EMA of weights decay (0=비활성, 0.9999=ConvNeXt-V2 official) |
| `--use_mixup` | False | mixup augmentation |
| `--mixup_alpha` | 0.2 | mixup strength |
| `--ohem_ratio` | 0.0 | Online Hard Example Mining (0.75 = top 75%) |

### 2.4 Data / Mode

| 옵션 | default | 설명 |
|---|---|---|
| `--normal_ratio` | 0 | binary 모드 정상 샘플 수 (0 = 전체). 700/1400/2100/2800 추천 |
| `--max_per_class` | 0 | 클래스당 최대 샘플 (0 = 제한 없음) |
| `--mode` | binary | `binary` / `multiclass` / `anomaly_type` |
| `--seed` | 42 | random seed |
| `--normal_threshold` | 0.9 | selected inference/reporting NT. per-run test tuning 금지 |

### 2.5 Model

| 옵션 | default | 설명 |
|---|---|---|
| `--model_name` | convnextv2_tiny.fcmae_ft_in22k_in1k | timm model id |
| `--freeze_backbone_epochs` | 0 | 초기 N epoch backbone freeze |

### 2.6 Runtime / Precision

| 옵션 | default | 설명 |
|---|---|---|
| `--precision` | fp16 | `fp16` / `bf16` / `fp32` — 자세히는 §4 참조 |
| `--use_amp` | True | automatic mixed precision (fp16 시 GradScaler 사용) |
| `--compile` | off | torch.compile (H100/H200 에서 20~50% 가속) |
| `--num_workers` | 4 | DataLoader worker 수. Linux 서버: 8~16 |
| `--prefetch_factor` | 4 | DataLoader prefetch. 서버: 8~16 |

### 2.7 Output

| 옵션 | default | 설명 |
|---|---|---|
| `--log_dir` | logs | 출력 폴더. 매 실험 새 이름 필수 (삭제 금지) |
| `--config` | dataset.yaml | YAML 설정 경로 |

---

## 3. 추천 명령어 (복사 → 붙여넣기)

### 3.1 기본 — v9 seed 1 n=700 (winning minimal)

```bash
python train.py \
    --normal_ratio 700 --seed 1 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --min_epochs 10 --patience 20 \
    --normal_threshold 0.9 \
    --log_dir logs/v9_lr3_n700_s1_p20
```

### 3.2 최고 성능 확인 — n=2800 seed 1 (1 error 재현)

```bash
python train.py \
    --normal_ratio 2800 --seed 1 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --min_epochs 10 --patience 20 \
    --normal_threshold 0.9 \
    --log_dir logs/v9_lr3_n2800_s1_p20
```

### 3.3 avg_last_n 5 (val 포화 회피) — n=1400 seed 1

```bash
python train.py \
    --normal_ratio 1400 --seed 1 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --min_epochs 10 --avg_last_n 5 \
    --normal_threshold 0.9 \
    --log_dir logs/v9_lr3_n1400_s1_avg5
```

### 3.4 Seed 안정성 확인 — 4 seed 병렬

```bash
for s in 1 2 4 42; do
  python train.py \
      --normal_ratio 700 --seed $s \
      --lr_backbone 3e-5 --lr_head 3e-4 \
      --min_epochs 10 --patience 20 \
      --normal_threshold 0.9 \
      --log_dir logs/v9_lr3_n700_s${s}_p20
done
```

### 3.5 H200 서버 최적 — bf16 + compile + 대용량 batch

```bash
python train.py \
    --normal_ratio 2800 --seed 1 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --min_epochs 10 --patience 20 \
    --normal_threshold 0.9 \
    --precision bf16 --compile \
    --batch_size 64 \
    --num_workers 16 --prefetch_factor 16 \
    --log_dir logs/v9_lr3_n2800_s1_h200
```

### 3.6 EMA of weights 실험 — ConvNeXt-V2 official style

```bash
python train.py \
    --normal_ratio 2800 --seed 1 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --min_epochs 10 --patience 20 \
    --normal_threshold 0.9 \
    --ema_decay 0.9999 \
    --log_dir logs/v9_lr3_ema9999_n2800_s1
```

### 3.7 6-class multiclass (Phase 2 확장)

```bash
python train.py \
    --mode multiclass \
    --normal_ratio 0 --seed 1 \
    --lr_backbone 2e-5 --lr_head 2e-4 \
    --min_epochs 10 --patience 10 \
    --label_smoothing 0.05 \
    --normal_threshold 0.9 \
    --log_dir logs/v9_6cls_s1
```

### 3.8 매 epoch test 평가 (trajectory 추적)

```bash
python train.py \
    --normal_ratio 700 --seed 1 \
    --lr_backbone 3e-5 --lr_head 3e-4 \
    --min_epochs 10 --patience 20 \
    --normal_threshold 0.9 \
    --eval_test_every_epoch \
    --log_dir logs/v9_lr3_n700_s1_evalall
```

### 3.9 2-GPU 병렬 sweep (스크립트)

```bash
bash run_sweep_2gpu.sh
# 또는 nohup background
nohup bash run_sweep_2gpu.sh > sweep.log 2>&1 &
tail -f sweep.log
```

---

## 4. precision 옵션 설명 (fp16 / bf16 / fp32)

### 4.1 용어

**부동소수점 정밀도**. 숫자를 몇 비트로 표현하느냐의 차이.

| 타입 | 총 비트 | 지수 (exponent) | 가수 (mantissa) | 표현 범위 | 정밀도 |
|---|---|---|---|---|---|
| **fp32** | 32 | 8 bit | 23 bit | ±10^38 | ~7 자리 유효숫자 |
| **fp16** | 16 | 5 bit | 10 bit | **±6.5×10⁴** (좁음) | ~3 자리 유효숫자 |
| **bf16** | 16 | **8 bit** | 7 bit | ±10^38 (fp32 동일) | ~2 자리 유효숫자 |

### 4.2 핵심 차이

- **fp32 = 표준**. 정확하지만 메모리 2배, 속도 느림.
- **fp16** = "half precision". 메모리 절반, 속도 2~3배. **단점**: 표현 범위가 좁아서 gradient 값이 작으면 **0 으로 underflow**, 크면 **overflow (inf/nan)**. 그래서 **GradScaler** 로 gradient 를 크게 스케일링한 뒤 unscale 하는 보정이 필수 (PyTorch AMP 가 자동 처리).
- **bf16** = "brain float 16" (Google Brain 개발). fp16 과 같은 16-bit 지만 **지수 8 bit (fp32 와 동일)** — 표현 범위가 fp32 수준이라 overflow/underflow 거의 없음. 가수 bit 는 적지만 학습에는 정밀도보다 범위가 더 중요해서 사실상 문제 없음. **GradScaler 불필요**.

### 4.3 하드웨어 지원

| GPU | fp16 | bf16 |
|---|---|---|
| RTX 20xx (Turing) | ✓ | ✗ (시뮬레이션만) |
| RTX 30xx (Ampere) | ✓ | **✓** (Ampere 부터 native) |
| RTX 40xx (Ada) | ✓ | ✓ |
| **RTX 4060 Ti** (이 프로젝트) | ✓ | **✓** |
| A100 / H100 / **H200** | ✓ | **✓** (최적화) |
| TPU | ✗ | ✓ (Google TPU 기본) |

### 4.4 권장

| 상황 | 추천 | 이유 |
|---|---|---|
| **RTX 4060 Ti** 로컬 | `fp16` (default) | 안정적, GradScaler 자동 |
| **H100 / H200** 서버 | **`bf16`** | overflow 걱정 없음, Tensor Core 최적화 |
| 학습 불안정 (nan/inf) | `bf16` 또는 `fp32` | fp16 overflow 회피 |
| 수치 민감 연구 | `fp32` | 정확성 최우선 |

### 4.5 이 프로젝트 사용법

```bash
# 로컬 RTX 4060 Ti (default)
python train.py ... --precision fp16

# H200 서버
python train.py ... --precision bf16 --compile

# 디버깅 (nan 발생 시)
python train.py ... --precision fp32
```

---

## 5. 결과 확인

### 5.1 단일 실험 요약

```bash
python -c "
import json
d = json.load(open('logs/v9_lr3_n2800_s1_p20/best_info.json'))
fn = round((1-d['test_metrics']['abnormal']['recall']) * 750)
fp = round((1-d['test_metrics']['normal']['recall']) * 750)
print(f'ep={d[\"epoch\"]} test_f1={d[\"test_f1\"]:.4f}  FN={fn} FP={fp} total={fn+fp}')
for e in d.get('test_history', []):
    print(f'  ep{e[\"epoch\"]} {e[\"event\"]}: sm={e[\"val_target_smoothed\"]:.4f} FN={e[\"fn\"]} FP={e[\"fp\"]}')
"
```

### 5.2 모든 실험 랭킹

```bash
python -c "
import os, json
rows = []
for d in sorted(os.listdir('logs')):
    if not d.startswith('v9_'): continue
    bf = f'logs/{d}/best_info.json'
    if not os.path.exists(bf): continue
    try:
        j = json.load(open(bf))
        fn = round((1-j['test_metrics']['abnormal']['recall'])*750)
        fp = round((1-j['test_metrics']['normal']['recall'])*750)
        rows.append((d, fn, fp, fn+fp, j.get('test_f1', 0)))
    except: pass
rows.sort(key=lambda r: r[3])
for r in rows[:20]:
    print(f'{r[0]:45s} FN={r[1]:3d} FP={r[2]:3d} tot={r[3]:3d} f1={r[4]:.4f}')
"
```

### 5.3 결과 폴더 구조

```
logs/<run_dir>/
├── best_info.json             # hparams + test_metrics + test_history
├── best_model.pth             # best weights (strict > 또는 avg_last_n 결과)
├── history.json               # epoch 별 train/val 기록
├── test_history.json          # best update 시점의 test 평가 누적
├── training_curves.png        # loss/f1/lr 곡선
├── confusion_matrix.png       # argmax CM
├── confusion_matrix_nt.png    # selected NT CM (default NT=0.9)
├── train.py                   # 실험 당시 train.py 스냅샷 (스크립트가 복사)
└── predictions/
    ├── tn_normal/             # True Negative (cap 100)
    ├── fn_abnormal/           # False Negative (전부)
    ├── fp_normal/             # False Positive (전부)
    └── tp_abnormal/           # True Positive (cap 100)
```

---

## 6. 주의사항 (절대 규칙)

1. **NT=0.9 고정**: reporting/inference 기준. run마다 test 결과를 보고 threshold를 바꾸지 말 것.
2. **학습 결과 폴더 삭제 금지**: `logs/<run_dir>/` 은 보존. 새 실험은 새 이름.
3. **Binary 모드 우선**: abnormal recall 최우선 지표. `--mode binary`.
4. **GPU 확인**: `python -c "import torch; print(torch.cuda.is_available())"` → `True` 확인.
5. **데이터 변경 시 이미지 재생성**: `generate_data.py` → `generate_images.py` 순.
6. **성능 보고 형식**: 항상 Binary (abn_R / nor_R) + **FN/FP 숫자** 같이.

---

## 7. 알려진 실험 결과 (v9 dataset, ConvNeXtV2-Tiny, lrB=3e-5)

| 실험 | n | seed | method | FN | FP | Total |
|---|---|---|---|---|---|---|
| 최고 1 | 2800 | 1 | strict p20 | 1 | 0 | **1** ⭐ |
| 최고 2 | 1400 | 1 | avg_last_n 5 | 0 | 1 | **1** ⭐ |
| n=700 | 700 | 1 | strict p20 | 2 | 1 | 3 |
| n=2100 | 2100 | 1 | avg_last_n 5 | 2 | 1 | 3 |
| seed 2 | 700 | 2 | avg_last_n 5 | 3 | 0 | 3 |
| seed 42 | 700 | 42 | strict p20 | 1 | 3 | 4 |
| seed 4 | 700 | 4 | avg_last_n 5 | 4 | 2 | 6 (s4 는 어려움) |

**핵심 발견**:
- `lrB=3e-5 / lrH=3e-4` 가 val spike 최소화
- `--patience 20` 으로 full 20 epoch 강제 필요
- `strict +p20` 과 `--avg_last_n 5` 는 **seed/n 조합에 따라 winner 가 다름** (trade-off)
- cudnn.benchmark=True → **non-deterministic** (매 run 다른 궤적)
- seed 4 는 v9 dataset 에서 worst-case (n=700 기준 5~9 errors)
