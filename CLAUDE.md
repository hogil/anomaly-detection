# CLAUDE.md

## 1. 프로젝트 목적

반도체 Fab L1 계측 데이터에서 이상을 자동 감지하여 품질 사고를 사전 차단한다.

Metrology Tool(optical CD 등)에서 나오는 시계열 측정값을 이미지로 변환하고,
딥러닝 분류 모델로 Normal/Anomaly를 판별한다.

**최종 목표: End-to-End Fab Quality Orchestration**
- Phase 1: L1 이상감지 (이미지 분류) ← **현재 단계**
- Phase 2: MTS 구조 & 연관 교호작용 (FT-Transformer, GNN)
- Phase 3: 통합 플랫폼 (FastAPI + React/Vue 대시보드)

---

## 2. 도메인 지식

### 계측 데이터 구조

반도체 계측 데이터는 2종류의 메타데이터로 구성된다:

**Chart 정의 (3개 컬럼):**
- **device**: 제품/디바이스 (DEV_A, DEV_B, ...)
- **step**: 공정 단계 (S01, S02, ...)
- **item**: 측정 항목 (CD1, CD2, THK1, OVL1, DEF1, ...)

→ device + step + item = **1개 chart** (1개 trend)

**Context 컬럼 (3개 컬럼):**
- **eqp_id**: 설비 (EQP_01, EQP_02, ...)
- **chamber**: 챔버 (CH_A, CH_B, ...)
- **recipe**: 레시피 (RCP_X1, RCP_X2, ...)

→ 같은 chart 안에서 eqp/chamber/recipe별로 fleet 비교

### Tabular 데이터 형식

```
timeseries.csv:
  chart_id, time_index, device, step, item, eqp_id, chamber, recipe, value

scenarios.csv:
  chart_id, class, device, step, item, context_column, target, contexts, defect_start_idx, defect_params, split
```

### 시계열 특성
- **주기성/계절성 없음** (절대 금지: 사인파 등)
- 불규칙 샘플링
- 밀집/희소/결핍 영역이 자연적으로 발생
- **전체 결핍 구간은 모든 멤버 공유** — 멤버별로는 독립적 dropout/thin/densify 패치 존재

### 불량 5종

| 유형 | 설명 | 불량 강도 |
|------|------|----------|
| Mean Shift | 평균이 갑자기 이동 (detrend 적용) | baseline_std × 3.2~4.0, boost 1.1~1.25 |
| Standard Deviation | 산포 급격히 증가 (detrend 적용) | scale 2.5~3.2배 |
| Spike | 순간적 이상값 (최소 3개, 크게) | baseline_std × 8.0~12.0, ratio 0.03~0.10, min 3개 |
| Drift | 점진적 한 방향 이동 | baseline_std × 0.05~0.08/step, 영역 20~35%, 최소 3σ 보장 |
| Context | 개별 설비는 정상이나 fleet 대비 유의차 | fleet_std × 2.5~4.0 |

**모든 불량 강도는 baseline_std에 비례** — 노이즈 강하면 불량도 강하게.

**Normal 클래스 품질 보장:**
- target의 산포(std)가 fleet 평균의 1.2배 초과 시 자동 축소
- target의 평균이 fleet center에서 과도하게 이탈 시 재정렬
- → Normal이 std/context 불량으로 오인되는 것 방지

---

## 3. 데이터 파이프라인

### 3.1 전체 흐름

```
config.yaml → generate_data.py → data/timeseries.csv + data/scenarios.csv
                                           ↓
                               generate_images.py → images/ + display/
                                           ↓
                                    train.py → weights/best_model.pth + logs/
```

### 3.2 이미지 포맷 (ALL overlay 통일)

**절대 규칙: 모든 클래스가 overlay 포맷**
- 1 이미지 = target(하이라이트) + fleet(배경)
- 학습/추론 동일 구조

**images/ (모델 입력):**
- 224×224, 축 없음
- target=하이라이트색, fleet=회색
- legend 없음

**display/ (사람 확인):**
- 원본 스케일, 축/legend
- 제목: device / step / item
- 정상 멤버=연한 고유색, 불량 멤버=빨강
- 경계선 (mean_shift/std/spike/drift)

### 3.3 추론 파이프라인 (절대 규칙)

**실전에서는 tabular 데이터만 입력받는다:**
1. tabular 데이터 수신 (라벨 없음)
2. 컬럼별 × 종류별 overlay 이미지 **2종 모두** 생성
   - images/: 모델 입력용
   - display/: 엔지니어 확인용
3. 모델이 images/ 분류 (6 class)
4. 불량 리스트 생성 + display/ 이미지 링크

**1 chart당 이미지 수 = eqp종류 + chamber종류 + recipe종류**

예: eqp 5종, chamber 4종, recipe 4종 → 13장 이미지 생성

### 3.4 Test 난이도

- test_difficulty_scale: 0.7
- test 불량 강도 = train × 0.7 (더 미묘)

---

## 4. 모델 설계

### 4.1 아키텍처

**ConvNeXtV2-Tiny (pretrained, file-based)**
```
Input Image (224x224)
    │
    └── ConvNeXtV2-Tiny (28.6M params, HuggingFace)
            │
            └── Classification Head
                Dropout(0.5) → Linear(768→512) → ReLU → Linear(512→6)
```

- 가중치: `weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth` (HF model id 그대로, `python download.py` 로 받음)
- GPU: RTX 4060 Ti (16GB VRAM)

### 4.2 학습 설정

```
Optimizer:    AdamW (weight_decay=0.01)
LR:           backbone=1e-4, head=1e-3
Scheduler:    CosineAnnealingLR (T_max=50, eta_min=1e-6)
Loss:         FocalLoss (gamma=2.0, alpha=inverse class frequency)
Batch Size:   32
Early Stop:   val recall 기준, patience=10
Epochs:       최대 50
```

### 4.3 Normal Threshold (Inference 전용)

```python
# 학습 때는 적용 금지
softmax_probs = F.softmax(logits, dim=1)
normal_prob = softmax_probs[:, 0]
is_normal = normal_prob > 0.7
predicted = torch.where(is_normal, 0, torch.argmax(softmax_probs[:, 1:], dim=1) + 1)
```

### 4.4 Augmentation

**사용:** ColorJitter(±10%), GaussianBlur(k=3), RandomErasing(10%)
**금지:** Horizontal/Vertical Flip, Rotation (시계열 구조 파괴)

### 4.5 학습 산출물

```
weights/best_model.pth          # 최고 모델 가중치
logs/best_info.json             # 하이퍼파라미터 + best 조건 + 성능
logs/history.json               # epoch별 상세 기록
logs/training_curves.png        # loss/accuracy/recall 곡선
logs/confusion_matrix.png       # best model confusion matrix
logs/confusion_matrix_nt.png    # normal threshold 적용 CM
```

---

## 5. 모듈 구조

```
src/
├── data/
│   ├── baseline_generator.py   # 에피소드 기반 시계열 + 공유 mask
│   ├── defect_synthesizer.py   # 불량 주입 (baseline_std 비례, detrend)
│   ├── scenario_generator.py   # 통합 시나리오 생성 (chart+context)
│   └── image_renderer.py       # overlay 이미지 렌더링 (학습+display)
│
├── models/
│   └── focal_loss.py           # Focal Loss
│
generate_data.py                # tabular CSV 생성
generate_images.py              # CSV → overlay 이미지
train.py                        # 학습 + 평가 + 로깅
config.yaml                     # 전체 설정
```

---

## 6. 절대 규칙

1. **베이스라인에 주기성/계절성 절대 금지**
2. **dispersion 사용 금지 → standard_deviation**
3. **GPU 확인 후 있으면 반드시 GPU 사용**
4. **추론은 tabular 입력만 → 이미지 생성 → 분류 → 불량 리스트**
5. **이미지는 항상 2종 (images/ + display/) 모두 생성**
6. **Context = 1 컬럼 그룹핑 × 멤버별 하이라이트 (cross-product 금지)**
7. **불량 강도는 baseline_std에 비례**
8. **mean_shift 주입 시 detrend 금지** (자연 변동 보존, std는 절대값 방식)
9. **설계/튜닝 시 소수(10~20개) 빠르게 생성 → 확인 → 조정**
10. **skill/agent 사용 시 반드시 사전 고지**
11. **학습/val은 binary 모드** (abnormal recall이 최우선 지표)
12. **성능은 항상 2개 보고**: Binary(abn_R/nor_R) + 6클래스(개별 recall)
13. **모든 변경사항 즉시 skill/memory 업데이트** (config, 코드, 실험 결과, best model — 미루기 절대 금지)
14. **⛔ 학습 결과 폴더 삭제 절대 금지** (`logs/<run_dir>/`). 새 실험은 무조건 새 폴더명 사용. 설정 변경 시 폴더명에 버전 식별자 포함 (예: `v9_noise25_n700_s42`). `rm -rf logs/...`, `shutil.rmtree(log_dir)` 절대 금지. 사용자가 명시적으로 삭제 요청 시에만 가능.
15. **⭐ Best 업데이트 시작점 / Early stop 시작점 분리** (2026-04-09 확정):
    - **Best 업데이트 시작**: smoothing 방식에 따라 다름
      - `smooth_window=1` (single / raw val_f1): **ep 10 부터**
      - `smooth_window=3` (med3) / `avg5` / 기타 smoothing: **ep 7 부터**
    - **Early stop patience counter 시작**: **항상 ep 10 (고정)**
    - **최소 학습 epoch = `10 + patience`**. 이 전엔 절대 종료 금지:
      - patience 5 → 최소 epoch 15
      - patience 10 → 최소 epoch 20
    - **매 epoch test 평가 + test_history 기록**: **ep 10 부터 강제** (best 갱신 여부 무관)
    - Why: val_f1 가 min_epochs 이전에 1.0 포화되면 strict > 1.0 이 불가능해져 첫 epoch 에서 freeze 되는 lottery 문제 방지. smoothing 이 있으면 ep 7 부터 저장 허용해서 더 좋은 early convergence 포착. test_history 는 사후 분석용 (어느 epoch 이 진짜 best 였는지 복원 가능).

---

## 7. 개발 환경

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 4060 Ti (16GB VRAM)
- **Python**: 3.13
- **주요 라이브러리**: PyTorch, torchvision, timm, numpy, pandas, matplotlib, seaborn, scikit-learn, tqdm
- **실행**:
  - `python generate_data.py` → tabular 데이터 생성
  - `python generate_images.py` → 이미지 생성
  - `python train.py --epochs 50` → 학습

---

## 8. 커밋 컨벤션

```
feat: 새 기능
fix: 버그 수정
refactor: 리팩토링
perf: 성능 개선
docs: 문서
test: 테스트
chore: 빌드/설정
```
