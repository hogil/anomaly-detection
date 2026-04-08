# HOW TO RUN — 파일별 역할 + 실행 방법

대상: **Ubuntu 24, H200 × 2 (각 141GB), 32 코어, 384GB RAM, 사내 폐쇄망**

---

## 1. 파일별 역할 한눈에

| 단계 | 파일 | 역할 | 입력 | 출력 |
|---|---|---|---|---|
| 1 | `generate_data.py` | tabular 데이터 생성 (시계열 + 시나리오) | `config.yaml` | `data/timeseries.csv`, `data/scenarios.csv` |
| 2 | `generate_images.py` | overlay 이미지 렌더링 (학습용 + 사람 확인용) | `data/*.csv` | `images/{train,val,test}/{cls}/*.png`, `display/...` |
| 3 | `train.py` | ConvNeXtV2-Tiny 학습 (1 실험) | `images/`, `data/scenarios.csv` | `logs/<exp>/{best_model.pth, best_info.json, ...}` |
| 4 | `run_experiments.py` | 24 실험 한 방에 (sweep/reg/lr/mc) | train.py 호출 | `logs/<exp>/`, `logs/experiments_summary.json` |
| 5 | `run_pipeline.sh` | 1~4 전체 orchestrator (env check → data → images → 24 실험 → summary) | (nothing) | 위 모든 결과 |

**보조 스크립트**:

| 파일 | 역할 |
|---|---|
| `download.py` | timm/HF 에서 backbone weights 다운 → `weights/<short>.pth` (인터넷 머신용, 1회 실행) |

**설정 파일**:

| 파일 | 역할 |
|---|---|
| `config.yaml` | 데이터 생성 + 학습 설정 (현재 v9: noise +25%, sparse 62%) |

**weights/** (gitignored — `python download.py` 로 받기, git에 절대 커밋 X):

| 파일 | 사이즈 | 받는 법 |
|---|---:|---|
| `weights/convnextv2_tiny.pth` | ~110 MB | `python download.py` (default) |
| `weights/convnextv2_base.pth` | ~340 MB | `python download.py --all` |
| `weights/efficientnetv2_s.pth` | ~85 MB | `python download.py --all` |
| `weights/swin_tiny.pth` | ~110 MB | `python download.py --all` |
| `weights/maxvit_tiny.pth` | ~120 MB | `python download.py --all` |
| `weights/clip_vit_b16.pth` | ~340 MB | `python download.py --all` |

---

## 2. 실행 순서 (TL;DR)

**Step 0**: pretrained weights 다운 (1회만, 인터넷 필요)
```bash
python download.py            # convnextv2_tiny만 (~110MB)
# 또는 backbone 비교까지 할 거면
python download.py --all       # 6 backbone (~1.2GB)
```

**Step 1~4**: 데이터 → 이미지 → 24 실험 → 요약 (한 방에)
```bash
bash run_pipeline.sh
```

폐쇄망 H200 서버라면 weights를 인터넷 머신에서 받아 USB로 옮긴 뒤 위 명령 실행.

---

## 3. 한 방 실행 상세

### Ubuntu 24 H200 서버 (24 실험 전부)

```bash
cd anomaly-detection
bash run_pipeline.sh
```

이 한 줄이 다음을 자동으로 한다:
1. **환경 검증** (GPU 2장, weights 파일 확인)
2. **데이터 생성** (`generate_data.py --workers 24`) — 32 코어 중 24개 사용
3. **이미지 생성** (`generate_images.py --workers 24`)
4. **24 실험 실행** (`run_experiments.py --server h200 --gpus 2`) — H200 2장에 자동 분배
5. **요약 생성** (`logs/experiments_summary.json`)

**예상 시간**: 데이터 ~3분 → 이미지 ~3분 → 실험 ~12분 → 총 **~18분**.

### 부분 실행

```bash
# data + images는 이미 있고 학습만 (재학습 또는 추가 그룹)
bash run_pipeline.sh skip-data skip-img

# sweep 그룹만 (15 runs, 가장 중요한 sweet spot 검증)
bash run_pipeline.sh skip-data skip-img sweep

# 정규화 ablation만
bash run_pipeline.sh skip-data skip-img reg

# 데이터/이미지 다시 만들고 싶으면 폴더 비우기
rm -rf data/ images/ display/
bash run_pipeline.sh
```

---

## 3. 단계별 수동 실행 (디버깅용)

### Step 1 — 데이터 생성

```bash
# H200 서버 (32 코어 → 24 worker 병렬)
python generate_data.py --workers 24

# 노트북 (순차, 4060 Ti)
python generate_data.py --workers 1
# 또는 그냥
python generate_data.py
```

CLI 옵션:
- `--workers 1` : 순차 (default, 노트북 호환)
- `--workers 0` : auto (cpu_count - 1)
- `--workers N` : N process 병렬
- `--config config.yaml` : 다른 config 파일

산출물:
```
data/
├── timeseries.csv      # ~14M rows (chart_id × time × value)
└── scenarios.csv       # ~7000 charts (chart_id, class, target, contexts, ...)
```

### Step 2 — 이미지 생성

```bash
# H200 서버
python generate_images.py --workers 24

# 노트북
python generate_images.py
```

CLI 옵션:
- `--workers 0` : auto (default, cpu_count - 1)
- `--workers N` : N process 병렬

산출물:
```
images/                 # 모델 입력 (224x224, 축 없음)
├── train/{normal,mean_shift,...}/*.png
├── val/...
└── test/...

display/                # 사람 확인용 (원본 스케일, 축/legend)
├── train/...
└── ...
```

### Step 3 — 단일 실험 학습

```bash
# H200 서버 — bf16 + compile + bs 256, 1장만 사용
CUDA_VISIBLE_DEVICES=0 python train.py \
    --precision bf16 --compile \
    --batch_size 256 --num_workers 16 --prefetch_factor 8 \
    --normal_ratio 2800 --seed 42 \
    --log_dir logs/v9_test_n2800_s42

# 노트북 — winning config 그대로 (default)
python train.py --normal_ratio 2800 --seed 42 --log_dir logs/v9_test
```

CLI 옵션 (자주 쓰는 것):
- `--epochs 20` : 학습 epoch 수
- `--mode binary|multiclass` : 학습 모드
- `--normal_ratio N` : binary 학습 시 normal 샘플 수 (700~3500)
- `--seed N` : random seed (재현성)
- `--log_dir logs/<name>` : 결과 폴더 (절대 기존 폴더 덮어쓰지 말 것)
- `--precision fp16|bf16|fp32` : 정밀도 (H200: bf16 권장)
- `--compile` : torch.compile 활성화
- `--batch_size N` : batch (H200: 256, 노트북: 32)
- `--num_workers N` : DataLoader workers (H200: 16, 노트북: 4)

산출물 (한 실험당):
```
logs/<exp_name>/
├── best_model.pth          # 모델 가중치
├── best_info.json          # 성능 + hparams + timing
├── history.json            # epoch별 기록
├── training_curves.png     # loss/f1/lr 곡선
├── confusion_matrix.png
├── confusion_matrix_nt.png # normal threshold 적용
├── run.log                 # subprocess 출력 (run_experiments.py 통해 실행 시)
└── predictions/
    ├── tn_normal/          # True Negative (cap 100)
    ├── fn_abnormal/        # False Negative 전부 (놓친 불량) ★
    ├── fp_normal/          # False Positive 전부 (false alarm) ★
    └── tp_abnormal/        # True Positive (cap 100)
```

### Step 4 — 24 실험 한 방에

```bash
# H200 서버 — 2 GPU 병렬, ~12분
python run_experiments.py --server h200 --gpus 2

# 노트북 — 순차, ~120분
python run_experiments.py

# 그룹 선택 (sweep만)
python run_experiments.py --server h200 --gpus 2 --groups sweep

# 명령만 미리 보기 (학습 안 함)
python run_experiments.py --server h200 --gpus 2 --dry-run

# 기존 결과 요약만 다시 출력
python run_experiments.py --only-summary
```

CLI 옵션:
- `--server laptop|h200` : 서버 프로파일 (h200은 bf16+compile+bs256+workers16 자동)
- `--gpus N` : 병렬 GPU 수 (h200 default 2, laptop default 1)
- `--groups sweep reg lr mc` : 실행 그룹 선택 (default: 전부)
- `--dry-run` : 명령만 출력
- `--force` : 기존 done 폴더도 재실행 (보통 안 씀, 절대 폴더 삭제는 안 함)
- `--only-summary` : 학습 안 하고 기존 결과만 요약

### Step 5 — 실험 그룹

24 runs 전체:

| 그룹 | runs | 목적 | 이름 |
|---|---:|---|---|
| sweep | 15 | normal_ratio sweet spot 재검증 (n × seed) | `v9x_n{700,1400,2100,2800,3500}_s{1,2,42}` |
| reg | 5 | 정규화 ablation (label_smooth/mixup/dropout/focal/wd) | `v9reg_*_n2800_s42` |
| lr | 3 | LR 민감도 (3e-5/1e-4/warmup8) | `v9lr_*_n2800_s42` |
| mc | 1 | multiclass 보조 (6-class 개별 recall) | `v9mc_n2800_s42` |

상세는 [`EXPERIMENTS.md`](EXPERIMENTS.md) 참조.

---

## 5. 폐쇄망 H200 서버 셋업 (1회)

### 5-1. 인터넷 머신에서

```bash
# 1) repo clone
git clone https://github.com/hogil/anomaly-detection.git
cd anomaly-detection

# 2) Python 패키지 wheel 모음 (scipy/sklearn 제외 — Ubuntu 24 ABI 충돌 회피)
mkdir -p ../offline_wheels
pip download -d ../offline_wheels \
    "torch>=2.4" "torchvision>=0.19" "timm>=1.0.9" \
    "numpy>=1.24,<2.2" "pandas>=2.0" "matplotlib>=3.7" \
    "Pillow>=10.0" "tqdm>=4.65" "pyyaml>=6.0" "seaborn>=0.13"

# 3) Pretrained backbone 다운 (필요한 만큼)
pip install timm torch                         # 임시 설치 (다운용)
python download.py                              # default: convnextv2_tiny (~110MB)
# 또는 backbone 비교 실험까지 할 거면
python download.py --all                        # 6 backbone (~1.2GB)

# 4) 전체 bundle (repo + offline_wheels)
cd ..
tar czf bundle.tar.gz anomaly-detection offline_wheels
# anomaly-detection/weights/ 에 다운받은 .pth 들 포함됨
```

### 4-2. USB / jump host로 폐쇄망 서버에 옮긴 뒤

```bash
tar xzf bundle.tar.gz
cd anomaly-detection

# 가상환경 + offline 설치
python3 -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links=../offline_wheels \
    torch torchvision timm numpy pandas matplotlib scipy scikit-learn pillow tqdm pyyaml seaborn

# 환경 검증
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# → True 2

ls weights/convnextv2_tiny.pth
# → 파일 존재 (~110MB)
```

### 4-3. 한 방 실행

```bash
bash run_pipeline.sh
```

자세한 설치 절차 + 트러블슈팅: [`SERVER_SETUP.md`](SERVER_SETUP.md).

---

## 5. 자주 쓰는 명령 모음 (cheat sheet)

```bash
# === 노트북 (Windows / Linux 4060 Ti) ===
python generate_data.py                       # 데이터 (순차)
python generate_images.py                     # 이미지 (auto workers)
python train.py --normal_ratio 2800           # 학습 1 run
python run_experiments.py                     # 24 실험 순차

# === H200 서버 (Ubuntu 24) ===
bash run_pipeline.sh                          # 한 방에 다 (권장)
bash run_pipeline.sh skip-data skip-img sweep # sweep만 재학습

python generate_data.py --workers 24          # 데이터만 (병렬)
python generate_images.py --workers 24        # 이미지만 (병렬)
python run_experiments.py --server h200 --gpus 2  # 24 실험 H200 2장 병렬

# === 결과 확인 ===
python run_experiments.py --only-summary      # 요약 표 출력
ls logs/*/best_info.json                       # 완료된 실험 목록
nvidia-smi -l 1                                # GPU 사용량 실시간

# === 결과 가져오기 (폐쇄망 → 외부) ===
tar czf results.tar.gz logs/v9*/best_info.json logs/experiments_summary.json
# → ~100KB, USB로 가져나오기 가능
```

---

## 6. 절대 규칙 (이 저장소)

1. **`logs/<run_dir>/` 절대 삭제 금지** — 새 실험은 무조건 새 폴더명
2. **`weights/` 는 git 에 절대 커밋 금지** (`.gitignore` 처리됨). 폐쇄망 서버는 인터넷 머신에서 `python download.py` 후 USB 로 옮길 것
3. **추론 입력은 tabular만** — images는 파이프라인 내부에서 생성
4. **Binary 학습 우선** — abn_R 최우선 지표
5. **성능 항상 2개 보고** — Binary(abn_R/nor_R/F1) + (mc 때만) 6-class 개별
6. **OOM 발생 시** — batch_size 줄이기, 폴더 삭제 금지

---

## 7. 다음 액션 (당장 할 일)

1. **인터넷 머신**에서 `git clone` + `pip download` 로 bundle 만들기
2. **폐쇄망 H200 서버**로 옮겨서 venv + offline install
3. `bash run_pipeline.sh` 실행
4. ~18분 후 `logs/experiments_summary.json` 확인
5. 결과 분석 → 최고 조합 확정 → 다음 단계 (EMA 적용, inference 검증, ...)
