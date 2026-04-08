# SERVER SETUP — H200 폐쇄망 배포 가이드

대상 서버:
- **OS**: Ubuntu 24.04
- **CPU**: 32 코어
- **RAM**: 384 GB
- **GPU**: NVIDIA H200 × 2 (각 141 GB VRAM)
- **네트워크**: 사내 폐쇄망 (인터넷 차단)

이 문서는 0부터 시작해서 학습 결과까지 한 번에 가는 절차다.

---

## 1. 사전 준비 — 인터넷 가능한 머신에서

폐쇄망 안에서는 `pip install`, `timm.create_model(pretrained=True)` 등 모든 외부 다운로드가 막혀 있다. 다음을 미리 빌드한다.

### 1-1. Python 패키지 wheel 모음

```bash
# 인터넷 머신 (서버와 동일 OS/Python 버전)
mkdir -p offline_wheels
pip download -d offline_wheels \
    "torch>=2.4" "torchvision>=0.19" "timm>=1.0.9" \
    "numpy>=1.24,<2.2" "pandas>=2.0" "matplotlib>=3.7" \
    "Pillow>=10.0" "tqdm>=4.65" "pyyaml>=6.0" "seaborn>=0.13"

# offline_wheels/ 폴더 전체를 USB로 옮김
```

> **권장**: 인터넷 머신도 Ubuntu 24 + Python 3.12 (서버와 동일). 다른 OS면 wheel 호환성 깨질 수 있음.
>
> **중요**: `scipy` / `scikit-learn` 는 **더 이상 필요 없음** (코드에서 제거됨).
> Ubuntu 24 + Python 3.12 + numpy 2.x 조합에서 구버전 scipy의 `_spropack`/`_propack` ABI 충돌이 있어, `train_test_split` 과 `confusion_matrix` 를 numpy-only 구현으로 대체했다.

### 1-2. Repo clone + pretrained weights 다운

```bash
git clone https://github.com/hogil/anomaly-detection.git
cd anomaly-detection
pip install timm torch  # 임시 (다운용)

# 옵션 A: convnextv2_tiny 만 (현재 winning config 모델, ~110MB)
python download.py

# 옵션 B: 6 backbone 비교 실험까지 (~1.2GB)
python download.py --all
```

다운된 파일은 `anomaly-detection/weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth` 등 HF model id 그대로 저장된다.
**중요**: `weights/` 는 git 에 절대 커밋하지 않는다 (`.gitignore` 처리됨).
폐쇄망 서버로는 인터넷 머신에서 받은 `weights/` 폴더를 USB/scp 로 통째로 옮긴다.

### 1-3. Repo 압축

```bash
# 인터넷 머신
cd ..
tar czf anomaly-detection.tar.gz anomaly-detection offline_wheels
# → USB 또는 jump host로 폐쇄망 서버에 옮김
```

---

## 2. 폐쇄망 H200 서버 — 환경 셋업

```bash
# 서버
tar xzf anomaly-detection.tar.gz
cd anomaly-detection

# Python 가상환경
python3 -m venv .venv
source .venv/bin/activate

# offline wheel 설치 (인터넷 사용 안 함)
pip install --no-index --find-links=../offline_wheels \
    torch torchvision timm \
    numpy pandas matplotlib pillow tqdm pyyaml seaborn
```

> scipy/sklearn 제외. 코드에서 사용 안 하며, 설치 시 numpy 2.x ABI 충돌(`_spropack`) 위험.

### 환경 검증

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 기대: True 2

python -c "import torch; [print(torch.cuda.get_device_name(i),
    torch.cuda.get_device_properties(i).total_memory/1024**3, 'GB')
    for i in range(torch.cuda.device_count())]"
# 기대: NVIDIA H200 ~141 GB × 2

ls weights/
# 기대: convnextv2_tiny.fcmae_ft_in22k_in1k.pth 존재 (~110MB)
```

---

## 3. 전체 파이프라인 한 방에 실행

```bash
bash run_pipeline.sh
```

이 한 줄이 다음을 순서대로 한다:
1. **환경 검증** (GPU 2장, weights 파일)
2. **데이터 생성** (`generate_data.py`) — `data/timeseries.csv` + `data/scenarios.csv`
3. **이미지 생성** (`generate_images.py --workers 24`) — `images/` + `display/`, 32 코어 중 24개 사용
4. **실험 실행** (`run_experiments.py --server h200 --gpus 2`) — 24 runs를 H200 2장에 자동 분배
5. **요약 재생성** (`logs/experiments_summary.json`)

### 부분 실행

```bash
# data + images는 이미 있고 학습만
bash run_pipeline.sh skip-data skip-img

# sweep 그룹만 (15 runs, 가장 중요)
bash run_pipeline.sh skip-data skip-img sweep

# 정규화 ablation만
bash run_pipeline.sh skip-data skip-img reg
```

### 로그 확인 (병렬 실행 중)

```bash
# 진행 중인 실험 로그 (실시간)
tail -f logs/v9x_n2800_s42/run.log

# 전체 GPU 사용량
nvidia-smi -l 1

# 완료된 실험 결과
ls logs/*/best_info.json
```

---

## 4. H200 최적화 (자동 적용)

`--server h200` 프로파일이 다음을 자동 주입:

| 옵션 | 값 | 효과 |
|---|---|---|
| `--precision` | bf16 | H200 네이티브, fp16 대비 overflow 없음, GradScaler 불필요 |
| `--compile` | (flag) | torch.compile max-autotune, 첫 epoch 컴파일 후 20~50% 가속 |
| `--batch_size` | 256 | 4060 Ti의 8배 (28M 모델 × 141GB VRAM 여유 충분) |
| `--num_workers` | 16 | 32 코어 절반 (GPU 2장 동시 실행 고려) |
| `--prefetch_factor` | 8 | 큰 배치 + 빠른 GPU에 맞춰 prefetch 증가 |

추가 환경변수 (`run_pipeline.sh`가 export):
- `TORCHINDUCTOR_CACHE_DIR=$PWD/.torch_compile_cache` — 컴파일 캐시 재사용 (다음 실험부터 즉시)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — 큰 배치 + bf16 메모리 단편화 완화

---

## 5. 예상 시간 (H200 × 2 vs 4060 Ti × 1)

| 단계 | 4060 Ti | H200 × 2 | 가속 |
|---|---:|---:|---:|
| 데이터 생성 | ~3 분 | ~3 분 | (CPU bound, 동일) |
| 이미지 생성 (5000 charts × 6 클래스) | ~10 분 | ~3 분 | 32 코어 ÷ 16 코어 |
| 1 학습 run (bs 32, fp16) | ~5 분 | – | – |
| 1 학습 run (bs 256, bf16, compile) | – | ~1 분 | bs 8× + 컴파일 |
| 24 runs 순차 (1 GPU) | ~120 분 | ~24 분 | – |
| **24 runs 병렬 (2 GPU)** | – | **~12 분** | **10× over laptop** |
| **전체 파이프라인** | ~135 분 | **~18 분** | **7.5×** |

> 첫 실험에서 `torch.compile`이 1~2분 추가 소요 (max-autotune). 캐시 이후 즉시.

---

## 6. 트러블슈팅

### `cannot import name '_spropack' from 'scipy.sparse.linalg'`
```
ImportError: cannot import name '_spropack' from 'scipy.sparse.linalg'
  (.../scipy/sparse/linalg/_propack.cpython-312-x86_64-linux-gnu.so)
```
→ **Python 3.12 + numpy 2.x + 구버전 scipy/sklearn ABI 충돌.**
이 repo는 `scipy`/`sklearn` 을 runtime dep에서 제거했으므로, 해당 패키지 **미설치 상태로도 모든 기능 동작**. 이미 설치돼 있다면:

```bash
# 옵션 1: scipy/sklearn 제거 (가장 깔끔)
pip uninstall -y scipy scikit-learn

# 옵션 2: 호환 버전으로 재설치 (sklearn 다른 용도로 필요한 경우)
pip install --no-index --find-links=../offline_wheels \
    "scipy>=1.13" "scikit-learn>=1.5"
```

그 후 `bash run_pipeline.sh` 재실행.

### `torch.compile failed` 메시지
```
[WARN] torch.compile failed, falling back: ...
```
→ torch 2.4 이상 필요. 그래도 fallback으로 학습은 정상 진행 (속도만 ~30% 손해).

### CUDA OOM (Out of Memory)
H200 141GB로 bs 256은 안전하지만, 만약 OOM 발생 시:
```bash
# run_experiments.py SERVER_PROFILES["h200"]["extra_args"] 수정
"--batch_size", "128",  # 256 → 128
```

### `weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth not found`
→ 인터넷 머신에서 `python download.py` 실행 후 `weights/` 폴더 복사 안 됨. 1-2 절 참조.

### `nvidia-smi`에 GPU 1장만 보임
→ `CUDA_VISIBLE_DEVICES` 환경변수가 외부에서 제한되어 있을 수 있음. 확인:
```bash
echo $CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
nvidia-smi
```

### 한글 로그 깨짐
→ `LANG=ko_KR.UTF-8` 또는 `LANG=C.UTF-8` 설정. `run_pipeline.sh` 실행 전:
```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

### 디스크 사용량
- `data/`: ~1 GB (CSV)
- `images/` + `display/`: ~1.5 GB (PNG)
- `logs/` (24 runs × ~500 MB): ~12 GB
- `weights/`: ~110 MB

총 약 **15 GB** 필요.

---

## 7. 결과 가져오기 (폐쇄망 → 외부)

```bash
# 서버 (best_info.json만 가져가고 싶을 때 — 작음)
tar czf results_summary.tar.gz logs/*/best_info.json logs/experiments_summary.json
# → 약 100 KB

# 전체 (predictions, plots 포함 — 큼)
tar czf results_full.tar.gz logs/v9x_* logs/v9reg_* logs/v9lr_* logs/v9mc_*
# → 약 12 GB
```

---

## 8. 절대 규칙

1. **logs/<run_dir>/ 절대 삭제 금지** — 새 실험은 무조건 새 폴더명
2. **weights/ 파일 절대 삭제 금지** — 재다운로드 불가 (폐쇄망)
3. **실험 추가 시** — `run_experiments.py` 의 `build_experiments()`에 새 prefix로 추가
4. **OOM 발생 시** — bs 줄이기, 폴더 삭제 금지
