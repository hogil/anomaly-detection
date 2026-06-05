# How To Run

이 프로젝트는 **trend 이미지(시계열 차트)** 를 보고 normal/anomaly를 분류합니다. 흐름:

```
dataset.yaml  →  데이터/이미지 생성  →  학습  →  추론
                                       ↑
                  현업 CSV는 generate_inference_images.py 로 끼워넣음
```

문제 설정은 `docs/problem_setting.md`에 고정합니다. 기본 운영 목표는 `normal`/`abnormal` binary gate이고, multiclass는 defect type 분석용 보조 실험입니다.

## Python 3.11 환경

기본 서버/PC 검증 환경은 Python 3.11 + torch 2.3.1 + torchvision 0.18.1 + torchaudio 2.3.1 + numpy 1.26.4입니다. H200 서버만 별도로 torch 2.7.0 + torchvision 0.22.0 + torchaudio 2.7.0을 씁니다.

```bash
conda env create -f environment-py311.yml
conda activate anomaly-py311
python - <<'PY'
import torch, numpy, yaml
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version() if torch.cuda.is_available() else None)
print("numpy", numpy.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
python scripts/check_torch_runtime.py
```

사내망에서는 서버에 설정된 사내 PyPI/mirror에서 기본 cu121 wheel을 받습니다. H200만 `requirements-h200.txt`를 사용합니다. 공식 PyTorch 2.7.0 wheel은 cu126/cu128까지 제공되므로, `2.7.0 cu130`은 사내 mirror가 별도 build를 제공할 때만 가능합니다. 실행할 때는 같은 환경을 쓰도록 `--python "$(which python)"`을 붙이는 것을 권장합니다.

`RuntimeError: operator torchvision::nms does not exist`가 나오면 `torch`와 `torchvision` wheel이 서로 안 맞는 상태입니다. 서버에서는 다음 조합으로 맞춥니다.

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
rm -rf ~/.cache/pip
python -m pip install --no-cache-dir --force-reinstall \
  torch==2.3.1 \
  torchvision==0.18.1 \
  torchaudio==2.3.1
python -m pip install -r requirements.txt
python scripts/check_torch_runtime.py
```

H200에서만:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
rm -rf ~/.cache/pip
python -m pip install --no-cache-dir --force-reinstall \
  torch==2.7.0 \
  torchvision==0.22.0 \
  torchaudio==2.7.0
python -m pip install -r requirements-h200.txt
AD_TORCH_PROFILE=h200 python scripts/check_torch_runtime.py
```

목차:
1. 데이터셋 정의 (`dataset.yaml`)
2. 데이터/이미지 생성
3. 단일 학습 (`train.py`)
4. 논문 sweep — 한 번에 전부 돌리기
5. **stage 부분만 돌리기 — 구체 예시**
6. 추론 (`inference.py`) + abnormal/normal 텍스트 리스트
7. 현업 CSV 가져왔을 때
8. 추가 학습 (folder)
9. logs에서 표·plot 만들기
10. Grad-CAM
11. FP/FN 치우칠 때

---

## 1. 데이터셋 정의는 `dataset.yaml` 한 군데

**바꿀 것은 이 파일뿐**입니다 — episode 길이, 노이즈, anomaly 종류·세기, fleet 수, 색·alpha 등 데이터셋 정의가 전부 여기.

각 학습 run은 자기가 본 yaml 스냅샷(`logs/<run>/data_config_used.yaml`)을 따로 저장하므로, `dataset.yaml`을 나중에 바꿔도 옛 학습은 자기 스냅샷으로 재현됩니다.

---

## 2. 데이터/이미지 생성

```bash
python generate_data.py   --config dataset.yaml --workers 24
python generate_images.py --config dataset.yaml --workers 24
```

출력:

```
data/scenarios.csv, data/timeseries.csv      # 시나리오·시계열
images/   <- 모델 입력용
display/  <- 사람이 보는 확인용
```

`run_paper_server_all.sh`가 이미지가 없으면 자동으로 위 두 명령을 호출합니다. 별도로 미리 만들어둘 필요 없음.

---

## 3. 단일 학습 (`train.py`)

가장 단순한 호출:

```bash
python train.py \
  --config dataset.yaml \
  --mode binary \
  --epochs 20 \
  --batch_size 32 \
  --precision fp16 \
  --normal_ratio 700 \
  --seed 42 \
  --log_dir my_run
```

자주 쓰는 옵션:

| 옵션 | 의미 | 예시 |
|---|---|---|
| `--model_name` | timm 백본 이름. `weights/<이름>.pth` 가 있어야 함 | `--model_name swinv2_cr_tiny_ns_224.sw_in1k` |
| `--lr_backbone` / `--lr_head` | LR. ConvNeXtV2-Tiny 기본 `2e-5 / 2e-4` | `--lr_backbone 3e-5 --lr_head 3e-4` |
| `--warmup_epochs` | warmup epoch 수 | `--warmup_epochs 3` |
| `--grad_clip` | gradient clip max_norm. `0` 이면 끔 | `--grad_clip 0.5` |
| `--label_smoothing` | label smoothing | `--label_smoothing 0.02` |
| `--asl_gamma_neg` / `--asl_clip` | asymmetric loss 축. ASL이 켜지면 loss-family 단일 조건으로 해석 | `--asl_gamma_neg 2.0 --asl_clip 0.05` |
| `--stochastic_depth_rate` | stochastic depth | `--stochastic_depth_rate 0.05` |
| `--focal_gamma` | focal loss γ | `--focal_gamma 2.0` |
| `--abnormal_weight` | anomaly class 가중치 | `--abnormal_weight 1.5` |
| `--ema_decay` | EMA. `0` 이면 끔 | `--ema_decay 0.95` |
| `--allow_tie_save` | val_f1 tie 시도 저장 허용 | `--allow_tie_save` |
| `--filter_nonfinite_loss` | NaN/Inf loss 샘플 step skip | `--filter_nonfinite_loss` |
| `--log_dir_group` | `logs/<group>/<run>/` 형식으로 묶음 (group 명에 시각이 앞에 와야 sort가 깨지지 않음) | `--log_dir_group 20260430_120000_run_paper` |

출력 폴더 (자동으로 시작 시각과 best F1/Recall이 붙음):

```
logs/                                                       # 단일 학습 (그룹 없음)
└── <YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/
       best_model.pth, best_info.json, history.json
       confusion_matrix.png, training_curves.png
       data_config_used.yaml   <- 이 run이 본 데이터셋 정의
       train_config_used.yaml  <- 이 run의 학습 인자 전부
       predictions/

logs/                                                       # 00_all.sh / batch 실행
└── <YYYYMMDD_HHMMSS>_run_paper/                            #   group 폴더 (시각 prefix가 먼저)
        ├── <YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/
        ├── <YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/
        └── ...
```

`--log_dir`로 넣는 값은 **topic(조건명)** 이고, 폴더명에 들어가는 시각·F1/Recall은 train.py가 자동으로 붙입니다 (best 갱신 시마다 F/R 숫자 갱신).

BKM combined 한 번 돌리는 단일 명령 예시 (`05_bkm_combined`와 같은 조합):

```bash
python train.py \
  --config dataset.yaml --mode binary --epochs 20 --batch_size 32 --precision fp16 \
  --normal_ratio 3300 \
  --label_smoothing 0.02 \
  --stochastic_depth_rate 0.05 \
  --focal_gamma 2.0 \
  --abnormal_weight 1.5 \
  --ema_decay 0.95 \
  --allow_tie_save \
  --seed 42 --log_dir bkm_combined_s42
```

---

## 4. 논문 sweep — 한 번에 전부 돌리기

```bash
bash scripts/sweeps_server/00_all.sh
```

내부 순서:

| # | stage | 한 줄 설명 |
|---|---|---|
| 0 | weights/dataset 준비 | 없으면 `download.py`, `generate_data.py`, `generate_images.py` 자동 |
| 1 | baseline | 같은 baseline 5-seed 재확인 |
| 2 | axis sweep (core) | lr/warmup/normal_ratio/per_class/weight_decay/smoothing/label_smoothing/asl/stochastic_depth/focal_gamma/abnormal_weight/ema/allow_tie_save 한 축씩만 |
| 3 | color | trend·fleet 색·alpha 축 |
| 4 | sample_skip | nonfinite-loss 샘플 step-skip 안전 실험 1-seed |
| 5 | backbone | `download.py::MODELS` 순서의 백본을 우선 학습하고, `weights/`의 추가 non-deprecated `*.pth`를 뒤에 학습 |
| 6 | logical_train | member별 logical 데이터셋 + 학습 |
| 7 | bkm_combined | BKM 값 다 적용한 candidate 1개 × 5-seed |
| 8 | postprocess | 종합 리포트 + 축별 plot + instability/trend |

GPU 메모리·CPU 수로 자동 프로필 결정:

| 프로필 | 조건 | num_workers | prefetch | 동시 launch |
|---|---|---:|---:|---:|
| `server` | GPU ≥ 40 GB | 48 | 4 | 무제한 |
| `pc` | GPU ≥ 12 GB | 2 | 2 | 무제한 |
| `minimal` | 그 외 | 0 | 1 | 무제한 |

`--num-workers`, `--prefetch-factor`, `--max-launched`, `--log-dir-group` 직접 지정하면 위 자동값 덮어씀.

`00_all.sh` 시작할 때 `LOG_DIR_GROUP=<YYYYMMDD_HHMMSS>_run_paper` 한 번 만들어서 모든 stage가 `logs/<YYYYMMDD_HHMMSS>_run_paper/<YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/` 아래로 모입니다 (시각이 앞에 와야 `ls logs/` 시간순 정렬). 단독 stage 실행도 같은 패턴: `13_sample_skip.sh` → `<timestamp>_sample_skip`, `14_backbone.sh` → `<timestamp>_backbone`, `15_logical_train.sh` → `<timestamp>_logical_train`, `17_bkm_combined.sh` → `<timestamp>_bkm_combined`.

---

## 5. stage 부분만 돌리기 — 구체 예시

### 한 축만

```bash
bash scripts/run_paper_server_all.sh --round1-include-axes lr --skip-post
bash scripts/run_paper_server_all.sh --round1-include-axes asl --skip-post
bash scripts/sweeps_server/01_baseline.sh
```

축 이름: `lr, warmup, normal_ratio, per_class, weight_decay, smoothing, label_smoothing, asl, stochastic_depth, focal_gamma, abnormal_weight, ema, color, allow_tie_save, baseline`.

### stage 1~3만 (baseline + 모든 core 축 + color)

축마다 호출하거나:

```bash
for axis in baseline lr warmup normal_ratio per_class weight_decay smoothing label_smoothing asl \
            stochastic_depth focal_gamma abnormal_weight ema allow_tie_save color; do
  if [ "$axis" = baseline ]; then
    bash scripts/sweeps_server/01_baseline.sh
  else
    bash scripts/run_paper_server_all.sh --round1-include-axes "$axis" --skip-post
  fi
done
```

또는 한 줄:

```bash
bash scripts/run_paper_server_all.sh \
  --skip-weights --skip-dataset \
  --round1-include-axes lr,warmup,normal_ratio,per_class,weight_decay,smoothing,label_smoothing,asl,stochastic_depth,focal_gamma,abnormal_weight,ema,allow_tie_save,color \
  --skip-post
```

### stage 0,4만 (데이터 준비 + sample_skip)

```bash
bash scripts/run_paper_server_all.sh --skip-refcheck --skip-round1 --skip-post
bash scripts/sweeps_server/sample_skip.sh
```

### stage 5,8만 (backbone + bkm_combined)

```bash
bash scripts/sweeps_server/backbone.sh
bash scripts/sweeps_server/bkm_combined.sh
```

### stage 9만 (이미 결과 있을 때 표/plot만 갱신)

```bash
bash scripts/run_paper_server_all.sh \
  --skip-weights --skip-dataset --skip-refcheck --skip-round1
```

### 데이터셋 변형 yaml 7종

base + 6개 변형 (×1.15와 ×1.30 두 강도 × 3 종류 변형). 각 yaml의 `output.*_dir` 이 분리돼서 동시 생성도 안 부딪힘:

| yaml | 변형 | 출력 폴더 |
|---|---|---|
| `dataset.yaml` | base | `data/` |
| `dataset1_noise_15.yaml` | noise ×1.15 | `data_noise_15/` |
| `dataset2_noise_30.yaml` | noise ×1.30 | `data_noise_30/` |
| `dataset3_anomaly_15.yaml` | anomaly ×1.15 | `data_anomaly_15/` |
| `dataset4_anomaly_30.yaml` | anomaly ×1.30 | `data_anomaly_30/` |
| `dataset5_all_15.yaml` | noise ×1.15 **and** anomaly ×1.15 | `data_all_15/` |
| `dataset6_all_30.yaml` | noise ×1.30 **and** anomaly ×1.30 | `data_all_30/` |

scaling 적용 항목:
- noise: `gaussian.sigma_range`, `laplacian.b_range`, `correlated.sigma_range` 전부에 factor 곱함
- anomaly: `mean_shift.shift_sigma_range`, `standard_deviation.scale_range`, `spike.magnitude_sigma_range` (+ `min_magnitude_sigma`), `drift.slope_sigma_range` (+ `min_max_drift_sigma`, `visual_floor_sigma`), 그리고 `defect.enforcement.*_floor_sigma` 도 같이 올려서 약한 case 가 floor 에 잘리지 않게 함

### 한방 — 모든 dataset × 모든 axis × 모든 backbone 한 번에

현재 `scripts/all-dataset-backbone.sh` 기본 dataset은 4개입니다: `dataset.yaml`, `dataset1_noise_15.yaml`, `dataset3_anomaly_10.yaml`, `dataset5_all_a10n15.yaml`. 각 yaml의 image class는 6개(`normal`, `mean_shift`, `standard_deviation`, `spike`, `drift`, `context`)이고, `--mode binary` 학습에서는 이를 `normal` vs `abnormal` 2클래스로 접습니다.

서버에서 같이 돌아가려면 repo 코드 외에 각 config의 `output.data_dir/timeseries.csv`, `output.data_dir/scenarios.csv`, `output.image_dir/<split>/<class>/*.png`가 필요합니다. 없으면 prep 단계가 `generate_data.py`와 `generate_images.py`로 다시 만들고, cross-product backbone mode에서는 요청한 `weights/<model_name>.pth`가 먼저 있어야 합니다.

`scripts/all-dataset-backbone.sh -x` 한 줄. Cross-product mode는 한 실행 폴더 아래에 timestamp-prefixed dataset 폴더를 만들고, 그 아래 timestamp-prefixed backbone 폴더를 둡니다. 각 dataset/backbone cell이 끝날 때마다 `<ts>_cross_dataset_report/` 를 갱신하고, 전부 끝난 뒤 dataset/global 평균 BKM을 추가로 돌린 다음 최종 비교 표·plot 을 다시 생성합니다.

BKM은 3층입니다.

| 층 | 의미 | 실행 위치 |
|---|---|---|
| cell BKM | dataset-backbone cell 안에서 축별 best 조건 결합 | 각 `<ts>_<backbone>/05_bkm_combined_*` |
| dataset BKM | 같은 dataset의 backbone 4개 평균으로 축별 best 조건 선택 후, 그 dataset의 backbone 4개 재학습 | `<ts>_cross_dataset_report/consensus_bkm/dataset_*/` |
| global BKM | 4 dataset × 4 backbone 전체 평균으로 축별 best 조건 선택 후, 16 cell 재학습 | `<ts>_cross_dataset_report/consensus_bkm/global/` |

`-x` 기본 checkpoint 정책은 `--checkpoint-retention dataset-backbone-best --checkpoint-retention-scope log-group` 입니다. 그래서 각 log group 안에서 `best_model.pth`는 dataset config + backbone별 winner만 남고, 나머지 sweep run의 `.pth`는 삭제됩니다. `best_info.json`, 결과 json/md/csv는 유지됩니다.

기본값은 prep 단계에서 `download.py`를 실행해 `weights/{model_name}.pth`를 준비합니다. 폐쇄망 서버처럼 `weights/*.pth`를 이미 복사해 둔 경우에만 `--skip-weights`를 붙입니다.

```bash
bash scripts/all-dataset-backbone.sh -x
# 기본 4 yaml 순차: dataset.yaml, dataset1_noise_15.yaml, dataset3_anomaly_10.yaml, dataset5_all_a10n15.yaml
# 끝나고:
#   validations/<ts>_all_dataset_backbone/<ts>_cross_dataset_report/cross_dataset_summary.md
#   validations/<ts>_all_dataset_backbone/<ts>_cross_dataset_report/cross_dataset_summary.csv
#   validations/<ts>_all_dataset_backbone/<ts>_cross_dataset_report/cross_dataset_overall.csv
#   validations/<ts>_all_dataset_backbone/<ts>_cross_dataset_report/cross_dataset_f1.png
#   validations/<ts>_all_dataset_backbone/<ts>_cross_dataset_report/cross_dataset_backbone.png
#   validations/<ts>_all_dataset_backbone/<ts>_cross_dataset_report/cross_dataset_overall.png
#   validations/<ts>_all_dataset_backbone/<ts>_cross_dataset_report/consensus_bkm/
```

`-x` cross-product 결과 폴더 예시:

```text
validations/
└── 20260529_123456_all_dataset_backbone/
    ├── 20260529_123500_dataset/
    │   ├── 20260529_123500_prep/
    │   ├── 20260529_123620_convnexttiny/
    │   ├── 20260529_140515_convnextv2base/
    │   ├── 20260529_153044_convnextv2tiny/
    │   └── 20260529_170201_vitbasepatch16clip224/
    ├── 20260529_184030_dataset1_noise_15/
    │   ├── 20260529_184030_prep/
    │   ├── 20260529_184155_convnexttiny/
    │   ├── 20260529_201002_convnextv2base/
    │   ├── 20260529_213540_convnextv2tiny/
    │   └── 20260529_230711_vitbasepatch16clip224/
    ├── 20260530_003300_dataset3_anomaly_10/
    │   └── ...
    ├── 20260530_073455_dataset5_all_a10n15/
    │   └── ...
    └── 20260529_123456_cross_dataset_report/
        ├── consensus_bkm/
        │   ├── consensus_bkm_selection.md
        │   ├── dataset_dataset/
        │   ├── dataset_dataset1noise15/
        │   ├── dataset_dataset3anomaly10/
        │   ├── dataset_dataset5alla10n15/
        │   └── global/
        ├── cross_dataset_summary.md
        ├── cross_dataset_summary.csv
        ├── cross_dataset_overall.csv
        ├── cross_dataset_f1.png
        ├── cross_dataset_backbone.png
        └── cross_dataset_overall.png
```

옵션:
```bash
# yaml 부분집합만
bash scripts/all-dataset-backbone.sh --datasets dataset.yaml,dataset1_noise_15.yaml

# prep 이미 끝났을 때 (weights/data/baseline 모두 있음)
bash scripts/all-dataset-backbone.sh --skip-prep

# 폐쇄망: weights/*.pth가 이미 있을 때만 다운로드 생략
bash scripts/all-dataset-backbone.sh --skip-weights

# 끝의 cross-dataset 리포트만 다시
bash scripts/all-dataset-backbone.sh --skip-prep --skip-full

# dataset/global 평균 BKM은 생략하고 cell BKM + cross report만
bash scripts/all-dataset-backbone.sh -x --skip-consensus-bkm

# 00_all.sh 로 forward 할 인자는 `--` 뒤에
bash scripts/all-dataset-backbone.sh -- --max-launched 1 --force
```

서버에서 SSH 끊어도 계속:
```bash
nohup bash scripts/all-dataset-backbone.sh > /tmp/all_dsbk.log 2>&1 &
disown
tail -f /tmp/all_dsbk.log
```

GPU 카드 분배(yaml 별로 GPU 다르게):
```bash
nohup env CUDA_VISIBLE_DEVICES=0 bash scripts/all-dataset-backbone.sh \
  --datasets dataset.yaml,dataset1_noise_15.yaml,dataset2_noise_30.yaml \
  > /tmp/all_dsbk_gpu0.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 bash scripts/all-dataset-backbone.sh \
  --datasets dataset3_anomaly_15.yaml,dataset4_anomaly_30.yaml,dataset5_all_15.yaml,dataset6_all_30.yaml \
  > /tmp/all_dsbk_gpu1.log 2>&1 &
disown
```

각 dataset 의 group 폴더는 `<timestamp>_run_paper_<config-stem>/` (예: `logs/20260504_120000_run_paper_dataset1_noise_15/`, `validations/20260504_120000_run_paper_dataset1_noise_15/`).

### Multi-GPU (torchrun DDP 방식)

`scripts/all-dataset-backbone.sh -x` 한 줄. wrapper가 visible GPU 수를 확인해서 각 queued `train.py` run을 `torch.distributed.run`으로 실행합니다. `python train.py`를 직접 실행하면 `WORLD_SIZE=1`이라 DDP가 켜지지 않습니다. 다중 GPU가 보이는 상태에서 직접 실행하면 이제 느린 `nn.DataParallel`로 조용히 떨어지지 않고 에러로 멈춥니다.
- `--batch_size`는 global batch로 유지
- GPU 수가 `N`이면 rank-local micro-batch는 `batch_size / N`
- 각 rank가 sampler로 자기 shard를 읽고, DDP all-reduce 뒤 optimizer update는 global batch 1번
- `batch_size`가 GPU 수로 나누어떨어지지 않으면 실패시킴
- controller는 각 queued run마다 `127.0.0.1:<free_port>`를 명시해서 torchrun을 띄웁니다. DDP 초기 socket listen 실패가 나면 학습 epoch 진입 전 실패로 보고 기본 2회 재시도합니다 (`AD_TRAIN_DDP_INIT_RETRIES`로 조정). 각 run 종료 후에는 남은 torchrun/train process group을 정리하고 기본 5초 쉰 뒤 다음 조건으로 넘어갑니다 (`AD_TRAIN_RUN_CLEANUP_SLEEP`로 조정).

```bash
# 모든 visible GPU 사용
bash scripts/all-dataset-backbone.sh -x

# GPU 부분집합만 (4개)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/all-dataset-backbone.sh -x

# nohup 백그라운드
nohup bash scripts/all-dataset-backbone.sh -x > /tmp/all_dsbk_ddp.log 2>&1 &
disown
```

`args.batch_size` 는 그대로입니다. 예: batch_size=32 + 4 GPU = 각 rank가 8 samples 처리, optimizer는 batch 32 기준 한 step 적용. **LR/warmup 등 hparam 그대로 재사용 가능**.

직접 단일 run을 DDP로 띄울 때:

```bash
python -m torch.distributed.run --nproc-per-node=4 train.py --config dataset.yaml --mode binary --batch_size 256
```

정말로 느린 단일 process `nn.DataParallel`을 테스트하려면 명시적으로 허용해야 합니다:

```bash
AD_ALLOW_DATA_PARALLEL=1 python train.py --config dataset.yaml --mode binary
```

검증 명령: `python -m py_compile train.py scripts/adaptive_experiment_controller.py`, `AD_TRAIN_DDP_NPROC=2 python scripts/adaptive_experiment_controller.py --queue validations/01_baseline_queue.json --dry-run --force`.

### 주말 한 GPU 순차 실행 (7개 yaml 자동, GUI/터미널 끊어도 계속 돔)

```bash
ssh user@server     # GUI 터미널이든 PuTTY든 상관없음

nohup bash -c '
for cfg in dataset.yaml \
           dataset1_noise_15.yaml dataset2_noise_30.yaml \
           dataset3_anomaly_15.yaml dataset4_anomaly_30.yaml \
           dataset5_all_15.yaml dataset6_all_30.yaml; do
  bash scripts/run_paper_server_all.sh --config "$cfg"
done
' > /tmp/weekend.log 2>&1 &
disown

# 이제 SSH 끊거나 GUI 끄거나 노트북 덮어도 학습은 계속 돔
```

진행 확인 / 종료 / 살아있나 체크:

```bash
ssh user@server
tail -f /tmp/weekend.log               # 실시간 진행 (Ctrl+C 로 빠져나옴, 학습은 안 죽음)
ps -ef | grep run_paper                # 살아있는 프로세스 확인
nvidia-smi                             # GPU 사용 중인지 확인
pkill -f run_paper_server_all.sh       # 강제 종료
```

`all-dataset-backbone.sh -x` 결과는 한 실행 루트 아래 dataset/backbone 하위폴더로 모입니다:

```
validations/
└── 20260529_123456_all_dataset_backbone/
    ├── 20260529_123500_dataset/
    │   ├── 20260529_123500_prep/
    │   └── 20260529_153044_convnextv2tiny/
    ├── 20260529_184030_dataset1_noise_15/
    │   └── 20260529_213540_convnextv2tiny/
    ├── 20260530_003300_dataset3_anomaly_10/
    │   └── 20260530_032912_convnextv2tiny/
    ├── 20260530_073455_dataset5_all_a10n15/
    │   └── 20260530_103218_convnextv2tiny/
    └── 20260529_123456_cross_dataset_report/
logs/
└── 20260529_123456_all_dataset_backbone/
    └── 20260529_123500_dataset/
        └── 20260529_153044_convnextv2tiny/
```

데이터셋/백본 group 폴더만 따로 정리해서 보고 싶으면:

```bash
python scripts/generate_group_report.py --group-dir logs/20260529_123456_all_dataset_backbone/20260529_123500_dataset/20260529_153044_convnextv2tiny
python scripts/generate_group_report.py --group-dir logs/20260529_123456_all_dataset_backbone/20260529_184030_dataset1_noise_15/20260529_213540_convnextv2tiny
# ...
```

### 데이터셋 여러 개 **순차** 실행 (한 GPU 자동 자율 실행)

가장 단순:

```bash
bash scripts/run_paper_server_all.sh --config dataset.yaml && \
bash scripts/run_paper_server_all.sh --config dataset1.yaml && \
bash scripts/run_paper_server_all.sh --config dataset2.yaml
```

`&&` = 앞이 성공해야 다음 실행. `;` 로 바꾸면 실패해도 계속.

⚠️ `cmd1 & cmd2` 는 **순차 아님** — `&`는 백그라운드 실행이라 둘이 동시에 시작합니다. 한 GPU면 OOM.

SSH 끊어도 끝까지 자동으로 돌리고 싶으면 `nohup` + 로그:

```bash
nohup bash -c '
for cfg in dataset.yaml dataset1.yaml dataset2.yaml; do
  bash scripts/run_paper_server_all.sh --config "$cfg"
done
' > /tmp/run_seq.log 2>&1 &
disown
```

확인 / 종료:

```bash
tail -f /tmp/run_seq.log               # 실시간 진행
ps -ef | grep run_paper                # 살아있나
pkill -f run_paper_server_all.sh       # 종료
```

각 yaml 결과는 자동으로 `validations/<timestamp>_run_paper_<yaml-stem>/`에 분리되어 쌓입니다 (예: `validations/20260430_120000_run_paper_dataset/`, `validations/20260430_133000_run_paper_dataset1/`).

### 데이터셋 여러 개 **병렬** (다른 yaml + 다른 GPU)

`--config` 로 yaml을 따로 지정하면 됩니다. 기본 `LOG_DIR_GROUP`이 yaml 파일명을 자동으로 포함하므로 같은 초에 launch해도 group 폴더가 충돌하지 않습니다 (`<timestamp>_run_paper_dataset`, `<timestamp>_run_paper_dataset1`처럼).

⚠️ 각 yaml의 `output.data_dir`/`image_dir`/`display_dir`이 **서로 달라야** 데이터 생성도 충돌하지 않습니다 (예: `dataset.yaml` → `data/`, `dataset1.yaml` → `data1/`).

⚠️ 같은 GPU에 두 학습이 들어가면 OOM. GPU 카드별로 분리:

```bash
# GPU 0 → dataset.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_server_all.sh --config dataset.yaml &

# GPU 1 → dataset1.yaml
CUDA_VISIBLE_DEVICES=1 bash scripts/run_paper_server_all.sh --config dataset1.yaml &

wait    # 둘 다 끝날 때까지 대기
```

각각 `validations/<timestamp>_run_paper_dataset/`, `validations/<timestamp>_run_paper_dataset1/` 에 결과가 분리되어 쌓입니다.

GPU 카드가 2개 이상이면 nohup으로 백그라운드 + 로그 분리:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_server_all.sh --config dataset.yaml  > /tmp/run_ds0.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 bash scripts/run_paper_server_all.sh --config dataset1.yaml > /tmp/run_ds1.log 2>&1 &
disown
```

### 같은 group 으로 묶어 돌리기

여러 명령을 한 묶음으로 보고 싶을 때 group 명을 직접 지정:

```bash
GROUP=run_$(date +%Y%m%d_%H%M%S)
bash scripts/sweeps_server/01_baseline.sh --log-dir-group "$GROUP"
bash scripts/run_paper_server_all.sh --round1-include-axes lr --skip-post --log-dir-group "$GROUP"
bash scripts/sweeps_server/backbone.sh       --log-dir-group "$GROUP"
```

전 stage 결과가 `logs/$GROUP/<run>/` 아래로 모입니다.

### 1 run만 검증 (학습 launch 안 함)

```bash
bash scripts/sweeps_server/backbone.sh --prepare-only          # queue/active만 만듦
bash scripts/run_paper_server_all.sh --round1-include-axes lr --max-launched 1 --skip-post
```

---

## 6. 추론 (`inference.py`) + 텍스트 리스트

```bash
python inference.py \
  --model logs/<group>/<run>/best_model.pth \
  --output_dir my_inference
```

출력 폴더는 시각 prefix 자동 부여: `<YYMMDD_HHMMSS>_my_inference/` (덮어쓰기 방지, ls 시간순 정렬):

```
<YYMMDD_HHMMSS>_my_inference/
├── abnormal/             <- 불량 판정 display 이미지
├── normal/               <- 정상 판정 display 이미지
├── predictions.csv       <- 모든 chart, p_abnormal, predicted, p_normal
├── predictions.txt       <- 통합 텍스트 (ABNORMAL 위, NORMAL 아래)
├── abnormal_list.txt     <- 불량 chart_id 목록만
└── normal_list.txt       <- 정상 chart_id 목록만
```

`abnormal_list.txt` / `normal_list.txt` 형식:

```
ch_09100   p_abn=0.9991   ch_09100.png
ch_09572   p_abn=0.0017   ch_09572.png
```

특정 split만 보고 싶으면:

```bash
python inference.py --model logs/<run>/best_model.pth --split test
```

서버 batch 추론 (모델 폴더 통째로):

```bash
python scripts/server_batch_predict.py --model-run logs/<group>/<run>
```

2-stage 추론은 1차 binary 모델과 2차 `anomaly_type` 모델을 따로 학습해 둔 뒤 실행합니다. 1차에서 `p_normal > normal_threshold`이면 normal로 종료하고, 그 외 sample만 2차 모델이 `mean_shift`, `spike`, `drift`, `context` 같은 defect type을 붙입니다. 상세한 해석과 FP/FN 진단법은 `docs/two_stage_workflow.md`를 봅니다.

```bash
# 1차 pass/fail gate
python train.py --config dataset.yaml --mode binary --log_dir binary_gate

# 2차 abnormal-only type classifier
python train.py --config dataset.yaml --mode anomaly_type --log_dir type_classifier

# 1차 결과가 abnormal인 chart만 2차로 분류
python scripts/two_stage_predict.py \
  --binary-model-run logs/<binary_gate_run> \
  --type-model-run logs/<type_classifier_run> \
  --dataset-dir data \
  --split test \
  --normal-threshold 0.9 \
  --output-dir two_stage_test \
  --device cpu
```

출력은 `two_stage_predictions.csv`와 `summary.json`입니다. CSV에는 `p_normal`, `p_abnormal`, `binary_pred`, `stage2_ran`, `stage2_pred`, `final_pred`, `bucket`이 들어가므로 binary FN/FP와 defect type별 실패 원인을 같이 볼 수 있습니다.

---

## 7. 현업 CSV 가져왔을 때

`all-dataset-backbone.sh -x` 완료 후 BKM 모델로 현업 CSV predict만 할 때는
[`docs/bkm_field_predict.md`](docs/bkm_field_predict.md)를 기준으로 본다.

현업 데이터는 **`timeseries.csv` 한 개만** 있고 `scenarios.csv`는 없습니다(라벨 모름). production 기본 흐름:

```bash
MODEL_RUN="logs/<all_dataset_backbone>/<...>_bkm_global/<run_folder>"
bash scripts/run_field_predict.sh fab_export/timeseries.csv "$MODEL_RUN"
```

wrapper 내부 순서:

```text
scripts/generate_field_images.py  -> 현업 CSV를 model input image + manifest.csv로 렌더링
scripts/predict_images.py         -> BKM best_model.pth로 normal/abnormal 판정
```

출력:

```text
field_runs/<YYMMDD_HHMMSS>_images/
field_runs/<YYMMDD_HHMMSS>_predictions/predictions.csv
```

라벨 없는 production CSV에서는 `predicted`, `p_normal`, `p_abnormal`만 해석한다. `FN/FP/F1`은 정답 컬럼이 없으면 계산하지 않는다.

---

## 8. normal/abnormal 폴더로 추가 학습

이미 분류된 폴더가 있을 때 fine-tune:

```
extra_images/
  normal/
  abnormal/
```

```bash
python scripts/add_training_from_folders.py \
  --model-run logs/<group>/<run> \
  --image-root extra_images \
  --epochs 3 --lr 1e-5 --scheduler cosine
```

`best_model.pth`에서 weight만 불러와서 fine-tune. 출력은 `logs/addtrain_*/`. 추가 학습 입력은 **모델 입력용 이미지** (display 아님).

---

## 9. logs에서 표·plot 만들기

```bash
python scripts/generate_log_history_report.py \
  --logs-dir logs \
  --out-prefix validations/log_history_report_rawbase \
  --contains rawbase \
  --top-k 30
```

`logs/<group>/<run>/history.json` 까지 자동으로 추적합니다. 특정 group만 보고 싶으면 `--contains 20260430_120000_run_paper` 처럼 지정.

출력: markdown / CSV / PNG (candidate F1 막대, val_f1 곡선, grad p99 곡선, FN/FP 산점).

Labeled inference output에서 AUROC와 threshold별 FN/FP를 바로 보고 싶으면:

```bash
python scripts/binary_threshold_report.py \
  --predictions <inference_output>/predictions.csv
```

`normal_threshold=0.9`는 `p_normal > 0.9`일 때만 normal로 통과시키는 train.py 기준입니다. 즉 `p_abnormal >= 0.1`이면 abnormal로 보내는 운영점입니다.

---

## 10. Grad-CAM

```bash
python scripts/gradcam_report.py \
  --model-run logs/<group>/<run> \
  --image-root images/test \
  --out-dir validations/gradcam_probe \
  --include-classes normal,mean_shift,standard_deviation,spike,drift,context \
  --limit-per-class 6 \
  --save-heat-only \
  --heat-threshold 0.0 --heat-min-alpha 0.18
```

CAM 은 모델 근거 위치를 보여주는 것이지 실제 anomaly 위치가 아닙니다. 후처리 룰로는 쓰지 않음.

FP/FN만 따로 보고 싶으면 `gradcam_error_report.py`.

---

## 11. FP / FN 한쪽으로 치우칠 때

| 현상 | 우선 시도 |
|---|---|
| FP가 많다 (정상을 anomaly로 잡음) | `normal_ratio`↑ 또는 `--max_per_class`↑, `abnormal_weight`↓ 또는 `focal_gamma`↓ |
| FN이 많다 (불량을 정상으로 놓침) | `abnormal_weight`↑, `focal_gamma`↑, `label_smoothing` 주변값 |

한 번에 여러 축 동시에 바꾸지 말고 한 축씩만 비교 (이게 `02_sweep_queue.json` 의 핵심 원칙).

---

## 결과 파일 (`validations/`)

template 큐는 batch끼리 공유하므로 root에:

| 파일 | 뜻 |
|---|---|
| `validations/01_baseline_queue.json` | baseline 5-seed 입력 |
| `validations/02_sweep_queue.json` | 축별 sweep 입력 |
| `validations/03_sample_skip_queue.json` | sample-skip 1-run 입력 |

**출력은 batch마다 분리**돼서 `validations/<group>/`에 쌓입니다. `all-dataset-backbone.sh -x`는 group이 `<ts>_all_dataset_backbone/<ts>_<dataset>/<ts>_<backbone>` 형태입니다.

| 파일 (`<group>/` 안) | 뜻 |
|---|---|
| `01_baseline_active.json` / `_results.{json,md}` | baseline 실행 큐 + 결과 |
| `02_sweep_active.json` / `_results.{json,md}` | sweep 실행 큐 + live 결과 |
| `02_sweep_report.md` / `02_sweep_plots/` | postprocess 종합 리포트 + 축별 plot |
| `03_sample_skip_active.json` / `_results.{json,md}` / `_plot.png` | sample-skip + 비교 plot |
| `04_backbone_queue.json` / `_active.json` / `_results.{json,md}` / `_plot.png` | backbone + 비교 plot |
| `05_bkm_combined_queue.json` / `_active.json` / `_results.{json,md}` / `_plot.png` | BKM combined + 비교 plot |
| `15_logical_train_queue.json` / `_results.{json,md}` | logical train |
| `instability_cases.{csv,json,md}` / `prediction_trend_latest.*` | postprocess 분석 |
| `run.log` | 이 batch 의 통합 실행 로그 |

여러 번 `00_all.sh` 돌리면 group 폴더가 시간순으로 정렬되고 서로 안 덮어씁니다.
