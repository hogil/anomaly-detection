# HOW TO RUN V11

현재 프로젝트의 기준 파이프라인은 `v11` 입니다. 이 문서는 회사 환경에서 데이터셋 생성부터 학습, sweep, ablation까지 다시 돌릴 때 필요한 파일과 명령을 정리한 실행 문서입니다.

## 1. 현재 기준 ref

- 데이터셋: `dataset.yaml`
- 학습 엔트리: `train.py`
- sweep 엔트리: `run_experiments_v11.py`
- 현재 기준 ref 설정:
  - `v11`에서 선택된 LR baseline
  - backbone LR `2e-5`
  - head LR `2e-4`
  - warmup `5`
  - mode: `binary`
  - `normal_ratio=700`
  - 나머지 하이퍼파라미터는 `configs/train/winning.yaml` 기본값 사용
- 현재 ref 성능:
  - `fresh0413_reset` sweep 기준 `n=700` 5-seed mean `F1=0.9969 ± 0.0014`
  - best single run: `F1=0.9993`

## 2. 단계별 파일

| 단계 | 파일 | 역할 |
|---|---|---|
| pretrained weights 준비 | `download.py` | timm backbone weights 다운로드 |
| 데이터셋 생성 | `generate_data.py` | `data/scenarios.csv`, `data/timeseries.csv` 생성 |
| 이미지 생성 | `generate_images.py` | `images/`, `display/` 렌더링 |
| 데이터셋 검증 | `scripts/validate_dataset.py` | 생성된 anomaly 강도와 weak case 검증 |
| 단일 학습 | `train.py` | 1개 설정 학습 |
| 전체 sweep/ablation | `run_experiments_v11.py` | `sweep`, `lr`, `gc`, `smooth`, `reg`, `color` 그룹 실행 |

## 3. 사전 준비

### Python 환경

프로젝트 루트에서 아래가 동작해야 합니다.

```bash
python --version
python -c "import torch, timm, pandas, yaml"
```

### pretrained weights

인터넷이 되는 머신에서 먼저 받거나, 이미 받은 `weights/` 폴더를 사내 서버로 복사합니다.

```bash
python download.py
```

필수 파일:

```text
weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth
```

## 4. 원본 Python 명령

### 4.1 데이터셋 생성

```bash
python generate_data.py --config dataset.yaml --workers 1
python generate_images.py --config dataset.yaml --workers 1
python scripts/validate_dataset.py --config dataset.yaml
```

리눅스 서버에서는 worker 수를 더 키워도 됩니다.

```bash
python generate_data.py --config dataset.yaml --workers 8
python generate_images.py --config dataset.yaml --workers 8
python scripts/validate_dataset.py --config dataset.yaml
```

### 4.2 단일 학습

현재 ref를 그대로 1회 돌리려면:

```bash
python train.py ^
  --config dataset.yaml ^
  --normal_ratio 700 ^
  --seed 42 ^
  --num_workers 1 ^
  --prefetch_factor 4 ^
  --log_dir company_ref_n700_s42
```

리눅스/bash에서는:

```bash
python train.py \
  --config dataset.yaml \
  --normal_ratio 700 \
  --seed 42 \
  --num_workers 1 \
  --prefetch_factor 4 \
  --log_dir company_ref_n700_s42
```

### 4.3 Group B sweep

`normal_ratio`별 5-seed sweep:

```bash
python run_experiments_v11.py --groups sweep --num_workers 1 --name-prefix company_run
```

### 4.4 Group C/D/F/G ablation

현재 기준 추천 `base_n`은 `700` 입니다.

```bash
python run_experiments_v11.py \
  --groups lr gc smooth reg \
  --base_n 700 \
  --num_workers 1 \
  --name-prefix company_run
```

### 4.5 Train per-class count sweep

train split에서 original class(`normal`, `mean_shift`, `std`, `spike`, `drift`, `context`) 각각을 같은 개수로 제한하는 실험입니다.

```bash
python run_experiments_v11.py \
  --groups perclass \
  --num_workers 1 \
  --name-prefix company_run
```

이 그룹은 아래 count를 5-seed로 평가합니다.

```text
100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
```

### 4.6 결과 summary만 다시 보기

```bash
python run_experiments_v11.py --only-summary --base_n 700 --name-prefix company_run
```

### 4.7 논문용 ablation 표 생성 / combo follow-up

현재까지 완료된 결과로 paper-style 표를 만들려면:

```bash
python scripts/paper_followup_v11.py --prefix company_run --base-n 700
```

ablation이 끝날 때까지 계속 감시하고, 끝나면 best item들을 묶은 `combo` run까지 자동 발사하려면:

```bash
python scripts/paper_followup_v11.py \
  --prefix company_run \
  --base-n 700 \
  --num-workers 1 \
  --watch \
  --launch-combo
```

## 5. 래퍼 스크립트

직접 긴 명령을 치지 않게 아래 스크립트를 추가했습니다.

### Windows PowerShell

파일: `scripts/run_v11_pipeline.ps1`

예시:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_v11_pipeline.ps1 -Stage weights

powershell -ExecutionPolicy Bypass -File .\scripts\run_v11_pipeline.ps1 -Stage dataset -Workers 1

powershell -ExecutionPolicy Bypass -File .\scripts\run_v11_pipeline.ps1 -Stage train -NormalRatio 700 -Seed 42 -LogName company_ref_n700_s42

powershell -ExecutionPolicy Bypass -File .\scripts\run_v11_pipeline.ps1 -Stage sweep -NamePrefix company_run -NumWorkers 1

powershell -ExecutionPolicy Bypass -File .\scripts\run_v11_pipeline.ps1 -Stage perclass -NamePrefix company_run -NumWorkers 1

powershell -ExecutionPolicy Bypass -File .\scripts\run_v11_pipeline.ps1 -Stage ablation -BaseN 700 -NamePrefix company_run -NumWorkers 1

powershell -ExecutionPolicy Bypass -File .\scripts\run_v11_pipeline.ps1 -Stage summary -BaseN 700 -NamePrefix company_run

powershell -ExecutionPolicy Bypass -File .\scripts\run_v11_pipeline.ps1 -Stage paper -BaseN 700 -NamePrefix company_run
```

### Linux / bash

파일: `scripts/run_v11_pipeline.sh`

예시:

```bash
bash scripts/run_v11_pipeline.sh weights

bash scripts/run_v11_pipeline.sh dataset --workers 8

bash scripts/run_v11_pipeline.sh train --normal-ratio 700 --seed 42 --log-name company_ref_n700_s42

bash scripts/run_v11_pipeline.sh sweep --name-prefix company_run --num-workers 8

bash scripts/run_v11_pipeline.sh perclass --name-prefix company_run --num-workers 8

bash scripts/run_v11_pipeline.sh ablation --base-n 700 --name-prefix company_run --num-workers 8

bash scripts/run_v11_pipeline.sh summary --base-n 700 --name-prefix company_run

bash scripts/run_v11_pipeline.sh paper --base-n 700 --name-prefix company_run
```

## 6. 권장 실행 순서

### 회사 서버에서 처음부터 다시 돌릴 때

1. `weights/` 준비
2. `dataset` stage 실행
3. `train` stage로 ref 1개 확인
4. `sweep` stage 실행
5. `summary`로 best `n` 확인
6. 필요하면 `perclass` stage 실행
7. `ablation` stage 실행
8. `paper` stage로 개별 best item / combo 표 생성

### 가장 짧은 검증 루프

```bash
bash scripts/run_v11_pipeline.sh train --normal-ratio 700 --seed 42 --log-name smoke_ref
```

## 7. 결과 위치

- 데이터:
  - `data/scenarios.csv`
  - `data/timeseries.csv`
- 이미지:
  - `images/`
  - `display/`
- 검증 결과:
  - `validations/<timestamp>/`
- 학습 결과:
  - `logs/YYMMDD_HHMMSS_<name>_F<test_f1>_R<test_recall>/`
- 실험 스펙 snapshot:
  - `configs/runs/*.yaml`
- 실험 요약:
  - `logs/v11_experiments_summary_<prefix>.json`
- 논문용 표:
  - `logs/paper_tables/<prefix>_n<base_n>_paper_table.md`
  - `logs/paper_tables/<prefix>_n<base_n>_paper_table.csv`

## 8. 운영 메모

- Windows에서는 현재 `--num_workers 1`이 가장 안전했습니다.
- `normal_ratio`만 키우면 현재 데이터셋에서는 `FP`는 줄고 `FN`이 늘어나는 경향이 있었습니다.
- 그래서 현재 후속 실험 기준점은 `base_n=700` 입니다.
- `HOW_TO_RUN.md`는 예전 러너 기준 문서이고, 현재 `v11` 실행은 이 문서 기준으로 보는 것이 맞습니다.
