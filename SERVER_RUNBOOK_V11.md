# SERVER RUNBOOK V11

이 문서는 다른 서버에서 `v11` 실험을 처음부터 다시 돌릴 때 필요한 최소 절차를 정리한 실행 문서입니다.

## 1. 목적

- `ref` 기준 실행
- `normal_ratio` sweep 실행
- `train per-class count` sweep 실행
- `best_n=700` 기준 `lr/gc/smooth/reg` ablation 실행

## 2. 서버 준비

### 2.1 코드 받기

```bash
git clone https://github.com/hogil/anomaly-detection.git
cd anomaly-detection
git checkout master
git pull origin master
```

별도 복사 없이 repo root의 `dataset.yaml` 을 바로 사용하면 됩니다.
기본 설정은 `configs/datasets/v11.yaml` 과 동일합니다.

### 2.2 Python 환경

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 기본 확인

```bash
python --version
python -c "import torch, timm, pandas, yaml"
nvidia-smi
```

## 3. pretrained weights

외부 인터넷이 가능하면:

```bash
python download.py
```

이미 받은 파일을 복사하는 경우 최소 파일:

```text
weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth
```

## 4. 가장 간단한 전체 실행

가장 단순한 방식은 wrapper의 `all` stage 입니다.

```bash
bash scripts/run_v11_pipeline.sh all \
  --config dataset.yaml \
  --workers 8 \
  --num-workers 8 \
  --base-n 700 \
  --name-prefix server_run
```

이 한 줄이 아래를 순서대로 실행합니다.

1. dataset 생성
2. normal_ratio sweep
3. perclass sweep
4. ablation
5. paper table 생성

## 5. 데이터셋 생성

```bash
bash scripts/run_v11_pipeline.sh dataset --workers 8
```

직접 명령:

```bash
python generate_data.py --config dataset.yaml --workers 8
python generate_images.py --config dataset.yaml --workers 8
python scripts/validate_dataset.py --config dataset.yaml
```

## 6. ref 1회 확인

현재 ref는 `v11`에서 선택된 LR baseline입니다.

- backbone LR `2e-5`
- head LR `2e-4`
- warmup `5`
- `normal_ratio=700`

```bash
bash scripts/run_v11_pipeline.sh train \
  --normal-ratio 700 \
  --seed 42 \
  --log-name server_ref_n700_s42 \
  --num-workers 8
```

## 7. normal_ratio sweep

```bash
bash scripts/run_v11_pipeline.sh sweep \
  --name-prefix server_run \
  --num-workers 8
```

직접 명령:

```bash
python run_experiments_v11.py \
  --groups sweep \
  --server h200 \
  --gpus 2 \
  --num_workers 8 \
  --name-prefix server_run
```

## 8. train per-class count sweep

이 실험은 train split에서 original class별 최대 개수를 동일하게 맞춥니다.

- 대상 count: `100, 200, 300, 400, 500, 600, 700, 800, 900, 1000`
- 적용 범위: train split only
- val/test: 그대로 유지

```bash
bash scripts/run_v11_pipeline.sh perclass \
  --name-prefix server_run \
  --num-workers 8
```

직접 명령:

```bash
python run_experiments_v11.py \
  --groups perclass \
  --server h200 \
  --gpus 2 \
  --num_workers 8 \
  --name-prefix server_run
```

## 9. ablation

현재 기준 `best_n=700`으로 진행합니다.

```bash
bash scripts/run_v11_pipeline.sh ablation \
  --base-n 700 \
  --name-prefix server_run \
  --num-workers 8
```

직접 명령:

```bash
python run_experiments_v11.py \
  --groups lr gc smooth reg \
  --base_n 700 \
  --server h200 \
  --gpus 2 \
  --num_workers 8 \
  --name-prefix server_run
```

## 10. summary / paper table

```bash
bash scripts/run_v11_pipeline.sh summary --base-n 700 --name-prefix server_run
bash scripts/run_v11_pipeline.sh paper --base-n 700 --name-prefix server_run
```

직접 명령:

```bash
python run_experiments_v11.py --only-summary --base_n 700 --name-prefix server_run
python scripts/paper_followup_v11.py --prefix server_run --base-n 700
```

## 11. 백그라운드 실행 예시

### tmux

```bash
tmux new -s v11
bash scripts/run_v11_pipeline.sh sweep --name-prefix server_run --num-workers 8
```

분리:

```bash
Ctrl-b d
```

다시 붙기:

```bash
tmux attach -t v11
```

### nohup

```bash
nohup bash scripts/run_v11_pipeline.sh perclass --name-prefix server_run --num-workers 8 \
  > logs/server_perclass.out 2>&1 &
```

## 12. 결과 위치

- data: `data/scenarios.csv`, `data/timeseries.csv`
- images: `images/`, `display/`
- train 결과: `logs/YYMMDD_HHMMSS_<run_name>_F<test_f1>_R<test_recall>/`
- paper table: `logs/paper_tables/`

## 13. 운영 메모

- Windows에서는 `num_workers=1`이 안전했고, Linux 서버는 `8`부터 시작하는 것이 합리적입니다.
- `normal_ratio` sweep과 `perclass` sweep은 다른 실험입니다.
  - `sweep`: normal 개수만 바꿈
  - `perclass`: train split original class 전체를 같은 cap으로 제한
- `ref`는 현재 LR 재탐색 결과가 아니라, `v11`에서 이미 선택된 LR baseline입니다.
