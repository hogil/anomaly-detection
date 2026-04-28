# anomaly-detection

시계열 anomaly 데이터 생성, 이미지 렌더링, 학습, 추론, 로그 리포트 생성 repo입니다.

## Core Flow

| step | command | output |
| --- | --- | --- |
| data | `python generate_data.py --config dataset.yaml --workers 24` | `data/timeseries.csv`, `data/scenarios.csv` |
| images | `python generate_images.py --config dataset.yaml --workers 24` | `images/`, `display/` |
| train | `python train.py --config dataset.yaml --log_dir my_run` | `logs/<run>/best_model.pth`, `best_info.json`, `history.json` |
| inference | `python inference.py --model logs/<run>/best_model.pth` | predictions/metrics |
| batch inference | `python scripts/server_batch_predict.py --model-run logs/<run>` | server inference outputs |
| log report | `python scripts/generate_log_history_report.py --logs-dir logs --out-prefix validations/log_history_report --contains rawbase` | markdown, CSV, PNG plots |

## Main Files

- `dataset.yaml`: active dataset/rendering config
- `config.yaml`: default training config
- `generate_data.py`: tabular data generator
- `generate_images.py`: training/display image renderer
- `train.py`: training entrypoint
- `inference.py`: single-model inference
- `scripts/server_batch_predict.py`: batch inference
- `scripts/generate_log_history_report.py`: tables and plots from `logs/`
- `scripts/sweeps_server/00_all.sh`: current server experiment resume
- `docs/summary.md`: current experiment summary

Generated folders such as `data/`, `images/`, `display/`, `logs/`, `weights/`, and `validations/` are gitignored.
