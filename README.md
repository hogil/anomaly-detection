# anomaly-detection

시계열 anomaly 데이터 생성, 이미지 렌더링, 학습, 추론, 로그 리포트 생성 repo입니다.

## Core Flow

| step | command | output |
| --- | --- | --- |
| data | `python generate_data.py --config dataset.yaml --workers 24` | `data/timeseries.csv`, `data/scenarios.csv` |
| images | `python generate_images.py --config dataset.yaml --workers 24` | `images/`, `display/` |
| train | `python train.py --config dataset.yaml --log_dir my_run` | `logs/<run>/best_model.pth`, `best_info.json`, `history.json` |
| grouped train | `python train.py --config dataset.yaml --log_dir my_run --log_dir_group run_YYYYMMDD_HHMMSS` | `logs/<group>/<run>/best_model.pth`, grouped paper-run logs |
| inference images | `python scripts/generate_inference_images.py --timeseries data/timeseries.csv --scenarios data/scenarios.csv --out-dir inference_inputs` | flat model-input images and manifest |
| inference | `python inference.py --model logs/<run>/best_model.pth` | predictions/metrics |
| add training | `python scripts/add_training_from_folders.py --model-run logs/<run> --image-root extra_images` | fine-tuned `logs/addtrain_*/best_model.pth` |
| batch inference | `python scripts/server_batch_predict.py --model-run logs/<run>` | server inference outputs |
| Grad-CAM | `python scripts/gradcam_report.py --model-run logs/<run> --image-root images/test --out-dir validations/gradcam_probe --save-heat-only` | trend+CAM overlays, transparent heat masks, heat CSV |
| FP/FN Grad-CAM | `python scripts/gradcam_error_report.py --model-run logs/<run> --error-type fp` | CAM overlays for false positives or false negatives |
| group report | `python scripts/generate_group_report.py --group-dir logs/run_YYYYMMDD_HHMMSS` | per-group candidate table + F1/val curves |
| log report | `python scripts/generate_log_history_report.py --logs-dir logs --out-prefix validations/log_history_report --contains rawbase` | markdown, CSV, PNG plots |

## Main Files

- `dataset.yaml`: active dataset/rendering config
- `generate_data.py`: tabular data generator
- `generate_images.py`: training/display image renderer
- `train.py`: training entrypoint
- `inference.py`: single-model inference
- `scripts/server_batch_predict.py`: batch inference
- `scripts/generate_inference_images.py`: inference image renderer from existing trend CSVs
- `scripts/add_training_from_folders.py`: fine-tune a best model from `normal/` and `abnormal/` image folders
- `scripts/gradcam_report.py`: Grad-CAM overlays on trend images and transparent heat masks
- `scripts/gradcam_error_report.py`: Grad-CAM overlays for FP/FN samples
- `scripts/generate_log_history_report.py`: tables and plots from flat or grouped `logs/**/`
- `scripts/sweeps_server/00_all.sh`: current paper experiment pipeline, ending with color -> sample_skip -> backbone -> logical_train -> gc -> BKM combined
- `docs/summary.md`: current experiment summary

Generated folders such as `data/`, `images/`, `display/`, `logs/`, `weights/`, and `validations/` are gitignored.
