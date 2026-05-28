# anomaly-detection

시계열 anomaly 데이터 생성, 이미지 렌더링, 학습, 추론, 로그 리포트 생성 repo입니다.

## Core Flow

서버에서 처음부터 전체 재실행하거나 현업 CSV를 추론/추가학습에 쓰는 짧은 runbook은
[`docs/server_full_run_and_field_data.md`](docs/server_full_run_and_field_data.md)를 봅니다.

| step | command | output |
| --- | --- | --- |
| problem setting | see `docs/problem_setting.md` | binary gate as primary, multiclass/anomaly baselines as secondary |
| data | `python generate_data.py --config dataset.yaml --workers 24` | `data/timeseries.csv`, `data/scenarios.csv` |
| images | `python generate_images.py --config dataset.yaml --workers 24` | `images/`, `display/` |
| train | `python train.py --config dataset.yaml --mode binary --log_dir my_run` | `logs/<run>/best_model.pth`, `best_info.json`, `history.json` |
| train stage2 type | `python train.py --config dataset.yaml --mode anomaly_type --log_dir type_run` | abnormal-only defect type classifier |
| grouped train | `python train.py --config dataset.yaml --log_dir my_run --log_dir_group run_YYYYMMDD_HHMMSS` | `logs/<group>/<run>/best_model.pth`, grouped paper-run logs |
| inference images | `python scripts/generate_inference_images.py --timeseries data/timeseries.csv --scenarios data/scenarios.csv --out-dir inference_inputs` | flat model-input images and manifest |
| inference | `python inference.py --model logs/<run>/best_model.pth` | predictions/metrics |
| two-stage inference | `python scripts/two_stage_predict.py --binary-model-run logs/<binary_run> --type-model-run logs/<type_run>` | binary gate + defect type CSV |
| binary threshold report | `python scripts/binary_threshold_report.py --predictions <inference_output>/predictions.csv` | AUROC + threshold별 FN/FP table |
| add training | `python scripts/add_training_from_folders.py --model-run logs/<run> --image-root extra_images` | fine-tuned `logs/addtrain_*/best_model.pth` |
| batch inference | `python scripts/server_batch_predict.py --model-run logs/<run>` | server inference outputs |
| Grad-CAM | `python scripts/gradcam_report.py --model-run logs/<run> --image-root images/test --out-dir validations/gradcam_probe --save-heat-only` | trend+CAM overlays, transparent heat masks, heat CSV |
| FP/FN Grad-CAM | `python scripts/gradcam_error_report.py --model-run logs/<run> --error-type fp` | CAM overlays for false positives or false negatives |
| group report | `python scripts/generate_group_report.py --group-dir logs/run_YYYYMMDD_HHMMSS` | per-group candidate table + F1/val curves |
| log report | `python scripts/generate_log_history_report.py --logs-dir logs --out-prefix validations/log_history_report --contains rawbase` | markdown, CSV, PNG plots |

## Main Files

- `dataset.yaml`: active dataset/rendering config
- `docs/problem_setting.md`: binary/multiclass/anomaly problem framing and evaluation gates
- `docs/two_stage_workflow.md`: detailed two-stage training, inference, output, and FP/FN diagnosis workflow
- `generate_data.py`: tabular data generator
- `generate_images.py`: training/display image renderer
- `train.py`: training entrypoint
- `inference.py`: single-model inference
- `scripts/two_stage_predict.py`: stage-1 binary gate followed by stage-2 defect type classification
- `scripts/binary_threshold_report.py`: AUROC and threshold sweep report from labeled inference predictions
- `scripts/server_batch_predict.py`: batch inference
- `scripts/generate_inference_images.py`: inference image renderer from existing trend CSVs
- `scripts/add_training_from_folders.py`: fine-tune a best model from `normal/` and `abnormal/` image folders
- `scripts/gradcam_report.py`: Grad-CAM overlays on trend images and transparent heat masks
- `scripts/gradcam_error_report.py`: Grad-CAM overlays for FP/FN samples
- `scripts/generate_log_history_report.py`: tables and plots from flat or grouped `logs/**/`
- `scripts/sweeps_server/00_all.sh`: current paper experiment pipeline, ending with color -> sample_skip -> backbone -> logical_train -> BKM combined
- `scripts/all-dataset-backbone.sh`: one-shot wrapper that runs the full sweep for every dataset yaml (weights/data/baseline prep, all axes including ASL, all backbones, bkm_combined, postprocess) and refreshes the cross-dataset comparison report after each dataset and at the end
- `scripts/all-dataset-backbone-ddp.sh`: same wrapper, but queued train.py runs launch through torchrun DDP; `--batch_size` stays global and each GPU gets a rank-local micro-batch
- `scripts/generate_cross_dataset_report.py`: per-dataset and overall baseline/BKM/backbone comparison tables and bar plots
- `docs/summary.md`: current experiment summary

Generated folders such as `data/`, `images/`, `display/`, `logs/`, `weights/`, and `validations/<group>/` are gitignored. Root queue templates under `validations/*.json` that server sweeps require are tracked.

Server sweep wrappers keep checkpoint storage bounded by default: after each controlled run, `best_model.pth` is retained only for the global best run and the best run for each dataset-config + backbone pair. Metrics, summaries, plots, and `best_info.json` remain for every run.
