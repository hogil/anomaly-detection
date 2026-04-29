# scripts

## Server Experiments

- `sweeps_server/00_all.sh`: current needed-only server resume
- `run_paper_server_all.sh`: lower-level server pipeline runner
- `prepare_server_queue.py`: server queue preparation
- `adaptive_experiment_controller.py`: sequential train runner
- `watch_refcheck_then_round1.py`: Windows/local watcher
- `sweeps_server/06_sample_skip.sh`: separate sample-skip pilot

## Reports From Logs

```bash
python scripts/generate_log_history_report.py \
  --logs-dir logs \
  --out-prefix validations/log_history_report_rawbase \
  --contains rawbase \
  --top-k 30
```

Outputs markdown, CSV tables, and PNG plots from `logs/*/history.json` and `best_info.json`.

## Inference

- `generate_inference_images.py`: render flat inference images from existing `timeseries.csv` and `scenarios.csv`
- `server_batch_predict.py`: batch inference using a saved `best_model.pth`
- `run_server_batch_predict.sh`: shell wrapper
- `add_training_from_folders.py`: load a saved best model and fine-tune from `normal/` and `abnormal/` image folders

## Dataset / Validation

- `generate_per_member_images.py`
- `validate_dataset.py`

## Analysis Helpers

- `gradcam_report.py`: Grad-CAM overlays and transparent heat-only summaries for saved models
- `right_crop_postprocess_report.py`: test right-crop postprocess rules against labeled FP/FN
- `gradcam_normal_rescue_report.py`: test normal-prediction Grad-CAM rescue rules against labeled FP/FN
- `generate_strict_one_factor_report.py`
- `update_live_summary_doc.py`
- `collect_instability_cases.py`
- `analyze_prediction_trends.py`
- `select_strict_single_factor_refinements.py`
