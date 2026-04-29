# scripts

## Pipeline Entry Points

- `sweeps_server/00_all.sh`: full paper pipeline. Auto-detects runtime profile (server/pc/minimal) from GPU memory + CPU count, then runs core axes -> color -> sample_skip -> backbone -> logical_train -> gc -> BKM combined -> postprocess.
- `run_paper_server_all.sh`: lower-level runner used by every wrapper above. Handles weights download, dataset/image generation, baseline recheck, axis sweep, and post-analysis.
- `sweeps_server/axis.sh`: single-axis wrapper. See `sweeps_server/README.md`.
- `sweeps_server/backbone.sh`: auto-detects every `weights/*.pth` (excluding `best_model.pth` and `*.fp16.pth`) and rotates through whatever it finds. No hardcoded list.

## Queue Helpers

- `prepare_server_queue.py`: rewrite a template queue for the active server with rawbase tags, axis-only edits, and skip-list filtering.
- `adaptive_experiment_controller.py`: sequential train runner. Reads an active queue, launches `train.py` per run, updates the live summary JSON/MD after each run.

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

- `gradcam_report.py`: Grad-CAM overlays on trend images plus transparent heat masks for saved models
- `gradcam_error_report.py`: Grad-CAM overlays for FP/FN samples
- `generate_strict_one_factor_report.py`: build the strict one-factor markdown + axis plots
- `generate_stage_comparison.py`: comparison .md + bar plot for backbone / BKM-combined / sample-skip stages
- `generate_group_report.py`: build a candidate table + plots from a single `logs/<group>/` folder (CLI)
- `collect_instability_cases.py`: scan logs for collapse / oscillation / spike instabilities
- `analyze_prediction_trends.py`: FN/FP trend tables across high-F1 candidates
