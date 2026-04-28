# Server Sweeps

This folder is the active server entrypoint for the paper experiment pipeline.

Current server default:

```bash
bash scripts/sweeps_server/00_all.sh
```

This runs the still-needed rawbase round1 axes used by `docs/summary.md`. It skips weights, dataset generation, refcheck, round2, and post-processing.

## Commands

```bash
bash scripts/sweeps_server/00_all.sh
bash scripts/sweeps_server/01_refcheck.sh
bash scripts/sweeps_server/10_lr.sh
bash scripts/sweeps_server/11_warmup.sh
bash scripts/sweeps_server/20_normal_ratio.sh
bash scripts/sweeps_server/21_per_class.sh
bash scripts/sweeps_server/30_label_smoothing.sh
bash scripts/sweeps_server/31_stochastic_depth.sh
bash scripts/sweeps_server/32_focal_gamma.sh
bash scripts/sweeps_server/33_abnormal_weight.sh
bash scripts/sweeps_server/34_ema.sh
bash scripts/sweeps_server/40_color.sh
bash scripts/sweeps_server/41_allow_tie_save.sh
bash scripts/sweeps_server/90_gc.sh
bash scripts/sweeps_server/06_sample_skip.sh
```

All options are passed through to `scripts/run_paper_server_all.sh`.

Useful resume/debug examples:

```bash
bash scripts/sweeps_server/00_all.sh --max-launched 1
bash scripts/sweeps_server/06_sample_skip.sh
```

Defaults are inherited from `run_paper_server_all.sh`: data/image generation workers `24`, training DataLoader workers `24`, and prefetch factor `4`.

The active server baseline is raw: `grad_clip=0.0`, `smooth_window=1`, `smooth_method=median`. Prepared server queues rewrite run tags with `fresh0412_v11_rawbase_...` so old GC/smoothed logs are not reused. GC is limited to 5 conditions by `scripts/prepare_server_queue.py`.

Round1 queue preparation removes tags already marked `complete` or `skipped` in `validations/server_paper_rawbase_strict_single_factor_summary.json` by default. The current resume order is `lr`, `warmup`, `normal_ratio`, `per_class`, `label_smoothing`, `stochastic_depth`, `focal_gamma`, `abnormal_weight`, `ema`, `color`, `allow_tie_save`, then `gc` last. Use `--round1-keep-completed` only when intentionally rebuilding the full prepared queue.

Main regularization/loss axes use 5 candidate conditions each: `label_smoothing`, `stochastic_depth`, `focal_gamma`, `abnormal_weight`, and `ema`.

For local/Windows chaining without bash, use:

```powershell
python scripts\watch_refcheck_then_round1.py
```

It waits for `validations/server_paper_refcheck_raw_summary.json` to finish 5 raw refcheck runs, prepares `validations/server_paper_rawbase_strict_single_factor_queue.json`, and launches the same needed-only rawbase remainder. In controller or `tee` logs, `train.py` disables tqdm bars automatically so progress updates do not become one line per refresh.

To build summary-style tables and plots directly from run histories:

```bash
python scripts/generate_log_history_report.py --logs-dir logs --out-prefix validations/log_history_report --contains fresh0412_v11
```
