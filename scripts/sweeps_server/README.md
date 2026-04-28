# Server Sweeps

This folder is the active server entrypoint for the paper experiment pipeline.
Old pre-paper grid sweeps were moved to `legacy/`.

Current server default:

```bash
bash scripts/sweeps_server/00_all.sh
```

This runs only the still-needed rawbase round1 remainder. It skips weights, dataset generation, refcheck, GC, warmup, round2, and post-processing.

## Commands

```bash
bash scripts/sweeps_server/00_all.sh
bash scripts/sweeps_server/01_refcheck.sh
bash scripts/sweeps_server/02_round1.sh
bash scripts/sweeps_server/03_round1_after_gc.sh
bash scripts/sweeps_server/04_round2.sh
bash scripts/sweeps_server/05_post.sh
bash scripts/sweeps_server/06_sample_skip.sh
```

All options are passed through to `scripts/run_paper_server_all.sh`.

Useful resume/debug examples:

```bash
bash scripts/sweeps_server/02_round1.sh --max-launched 1
bash scripts/sweeps_server/06_sample_skip.sh
bash scripts/sweeps_server/04_round2.sh
bash scripts/sweeps_server/05_post.sh --candidate-prefix fresh0412_v11
```

Defaults are inherited from `run_paper_server_all.sh`: data/image generation workers `24`, training DataLoader workers `24`, and prefetch factor `4`.

The active server baseline is raw: `grad_clip=0.0`, `smooth_window=1`, `smooth_method=median`. Prepared server queues rewrite run tags with `fresh0412_v11_rawbase_...` so old GC/smoothed logs are not reused. GC has already been covered by the dense historical sweep and rawbase partial checks, so the current server default does not schedule GC.

Round1 queue preparation removes tags already marked `complete` or `skipped` in `validations/server_paper_rawbase_strict_single_factor_summary.json` by default. The current needed-only resume keeps only `stochastic_depth`, `focal_gamma`, `abnormal_weight`, `ema`, `color`, and `allow_tie_save`. Use `--round1-keep-completed` only when intentionally rebuilding the full prepared queue.

For local/Windows chaining without bash, use:

```powershell
python scripts\watch_refcheck_then_round1.py
```

It waits for `validations/server_paper_refcheck_raw_summary.json` to finish 5 raw refcheck runs, prepares `validations/server_paper_rawbase_strict_single_factor_queue.json`, and launches the same needed-only rawbase remainder. In controller or `tee` logs, `train.py` disables tqdm bars automatically so progress updates do not become one line per refresh.

To build summary-style tables and plots directly from run histories:

```bash
python scripts/generate_log_history_report.py --logs-dir logs --out-prefix validations/log_history_report --contains fresh0412_v11
```
