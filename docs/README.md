# docs

- `summary.md`: current experiment summary
- `plots/`: summary plots
- `sample_overview_train.png`: training image examples
- `sample_overview_display.png`: display image examples
- `logical_member_targets_ch09100.png`: logical member class example

Server log report command:

```bash
python scripts/generate_log_history_report.py \
  --logs-dir logs \
  --out-prefix validations/log_history_report_rawbase \
  --contains rawbase \
  --top-k 30
```
