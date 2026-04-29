# docs

- `summary.md`: current experiment summary
- `plots/`: summary plots
- `images/`: report example images and Grad-CAM galleries
- `data/one_factor_latest.json`: latest paper-facing one-factor aggregate table used to regenerate `summary.md` and plots
- `images/sample_overview_train.png`: training image examples
- `images/sample_overview_display.png`: display image examples
- `images/sample_overview_legend_axis.png`: display image examples grouped by legend_axis
- `images/logical_member_targets_ch09100.png`: logical member class example
- `images/color_samples/`: representative color-rendering samples
- `images/logical_members_ch09100/`: logical member examples split by target member

Server log report command:

```bash
python scripts/generate_log_history_report.py \
  --logs-dir logs \
  --out-prefix validations/log_history_report_rawbase \
  --contains rawbase \
  --top-k 30
```
