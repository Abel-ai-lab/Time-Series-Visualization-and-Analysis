# Agent Guide

Use this repository when an agent needs reproducible time-series visualization and data analysis.

## Quick Invocation

For a CPU-only report run:

```bash
CUDA_VISIBLE_DEVICES= python scripts/run_visual_report.py \
  --raw-csv /path/to/raw.csv \
  --logret-csv /path/to/logret.csv \
  --output-dir outputs/my_report \
  --ticker-book selected
```

For the finance1000 cloud reproduction on Pro6000:

```bash
CUDA_VISIBLE_DEVICES= bash scripts/run_finance1000_cloud.sh
```

Always set `CUDA_VISIBLE_DEVICES=` for report generation unless the user explicitly requests GPU use. The visualizer is CPU-oriented and should not disturb training jobs.

## Expected Inputs

Read `references/data_schema.md` before adapting the workflow to a new dataset. The default schema is a wide CSV with `date`, `*_close`, and `*_volume` columns. A matching log-return CSV can be supplied; otherwise the tool derives log returns from raw close values.

## Expected Outputs

The output directory includes a PDF report, PNG figures, CSV tables, `summary.json`, and `manifest.json`. Use `--ticker-book all` when the user needs a page for every series; use `selected` for a compact book.
