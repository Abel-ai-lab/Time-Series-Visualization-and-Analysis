# Portable Skill Instructions

A Codex/agent skill can point to this repository and run `scripts/run_visual_report.py` as its stable entrypoint.

Recommended skill description:

> Generate reproducible ModernTSF-style visual data analysis reports for wide time-series CSV datasets, especially panels with date, `*_close`, and `*_volume` columns. Use when the agent needs raw data inspection, log-return diagnostics, empirical distributions, real curves, cross-sectional structure, per-series heterogeneity, heatmaps, or PDF/PNG/CSV report artifacts.

Workflow:

1. Identify raw CSV, optional log-return CSV, output directory, date column, and suffixes.
2. Read `references/data_schema.md` if the schema is unfamiliar.
3. Keep large raw datasets in place; copy back only report artifacts unless asked otherwise.
4. Run with `CUDA_VISIBLE_DEVICES=` by default.
5. Return links/paths to the generated PDF, figures, tables, `summary.json`, and `manifest.json`.
