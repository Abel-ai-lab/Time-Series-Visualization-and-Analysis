# Finance1000 Report Notes

This repository includes generated report artifacts for the finance1000 hourly ticker panel.

## Dataset

- Universe: 1000 US equity tickers.
- Timestamps: 28,510 hourly rows.
- Correct plotting frequency: native hourly rows for curve and ticker-book views.
- Time span: 2015-01-02 to 2026-02-25 UTC.
- Raw table: `date` plus `1000 *_close` and `1000 *_volume` columns.
- Log-return table: same shape, with close log returns and volume log changes.

Raw data is intentionally not committed. On Pro6000 it lives under:

```text
/data/jm/tsorchestra_repro_20260610/finance1000_source/
```

## Included Artifacts

- `reports/finance1000/finance1000_figure_first_report.pdf`: primary figure-first report with interpretation notes.
- `reports/finance1000/modern_tsf_visual_data_analysis.pdf`: compact auto-generated report.
- `reports/finance1000/ticker_book_native_selected_12.pdf`: committed hourly selected-ticker book.
- Full hourly 1000-ticker book: `/data/jm/tsorchestra_repro_20260610/analysis_outputs/finance1000_modern_tsf_visualizer_hourly_20260614/ticker_book_native_all_1000.pdf` on Pro6000; not committed because it is about 239 MB.
- `reports/finance1000/ticker_book_all_1000.pdf`: legacy daily-resampled book, kept for comparison only.
- `reports/finance1000/figures/*.png`: key reusable figures.
- `reports/finance1000/tables/*.csv`: distribution, cross-section, and per-ticker stats.
- `reports/finance1000/summary.json`: machine-readable summary.
- `reports/finance1000/manifest.json`: generated output manifest.

## Key Statistics

| Group | Count | Mean | Std | p1 | p50 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Raw close | 7,000,000 | 25,674,206.2471 | 2,160,315,233.3669 | 0.0000 | 33.0600 | 15,300.0000 |
| Raw volume | 7,000,000 | 363,623.3694 | 1,273,870.3358 | 0.0000 | 89,412.0000 | 4,325,528.0000 |
| Close log return | 7,000,000 | 0.0000147 | 0.0169491 | -0.0373178 | 0.0000 | 0.0381934 |
| Volume log change | 7,000,000 | 0.0035401 | 1.3289781 | -2.1679102 | 0.0000 | 2.2776405 |
