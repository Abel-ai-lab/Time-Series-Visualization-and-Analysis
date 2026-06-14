# Time-Series Visualization and Analysis

Reproducible ModernTSF-style visualization and data analysis for wide time-series panels. The repository includes a reusable Python CLI plus generated finance1000 report artifacts.

## What This Produces

The visualizer generates real data diagnostics, not only text summaries:

- raw close/value and volume distributions;
- log-return and volume-change distributions;
- true historical close, volume, and native-frequency log-return curves;
- log-return QQ plot, tail CCDF, and tail event frequencies;
- cross-sectional close and volume structure over time;
- per-ticker volatility, volume, and long-run return heterogeneity;
- daily log-return and rolling-volatility heatmaps;
- figure-first interpreted PDF report and optional all-ticker PDF book.

## Repository Layout

```text
modern_tsf_visualizer/          Python package and CLI
scripts/                        Stable entrypoints for humans and agents
configs/                        Example JSON configs
references/data_schema.md       Input schema reference
reports/finance1000/            Generated finance1000 figures, PDFs, tables, summary
AGENTS.md                       Agent-facing invocation guide
docs/                           Report notes and portable skill instructions
```

## Install

```bash
pip install -e .
```

The code uses `matplotlib`, `numpy`, and `pandas`. It is CPU-oriented; use `CUDA_VISIBLE_DEVICES=` on shared GPU servers.

## Run On A Generic Wide CSV

```bash
CUDA_VISIBLE_DEVICES= python scripts/run_visual_report.py \
  --raw-csv /path/to/raw.csv \
  --logret-csv /path/to/logret.csv \
  --output-dir outputs/my_report \
  --date-col date \
  --close-suffix _close \
  --volume-suffix _volume \
  --curve-frequency native \
  --ticker-book selected
```

If no `--logret-csv` is supplied, the tool derives close log returns from `diff(log(close))` and volume changes from `diff(log1p(volume))`.

## Reproduce Finance1000 On Pro6000

Raw data is not committed. On Pro6000 it is expected at:

```text
/data/jm/tsorchestra_repro_20260610/finance1000_source/
```

Run:

```bash
CUDA_VISIBLE_DEVICES= bash scripts/run_finance1000_cloud.sh
```

This uses `configs/finance1000.cloud.json` and writes to the configured cloud output directory.

## Included Finance1000 Artifacts

- [Figure-first interpreted PDF report](reports/finance1000/finance1000_figure_first_report.pdf)
- [Compact auto-generated PDF report](reports/finance1000/modern_tsf_visual_data_analysis.pdf)
- [Hourly selected-ticker PDF book](reports/finance1000/ticker_book_native_selected_12.pdf)
- Full hourly all-ticker PDF book: generated on Pro6000 at `/data/jm/tsorchestra_repro_20260610/analysis_outputs/finance1000_modern_tsf_visualizer_hourly_20260614/ticker_book_native_all_1000.pdf` (about 239 MB, not committed to GitHub).
- [Legacy daily all-ticker PDF book](reports/finance1000/ticker_book_all_1000.pdf)
- [Real curves by ticker](reports/finance1000/figures/real_curves_by_ticker.png)
- [Distribution panels](reports/finance1000/figures/real_distribution_panels.png)
- [Log-return diagnostics](reports/finance1000/figures/log_return_diagnostics.png)
- [Cross-section time structure](reports/finance1000/figures/cross_section_time_structure.png)
- [Per-ticker heterogeneity](reports/finance1000/figures/per_ticker_heterogeneity.png)
- [Log-return volatility heatmaps](reports/finance1000/figures/logret_volatility_heatmaps.png)
- [Per-ticker stats](reports/finance1000/tables/per_ticker_stats.csv)
- [Cross-section daily stats](reports/finance1000/tables/cross_section_daily.csv)
- [Distribution stats](reports/finance1000/tables/distribution_stats.csv)
- [Summary JSON](reports/finance1000/summary.json)
- [Manifest JSON](reports/finance1000/manifest.json)

## Finance1000 Snapshot

- Universe: 1000 US equity tickers.
- Timestamps: 28,510 hourly rows.
- Curve/ticker-book frequency: native hourly rows.
- Span: 2015-01-02 to 2026-02-25 UTC.
- Representative tickers in the compact report: AAPL, MSFT, AMZN, JPM, KMB, NVAX, GME, META, FCEL, MARA, CL, VZ.

| Group | Count | Mean | Std | p1 | p50 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Raw close | 7,000,000 | 25,674,206.2471 | 2,160,315,233.3669 | 0.0000 | 33.0600 | 15,300.0000 |
| Raw volume | 7,000,000 | 363,623.3694 | 1,273,870.3358 | 0.0000 | 89,412.0000 | 4,325,528.0000 |
| Close log return | 7,000,000 | 0.0000147 | 0.0169491 | -0.0373178 | 0.0000 | 0.0381934 |
| Volume log change | 7,000,000 | 0.0035401 | 1.3289781 | -2.1679102 | 0.0000 | 2.2776405 |

## Agent Compatibility

Agents should use `AGENTS.md` and `scripts/run_visual_report.py`. The wrapper keeps GPU use disabled by default and passes arguments through to `modern_tsf_visualizer.cli`.

Example:

```bash
CUDA_VISIBLE_DEVICES= python scripts/run_visual_report.py -- \
  --config configs/generic_wide_csv.json \
  --output-dir outputs/custom_report
```
