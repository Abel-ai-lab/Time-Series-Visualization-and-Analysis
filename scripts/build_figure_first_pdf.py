#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


FIGURE_SECTIONS = [
    {
        "file": "real_curves_by_ticker.png",
        "title": "1. Real curves: close, volume, and log return",
        "why": "This is the first sanity check: the report shows actual historical trajectories rather than synthetic examples.",
        "points": [
            "Each row is one representative ticker; columns are daily close, daily volume, and daily close log return.",
            "The close panels reveal regime shifts, split-like artifacts, outliers, and long-run trends that affect model scaling.",
            "The volume panels use a log scale because liquidity is extremely uneven across tickers and time.",
            "The log-return panels make volatility clustering and jump periods visible in a stationary-like representation.",
        ],
    },
    {
        "file": "real_distribution_panels.png",
        "title": "2. Empirical distributions: raw and transformed data",
        "why": "Raw prices and volumes are heavy-tailed; log-return views are needed for model diagnostics and robust comparison.",
        "points": [
            "Raw close and raw volume distributions are clipped at extreme quantiles for readability while preserving log-count scaling.",
            "The log1p(volume) panel compresses liquidity differences while preserving zero-volume structure.",
            "Close log returns are centered near zero but have heavy tails, which is typical for equity data.",
            "Absolute log returns expose the tail mass that drives forecasting error and risk-sensitive metrics.",
        ],
    },
    {
        "file": "log_return_diagnostics.png",
        "title": "3. Log-return diagnostics: normality and tails",
        "why": "This page checks whether the transformed series behaves like a simple Gaussian process. It does not.",
        "points": [
            "The QQ plot compares empirical close log returns with Normal quantiles; tail deviations indicate non-Gaussian behavior.",
            "The clipped histogram shows the central mass while preserving log-scale count differences.",
            "The absolute-return CCDF highlights tail decay and rare but material jumps.",
            "Tail event frequencies quantify how often moves exceed fixed log-return thresholds.",
        ],
    },
    {
        "file": "cross_section_time_structure.png",
        "title": "4. Cross-sectional structure over time",
        "why": "Foundation models see a panel, not one isolated ticker; cross-sectional scale changes matter for normalization and sampling.",
        "points": [
            "The close p10/p50/p90 band tracks how the price universe changes over calendar time.",
            "Aggregate volume shows market-wide liquidity regimes and calendar effects.",
            "Volume median and p90 show that liquidity is concentrated in a subset of tickers.",
            "These plots are useful for detecting broad data shifts before model evaluation.",
        ],
    },
    {
        "file": "per_ticker_heterogeneity.png",
        "title": "5. Per-ticker heterogeneity",
        "why": "A 1000-ticker dataset mixes liquid mega-caps, quiet names, and highly volatile small caps; model errors will not be uniform.",
        "points": [
            "Volatility versus total volume separates liquid stable names from thin or unstable names.",
            "Long-run return versus volatility identifies tickers with high realized trend and high uncertainty.",
            "Top-volatility and top-volume rankings provide concrete candidates for stress tests and qualitative inspection.",
            "This page helps choose representative subsets for demos and deeper error analysis.",
        ],
    },
    {
        "file": "logret_volatility_heatmaps.png",
        "title": "6. Heatmaps: return and rolling volatility regimes",
        "why": "Heatmaps make simultaneous time structure visible across representative and high-volatility tickers.",
        "points": [
            "The daily log-return heatmap shows co-moving shock periods and ticker-specific jumps.",
            "The rolling annualized volatility heatmap shows persistent risk regimes rather than isolated spikes.",
            "Representative tickers are mixed with high-volatility names to expose both normal and stressed behavior.",
            "This view is useful for selecting forecast windows for qualitative foundation-model comparisons.",
        ],
    },
]


def wrapped_lines(text: str, width: int = 96) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    size: float = 10.5,
    weight: str = "normal",
    width: int = 98,
) -> float:
    rendered = wrapped_lines(text, width=width)
    ax.text(
        x,
        y,
        rendered,
        ha="left",
        va="top",
        fontsize=size,
        fontweight=weight,
        transform=ax.transAxes,
    )
    return y - 0.038 * (rendered.count("\n") + 1)


def cover_page(pdf: PdfPages, summary: dict) -> None:
    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle("Finance1000 Visual Data Analysis", fontsize=24, fontweight="bold", y=0.93)
    y = 0.82
    y = add_text(ax, 0.06, y, "Figure-first report for a 1000-ticker hourly US equity panel.", 14, "bold")
    y -= 0.03
    lines = [
        f"Universe: {summary['num_tickers']} tickers",
        f"Timestamps: {summary['num_timestamps']:,} hourly rows",
        f"Span: {summary['time_start']} to {summary['time_end']}",
        f"Representative tickers: {', '.join(summary['representative_tickers'])}",
    ]
    for line in lines:
        y = add_text(ax, 0.08, y, line, 11.5)
    y -= 0.04
    y = add_text(ax, 0.06, y, "How to read this report", 13, "bold")
    for point in [
        "Each section is organized around one generated figure and a short interpretation.",
        "The goal is visual data validation: understand raw scale, transformed log-return behavior, tails, cross-sectional structure, and ticker heterogeneity before modeling.",
        "The separate ticker book contains one page per ticker for detailed inspection of all 1000 series.",
    ]:
        y = add_text(ax, 0.08, y, "- " + point, 10.8)
    pdf.savefig(fig)
    plt.close(fig)


def stats_page(pdf: PdfPages, summary: dict) -> None:
    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle("Distribution Statistics Snapshot", fontsize=18, fontweight="bold", y=0.94)
    cols = ["group", "count", "mean", "std", "p1", "p50", "p99"]
    groups = ["raw_close", "raw_volume", "close_log_return", "volume_log_change"]
    labels = {
        "raw_close": "Raw close",
        "raw_volume": "Raw volume",
        "close_log_return": "Close log return",
        "volume_log_change": "Volume log change",
    }
    rows = []
    for group in groups:
        stats = summary["distribution_stats"][group]
        rows.append(
            [
                labels[group],
                f"{int(stats['count']):,}",
                f"{stats['mean']:.6g}",
                f"{stats['std']:.6g}",
                f"{stats['p1']:.6g}",
                f"{stats['p50']:.6g}",
                f"{stats['p99']:.6g}",
            ]
        )
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.7)
    note = (
        "Interpretation: raw close and volume are highly skewed, while log returns are centered "
        "near zero but remain heavy-tailed. These statistics motivate reporting both raw-data "
        "views and log-return diagnostics."
    )
    ax.text(0.06, 0.18, wrapped_lines(note, 130), ha="left", va="top", fontsize=11, transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def figure_page(pdf: PdfPages, fig_path: Path, section: dict) -> None:
    fig = plt.figure(figsize=(13, 8.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.05)
    ax_img = fig.add_subplot(grid[0, 0])
    ax_txt = fig.add_subplot(grid[0, 1])
    ax_img.axis("off")
    ax_txt.axis("off")
    img = plt.imread(fig_path)
    ax_img.imshow(img)
    ax_img.set_title(section["title"], fontsize=14, fontweight="bold", pad=12)
    y = 0.95
    y = add_text(ax_txt, 0.02, y, "Why this figure matters", 12.5, "bold", width=46)
    y = add_text(ax_txt, 0.02, y, section["why"], 10.8, width=48)
    y -= 0.035
    y = add_text(ax_txt, 0.02, y, "Reading notes", 12.5, "bold", width=46)
    for point in section["points"]:
        y = add_text(ax_txt, 0.04, y, "- " + point, 10.2, width=50)
        y -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a figure-first interpreted PDF from generated report artifacts.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/finance1000"))
    parser.add_argument("--output", type=Path, default=Path("reports/finance1000/finance1000_figure_first_report.pdf"))
    args = parser.parse_args()
    summary = json.loads((args.report_dir / "summary.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        cover_page(pdf, summary)
        stats_page(pdf, summary)
        for section in FIGURE_SECTIONS:
            figure_page(pdf, args.report_dir / "figures" / section["file"], section)
    print(args.output)


if __name__ == "__main__":
    main()
