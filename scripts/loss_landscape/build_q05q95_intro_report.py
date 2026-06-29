from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path('/data/jm/tsorchestra_repro_20260610/analysis_outputs/zero_epsilon_q05q95_report_20260629_100tickers_b11_exact')
SHARDS = [BASE / f'shard_{i}' for i in range(4)]
OUT = BASE / 'merged_intro'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TARGETS = ['close_log_return', 'volume_log_return']
METRIC_KEYS = ['mae','mse','rmse','direction_accuracy','pearson_ic','spearman_rho','cosine_similarity','std_gap','l1_divergence','dwt_distance','kl_divergence','wasserstein']
SELECTED = Path('/data/jm/tsorchestra_repro_20260610/finance1000_source/selected_tickers.csv')

plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 180,
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.25,
})


def fmt(v: float, digits: int = 4) -> str:
    if not np.isfinite(v):
        return 'nan'
    if abs(v) < 1e-3 and v != 0:
        return f'{v:.2e}'
    return f'{v:.{digits}f}'


def avg_pair(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=METRIC_KEYS, columns=METRIC_KEYS, dtype=float)
    for a in METRIC_KEYS:
        for b in METRIC_KEYS:
            out.loc[a, b] = float(df[f'{a}__{b}'].mean())
    return out


def save_bar(df: pd.DataFrame, target: str, out_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.barh(df['metric_label'], df['mean_zero_percentile'], color='#4C78A8')
    ax.set_xlim(0, 1.02)
    ax.set_title(f'{target}: zero-baseline mean percentile by metric')
    ax.set_xlabel('percentile in candidate prediction space')
    fig.tight_layout()
    path = FIG / out_name
    fig.savefig(path)
    plt.close(fig)
    return path


def save_ladder(pivot: pd.DataFrame, target: str, out_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    focus_cols = [c for c in ['MAE','MSE','Direction Accuracy','Pearson / IC','Spearman','Cosine Similarity','Wasserstein','KL Divergence'] if c in pivot.columns]
    for col in focus_cols:
        ax.plot(pivot.index, pivot[col], marker='o', linewidth=1.8, label=col)
    ax.set_ylim(0, 1.02)
    ax.set_title(f'{target}: epsilon ladder percentile curves')
    ax.set_xlabel('epsilon index')
    ax.set_ylabel('mean percentile')
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = FIG / out_name
    fig.savefig(path)
    plt.close(fig)
    return path


def save_heatmap(df: pd.DataFrame, title: str, out_name: str, cmap: str, vmin: float, vmax: float) -> Path:
    fig, ax = plt.subplots(figsize=(9.8, 8.4))
    im = ax.imshow(df.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f'{df.iat[i,j]:.2f}', ha='center', va='center', fontsize=6, color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIG / out_name
    fig.savefig(path)
    plt.close(fig)
    return path


def save_case_buckets(case_df: pd.DataFrame, out_name: str) -> Path:
    fig, axes = plt.subplots(len(case_df), 1, figsize=(11.5, 3.0 * len(case_df)), constrained_layout=True)
    if len(case_df) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, case_df.iterrows()):
        buckets = row['bucket_values']
        xs = np.arange(len(buckets))
        colors = ['#4C78A8' if abs(v) <= row['bucket_step'] + 1e-12 else '#B8C4D6' for v in buckets]
        ax.bar(xs, buckets, color=colors)
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set_title(f"{row['symbol']} | std={row['daily_std']:.4f} | q05={row['q05']:.4f} q95={row['q95']:.4f}")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs])
        ax.set_ylabel('bucket center')
    path = FIG / out_name
    fig.savefig(path)
    plt.close(fig)
    return path


def epsilon_table_html(ladder_df: pd.DataFrame) -> str:
    pivot = ladder_df.pivot_table(index='epsilon_index', columns='metric_label', values='baseline_percentile', aggfunc='mean')
    header = ''.join(f'<th>{c}</th>' for c in pivot.columns)
    rows = []
    for idx, row in pivot.iterrows():
        body = ''.join(f'<td>{fmt(float(v))}</td>' for v in row.values)
        rows.append(f'<tr><td>{idx}</td>{body}</tr>')
    return '<table><thead><tr><th>epsilon_index</th>' + header + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'


def takeaway_html(target: str, zero_summary: pd.DataFrame) -> str:
    metric_map = {r['metric_label']: float(r['mean_zero_percentile']) for _, r in zero_summary.iterrows()}
    def g(name: str) -> str:
        return fmt(metric_map.get(name, float('nan')))
    if target == 'close_log_return':
        return (
            "<div class='takeaway'><h3>Read This First</h3>"
            f"<p>After trimming each ticker to its own q05/q95 range, the zero bucket still dominates amplitude losses for <b>close log return</b>: <b>MAE={g('MAE')}</b>, <b>MSE={g('MSE')}</b>, <b>RMSE={g('RMSE')}</b>, and <b>Wasserstein={g('Wasserstein')}</b> remain extremely strong. But <b>Direction Accuracy={g('Direction Accuracy')}</b> stays weak and <b>Pearson / IC={g('Pearson / IC')}</b> stays neutral. <b>Spearman={g('Spearman')}</b> and <b>Cosine={g('Cosine Similarity')}</b> are higher than Pearson/IC, which suggests some rank or orientation structure survives without turning into useful signed forecasts.</p>"
            "<p>Takeaway: clipping away the tails does not remove the strong-zero-baseline story for close. It mainly makes the mismatch between amplitude losses and directional or correlational objectives easier to see.</p></div>"
        )
    return (
        "<div class='takeaway'><h3>Read This First</h3>"
        f"<p>For <b>volume log return</b>, the zero bucket is still strong on amplitude losses, but the dominance is softer than for close. <b>MAE={g('MAE')}</b>, <b>MSE={g('MSE')}</b>, and <b>RMSE={g('RMSE')}</b> remain high, while <b>Wasserstein={g('Wasserstein')}</b>, <b>Std Gap={g('Std Gap')}</b>, and <b>KL={g('KL Divergence')}</b> are less extreme. <b>Direction Accuracy={g('Direction Accuracy')}</b> remains weak and <b>Pearson / IC={g('Pearson / IC')}</b> remains neutral.</p>"
        "<p>Takeaway: trimming to q05/q95 weakens the extreme-zero picture for volume more than for close, but it still does not make directional or correlation metrics align with amplitude-style objectives.</p></div>"
    )


report = {'targets': {}}
fig_paths = {}
all_ticker_meta = []
for target in TARGETS:
    ladders = []
    corrs = []
    overlaps = []
    tickers = []
    total_windows = 0
    bucket_count = None
    num_predictions = None
    for shard in SHARDS:
        ladders.append(pd.read_csv(shard / f'{target}_epsilon_ladder.csv'))
        corrs.append(pd.read_csv(shard / f'{target}_metric_corr.csv'))
        overlaps.append(pd.read_csv(shard / f'{target}_metric_overlap.csv'))
        shard_report = json.loads((shard / 'report.json').read_text())
        block = shard_report['targets'][target]
        tickers.extend(block['tickers'])
        total_windows += int(block['summary']['num_windows'])
        bucket_count = int(block['summary']['bucket_count'])
        num_predictions = max(int(t['num_predictions']) for t in block['tickers'])

    ladder_df = pd.concat(ladders, ignore_index=True)
    corr_df = pd.concat(corrs, ignore_index=True)
    overlap_df = pd.concat(overlaps, ignore_index=True)
    ladder_df.to_csv(OUT / f'{target}_epsilon_ladder.csv', index=False)
    corr_df.to_csv(OUT / f'{target}_metric_corr.csv', index=False)
    overlap_df.to_csv(OUT / f'{target}_metric_overlap.csv', index=False)

    corr_mean = avg_pair(corr_df.drop(columns=['target','symbol']))
    overlap_mean = avg_pair(overlap_df.drop(columns=['target','symbol']))
    corr_mean.to_csv(OUT / f'{target}_metric_corr_mean.csv')
    overlap_mean.to_csv(OUT / f'{target}_metric_overlap_mean.csv')

    zero_summary = (
        ladder_df[ladder_df['epsilon_index'] == 0]
        .groupby('metric_label')['baseline_percentile']
        .mean().reset_index()
        .rename(columns={'baseline_percentile': 'mean_zero_percentile'})
        .sort_values('mean_zero_percentile', ascending=False)
    )
    zero_summary.to_csv(OUT / f'{target}_zero_summary.csv', index=False)

    pivot = ladder_df.pivot_table(index='epsilon_index', columns='metric_label', values='baseline_percentile', aggfunc='mean')
    pivot.to_csv(OUT / f'{target}_epsilon_ladder_pivot.csv')

    meta_df = pd.DataFrame(tickers)
    meta_df.insert(0, 'target', target)
    meta_df.to_csv(OUT / f'{target}_ticker_meta.csv', index=False)
    all_ticker_meta.append(meta_df)

    fig_paths[f'{target}_bar'] = save_bar(zero_summary, target, f'{target}_zero_bar.png')
    fig_paths[f'{target}_ladder'] = save_ladder(pivot, target, f'{target}_ladder.png')
    fig_paths[f'{target}_corr'] = save_heatmap(corr_mean, f'{target}: mean metric correlation', f'{target}_corr_heatmap.png', 'coolwarm', -1, 1)
    fig_paths[f'{target}_overlap'] = save_heatmap(overlap_mean, f'{target}: mean top-20% overlap', f'{target}_overlap_heatmap.png', 'Blues', 0, 1)

    report['targets'][target] = {
        'summary': {'num_tickers': len(tickers), 'num_windows': total_windows, 'bucket_count': bucket_count, 'num_predictions_max': num_predictions},
        'zero_summary': zero_summary.to_dict(orient='records'),
        'epsilon_table_html': epsilon_table_html(ladder_df),
        'corr_mean': corr_mean.to_dict(),
        'overlap_mean': overlap_mean.to_dict(),
    }

all_meta = pd.concat(all_ticker_meta, ignore_index=True)
all_meta.to_csv(OUT / 'all_ticker_meta.csv', index=False)
selected = pd.read_csv(SELECTED)[['symbol','close_logret_std','close_logret_std_rank_pct']]
case_df = (all_meta[all_meta['target'] == 'close_log_return']
           .merge(selected, on='symbol', how='left')
           .sort_values('close_logret_std'))
case_pick = pd.concat([
    case_df.head(2),
    case_df.iloc[[len(case_df)//2 - 1, len(case_df)//2]],
    case_df.tail(2)
]).drop_duplicates(subset=['symbol'])
fig_paths['case_buckets'] = save_case_buckets(case_pick, 'close_quantile_bucket_cases.png')

(OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

rows = []
for target, block in report['targets'].items():
    rows.append(f"<tr><td>{target}</td><td>{block['summary']['num_tickers']}</td><td>{block['summary']['num_windows']}</td><td>{block['summary']['bucket_count']}</td><td>{block['summary']['num_predictions_max']}</td></tr>")

case_rows = ''.join(
    f"<tr><td>{r['symbol']}</td><td>{fmt(float(r['daily_std']))}</td><td>{fmt(float(r['q05']))}</td><td>{fmt(float(r['q95']))}</td><td>{len(r['bucket_values'])}</td><td>{fmt(float(r['bucket_step']))}</td></tr>"
    for _, r in case_pick.iterrows()
)

sections = []
for target, block in report['targets'].items():
    zero_rows = ''.join(
        f"<tr><td>{r['metric_label']}</td><td>{fmt(r['mean_zero_percentile'])}</td></tr>"
        for r in block['zero_summary']
    )
    sections.append(
        f"<section class='card'><h2>{target}</h2>"
        f"<p>Each ticker uses its own q05/q95 range, then a zero-centered 11-bucket ladder clipped to that interval. The tables and charts below show whether trimming out the extreme tails changes the conclusion that near-zero forecasts dominate amplitude losses but not directional or correlational metrics.</p>"
        f"{takeaway_html(target, pd.DataFrame(block['zero_summary']))}"
        f"<div class='grid2'><div><h3>Each epsilon-baseline mean percentile</h3>{block['epsilon_table_html']}</div><div><img src='figures/{fig_paths[f'{target}_bar'].name}' alt='bar'></div></div>"
        f"<div class='grid2'><div><img src='figures/{fig_paths[f'{target}_ladder'].name}' alt='ladder'></div><div><img src='figures/{fig_paths[f'{target}_corr'].name}' alt='corr'></div></div>"
        f"<div class='grid2'><div><img src='figures/{fig_paths[f'{target}_overlap'].name}' alt='overlap'></div><div><h3>Zero-baseline mean percentile</h3><table><thead><tr><th>metric</th><th>pct</th></tr></thead><tbody>{zero_rows}</tbody></table></div></div></section>"
    )

html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>q05/q95 Zero-Baseline Intro Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f6f7fb; }}
    h1,h2,h3 {{ margin: 0 0 10px; }}
    p {{ line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }}
    th,td {{ border: 1px solid #d9dee7; padding: 6px 8px; font-size: 12px; text-align: right; }}
    th:first-child,td:first-child {{ text-align: left; }}
    thead th {{ background: #eef2f7; }}
    .card {{ background: white; border: 1px solid #d9dee7; padding: 16px; margin-bottom: 22px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; margin-bottom: 14px; }}
    .takeaway {{ background: #eef4fb; border: 1px solid #c9d8ea; padding: 14px 16px; margin: 14px 0 18px; }}
    .takeaway h3 {{ margin-bottom: 8px; }}
    img {{ width: 100%; border: 1px solid #d9dee7; background: white; }}
  </style>
</head>
<body>
  <h1>q05/q95 Zero-Baseline Intro Report</h1>
  <div class='card'>
    <p>This introductory report covers the <b>100 ticker</b> q05/q95 experiment. Compared with the older min/max range, each ticker-target pair now trims the value range to the historical 5th and 95th percentiles before constructing a zero-centered 11-bucket ladder. This is intended to reduce the influence of extreme tails and test whether the strong-zero-baseline phenomenon still survives when candidate values are constrained to the central body of the historical distribution.</p>
    <table><thead><tr><th>target</th><th>tickers</th><th>weekly windows</th><th>bucket count</th><th>max prediction count</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  </div>
  <section class='card'>
    <h2>Method</h2>
    <p><b>1. Ticker-wise q05/q95 buckets.</b> For each ticker, we aggregate hourly log returns into daily 7-step non-overlapping weekly windows. Then we estimate that ticker's own historical 5th and 95th percentiles and build a zero-centered 11-bucket ladder inside that trimmed interval. This means the candidate space is no longer dominated by the most extreme historical tails.</p>
    <p><b>2. Epsilon-baseline meaning.</b> Epsilon-baseline is not hand-designed separately. It comes directly from the bucket ladder. <code>epsilon_index = 0</code> means only the zero bucket is allowed. <code>epsilon_index = 1</code> means all bucket centers with absolute value no larger than one bucket step from zero are allowed, and so on. The table below reports the mean percentile of each epsilon-baseline in the candidate prediction space.</p>
    <p><b>3. Case examples.</b> The following cases show what ticker-wise q05/q95 bucket ladders look like for low, middle, and high volatility names.</p>
    <div class='grid2'>
      <div><table><thead><tr><th>symbol</th><th>daily std</th><th>q05</th><th>q95</th><th># buckets</th><th>step</th></tr></thead><tbody>{case_rows}</tbody></table></div>
      <div><img src='figures/{fig_paths['case_buckets'].name}' alt='case buckets'></div>
    </div>
  </section>
  {''.join(sections)}
</body>
</html>"""

(OUT / 'intro_report.html').write_text(html, encoding='utf-8')
print(json.dumps({'merged_intro_dir': str(OUT), 'intro_html': str(OUT / 'intro_report.html')}, indent=2), flush=True)
