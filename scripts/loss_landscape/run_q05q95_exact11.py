from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/data/jm/tsorchestra_repro_20260610')
SOURCE = ROOT / 'finance1000_source/financial_hourly_1000_logret.csv'
SELECTED = ROOT / 'finance1000_source/selected_tickers.csv'
DEFAULT_OUT = ROOT / 'analysis_outputs/zero_epsilon_q05q95_report_20260629_100tickers_b11_exact'


@dataclass
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool


METRICS: list[MetricSpec] = [
    MetricSpec('mae', 'MAE', False),
    MetricSpec('mse', 'MSE', False),
    MetricSpec('rmse', 'RMSE', False),
    MetricSpec('direction_accuracy', 'Direction Accuracy', True),
    MetricSpec('pearson_ic', 'Pearson / IC', True),
    MetricSpec('spearman_rho', 'Spearman', True),
    MetricSpec('cosine_similarity', 'Cosine Similarity', True),
    MetricSpec('std_gap', 'Std Gap', False),
    MetricSpec('l1_divergence', 'L1 Divergence', False),
    MetricSpec('dwt_distance', 'DWT Distance', False),
    MetricSpec('kl_divergence', 'KL Divergence', False),
    MetricSpec('wasserstein', 'Wasserstein', False),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def choose_tickers(selected: pd.DataFrame, count: int, offset: int = 0, limit: int | None = None) -> list[str]:
    selected = selected.sort_values('close_logret_std').reset_index(drop=True)
    if count >= len(selected):
        symbols = selected['symbol'].tolist()
    else:
        positions = np.linspace(0, len(selected) - 1, count)
        idx = np.unique(np.round(positions).astype(int))
        while idx.size < count:
            extras = np.setdiff1d(np.arange(len(selected)), idx)
            idx = np.sort(np.concatenate([idx, extras[: count - idx.size]]))
        symbols = selected.iloc[idx[:count]]['symbol'].tolist()
    if limit is None:
        return symbols[offset:]
    return symbols[offset:offset + limit]


def load_target_frame(symbols: list[str]) -> pd.DataFrame:
    usecols = ['date']
    for s in symbols:
        usecols.append(f'{s}_close')
        usecols.append(f'{s}_volume')
    return pd.read_csv(SOURCE, usecols=usecols)


def daily_from_hourly_sum(values: np.ndarray) -> np.ndarray:
    usable = (values.size // 24) * 24
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    return values[:usable].reshape(-1, 24).sum(axis=1, dtype=np.float32)


def weekly_non_overlap(values: np.ndarray, length: int = 7) -> np.ndarray:
    usable = (values.size // length) * length
    if usable == 0:
        return np.empty((0, length), dtype=np.float32)
    return values[:usable].reshape(-1, length)


def build_bucket_grid_from_quantiles(values: np.ndarray, bucket_count: int = 11) -> tuple[np.ndarray, float, int, float, float]:
    q05 = float(np.quantile(values, 0.05))
    q95 = float(np.quantile(values, 0.95))
    radius_steps = max(1, (bucket_count - 1) // 2)
    max_abs = max(abs(q05), abs(q95))
    step = max_abs / radius_steps if radius_steps > 0 else max_abs
    if step == 0.0:
        step = 1e-6
    centers = np.arange(-radius_steps, radius_steps + 1, dtype=np.float32) * step
    kept = centers[(centers >= q05 - 1e-12) & (centers <= q95 + 1e-12)]
    if not np.any(np.isclose(kept, 0.0)):
        kept = np.sort(np.unique(np.concatenate([kept, np.array([0.0], dtype=np.float32)])))
    neg_levels = int(np.sum(kept < 0))
    pos_levels = int(np.sum(kept > 0))
    symmetric_levels = min(neg_levels, pos_levels)
    return kept.astype(np.float32), float(step), symmetric_levels, q05, q95


def decode_prediction_chunk(start: int, end: int, bucket_values: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    base = int(bucket_values.shape[0])
    ids = torch.arange(start, end, device=device, dtype=torch.long)
    digits = []
    tmp = ids.clone()
    for _ in range(length):
        digits.append(tmp % base)
        tmp = tmp // base
    digits = torch.stack(digits[::-1], dim=1)
    return bucket_values[digits]


def torch_percentile(distribution: np.ndarray, value: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return float(np.mean(distribution <= value))
    return float(np.mean(distribution >= value))


def metric_corr_matrix(values_by_metric: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    keys = list(values_by_metric)
    out: dict[str, dict[str, float]] = {}
    for a in keys:
        xa = values_by_metric[a].astype(np.float64)
        row: dict[str, float] = {}
        for b in keys:
            xb = values_by_metric[b].astype(np.float64)
            if np.allclose(xa, xa[0]) or np.allclose(xb, xb[0]):
                row[b] = 0.0
            else:
                v = np.corrcoef(xa, xb)[0, 1]
                row[b] = float(v) if np.isfinite(v) else 0.0
        out[a] = row
    return out


def top_overlap_matrix(values_by_metric: dict[str, np.ndarray], top_frac: float = 0.2) -> dict[str, dict[str, float]]:
    keys = list(values_by_metric)
    n = len(next(iter(values_by_metric.values())))
    top_k = max(1, int(round(n * top_frac)))
    masks: dict[str, np.ndarray] = {}
    for spec in METRICS:
        arr = values_by_metric[spec.key]
        order = np.argsort(-arr) if spec.higher_is_better else np.argsort(arr)
        mask = np.zeros(n, dtype=bool)
        mask[order[:top_k]] = True
        masks[spec.key] = mask
    out: dict[str, dict[str, float]] = {}
    for a in keys:
        denom = max(1, int(masks[a].sum()))
        row: dict[str, float] = {}
        for b in keys:
            row[b] = float(np.sum(masks[a] & masks[b]) / denom)
        out[a] = row
    return out


def rank_tensor(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, dim=2)
    ranks = torch.zeros_like(x)
    rank_vals = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).view(1, 1, -1).expand_as(order)
    ranks.scatter_(2, order, rank_vals)
    return ranks


def compute_scores(pred: torch.Tensor, true_windows: torch.Tensor, bins: torch.Tensor) -> dict[str, np.ndarray]:
    pred3 = pred.unsqueeze(0)
    true3 = true_windows.unsqueeze(1)
    diff = pred3 - true3
    abs_diff = diff.abs()
    scores: dict[str, torch.Tensor] = {}
    mse_map = diff.pow(2).mean(dim=2)
    scores['mae'] = abs_diff.mean(dim=2).mean(dim=0)
    scores['mse'] = mse_map.mean(dim=0)
    scores['rmse'] = torch.sqrt(scores['mse'].clamp_min(0.0))
    scores['direction_accuracy'] = (torch.sign(pred3) == torch.sign(true3)).float().mean(dim=2).mean(dim=0)
    pred_center = pred3 - pred3.mean(dim=2, keepdim=True)
    true_center = true3 - true3.mean(dim=2, keepdim=True)
    cov = (pred_center * true_center).mean(dim=2)
    pred_std = pred_center.pow(2).mean(dim=2).sqrt()
    true_std = true_center.pow(2).mean(dim=2).sqrt()
    denom = pred_std * true_std
    pearson = torch.where(denom > 1e-12, cov / denom, torch.zeros_like(cov))
    scores['pearson_ic'] = torch.nan_to_num(pearson, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=0)
    pred_rank = rank_tensor(pred3.expand(true3.shape[0], pred3.shape[1], pred3.shape[2]))
    true_rank = rank_tensor(true3.expand(true3.shape[0], pred3.shape[1], pred3.shape[2]))
    pred_rank_center = pred_rank - pred_rank.mean(dim=2, keepdim=True)
    true_rank_center = true_rank - true_rank.mean(dim=2, keepdim=True)
    rho_cov = (pred_rank_center * true_rank_center).mean(dim=2)
    rho_den = pred_rank_center.pow(2).mean(dim=2).sqrt() * true_rank_center.pow(2).mean(dim=2).sqrt()
    rho = torch.where(rho_den > 1e-12, rho_cov / rho_den, torch.zeros_like(rho_cov))
    scores['spearman_rho'] = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=0)
    pred_norm = pred3.pow(2).sum(dim=2).sqrt()
    true_norm = true3.pow(2).sum(dim=2).sqrt()
    cos = torch.where(pred_norm * true_norm > 1e-12, (pred3 * true3).sum(dim=2) / (pred_norm * true_norm), torch.zeros_like(pred_norm))
    scores['cosine_similarity'] = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=0)
    scores['std_gap'] = (pred_std - true_std).abs().mean(dim=0)
    scores['l1_divergence'] = (pred_center - true_center).abs().mean(dim=2).mean(dim=0)
    pred_even = pred3[..., : pred3.shape[2] - (pred3.shape[2] % 2)]
    true_even = true3[..., : true3.shape[2] - (true3.shape[2] % 2)]
    pred_avg = (pred_even[..., 0::2] + pred_even[..., 1::2]) * 0.5
    true_avg = (true_even[..., 0::2] + true_even[..., 1::2]) * 0.5
    pred_detail = (pred_even[..., 0::2] - pred_even[..., 1::2]) * 0.5
    true_detail = (true_even[..., 0::2] - true_even[..., 1::2]) * 0.5
    dwt_pred = torch.cat([pred_avg, pred_detail], dim=2)
    dwt_true = torch.cat([true_avg, true_detail], dim=2)
    scores['dwt_distance'] = torch.linalg.norm(dwt_pred - dwt_true, dim=2).mean(dim=0)
    pred_hist = []
    true_hist = []
    for left, right in zip(bins[:-1], bins[1:]):
        pred_hist.append(((pred3 >= left) & (pred3 < right)).float().sum(dim=2))
        true_hist.append(((true3 >= left) & (true3 < right)).float().sum(dim=2))
    pred_hist_t = torch.stack(pred_hist, dim=2) + 1e-12
    true_hist_t = torch.stack(true_hist, dim=2) + 1e-12
    pred_prob = pred_hist_t / pred_hist_t.sum(dim=2, keepdim=True)
    true_prob = true_hist_t / true_hist_t.sum(dim=2, keepdim=True)
    scores['kl_divergence'] = (true_prob * (true_prob / pred_prob).log()).sum(dim=2).mean(dim=0)
    pred_sorted = torch.sort(pred3, dim=2).values
    true_sorted = torch.sort(true3.expand(true3.shape[0], pred3.shape[1], pred3.shape[2]), dim=2).values
    scores['wasserstein'] = (pred_sorted - true_sorted).abs().mean(dim=2).mean(dim=0)
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in scores.items()}


def analyze_target(series_map: dict[str, np.ndarray], target_name: str, tickers: list[str], bucket_count: int, chunk_size: int, device: torch.device, max_windows_per_series: int | None):
    ticker_reports = []
    ladder_rows = []
    corr_rows = []
    overlap_rows = []
    total_predictions_record = None
    for ticker_idx, symbol in enumerate(tickers, start=1):
        t0 = time.time()
        daily = daily_from_hourly_sum(series_map[symbol])
        weekly = weekly_non_overlap(daily, length=7)
        if max_windows_per_series is not None:
            weekly = weekly[:max_windows_per_series]
        if weekly.size == 0:
            continue
        bucket_values_np, step, symmetric_levels, q05, q95 = build_bucket_grid_from_quantiles(daily, bucket_count=bucket_count)
        epsilon_levels = [0.0] + [float(step * i) for i in range(1, symmetric_levels + 1)]
        total_predictions = int(len(bucket_values_np) ** 7)
        total_predictions_record = total_predictions
        total_chunks = math.ceil(total_predictions / chunk_size)
        log(f'[{target_name}] {ticker_idx}/{len(tickers)} {symbol}: windows={weekly.shape[0]} q05={q05:.6g} q95={q95:.6g} buckets={len(bucket_values_np)} predictions={total_predictions} chunks={total_chunks}')
        true_windows = torch.tensor(weekly, device=device)
        bins = torch.tensor(np.linspace(q05, q95, max(24, bucket_count + 9), dtype=np.float32), device=device)
        bucket_values = torch.tensor(bucket_values_np, device=device)
        metric_scores = {m.key: np.empty(total_predictions, dtype=np.float32) for m in METRICS}
        write_at = 0
        for chunk_idx, start in enumerate(range(0, total_predictions, chunk_size), start=1):
            end = min(start + chunk_size, total_predictions)
            pred = decode_prediction_chunk(start, end, bucket_values, 7, device)
            chunk_scores = compute_scores(pred, true_windows, bins)
            for key, val in chunk_scores.items():
                metric_scores[key][write_at:write_at + val.shape[0]] = val
            write_at += end - start
            if chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % 10 == 0:
                log(f'[{target_name}] {symbol}: chunk {chunk_idx}/{total_chunks} ({100.0*chunk_idx/total_chunks:.1f}%) done')
        base = len(bucket_values_np)
        ids = np.arange(total_predictions, dtype=np.int64)
        digit_cols = []
        tmp = ids.copy()
        for _ in range(7):
            digit_cols.append(tmp % base)
            tmp //= base
        digits = np.stack(digit_cols[::-1], axis=1)
        pred_abs = np.abs(bucket_values_np[digits])
        epsilon_masks = [np.all(pred_abs <= eps + 1e-12, axis=1) for eps in epsilon_levels]
        metric_summary = []
        for spec in METRICS:
            dist = metric_scores[spec.key]
            zero_value = float(dist[epsilon_masks[0]].mean())
            zero_pct = torch_percentile(dist, zero_value, spec.higher_is_better)
            metric_summary.append({'metric': spec.label, 'zero_value': zero_value, 'zero_percentile': zero_pct, 'space_mean': float(dist.mean()), 'space_std': float(dist.std())})
            for eps_idx, eps in enumerate(epsilon_levels):
                eps_value = float(dist[epsilon_masks[eps_idx]].mean())
                eps_pct = torch_percentile(dist, eps_value, spec.higher_is_better)
                ladder_rows.append({'symbol': symbol, 'epsilon_index': eps_idx, 'epsilon_value': eps, 'prediction_count': int(epsilon_masks[eps_idx].sum()), 'metric_key': spec.key, 'metric_label': spec.label, 'baseline_value': eps_value, 'baseline_percentile': eps_pct})
        ticker_reports.append({'symbol': symbol, 'daily_std': float(np.std(daily)), 'q05': q05, 'q95': q95, 'bucket_values': [float(v) for v in bucket_values_np.tolist()], 'bucket_step': step, 'num_predictions': total_predictions, 'num_windows': int(weekly.shape[0]), 'epsilon_levels': epsilon_levels, 'metric_summary': metric_summary})
        corr = metric_corr_matrix(metric_scores)
        overlap = top_overlap_matrix(metric_scores)
        corr_rows.append({'symbol': symbol, **{f'{a}__{b}': v for a, row in corr.items() for b, v in row.items()}})
        overlap_rows.append({'symbol': symbol, **{f'{a}__{b}': v for a, row in overlap.items() for b, v in row.items()}})
        log(f'[{target_name}] {ticker_idx}/{len(tickers)} {symbol}: finished in {(time.time()-t0)/60:.2f} min')
    ladder_df = pd.DataFrame(ladder_rows)
    corr_df = pd.DataFrame(corr_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    summary = {'num_tickers': int(len(ticker_reports)), 'num_windows': int(sum(t['num_windows'] for t in ticker_reports)), 'bucket_count': bucket_count, 'num_predictions': int(total_predictions_record or 0)}
    return summary, ticker_reports, ladder_df, corr_df, overlap_df


def average_pair_matrix(df: pd.DataFrame, metric_keys: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=metric_keys, columns=metric_keys, dtype=float)
    for a in metric_keys:
        for b in metric_keys:
            out.loc[a, b] = float(df[f'{a}__{b}'].mean())
    return out


def render_html(report: dict, out_path: Path) -> None:
    sections = []
    for target_name, target in report['targets'].items():
        ladder = pd.DataFrame(target['ladder'])
        pivot = ladder.pivot_table(index='epsilon_index', columns='metric_label', values='baseline_percentile', aggfunc='mean')
        ladder_rows = ['<tr><td>{}</td>{}</tr>'.format(idx, ''.join(f'<td>{float(v):.4f}</td>' for v in row.values)) for idx, row in pivot.iterrows()]
        ladder_cols = ''.join(f'<th>{c}</th>' for c in pivot.columns)
        zero_only = ladder[ladder['epsilon_index'] == 0].groupby('metric_label')['baseline_percentile'].mean().sort_values(ascending=False)
        zero_rows = ''.join(f'<tr><td>{idx}</td><td>{float(v):.4f}</td></tr>' for idx, v in zero_only.items())
        corr = pd.DataFrame(target['corr_mean'])
        overlap = pd.DataFrame(target['overlap_mean'])
        pair_cols = ''.join(f'<th>{c}</th>' for c in corr.columns)
        corr_rows = [f"<tr><th>{idx}</th>{''.join(f'<td>{float(v):.3f}</td>' for v in row.values)}</tr>" for idx, row in corr.iterrows()]
        overlap_rows = [f"<tr><th>{idx}</th>{''.join(f'<td>{float(v):.3f}</td>' for v in row.values)}</tr>" for idx, row in overlap.iterrows()]
        sections.append(f'''<section class="card"><h2>{target_name}</h2><p>tickers={target['summary']['num_tickers']} | weekly_windows={target['summary']['num_windows']} | q05/q95 bucket_count={target['summary']['bucket_count']} | full_exact_predictions={target['summary']['num_predictions']}</p><div class="top-grid"><div><h3>Zero-baseline mean percentile</h3><table><thead><tr><th>metric</th><th>pct</th></tr></thead><tbody>{zero_rows}</tbody></table></div><div><h3>Mean percentile by epsilon ladder</h3><table><thead><tr><th>epsilon_index</th>{ladder_cols}</tr></thead><tbody>{''.join(ladder_rows)}</tbody></table></div></div><div class="matrix-grid"><div><h3>Mean metric correlation</h3><table><thead><tr><th></th>{pair_cols}</tr></thead><tbody>{''.join(corr_rows)}</tbody></table></div><div><h3>Mean top-20% overlap</h3><table><thead><tr><th></th>{pair_cols}</tr></thead><tbody>{''.join(overlap_rows)}</tbody></table></div></div></section>''')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Zero / Epsilon q05/q95 Exact-11 GPU Report</title><style>body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f6f7fb; }} h1,h2,h3 {{ margin: 0 0 10px; }} p {{ line-height: 1.45; }} table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }} th,td {{ border: 1px solid #d9dee7; padding: 6px 8px; font-size: 12px; text-align: right; }} th:first-child,td:first-child {{ text-align: left; }} thead th {{ background: #eef2f7; }} .card {{ background: white; border: 1px solid #d9dee7; padding: 16px; margin-bottom: 20px; }} .matrix-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }} .top-grid {{ display: grid; grid-template-columns: 0.7fr 1.3fr; gap: 16px; align-items: start; }}</style></head><body><h1>Zero / Epsilon q05/q95 Exact-11 GPU Report</h1><div class="card"><p>This report rebuilds each ticker target range using its own empirical 5th and 95th percentiles instead of full-history min/max. The 11-center zero-centered bucket ladder is then clipped to that q05/q95 range. 100 tickers are selected evenly across the close-log-return volatility ranking and run in parallel on CUDA 0/1/2/3. The goal is to see whether the near-zero dominance story changes once extreme tails are trimmed out of the candidate value range.</p></div>{''.join(sections)}</body></html>'''
    out_path.write_text(html, encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='q05/q95 exact 11-bucket zero/epsilon ladder GPU report for finance1000.')
    parser.add_argument('--ticker-count', type=int, default=100)
    parser.add_argument('--ticker-offset', type=int, default=0)
    parser.add_argument('--ticker-limit', type=int)
    parser.add_argument('--bucket-count', type=int, default=11)
    parser.add_argument('--chunk-size', type=int, default=262144)
    parser.add_argument('--max-windows-per-series', type=int)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    selected = pd.read_csv(SELECTED)
    tickers = choose_tickers(selected, args.ticker_count, offset=args.ticker_offset, limit=args.ticker_limit)
    log(f'start q05q95 run device={device} tickers={len(tickers)} bucket_count={args.bucket_count} chunk_size={args.chunk_size}')
    frame = load_target_frame(tickers)
    close_map = {symbol: frame[f'{symbol}_close'].to_numpy(dtype=np.float32, copy=True) for symbol in tickers}
    volume_map = {symbol: frame[f'{symbol}_volume'].to_numpy(dtype=np.float32, copy=True) for symbol in tickers}
    report = {'device': str(device), 'tickers': tickers, 'targets': {}}
    metric_keys = [m.key for m in METRICS]
    for target_name, series_map in [('close_log_return', close_map), ('volume_log_return', volume_map)]:
        summary, ticker_reports, ladder_df, corr_df, overlap_df = analyze_target(series_map, target_name, tickers, args.bucket_count, args.chunk_size, device, args.max_windows_per_series)
        ladder_df.insert(0, 'target', target_name)
        corr_df.insert(0, 'target', target_name)
        overlap_df.insert(0, 'target', target_name)
        ladder_df.to_csv(args.output_dir / f'{target_name}_epsilon_ladder.csv', index=False)
        corr_df.to_csv(args.output_dir / f'{target_name}_metric_corr.csv', index=False)
        overlap_df.to_csv(args.output_dir / f'{target_name}_metric_overlap.csv', index=False)
        corr_mean = average_pair_matrix(corr_df.drop(columns=['target', 'symbol']), metric_keys)
        overlap_mean = average_pair_matrix(overlap_df.drop(columns=['target', 'symbol']), metric_keys)
        corr_mean.to_csv(args.output_dir / f'{target_name}_metric_corr_mean.csv')
        overlap_mean.to_csv(args.output_dir / f'{target_name}_metric_overlap_mean.csv')
        zero_summary = (ladder_df[ladder_df['epsilon_index'] == 0].groupby('metric_label')['baseline_percentile'].mean().reset_index().rename(columns={'baseline_percentile': 'mean_zero_percentile'}))
        zero_summary.to_csv(args.output_dir / f'{target_name}_zero_summary.csv', index=False)
        report['targets'][target_name] = {'summary': summary, 'tickers': ticker_reports, 'ladder': ladder_df.to_dict(orient='records'), 'corr_mean': corr_mean.to_dict(), 'overlap_mean': overlap_mean.to_dict()}
        log(f'finished target {target_name}')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    render_html(report, args.output_dir / 'report.html')
    print(json.dumps({'output_dir': str(args.output_dir), 'html': str(args.output_dir / 'report.html'), 'device': str(device), 'ticker_count': len(tickers), 'bucket_count': args.bucket_count}, indent=2), flush=True)

if __name__ == '__main__':
    main()
