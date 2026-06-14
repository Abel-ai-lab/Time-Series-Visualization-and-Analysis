from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=RuntimeWarning)


DEFAULT_PREFERRED_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "JPM",
    "KMB",
    "NVAX",
    "GME",
    "TSLA",
    "META",
    "XOM",
    "FCEL",
    "MARA",
]


@dataclass(frozen=True)
class DatasetColumns:
    close_cols: list[str]
    volume_cols: list[str]
    symbols: list[str]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 190,
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    pre_args, remaining = pre.parse_known_args(argv)
    defaults = {}
    if pre_args.config:
        defaults = json.loads(pre_args.config.read_text())

    parser = argparse.ArgumentParser(
        description="Generate ModernTSF-style visual reports for wide time-series CSV data."
    )
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument("--raw-csv", type=Path, required="raw_csv" not in defaults)
    parser.add_argument("--logret-csv", type=Path)
    parser.add_argument("--ticker-file", type=Path)
    parser.add_argument("--ticker-column", default="symbol")
    parser.add_argument("--tickers", help="Comma-separated ticker list for representative plots.")
    parser.add_argument("--output-dir", type=Path, required="output_dir" not in defaults)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--close-suffix", default="_close")
    parser.add_argument("--volume-suffix", default="_volume")
    parser.add_argument("--sample-rows", type=int, default=7000)
    parser.add_argument("--chunksize", type=int, default=900)
    parser.add_argument("--representative-count", type=int, default=12)
    parser.add_argument("--ticker-book", choices=["none", "selected", "all"], default="selected")
    parser.add_argument("--book-max-tickers", type=int, help="Optional cap for ticker-book pages.")
    parser.add_argument("--book-batch-size", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=10)
    parser.add_argument("--title", default="ModernTSF Real Data Visualization and Analysis")
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining if argv is not None else None)
    args.raw_csv = Path(args.raw_csv)
    args.output_dir = Path(args.output_dir)
    if args.logret_csv:
        args.logret_csv = Path(args.logret_csv)
    if args.ticker_file:
        args.ticker_file = Path(args.ticker_file)
    return args


def symbol_from_col(col: str, suffix: str) -> str:
    return col[: -len(suffix)] if suffix and col.endswith(suffix) else col


def column_for(symbol: str, suffix: str) -> str:
    return f"{symbol}{suffix}"


def discover_columns(path: Path, close_suffix: str, volume_suffix: str) -> DatasetColumns:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    close_cols = [c for c in cols if c.endswith(close_suffix)]
    volume_set = {c for c in cols if c.endswith(volume_suffix)}
    symbols = []
    volumes = []
    for close_col in close_cols:
        symbol = symbol_from_col(close_col, close_suffix)
        volume_col = column_for(symbol, volume_suffix)
        if volume_col in volume_set:
            symbols.append(symbol)
            volumes.append(volume_col)
    return DatasetColumns(
        close_cols=[column_for(s, close_suffix) for s in symbols],
        volume_cols=volumes,
        symbols=symbols,
    )


def load_ticker_file(path: Path | None, column: str) -> list[str]:
    if not path:
        return []
    df = pd.read_csv(path)
    if column not in df.columns:
        column = df.columns[0]
    return [str(x) for x in df[column].dropna().astype(str).tolist()]


def choose_representatives(args: argparse.Namespace, symbols: list[str]) -> list[str]:
    universe = set(symbols)
    requested = []
    if args.tickers:
        requested.extend([x.strip() for x in args.tickers.split(",") if x.strip()])
    requested.extend(DEFAULT_PREFERRED_TICKERS)
    requested.extend(load_ticker_file(args.ticker_file, args.ticker_column))
    requested.extend(symbols)

    reps = []
    for symbol in requested:
        if symbol in universe and symbol not in reps:
            reps.append(symbol)
        if len(reps) >= args.representative_count:
            break
    return reps


def count_data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_row_sample(
    path: Path,
    usecols: list[str],
    date_col: str,
    sample_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    total = count_data_rows(path)
    if sample_rows <= 0 or sample_rows >= total:
        df = pd.read_csv(path, usecols=usecols)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(np.arange(total), size=min(sample_rows, total), replace=False))
        keep = set(idx + 1)
        df = pd.read_csv(path, usecols=usecols, skiprows=lambda i: i > 0 and i not in keep)
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    return df, total


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64").ravel()
    return arr[np.isfinite(arr)]


def describe(values: np.ndarray) -> dict[str, float | int]:
    x = finite_values(values)
    if x.size == 0:
        return {"count": 0}
    q = np.quantile(x, [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std > 0:
        z = (x - mean) / std
        skew = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4) - 3)
    else:
        skew = 0.0
        kurtosis = 0.0
    return {
        "count": int(x.size),
        "mean": mean,
        "std": std,
        "min": float(np.min(x)),
        "p0.1": float(q[0]),
        "p1": float(q[1]),
        "p5": float(q[2]),
        "p25": float(q[3]),
        "p50": float(q[4]),
        "p75": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "p99.9": float(q[8]),
        "max": float(np.max(x)),
        "zero_frac": float(np.mean(x == 0)),
        "skew": skew,
        "kurtosis_excess": kurtosis,
    }


def derive_log_returns_from_raw(raw_df: pd.DataFrame, close_cols: list[str], volume_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    close = raw_df[close_cols].replace(0, np.nan).to_numpy(dtype="float64")
    volume = raw_df[volume_cols].clip(lower=0).to_numpy(dtype="float64")
    close_lr = np.diff(np.log(close), axis=0)
    volume_lr = np.diff(np.log1p(volume), axis=0)
    return finite_values(close_lr), finite_values(volume_lr)


def read_distribution_arrays(
    args: argparse.Namespace,
    columns: DatasetColumns,
) -> tuple[dict[str, np.ndarray], int]:
    usecols = [args.date_col] + columns.close_cols + columns.volume_cols
    raw_sample, total_rows = read_row_sample(args.raw_csv, usecols, args.date_col, args.sample_rows, args.random_seed)
    raw_close = finite_values(raw_sample[columns.close_cols].to_numpy(dtype="float64"))
    raw_volume = finite_values(raw_sample[columns.volume_cols].to_numpy(dtype="float64"))

    if args.logret_csv:
        log_sample, _ = read_row_sample(
            args.logret_csv, usecols, args.date_col, args.sample_rows, args.random_seed + 1
        )
        close_logret = finite_values(log_sample[columns.close_cols].to_numpy(dtype="float64"))
        volume_log_change = finite_values(log_sample[columns.volume_cols].to_numpy(dtype="float64"))
    else:
        close_logret, volume_log_change = derive_log_returns_from_raw(
            raw_sample, columns.close_cols, columns.volume_cols
        )

    return (
        {
            "raw_close": raw_close,
            "raw_volume": raw_volume,
            "close_log_return": close_logret,
            "volume_log_change": volume_log_change,
        },
        total_rows,
    )


def read_representatives(
    args: argparse.Namespace,
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close_cols = [column_for(s, args.close_suffix) for s in symbols]
    volume_cols = [column_for(s, args.volume_suffix) for s in symbols]
    raw = pd.read_csv(args.raw_csv, usecols=[args.date_col] + close_cols + volume_cols)
    raw[args.date_col] = pd.to_datetime(raw[args.date_col], utc=True, errors="coerce")
    raw = raw.set_index(args.date_col).sort_index()
    daily_close = raw[close_cols].resample("1D").last()
    daily_volume = raw[volume_cols].resample("1D").sum()

    if args.logret_csv:
        logr = pd.read_csv(args.logret_csv, usecols=[args.date_col] + close_cols)
        logr[args.date_col] = pd.to_datetime(logr[args.date_col], utc=True, errors="coerce")
        logr = logr.set_index(args.date_col).sort_index()
        daily_logret = logr[close_cols].resample("1D").sum()
    else:
        daily_logret = np.log(daily_close.replace(0, np.nan)).diff()
    return daily_close, daily_volume, daily_logret


def iter_raw_and_logret_chunks(
    args: argparse.Namespace,
    close_cols: list[str],
    volume_cols: list[str],
) -> Iterable[tuple[pd.DataFrame, pd.DataFrame | None]]:
    raw_iter = pd.read_csv(
        args.raw_csv,
        usecols=[args.date_col] + close_cols + volume_cols,
        chunksize=args.chunksize,
    )
    if args.logret_csv:
        log_iter: Iterable[pd.DataFrame | None] = pd.read_csv(
            args.logret_csv,
            usecols=[args.date_col] + close_cols,
            chunksize=args.chunksize,
        )
    else:
        log_iter = iter(lambda: None, None)

    if args.logret_csv:
        for raw_chunk, log_chunk in zip(raw_iter, log_iter):
            yield raw_chunk, log_chunk
    else:
        for raw_chunk in raw_iter:
            yield raw_chunk, None


def chunk_close_log_returns(
    raw_close: np.ndarray,
    log_chunk: pd.DataFrame | None,
    close_cols: list[str],
    previous_log_close: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if log_chunk is not None:
        return log_chunk[close_cols].to_numpy(dtype="float64"), previous_log_close

    close = np.where(raw_close > 0, raw_close, np.nan)
    log_close = np.log(close)
    stacked = log_close if previous_log_close is None else np.vstack([previous_log_close, log_close])
    returns = np.diff(stacked, axis=0)
    return returns, log_close[-1:, :]


def compute_per_ticker_stats(args: argparse.Namespace, columns: DatasetColumns) -> pd.DataFrame:
    symbols = columns.symbols
    n = len(symbols)
    count = np.zeros(n)
    sum_lr = np.zeros(n)
    sumsq_lr = np.zeros(n)
    abs_sum = np.zeros(n)
    zero_count = np.zeros(n)
    close_min = np.full(n, np.inf)
    close_max = np.full(n, -np.inf)
    first_close = np.full(n, np.nan)
    last_close = np.full(n, np.nan)
    vol_sum = np.zeros(n)
    vol_nonzero = np.zeros(n)
    vol_max = np.zeros(n)
    previous_log_close = None

    for raw, logr in iter_raw_and_logret_chunks(args, columns.close_cols, columns.volume_cols):
        c = raw[columns.close_cols].to_numpy(dtype="float64")
        v = raw[columns.volume_cols].to_numpy(dtype="float64")
        r, previous_log_close = chunk_close_log_returns(c, logr, columns.close_cols, previous_log_close)

        c_masked = np.where(np.isfinite(c), c, np.nan)
        close_min = np.minimum(close_min, np.nanmin(c_masked, axis=0))
        close_max = np.maximum(close_max, np.nanmax(c_masked, axis=0))
        for j in range(n):
            valid = c[:, j][np.isfinite(c[:, j]) & (c[:, j] != 0)]
            if valid.size:
                if np.isnan(first_close[j]):
                    first_close[j] = valid[0]
                last_close[j] = valid[-1]

        finite_r = np.isfinite(r)
        count += finite_r.sum(axis=0)
        safe_r = np.where(finite_r, r, 0)
        sum_lr += safe_r.sum(axis=0)
        sumsq_lr += (safe_r * safe_r).sum(axis=0)
        abs_sum += np.abs(safe_r).sum(axis=0)
        zero_count += np.where(finite_r & (r == 0), 1, 0).sum(axis=0)

        finite_v = np.isfinite(v)
        vol_sum += np.where(finite_v, v, 0).sum(axis=0)
        vol_nonzero += np.where(finite_v & (v > 0), 1, 0).sum(axis=0)
        vol_max = np.maximum(vol_max, np.nanmax(np.where(finite_v, v, np.nan), axis=0))

    mean_lr = sum_lr / np.maximum(count, 1)
    var_lr = sumsq_lr / np.maximum(count, 1) - mean_lr**2
    std_lr = np.sqrt(np.maximum(var_lr, 0))
    total_return = np.log(last_close / first_close)
    price_range_ratio = (close_max - close_min) / np.where(np.abs(first_close) > 0, np.abs(first_close), np.nan)

    return pd.DataFrame(
        {
            "symbol": symbols,
            "first_close": first_close,
            "last_close": last_close,
            "total_log_return": total_return,
            "close_min": close_min,
            "close_max": close_max,
            "price_range_ratio": price_range_ratio,
            "logret_mean": mean_lr,
            "logret_std": std_lr,
            "logret_abs_mean": abs_sum / np.maximum(count, 1),
            "logret_zero_frac": zero_count / np.maximum(count, 1),
            "volume_sum": vol_sum,
            "volume_nonzero_frac": vol_nonzero / np.maximum(count, 1),
            "volume_max": vol_max,
        }
    )


def compute_cross_section(args: argparse.Namespace, columns: DatasetColumns) -> pd.DataFrame:
    rows = []
    for chunk in pd.read_csv(
        args.raw_csv,
        usecols=[args.date_col] + columns.close_cols + columns.volume_cols,
        chunksize=args.chunksize,
    ):
        dates = pd.to_datetime(chunk[args.date_col], utc=True, errors="coerce")
        c = chunk[columns.close_cols].to_numpy(dtype="float64")
        v = chunk[columns.volume_cols].to_numpy(dtype="float64")
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "close_p10": np.nanquantile(c, 0.10, axis=1),
                    "close_p50": np.nanquantile(c, 0.50, axis=1),
                    "close_p90": np.nanquantile(c, 0.90, axis=1),
                    "volume_p50": np.nanquantile(v, 0.50, axis=1),
                    "volume_p90": np.nanquantile(v, 0.90, axis=1),
                    "volume_sum": np.nansum(v, axis=1),
                }
            )
        )
    out = pd.concat(rows).set_index("date").sort_index()
    return out.resample("1D").median()


def rolling_vol_panel(args: argparse.Namespace, symbols: list[str]) -> pd.DataFrame:
    close_cols = [column_for(s, args.close_suffix) for s in symbols]
    if args.logret_csv:
        logr = pd.read_csv(args.logret_csv, usecols=[args.date_col] + close_cols)
        logr[args.date_col] = pd.to_datetime(logr[args.date_col], utc=True, errors="coerce")
        daily = logr.set_index(args.date_col).sort_index()[close_cols].resample("1D").sum()
    else:
        raw = pd.read_csv(args.raw_csv, usecols=[args.date_col] + close_cols)
        raw[args.date_col] = pd.to_datetime(raw[args.date_col], utc=True, errors="coerce")
        daily_close = raw.set_index(args.date_col).sort_index()[close_cols].resample("1D").last()
        daily = np.log(daily_close.replace(0, np.nan)).diff()
    return daily.rolling(30, min_periods=10).std() * math.sqrt(252)


def save_fig(fig: plt.Figure, figures_dir: Path, name: str) -> Path:
    path = figures_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_representative_curves(
    symbols: list[str],
    daily_close: pd.DataFrame,
    daily_volume: pd.DataFrame,
    daily_logret: pd.DataFrame,
    args: argparse.Namespace,
    figures_dir: Path,
) -> Path:
    fig, axes = plt.subplots(len(symbols), 3, figsize=(15, 2.25 * len(symbols)), sharex="col", squeeze=False)
    for i, symbol in enumerate(symbols):
        close_col = column_for(symbol, args.close_suffix)
        volume_col = column_for(symbol, args.volume_suffix)
        c = daily_close[close_col].replace(0, np.nan)
        v = daily_volume[volume_col].replace(0, np.nan)
        r = daily_logret[close_col]
        axes[i, 0].plot(c.index, c, color="#245C7A", lw=0.8)
        axes[i, 0].set_title(f"{symbol} true close")
        axes[i, 0].set_ylabel("close")
        axes[i, 1].plot(v.index, v, color="#2D8F85", lw=0.75)
        axes[i, 1].set_yscale("log")
        axes[i, 1].set_title(f"{symbol} true volume")
        axes[i, 1].set_ylabel("volume log")
        axes[i, 2].plot(r.index, r, color="#A94E67", lw=0.65)
        axes[i, 2].axhline(0, color="black", lw=0.6)
        finite = r[np.isfinite(r)]
        if finite.size:
            lo, hi = np.nanquantile(finite, [0.005, 0.995])
            if lo < hi:
                axes[i, 2].set_ylim(lo, hi)
        axes[i, 2].set_title(f"{symbol} true daily log return")
        axes[i, 2].set_ylabel("logret")
    fig.suptitle("Real Historical Curves: Close, Volume, and Log Return", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return save_fig(fig, figures_dir, "real_curves_by_ticker.png")


def clipped(values: np.ndarray, q: tuple[float, float] | None) -> np.ndarray:
    x = finite_values(values)
    if x.size == 0 or q is None:
        return x
    lo, hi = np.nanquantile(x, q)
    return x[(x >= lo) & (x <= hi)]


def plot_distributions(arrays: dict[str, np.ndarray], figures_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    panels = [
        (axes[0, 0], arrays["raw_close"], "Raw close", "close", "#245C7A", (0.001, 0.999)),
        (axes[0, 1], arrays["raw_volume"], "Raw volume", "volume", "#2D8F85", (0.001, 0.999)),
        (axes[0, 2], np.log1p(np.clip(arrays["raw_volume"], 0, None)), "log1p(raw volume)", "log1p volume", "#508F4C", None),
        (axes[1, 0], arrays["close_log_return"], "Close log return", "log return", "#A94E67", (0.001, 0.999)),
        (axes[1, 1], arrays["volume_log_change"], "Volume log change", "log change", "#765A91", (0.001, 0.999)),
        (axes[1, 2], np.abs(arrays["close_log_return"]), "Abs close log return", "abs log return", "#C38342", (0, 0.999)),
    ]
    for ax, vals, title, xlabel, color, clip_q in panels:
        x = clipped(vals, clip_q)
        ax.hist(x, bins=180, color=color, alpha=0.86, edgecolor="white", linewidth=0.2)
        ax.set_yscale("log")
        ax.set_title(f"{title} clipped" if clip_q else title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count log scale")
    fig.suptitle("Empirical Distributions from Real Data", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_fig(fig, figures_dir, "real_distribution_panels.png")


def normal_quantiles(n: int) -> np.ndarray:
    nd = NormalDist()
    p = (np.arange(1, n + 1) - 0.5) / n
    return np.array([nd.inv_cdf(float(v)) for v in p], dtype="float64")


def plot_logret_diagnostics(close_logret: np.ndarray, figures_dir: Path) -> Path:
    x = finite_values(close_logret)
    rng = np.random.default_rng(42)
    sample_n = min(250_000, x.size)
    qq = np.sort(rng.choice(x, sample_n, replace=False)) if sample_n else np.array([])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    if qq.size:
        qn = normal_quantiles(qq.size)
        axes[0, 0].scatter(qn, qq, s=1.5, color="#765A91", alpha=0.5)
        slope, intercept = np.polyfit(qn, qq, 1)
        axes[0, 0].plot(qn, slope * qn + intercept, color="#A94E67", lw=1)
    axes[0, 0].set_title("QQ plot: close log returns vs Normal")
    axes[0, 0].set_xlabel("normal quantile")
    axes[0, 0].set_ylabel("empirical quantile")
    lo, hi = np.nanquantile(x, [0.0005, 0.9995]) if x.size else (0, 0)
    axes[0, 1].hist(x[(x >= lo) & (x <= hi)], bins=220, color="#A94E67", alpha=0.86)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Close log return histogram, p0.05-p99.95")
    axes[0, 1].set_xlabel("log return")
    absx = np.abs(x)
    sorted_abs = np.sort(absx[absx > 0])
    if sorted_abs.size:
        ccdf_y = 1 - np.arange(1, sorted_abs.size + 1) / sorted_abs.size
        sample = np.linspace(0, sorted_abs.size - 1, min(12000, sorted_abs.size)).astype(int)
        axes[1, 0].plot(sorted_abs[sample], ccdf_y[sample], color="#C38342", lw=1.2)
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("CCDF of absolute close log returns")
    axes[1, 0].set_xlabel("|log return|")
    axes[1, 0].set_ylabel("P(|r| > x)")
    thresholds = [0.005, 0.01, 0.02, 0.05, 0.10]
    freqs = [float(np.mean(absx > t)) if absx.size else 0.0 for t in thresholds]
    axes[1, 1].bar([str(t) for t in thresholds], freqs, color="#6B705C")
    axes[1, 1].set_title("Tail event frequency")
    axes[1, 1].set_xlabel("|log return| threshold")
    axes[1, 1].set_ylabel("fraction")
    fig.suptitle("Log Return Diagnostics: Distribution, Normality, and Tails", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_fig(fig, figures_dir, "log_return_diagnostics.png")


def plot_cross_section(cs_daily: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(cs_daily.index, cs_daily["close_p50"], color="#245C7A", lw=1.0, label="median")
    axes[0].fill_between(
        cs_daily.index,
        cs_daily["close_p10"],
        cs_daily["close_p90"],
        color="#245C7A",
        alpha=0.18,
        label="p10-p90",
    )
    axes[0].set_title("Cross-sectional close distribution over time")
    axes[0].set_ylabel("close")
    axes[0].legend()
    axes[1].plot(cs_daily.index, cs_daily["volume_sum"], color="#2D8F85", lw=0.9)
    axes[1].set_yscale("log")
    axes[1].set_title("Aggregate volume over tickers")
    axes[1].set_ylabel("sum volume")
    axes[2].plot(cs_daily.index, cs_daily["volume_p50"], color="#765A91", lw=0.9, label="median")
    axes[2].plot(cs_daily.index, cs_daily["volume_p90"], color="#A94E67", lw=0.9, label="p90")
    axes[2].set_yscale("log")
    axes[2].set_title("Cross-sectional volume median and p90")
    axes[2].set_ylabel("volume")
    axes[2].legend()
    fig.suptitle("Cross-Sectional Structure in Raw Data", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_fig(fig, figures_dir, "cross_section_time_structure.png")


def plot_ticker_stats(stats_df: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    s = stats_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["logret_std", "volume_sum", "total_log_return"]
    )
    axes[0, 0].scatter(s["logret_std"], np.log10(s["volume_sum"].clip(lower=1)), s=10, alpha=0.55, color="#245C7A")
    axes[0, 0].set_title("Ticker volatility vs total volume")
    axes[0, 0].set_xlabel("close log-return std")
    axes[0, 0].set_ylabel("log10 total volume")
    axes[0, 1].scatter(s["total_log_return"], s["logret_std"], s=10, alpha=0.55, color="#A94E67")
    axes[0, 1].axvline(0, color="black", lw=0.7)
    axes[0, 1].set_title("Long-run return vs volatility")
    axes[0, 1].set_xlabel("total log return")
    axes[0, 1].set_ylabel("log-return std")
    topv = s.nlargest(20, "logret_std").sort_values("logret_std")
    axes[1, 0].barh(topv["symbol"], topv["logret_std"], color="#C38342")
    axes[1, 0].set_title("Top 20 tickers by log-return volatility")
    axes[1, 0].set_xlabel("std")
    topvol = s.nlargest(20, "volume_sum").sort_values("volume_sum")
    axes[1, 1].barh(topvol["symbol"], topvol["volume_sum"], color="#2D8F85")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Top 20 tickers by total volume")
    axes[1, 1].set_xlabel("total volume log scale")
    fig.suptitle("Per-Ticker Heterogeneity", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_fig(fig, figures_dir, "per_ticker_heterogeneity.png")


def plot_heatmaps(
    symbols: list[str],
    daily_logret: pd.DataFrame,
    rolling_vol: pd.DataFrame,
    args: argparse.Namespace,
    figures_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    close_cols = [column_for(s, args.close_suffix) for s in symbols]
    lr = daily_logret[close_cols].T
    lr.index = symbols
    lim = np.nanquantile(np.abs(lr.to_numpy()), 0.995)
    im0 = axes[0].imshow(lr, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set_yticks(np.arange(len(symbols)))
    axes[0].set_yticklabels(symbols)
    axes[0].set_title("Daily log-return heatmap for representative tickers")
    fig.colorbar(im0, ax=axes[0], label="daily log return")
    rv = rolling_vol[close_cols].T
    rv.index = symbols
    im1 = axes[1].imshow(rv, aspect="auto", interpolation="nearest", cmap="magma")
    axes[1].set_yticks(np.arange(len(symbols)))
    axes[1].set_yticklabels(symbols)
    axes[1].set_title("30-day rolling annualized volatility heatmap")
    fig.colorbar(im1, ax=axes[1], label="annualized vol")
    dates = daily_logret.index
    if len(dates):
        ticks = np.linspace(0, len(dates) - 1, min(8, len(dates))).astype(int)
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels([str(dates[i].date()) for i in ticks], rotation=30, ha="right")
    fig.suptitle("Time Structure in Log Returns and Volatility", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_fig(fig, figures_dir, "logret_volatility_heatmaps.png")


def add_png_to_pdf(pdf: PdfPages, png: Path) -> None:
    img = plt.imread(png)
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def write_distribution_stats(arrays: dict[str, np.ndarray], tables_dir: Path) -> dict[str, dict[str, float | int]]:
    summary = {name: describe(values) for name, values in arrays.items()}
    rows = [
        {"group": group, "stat": stat, "value": value}
        for group, stats in summary.items()
        for stat, value in stats.items()
    ]
    pd.DataFrame(rows).to_csv(tables_dir / "distribution_stats.csv", index=False)
    return summary


def write_pdf_report(
    args: argparse.Namespace,
    pdf_path: Path,
    figure_paths: list[Path],
    summary: dict[str, object],
    ticker_book_path: Path | None,
) -> None:
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(13, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.suptitle(args.title, fontsize=17, fontweight="bold", y=0.95)
        lines = [
            "This report visualizes real time-series data in a ModernTSF-style data analysis workflow.",
            "",
            f"Raw CSV: {args.raw_csv}",
            f"Log-return CSV: {args.logret_csv if args.logret_csv else 'derived from raw close'}",
            f"Universe: {summary['num_tickers']} tickers",
            f"Timestamps: {summary['num_timestamps']}",
            f"Span: {summary['time_start']} to {summary['time_end']}",
            f"Representative tickers: {', '.join(summary['representative_tickers'])}",
            "",
            "Included figures:",
            "1. Real close, volume, and log-return curves by ticker",
            "2. Raw and log-return distribution panels",
            "3. Log-return QQ plot, tail CCDF, and tail event rates",
            "4. Cross-sectional close and volume structure over time",
            "5. Per-ticker volatility, volume, and return heterogeneity",
            "6. Log-return and rolling-volatility heatmaps",
        ]
        if ticker_book_path:
            lines.extend(["", f"Ticker book: {ticker_book_path.name}"])
        ax.text(0.06, 0.86, "\n".join(lines), va="top", ha="left", fontsize=10.5)
        pdf.savefig(fig)
        plt.close(fig)
        for path in figure_paths:
            add_png_to_pdf(pdf, path)


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def generate_ticker_book(
    args: argparse.Namespace,
    symbols: list[str],
    out_dir: Path,
) -> Path | None:
    if args.ticker_book == "none":
        return None
    book_symbols = list(symbols)
    if args.book_max_tickers:
        book_symbols = book_symbols[: args.book_max_tickers]
    if not book_symbols:
        return None

    path = out_dir / f"ticker_book_{args.ticker_book}_{len(book_symbols)}.pdf"
    with PdfPages(path) as pdf:
        for batch in batched(book_symbols, args.book_batch_size):
            daily_close, daily_volume, daily_logret = read_representatives(args, batch)
            for symbol in batch:
                close_col = column_for(symbol, args.close_suffix)
                volume_col = column_for(symbol, args.volume_suffix)
                fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
                c = daily_close[close_col].replace(0, np.nan)
                v = daily_volume[volume_col].replace(0, np.nan)
                r = daily_logret[close_col]
                axes[0].plot(c.index, c, color="#245C7A", lw=0.9)
                axes[0].set_title(f"{symbol} true close")
                axes[0].set_ylabel("close")
                axes[1].plot(v.index, v, color="#2D8F85", lw=0.8)
                axes[1].set_yscale("log")
                axes[1].set_title(f"{symbol} true volume")
                axes[1].set_ylabel("volume log")
                axes[2].plot(r.index, r, color="#A94E67", lw=0.65)
                axes[2].axhline(0, color="black", lw=0.6)
                finite = r[np.isfinite(r)]
                if finite.size:
                    lo, hi = np.nanquantile(finite, [0.005, 0.995])
                    if lo < hi:
                        axes[2].set_ylim(lo, hi)
                axes[2].set_title(f"{symbol} true daily log return")
                axes[2].set_ylabel("logret")
                fig.suptitle(f"{symbol}: Raw Close, Volume, and Log Return", fontsize=15, fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_matplotlib()

    out_dir = args.output_dir
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("discovering columns")
    columns = discover_columns(args.raw_csv, args.close_suffix, args.volume_suffix)
    if not columns.symbols:
        raise SystemExit("No ticker columns found. Check --close-suffix and --volume-suffix.")
    reps = choose_representatives(args, columns.symbols)

    print("loading distribution sample")
    arrays, total_rows = read_distribution_arrays(args, columns)
    dist_summary = write_distribution_stats(arrays, tables_dir)

    print("loading representative curves")
    daily_close, daily_volume, daily_logret = read_representatives(args, reps)

    print("computing per-ticker stats")
    stats_df = compute_per_ticker_stats(args, columns)
    stats_df.to_csv(tables_dir / "per_ticker_stats.csv", index=False)

    print("computing cross-section")
    cs_daily = compute_cross_section(args, columns)
    cs_daily.to_csv(tables_dir / "cross_section_daily.csv")

    print("computing rolling volatility")
    heatmap_symbols = reps[:8] + stats_df.nlargest(8, "logret_std")["symbol"].tolist()
    heatmap_symbols = list(dict.fromkeys([s for s in heatmap_symbols if s in columns.symbols]))
    _, _, daily_logret_h = read_representatives(args, heatmap_symbols)
    rolling_vol = rolling_vol_panel(args, heatmap_symbols)

    print("plotting figures")
    figure_paths = [
        plot_representative_curves(reps, daily_close, daily_volume, daily_logret, args, figures_dir),
        plot_distributions(arrays, figures_dir),
        plot_logret_diagnostics(arrays["close_log_return"], figures_dir),
        plot_cross_section(cs_daily, figures_dir),
        plot_ticker_stats(stats_df, figures_dir),
        plot_heatmaps(heatmap_symbols, daily_logret_h, rolling_vol, args, figures_dir),
    ]

    print("building ticker book")
    book_symbols = reps if args.ticker_book == "selected" else columns.symbols
    ticker_book_path = generate_ticker_book(args, book_symbols, out_dir)

    time_index = daily_close.index.dropna()
    summary: dict[str, object] = {
        "num_tickers": len(columns.symbols),
        "num_timestamps": total_rows,
        "time_start": str(time_index.min()) if len(time_index) else None,
        "time_end": str(time_index.max()) if len(time_index) else None,
        "representative_tickers": reps,
        "heatmap_tickers": heatmap_symbols,
        "distribution_stats": dist_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    pdf_path = out_dir / "modern_tsf_visual_data_analysis.pdf"
    print("writing PDF")
    write_pdf_report(args, pdf_path, figure_paths, summary, ticker_book_path)

    manifest = {
        "pdf": str(pdf_path),
        "ticker_book": str(ticker_book_path) if ticker_book_path else None,
        "figures": [str(p) for p in figure_paths],
        "tables": [
            str(tables_dir / "per_ticker_stats.csv"),
            str(tables_dir / "cross_section_daily.csv"),
            str(tables_dir / "distribution_stats.csv"),
        ],
        "summary": str(out_dir / "summary.json"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("REPORT", pdf_path)
    if ticker_book_path:
        print("TICKER_BOOK", ticker_book_path)
    print("OUT", out_dir)


if __name__ == "__main__":
    main()
