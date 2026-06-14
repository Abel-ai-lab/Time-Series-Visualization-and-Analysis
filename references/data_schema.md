# Data Schema

The visualizer expects a wide time-series CSV.

Required columns:

- One timestamp column, default `date`.
- One raw close/value column per series, default suffix `_close`.
- One raw volume/scale column per series, default suffix `_volume`.

Example:

```text
date,AAPL_close,AAPL_volume,MSFT_close,MSFT_volume
2024-01-02 14:30:00+00:00,185.5,8123456,374.1,6234567
```

Optional log-return CSV:

- Same timestamp column.
- Same `*_close` columns, interpreted as close log returns.
- Same `*_volume` columns, interpreted as log volume changes.

If no log-return CSV is supplied, close log returns are derived with `diff(log(close))`, and volume changes are derived with `diff(log1p(volume))`.

For very wide data, sample rows for empirical distributions but compute per-series and cross-sectional summaries with chunked passes.
