# Zero-Baseline Loss Landscape Workflow

This workflow studies whether near-zero forecasts are a strong baseline under different loss functions and similarity metrics.

Core ideas:

1. Work ticker-wise instead of pooling every series together.
2. Build a discrete candidate prediction space from ticker-specific value ranges.
3. Define epsilon baselines directly from the bucket ladder around zero.
4. Measure where each epsilon baseline ranks inside the candidate prediction space.
5. Compare whether metrics agree on what counts as a good prediction.

The current finance1000 implementation supports:

- full-history min/max bucket ranges;
- q05/q95 clipped bucket ranges;
- exact 11-bucket enumeration for short windows;
- multi-GPU sharding on shared servers.

The `100 ticker q05/q95` showcase report is included under:

- `reports/zero_baseline_q05q95_100tickers/`

The full merged `1000 ticker q05/q95` report is included under:

- `reports/zero_baseline_q05q95_1000tickers/`

Included workflow entrypoints in this repo:

- `scripts/loss_landscape/run_q05q95_exact11.py`
- `scripts/loss_landscape/build_q05q95_intro_report.py`

The current repository snapshot packages the finance1000-specific workflow and example report
artifacts first. A more generic adapter layer can be added on top of the same structure later.

Recommended execution pattern on Pro6000:

1. split tickers into shards;
2. run one shard per GPU with explicit `CUDA_VISIBLE_DEVICES`;
3. merge shard outputs;
4. build the intro/report HTML from merged outputs.

## Example: Finance1000 q05/q95 exact-11

The finance1000 experiment used:

- ticker selection by `close_logret_std`;
- ticker-wise `q05/q95` value ranges;
- zero-centered `11`-bucket ladders;
- non-overlapping `7`-step weekly windows;
- metrics:
  - `MAE`
  - `MSE`
  - `RMSE`
  - `Direction Accuracy`
  - `Pearson / IC`
  - `Spearman`
  - `Cosine Similarity`
  - `Std Gap`
  - `L1 Divergence`
  - `DWT Distance`
  - `KL Divergence`
  - `Wasserstein`

Conceptually, one shard can be run like this:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/loss_landscape/run_q05q95_exact11.py
```

Then merge and build the intro report:

```bash
python scripts/loss_landscape/build_q05q95_intro_report.py
```
