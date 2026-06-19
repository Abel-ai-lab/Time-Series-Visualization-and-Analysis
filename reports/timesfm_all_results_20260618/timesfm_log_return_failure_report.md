# TimesFM Finance1000 Log-Return Failure Report

Created from `reports/timesfm_all_results_20260618` on 2026-06-19.

## Cut

TimesFM does not learn useful close log-return signal in this experiment. Across 5,757 windows, 1,000 tickers, and 5,757,000 predictions per scale setting, the zero-shot close-return forecast is worse than predicting exactly zero, and full fine-tuning only produces a tiny RMSE gain while shrinking the predicted magnitudes even closer to zero.

The deck version is:

> For price log returns, the foundation model learns the safest answer: almost zero. That is not alpha. It is the baseline.

The attached paper, `Re(Visiting) Time Series Foundation Models in Finance.pdf`, uses out-of-sample R2 against a zero forecast. In that setup, zero is not a weak straw-man baseline. It is the finance baseline because individual stock returns are noisy and centered near zero; historical means can make weak models look better than they are. Our local TimesFM results reproduce the same failure mode on Finance1000 close log returns.

Important provenance note: the result bundle does not include a separate "all-zero baseline" run or row. The exact-zero baseline below is derived from the true-return columns already present in the result CSVs, using the paper's zero-forecast evaluation definition.

## Source Artifacts

- Result directory: `reports/timesfm_all_results_20260618`
- Main diagnostic table: `data/diagnostic_summary_unified_logret.csv`
- Zero-shot vs full fine-tune table: `data/zero_shot_vs_full_finetune_metrics.csv`
- Reference paper: `Re(Visiting) Time Series Foundation Models in Finance.pdf`
- Evaluation size for each zero-shot experiment: 5,757 windows x 1,000 tickers = 5,757,000 predictions
- Exact-zero baseline: derived, not a separate experiment artifact

## Baseline Definition

For the close log-return target, the relevant zero predictor is:

```text
y_hat = 0
MAE_zero = mean(abs(y_true))
RMSE_zero = sqrt(mean(y_true^2))
ROOS = 1 - RMSE_model^2 / RMSE_zero^2
```

This matches the attached paper's finance evaluation logic: out-of-sample R2 is measured against a zero forecast. If `ROOS < 0`, the model is worse than predicting zero.

This baseline is computed from the CSV columns `true_mean`, `true_std`, and `true_abs_mean`. It was not copied from a pre-existing baseline row.

For our close log-return data:

- True mean: -0.00002836
- True absolute mean: 0.00538284
- True standard deviation: 0.01686633
- Zero-predictor MAE: 0.00538284
- Zero-predictor RMSE: 0.01686635

The target itself is centered almost exactly at zero. A model that cannot discover stable directional or cross-sectional structure will minimize loss by collapsing toward zero.

## Headline Result: Close Log Returns

| Experiment | Mode | MAE | MAE vs zero | RMSE | ROOS vs zero | Nonzero direction | Flatten IC | Cross-section IC | Pred abs / true abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close_logret_x1 | zero-shot | 0.00567798 | +5.48% worse | 0.01713607 | -3.22% | 49.16% | -0.0303 | -0.0343 | 19.6% |
| close_logret_x1 | full fine-tune | 0.00552419 | +2.63% worse | 0.01681468 | +0.61% | 51.44% | 0.0809 | 0.1080 | 14.3% |

Interpretation:

- Zero-shot TimesFM is worse than exact zero on both MAE and RMSE.
- Fine-tuning does not solve the return problem. MAE is still worse than zero by 2.63%.
- The fine-tuned RMSE gain is only +0.61% ROOS, which is too small to support a strong predictive claim.
- Fine-tuning raises IC from negative to slightly positive, but at the cost of even more magnitude collapse: predicted absolute returns fall from 19.6% of true absolute returns to 14.3%.
- Nonzero direction accuracy improves from 49.16% to 51.44%, which is barely above coin flip and not enough to justify a foundation-model story.

## The Strongest Failure: Prediction Collapse Toward Zero

The most important diagnostic is not just low accuracy. It is scale collapse.

| Diagnostic | True close log return | Zero-shot TimesFM | Ratio |
|---|---:|---:|---:|
| Mean | -0.00002836 | -0.00005998 | 2.12x mean, still near zero |
| Standard deviation | 0.01686633 | 0.00256045 | 15.2% of true scale |
| Mean absolute value | 0.00538284 | 0.00105318 | 19.6% of true scale |
| Near-zero rate, abs < 1e-4 | 37.92% | 23.11% | Different shape, still tiny amplitude |

The model is not producing exactly zero for every row. The stronger, more defensible claim is:

> TimesFM outputs a narrow, low-amplitude band around zero, and that band is worse than just predicting exactly zero for close returns.

For deck language:

> The model does not forecast returns. It compresses them.

## Scale Test: x1, x100, x500 All Fail the Same Way

The close log-return target was tested at three scales and then converted back into the same unscaled evaluation space.

| Experiment | Conversion | MAE | RMSE | ROOS vs zero | Nonzero direction | Flatten IC | Cross-section IC | Pred std / true std | Pred abs / true abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| close_logret_x1 | pred, true | 0.00567798 | 0.01713607 | -3.22% | 49.16% | -0.0303 | -0.0343 | 15.2% | 19.6% |
| close_logret_x100 | pred / 100, true / 100 | 0.00567798 | 0.01713606 | -3.22% | 49.16% | -0.0303 | -0.0343 | 15.2% | 19.6% |
| close_logret_x500 | pred / 500, true / 500 | 0.00567798 | 0.01713607 | -3.22% | 49.16% | -0.0303 | -0.0343 | 15.2% | 19.6% |

The metric ranges across x1, x100, and x500 are effectively zero:

- MAE range: 0.000000000063
- RMSE range: 0.000000005344
- Nonzero direction range: 0.000000833
- Predicted absolute magnitude ratio range: 0.0000000106

This is the key experimental control. Scaling the return target by 100x or 500x does not uncover hidden signal. After conversion back to the real evaluation space, TimesFM returns the same failed forecast.

## Zero-Shot Details

| Experiment | Target space | MAE | RMSE | Direction accuracy | Nonzero direction | Flatten IC | Cross-section IC | Positive CS IC rate | Pred abs / true abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| close_logret_x1 | close_log_return_unscaled | 0.00567798 | 0.01713607 | 33.42% | 49.16% | -0.0303 | -0.0343 | 35.68% | 19.6% |
| close_logret_x100 | close_log_return_unscaled | 0.00567798 | 0.01713606 | 33.42% | 49.16% | -0.0303 | -0.0343 | 35.68% | 19.6% |
| close_logret_x500 | close_log_return_unscaled | 0.00567798 | 0.01713607 | 33.42% | 49.16% | -0.0303 | -0.0343 | 35.68% | 19.6% |

What this means:

- MAE and RMSE do not beat zero.
- Direction on nonzero moves is below 50%.
- Flatten IC is negative.
- Cross-section IC is negative.
- Only 35.68% of daily cross-sections have positive IC.
- The model outputs only one-fifth of the true absolute move size.

This is a clean failure across every close-return view we have.

## Fine-Tuning Does Not Rescue Close Returns

Fine-tuning improves some ranking diagnostics, but the economic/statistical story is still weak.

| Metric | Zero-shot close log return | Full fine-tune close log return | Change |
|---|---:|---:|---:|
| MAE | 0.00567798 | 0.00552419 | -2.71% |
| RMSE | 0.01713607 | 0.01681468 | -1.88% |
| ROOS vs zero | -3.22% | +0.61% | +3.84 pp |
| Nonzero direction | 49.16% | 51.44% | +2.28 pp |
| Flatten IC | -0.0303 | 0.0809 | +0.1112 |
| Cross-section IC | -0.0343 | 0.1080 | +0.1423 |
| Pred abs / true abs | 19.6% | 14.3% | -5.22 pp |

The fine-tuned model looks slightly less bad on RMSE and IC, but it still does not produce realistic returns. Its average absolute prediction is only 0.000772, while the average absolute realized close return is 0.005383. That is 14.3% of the true absolute return scale.

For deck language:

> Fine-tuning raises the score a little by becoming even more conservative. It predicts smaller numbers, not better causes.

## Why Volume Looks Better But Does Not Save the Return Result

The result bundle also includes volume log-return experiments. These are useful as a sanity check because they show the pipeline can find signal when the target has smoother, more persistent dynamics. But volume predictability is not price-return alpha.

| Experiment | Mode | MAE vs zero | ROOS vs zero | Nonzero direction | Flatten IC | Cross-section IC | Pred abs / true abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| volume_logret | zero-shot | -0.56% better | +5.85% | 62.77% | 0.2422 | 0.1552 | 26.9% |
| volume_logret | full fine-tune | -18.40% better | +34.09% | 74.53% | 0.5849 | 0.5475 | 65.6% |
| raw_volume -> volume_log_return | zero-shot | -6.80% better | +22.90% | 70.65% | 0.4940 | 0.5168 | 61.8% |
| raw_volume -> volume_log_return | full fine-tune | -17.75% better | +33.75% | 74.92% | 0.5854 | 0.5685 | 70.6% |

This contrast is useful:

- TimesFM can model smoother quantity dynamics such as volume.
- The same setup fails on close log returns.
- Therefore the close-return failure is not only a broken evaluation pipeline. It is the target: price returns are noisy, causally conditional, and weakly identified from univariate history.

## Connection To The Attached Paper

The attached paper supports three claims that matter for this deck:

1. Finance return forecasting should be evaluated against zero, not against a noisy historical mean.
2. Off-the-shelf TSFMs, including TimesFM, underperform specialized finance benchmarks in zero-shot daily return forecasting.
3. Fine-tuning helps only partially and does not close the gap to finance-specific benchmark models or portfolio performance.

Relevant paper anchors:

- Section 3.3.1 defines out-of-sample R2 against a zero prediction.
- The paper explains that zero is a stricter and economically meaningful baseline for individual stock returns because historical means are noisy.
- The paper's numerical summary reports TimesFM 500M at roughly -2.80% out-of-sample R2 with directional accuracy just below 50% in the zero-shot setting.
- The paper reports that fine-tuned TSFMs still have weak out-of-sample fit and that portfolio performance often deteriorates after fine-tuning.

Our local result is directionally consistent but even cleaner for the deck because we can show the actual collapse:

- Zero-shot close log-return ROOS: -3.22%
- Predicted close-return absolute scale: 19.6% of true
- Fine-tuned predicted close-return absolute scale: 14.3% of true
- Scale tests x1/x100/x500: identical failure after rescaling

## Why This Supports Causal Learning

A foundation model trained as a generic sequence forecaster is rewarded for fitting the marginal distribution of returns. In high-noise financial returns, the marginal mean is near zero, and stable predictive structure is conditional:

- market regime
- liquidity and volume shocks
- sector and factor exposures
- earnings and event timing
- macro policy and rates
- order-flow and positioning
- causal links between exogenous drivers and price response

Without those causal variables and mechanisms, the model sees a return series whose best unconditional forecast is nearly zero. The failure mode is exactly what we observe: it compresses the forecast toward zero and fails to rank or size price moves reliably.

Causal learning is necessary because the question is not:

> What comes next in this univariate sequence?

The finance question is:

> Which causal state changed, which assets are exposed, and how should that change propagate into returns?

## Deck-Ready Slide Copy

### Slide Title

Foundation models collapse to zero on log returns

### Main bullets

- Tested TimesFM on 5.76M close log-return predictions per scale setting.
- Zero-shot TimesFM is worse than exact zero: MAE +5.48%, RMSE +1.60%, ROOS -3.22%.
- Direction is not useful: 49.16% on nonzero moves, negative flatten IC (-0.0303), negative cross-section IC (-0.0343).
- Predicted return magnitude collapses: only 19.6% of true absolute returns and 15.2% of true volatility.
- Scaling returns by 100x or 500x changes nothing after rescaling; x1/x100/x500 all produce the same failed metrics.
- Fine-tuning does not solve it: MAE still worse than zero, ROOS only +0.61%, and predicted magnitude shrinks further to 14.3% of true.
- Volume is learnable; price returns are not, unless the model sees causal drivers.

### One-line takeaway

> A generic time-series foundation model does not learn financial return causality. It learns the zero baseline.

## Recommended Use In The Causal Learning Deck

Use this report as the empirical bridge:

1. Show the paper's baseline logic: zero is the serious benchmark for noisy stock returns.
2. Show our TimesFM result: zero-shot close returns lose to zero across 5.76M predictions.
3. Show the collapse diagnostic: predictions are only 19.6% of true absolute return scale; fine-tuning shrinks them to 14.3%.
4. Show the control: volume works better, so the pipeline is not simply broken.
5. Conclude: for alpha, sequence continuation is insufficient; we need causal state, causal features, and causal propagation.

## Most Important Numbers To Quote

| Claim | Number |
|---|---:|
| Predictions per close-return run | 5,757,000 |
| Zero-shot close-return MAE | 0.00567798 |
| Exact-zero MAE | 0.00538284 |
| Zero-shot MAE gap vs zero | +5.48% worse |
| Zero-shot close-return RMSE | 0.01713607 |
| Exact-zero RMSE | 0.01686635 |
| Zero-shot ROOS vs zero | -3.22% |
| Zero-shot nonzero direction accuracy | 49.16% |
| Zero-shot flatten IC | -0.0303 |
| Zero-shot cross-section IC | -0.0343 |
| Zero-shot predicted abs / true abs | 19.6% |
| Zero-shot predicted std / true std | 15.2% |
| Full fine-tune ROOS vs zero | +0.61% |
| Full fine-tune predicted abs / true abs | 14.3% |
| Close scale-test result | x1, x100, x500 are numerically identical after rescaling |

## Final Conclusion

The close log-return experiment is a strong negative example for generic foundation models in finance. TimesFM does not recover usable price-return signal from historical close returns. In zero-shot mode it is worse than the exact-zero baseline; after fine-tuning it becomes marginally less bad on RMSE but even more collapsed in magnitude.

This is precisely the opening for causal learning: the predictive content in finance is not sitting in the univariate return sequence waiting to be extrapolated. It is conditional on causal drivers, regimes, exposures, and shocks. The foundation model's collapse to near-zero is evidence that sequence modeling alone is the wrong abstraction for return prediction.
