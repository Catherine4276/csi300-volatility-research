# Validation and Evaluation Reference

## Validation designs

- Rolling walk-forward: fixed or expanding training window; forecast a future block; move origin forward; never reuse future blocks for fitting.
- Expanding walk-forward: preserve all past data; good for stable regimes and long history.
- Purging/embargo: remove training labels whose outcome horizon overlaps validation; use a gap when features are timestamped near boundaries.
- Nested time-series tuning: outer test blocks remain untouched; inner chronological folds choose hyperparameters.
- Multi-step forecasting: recursively generated predictions accumulate error; direct and recursive strategies should be stated.

Refit frequency is a design choice. Record it and test sensitivity if compute allows. Daily weight adaptation does not require daily model refitting, but weights must use only previously realized OOS losses.

## Regimes and dynamic ensembles

Use causal rolling quantiles, Markov switching fitted only on past windows, or economic state variables. Prevent rapid flips with a confirmation period, smoothing, hysteresis, or minimum dwell time; disclose the rule.

For two forecasters `f_A`, `f_B`, a constrained blend is:

`f = w f_A + (1-w) f_B`, with `0 <= w <= 1`.

Choose `w_t` from prior OOS losses:

`w_A,t = L_B,t-1 / (L_A,t-1 + L_B,t-1)`

then shrink toward a neutral/prior weight and enforce bounds. Use regime-specific history only when each state has enough observations. Include a static-weight control.

If theory motivates a directional regime rule, encode it transparently as a
bound or prior while still letting past OOS loss determine the magnitude. Test
the constrained rule against the unconstrained and static controls; report if
the constraint worsens predictive accuracy.

## Metrics

Let `sigma2_hat` be predicted variance, `sigma2` realized variance, and `e = sigma2_hat - sigma2`.

- MAE: `mean(abs(e))`
- RMSE: `sqrt(mean(e^2))`
- QLIKE: `mean(sigma2_hat/sigma2 - log(sigma2_hat/sigma2) - 1)`
- OOS R-squared: `1 - sum(e^2) / sum((sigma2 - mean(sigma2))^2)`

QLIKE is asymmetric and sensitive to low forecasts. Use a common support floor and disclose it. Do not compare a variance QLIKE to a volatility QLIKE.

## Statistical tests

For losses `L_A,t` and `L_B,t`, set:

`d_t = L_A,t - L_B,t`

Use a Newey-West/HAC long-run variance for the Diebold-Mariano test. A positive statistic means A has higher loss under the chosen difference order. Report the lag rule, lags used, alternative, and finite sample.

Model Confidence Set:

- define the loss matrix on a common finite sample;
- use a stationary or circular block bootstrap when losses are dependent;
- test pairwise loss-difference ranges and eliminate the worst model iteratively;
- report the implementation and p-values, especially when claiming an exact published MCS procedure.

Giacomini-White conditional predictive ability is relevant when model ranking may depend on observed states. Regress loss differences on past state instruments and use robust inference. A regime-by-regime comparison is descriptive; it is not a substitute for the conditional test.

## Sensitivity and robustness

Pre-specify and report:

- train window, refit frequency, and OOS start;
- state window, thresholds, smoothing/confirmation;
- ensemble lookback, shrinkage, bounds, and loss;
- target proxy/fallback and support floor;
- event windows and exclusion rules;
- random seeds for stochastic ML.

A positive main result is fragile if sign, ranking, or magnitude reverses under reasonable choices. Do not search a large grid and report only the best combination.

