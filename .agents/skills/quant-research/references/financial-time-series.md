# Financial Time-Series Reference

Use formulas only after fixing the observation frequency, price convention, and timestamp. State whether returns are simple or log and whether volatility is in daily variance, daily standard deviation, or annualized terms.

## Returns and realized variance

Simple return:

`r_t = P_t / P_{t-1} - 1`

Log return:

`r_t = log(P_t / P_{t-1})`

Intraday realized variance from `m` within-day returns:

`RV_t = sum_{j=1..m} r_{t,j}^2`

Exclude overnight returns unless the target explicitly includes them. Verify that the intraday sequence has no missing blocks, duplicate bars, crossed markets, or stale zero returns. Document bar frequency, session boundary, annualization, and minimum-bar rule.

If only OHLC daily data are available:

- close-to-close variance proxy: `r_t^2`;
- Parkinson: `log(H/L)^2 / (4 log 2)`;
- Garman-Klass: `0.5 log(H/L)^2 - (2 log 2 - 1) log(C/O)^2`.

These are range-based proxies, not high-frequency RV. They can be negative for some OHLC patterns depending on estimator and must be handled explicitly.

## HAR family

Standard HAR-RV:

`RV_{t+1} = beta_0 + beta_d RV_t + beta_w mean(RV_{t-4:t}) + beta_m mean(RV_{t-21:t}) + e_{t+1}`

Use only observations known at origin `t`. Estimate OLS with robust/HAC inference if residuals are autocorrelated or heteroskedastic; prediction should still protect the positive support of variance.

HAR-J separates continuous variation from jumps when a defensible jump measure is available:

`RV_{t+1} = beta_0 + beta_d C_t + beta_w C^w_t + beta_m C^m_t + beta_j J_t + e_{t+1}`

Do not manufacture a jump component from an arbitrary residual threshold without documenting it and validating sensitivity.

## Diagnostics

- ACF/PACF: dependence in returns and squared returns; report lags and sample.
- Ljung-Box: joint serial-correlation test; use on squared returns for volatility dependence.
- ADF: unit-root null under the chosen deterministic terms.
- KPSS: stationarity null under the chosen deterministic terms.
- ARCH-LM: test lagged squared-residual dependence; report lags and small-sample caveats.
- Rolling quantiles/regimes: use prior observations only, minimum history, and a confirmation or smoothing rule to limit noise-driven switches.
- Structural breaks: consider Quandt-Andrews, Bai-Perron, CUSUM, or rolling coefficient stability. Report searched break dates and avoid treating an in-sample break search as a real-time signal.
- Extreme risk: report tail quantiles, expected shortfall, and drawdown/event windows. Use EVT only with enough exceedances and explicit threshold diagnostics.

## Model-target cautions

Variance is positive, skewed, and persistent. Log transforms can stabilize the conditional mean but change the implied loss. When predicting `log(RV)`, exponentiate before calculating variance-space QLIKE, RMSE, and MAE. Bias correction for log transforms must be validated OOS.

HAR persistence can dominate random forests and neural networks. A complex model must beat HAR on a meaningful horizon and not only on a few crisis days.

