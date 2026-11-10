---
name: quant-research
description: Conduct auditable quantitative finance research on market data, especially return and volatility forecasting. Use for financial time-series analysis, leakage-safe backtests, HAR/ARCH/ML forecasting, regime analysis, forecast evaluation, and reproducible empirical reports; not for live trading or investment advice.
---

# Quant Research

Produce evidence that another researcher can reproduce and criticize. Treat data provenance, time ordering, and out-of-sample discipline as correctness requirements, not presentation details.

## Establish the research contract

Before training a model, state:

- hypothesis and decision the result should inform;
- target variable, units, frequency, and exact timestamp at which it becomes known;
- information set available at each forecast origin;
- observation period, missing-data treatment, and inclusion rules;
- benchmark, alternative models, and validation design;
- loss function and inferential test.

When the user asks for a complete study, implement and run it. Do not return only tutorials or illustrative pseudo-results.

## Apply non-negotiable gates

1. **Use real, documented data.** Record provider, endpoint or file, retrieval time, date range, symbol, adjustments, row count, and checksum. Mark synthetic data as test-only. Never invent missing observations, prices, bar data, significance, or performance.
2. **Respect the clock.** At origin `t`, features may use information available by `t`; the target must occur strictly after `t`. For supervised rows, the training target must already be realized before fitting a forecast at the current origin.
3. **Never randomize time-series splits.** Use chronological holdout, rolling forecast, expanding/rolling walk-forward, or blocked time-series cross-validation. Tune hyperparameters and model weights only inside past data.
4. **Separate proxy from truth.** Do not call a daily squared return, Parkinson, Garman-Klass, or range estimate high-frequency realized variance. Name the estimator and state its assumptions.
5. **Keep event labels diagnostic.** Crisis dates may define ex-post evaluation slices, but cannot inform features, online state, model selection, or weights before the event.
6. **Evaluate forecasts, not fitted values.** Metrics, tables, plots, and claims must use realized out-of-sample targets. Report all pre-specified models and unfavorable outcomes.
7. **Preserve reproducibility.** Record code version/checksum, config, random seed, feature order, hyperparameters, training dates, fit timestamp, package versions, output checksums, and the command used.
8. **Report uncertainty and failure.** Use effect sizes and dependence-aware tests where appropriate. A low loss does not prove economic value, causal mechanism, tradability, or direction prediction.

## Workflow

### 1. Audit the data

- Check schema, duplicates, monotonic timestamps, trading-calendar alignment, OHLC envelope, non-finite values, zero/negative volumes, stale prices, gaps, and outliers.
- Compare suspicious points with the provider or a second source when possible.
- Use causal imputation; avoid backfilling from future values. Prefer leaving an unavailable high-frequency target missing over substituting a different frequency without disclosure.
- Preserve raw data and write a manifest before transformations.

Read [references/data-quality.md](references/data-quality.md) for schemas, alignment rules, and leakage checks.

### 2. Diagnose before modeling

Quantify return distributions, volatility clustering, autocorrelation, stationarity, ARCH effects, tails, regime behavior, and structural instability. Diagnose both levels and transformed targets; transformation decisions must be justified by the forecast loss and model.

Read [references/financial-time-series.md](references/financial-time-series.md) for return/RV formulas, HAR/HAR-J, ADF/KPSS, ACF/Ljung-Box, ARCH-LM, break tests, and tail risk.

### 3. Build a model ladder

Start with a transparent baseline such as HAR-RV, then add economically motivated extensions and ML challengers. Choose model complexity from sample size, signal-to-noise, dimensionality, compute budget, and out-of-sample evidence.

- Linear models: interpretable HAR/AR baselines and low-dimensional extensions.
- Tree ensembles: useful nonlinear challengers when samples and features are moderate; use regularized XGBoost/LightGBM and causal tuning.
- Random forest: robust nonlinear benchmark, but less suitable for recentered rolling forecasts without frequent refits.
- LSTM/GRU/Transformer: only when there is enough history and repeated validation shows a material gain over strong simple baselines. Account for random seeds, early stopping, sequence construction, and leakage from normalization.

For volatility, target positive variance or log variance deliberately. If modeling log variance, convert back before QLIKE and variance-error metrics. Never compare metrics computed in different units.

### 4. Construct forecasts causally

- Define rolling training window and refit schedule.
- Tune on inner chronological folds or a prior validation block.
- Generate one-step or multi-step forecasts without resetting the information set.
- For regime-aware ensembles, optimize weights only from prior out-of-sample forecast losses, preferably by regime with minimum-sample safeguards.
- Use shrinkage, constraints, or fallback weights when history is sparse. Do not select the winning weight on the final test sample.
- Run threshold, window, weight, and loss sensitivity analyses.

Read [references/validation-and-evaluation.md](references/validation-and-evaluation.md) for walk-forward designs, regime detection, dynamic ensembles, metrics, DM, MCS, and conditional predictive ability.

### 5. Evaluate and test

At minimum report MAE, RMSE, QLIKE for variance targets, and an out-of-sample R-squared with an explicit denominator. Keep a common finite sample across models.

Use dependence-aware inference:

- Diebold-Mariano with Newey-West/HAC variance for unconditional loss differences;
- Giacomini-White for conditional predictive ability when state dependence is central;
- a documented Model Confidence Set procedure for multiple models;
- block bootstrap for serially dependent losses;
- multiple-testing caution across regimes, events, thresholds, and model variants.

Distinguish statistical significance from economic or operational value. Report confidence intervals or effect magnitudes when possible.

### 6. Deliver research artifacts

Deliver, as applicable:

- executable source code and automated tests;
- immutable raw data or a reproducible retrieval interface plus manifests;
- config and environment lock;
- all OOS predictions, metric tables, test results, model/refit logs, and failed experiments;
- a report with assumptions, methodology, results, limitations, and exact run instructions;
- a dashboard that reads persisted evaluated artifacts rather than embedding invented values.

Use [scripts/check_temporal_split.py](scripts/check_temporal_split.py) before backtests or after exporting forecasts:

```bash
python check_temporal_split.py forecasts.csv \
  --origin-column forecast_origin \
  --target-column target_date
```

Read [references/research-reporting.md](references/research-reporting.md) when producing a paper-style report, experiment registry, or model comparison.

## Decision rules

- Prefer the simplest model whose OOS gain is stable and defensible.
- If deep learning does not beat a strong HAR/ML baseline after repeated seeds and sensitivity analysis, omit it from the core claim.
- If results depend on an undisclosed change to sample, metric, or threshold, treat them as exploratory.
- If high-frequency history is unavailable, build a CSV import path and a clearly labeled daily baseline. Say what cannot be tested.
- If a bug or leakage is found, invalidate affected claims, fix the pipeline, rerun, and report the correction.
- If a user requests trade signals rather than research, clarify that this skill supports analysis and validation; it does not supply personalized investment advice or guarantee returns.
