# Data Quality and Leakage Reference

## Minimum data contract

For an OHLC daily dataset, require:

- unique, monotonic trading date;
- positive and finite open/high/low/close;
- `high >= max(open, close, low)` and `low <= min(open, close, high)`;
- non-negative volume and amount;
- symbol, timezone, provider, adjustment convention, extraction timestamp, and digest.

For intraday data, additionally require:

- timestamp includes session/date and timezone;
- unique bar endpoint convention;
- expected bar count per session and missing-bar report;
- volume-zero handling by session and instrument;
- mapping from bar date to the official trading calendar.

## Alignment rules

- Use explicit as-of joins. The right-side observation must be available no later than the left-side origin.
- News, fundamentals, and official statistics need publication timestamps, not period labels.
- Corporate actions must be handled consistently across prices and volume.
- Market closures and half-days must use the target market calendar.
- Avoid forward-filling a target or a predictor that would not be known in real time.

## Leakage checklist

Reject the experiment if any of these occur:

1. Random `train_test_split` on a time series.
2. Full-sample normalization, imputation, feature selection, or PCA.
3. Target encoding or hyperparameter selection using future validation rows.
4. Rolling windows centered on the observation.
5. Regime thresholds calculated over the full sample.
6. Ensemble weights fitted on the same observations used to claim OOS gain.
7. Crisis/event labels entering an allegedly real-time model.
8. Features that summarize the next day, next week, or full-sample distribution.
9. Multiple predictions stitched from models selected after seeing the outcome.
10. Confusing target date, feature availability date, and trade execution date.

## Reproducibility manifest

Record source URL/endpoint or filename, source version, query parameters, download time, checksum, row count, date range, schema, transformations, missingness, outliers, calendar, code checksum/version, random seed, dependency versions, and output checksums.

Keep a small test fixture for automated tests, but label it synthetic and exclude it from empirical claims.

