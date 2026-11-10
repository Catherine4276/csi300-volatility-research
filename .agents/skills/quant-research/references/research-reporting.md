# Research Reporting Reference

Write a report that lets a reader audit the claim without rerunning every notebook.

## Required sections

1. Research question and pre-specified hypotheses.
2. Data source, provenance, schema, period, frequency, adjustments, and limitations.
3. Exact target, units, availability time, and forecast horizon.
4. Feature information set and leakage defenses.
5. Benchmarks, models, validation design, tuning, and refit schedule.
6. Metrics and statistical tests, including assumptions and lags.
7. Full OOS results, event/regime slices, sensitivity results, and failures.
8. Current state or live prediction only if the same causal pipeline can generate it.
9. Limitations, threats to validity, and claims that are not supported.
10. Reproduction commands and artifact inventory.

## Result discipline

- Show `n` beside every metric.
- Use the same evaluation sample across model comparisons where possible.
- Distinguish annualized volatility from daily variance.
- State if an event window was selected ex post.
- Treat accuracy of volatility forecasts differently from accuracy of price direction.
- Avoid causal language absent a credible identification design.
- Avoid "profitable," "safe," or "best" unless the relevant economic and statistical tests are specified.

## Experiment registry

For each run store a stable run ID, code checksum, config checksum, data checksums, package lock, random seed, start/end timestamps, hardware/runtime where relevant, status, OOS predictions, metric tables, and errors. Never overwrite a reported run without retaining a trace.

## Dashboard

The dashboard should display persisted evaluated artifacts and clearly place its information cutoff in the interface. Show prediction horizon, units, target definition, risk-state algorithm, historical performance, and data limitations. Do not place invented future candles, point predictions without uncertainty, or performance values hardcoded in UI code.

