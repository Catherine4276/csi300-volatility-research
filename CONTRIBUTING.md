# Contributing

Contributions are welcome. This is a research project: correctness, provenance,
and out-of-sample discipline take priority over leaderboard performance.

## Before opening a pull request

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Run the full suite:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest -q
   ```

4. If forecasting logic changed, rerun the full research pipeline and include
   updated reports and experiment artifacts.
5. Describe data assumptions, evaluation changes, and negative results.

## Research integrity

- Do not shuffle time-series observations for train/test splitting.
- Do not use future targets, full-sample normalization, or future regime
  thresholds.
- Do not tune weights or hyperparameters on the final test sample.
- Do not hide failed models or unfavorable results.
- Clearly distinguish variance proxies, high-frequency realized variance, and
  annualized volatility.
- Volatility-forecast accuracy is not price-direction accuracy or investment
  performance.

## Data and credentials

- Never commit API keys, tokens, `.streamlit/secrets.toml`, or personal data.
- Raw provider data is excluded from Git. Download it with
  `scripts/prepare_data.py`.
- Do not add data files whose license does not permit public redistribution.

## License

By contributing, you agree that your contribution is licensed under the
project's MIT License.

