from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import (
    forecast_metrics,
    giacomini_white_test,
    newey_west_dm,
    qlike_loss,
)
from src.feature_engineering import build_features
from src.regime_detection import assign_volatility_regimes


def test_regime_filter_is_causal(synthetic_features, synthetic_daily):
    changed_daily = synthetic_daily.copy()
    future = changed_daily["date"] > changed_daily["date"].iloc[380]
    for column in ("open", "high", "low", "close"):
        changed_daily.loc[future, column] *= 2.0
    changed_features = build_features(changed_daily, vol_window=252)
    changed = assign_volatility_regimes(changed_features)
    cutoff = synthetic_daily["date"].iloc[380]
    expected = synthetic_features.loc[synthetic_features["date"] <= cutoff, "regime"].to_numpy()
    actual = changed.loc[changed["date"] <= cutoff, "regime"].to_numpy()
    np.testing.assert_array_equal(actual, expected)
    assert set(np.unique(actual)).issubset({0, 1, 2})


def test_qlike_and_metrics_have_expected_values():
    actual = np.array([1.0, 1.0])
    prediction = np.array([1.0, 2.0])
    losses = qlike_loss(actual, prediction)
    assert losses[0] == 0.0
    np.testing.assert_allclose(losses[1], 2.0 - np.log(2.0) - 1.0)
    metrics = forecast_metrics(actual, prediction)
    assert metrics["n"] == 2
    assert metrics["mae"] == 0.5
    assert metrics["rmse"] == np.sqrt(0.5)


def test_dm_returns_finite_result_for_dependent_losses():
    rng = np.random.default_rng(4)
    benchmark = rng.gamma(2.0, 0.2, 500)
    model = benchmark + rng.normal(0.02, 0.01, 500)
    result = newey_west_dm(model, benchmark, max_lags=10)
    assert np.isfinite(result["dm_stat"])
    assert 0 <= result["dm_pvalue"] <= 1


def test_giacomini_white_uses_origin_instruments():
    rng = np.random.default_rng(11)
    benchmark = rng.gamma(2.0, 0.2, 400)
    regime = rng.integers(0, 3, 400)
    model = benchmark + rng.normal(0.02, 0.01, 400)
    instruments = pd.DataFrame(
        {
            "state_high": (regime == 2).astype(float),
            "state_low": (regime == 0).astype(float),
        }
    )
    result = giacomini_white_test(model, benchmark, instruments, max_lags=12)
    assert np.isfinite(result["gw_stat"])
    assert 0 <= result["gw_pvalue"] <= 1
    assert result["gw_df"] == 3

