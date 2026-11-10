from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_daily_csv, validate_daily_frame
from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_features,
    calculate_intraday_realized_variance,
)


def test_csv_loading_and_validation(tmp_path, synthetic_daily):
    path = tmp_path / "daily.csv"
    synthetic_daily.to_csv(path, index=False)
    loaded = load_daily_csv(path)
    validate_daily_frame(loaded)
    assert len(loaded) == len(synthetic_daily)
    assert loaded["date"].is_monotonic_increasing


def test_invalid_ohlc_is_rejected(synthetic_daily):
    bad = synthetic_daily.copy()
    bad.loc[10, "high"] = bad.loc[10, "low"] - 1.0
    with pytest.raises(ValueError, match="Invalid OHLC"):
        validate_daily_frame(bad)


def test_intraday_rv_uses_within_day_squared_returns():
    rows = []
    for day, seed_prices in (
        ("2024-01-02", [100.0, 101.0, 100.0]),
        ("2024-01-03", [100.0, 99.0, 99.5]),
    ):
        prices = seed_prices + [99.5] * 17
        session_start = pd.Timestamp(f"{day} 09:35:00")
        for minute, price in enumerate(prices):
            rows.append(
                {
                    "date": session_start + pd.Timedelta(minutes=5 * minute),
                    "close": price,
                }
            )
    result = calculate_intraday_realized_variance(pd.DataFrame(rows))
    assert "2024-01-03" in result["date"].dt.date.astype(str).tolist()
    row = result.loc[result["date"].dt.strftime("%Y-%m-%d").eq("2024-01-03")].iloc[0]
    expected = np.log(99.0 / 100.0) ** 2 + np.log(99.5 / 99.0) ** 2
    assert row["intraday_rv"] == pytest.approx(expected)


def test_features_are_causal_for_all_existing_information(synthetic_daily):
    original = build_features(synthetic_daily, intraday_rv=None, vol_window=252)
    modified_daily = synthetic_daily.copy()
    future_mask = modified_daily["date"] > modified_daily["date"].iloc[320]
    for column in ("open", "high", "low", "close"):
        modified_daily.loc[future_mask, column] *= 1.7
    modified = build_features(modified_daily, intraday_rv=None, vol_window=252)
    cutoff = synthetic_daily["date"].iloc[320]
    left = original.loc[original["date"] <= cutoff, FEATURE_COLUMNS]
    right = modified.loc[modified["date"] <= cutoff, FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(left, right)


def test_target_is_strictly_next_trading_day(synthetic_features):
    complete = synthetic_features.dropna(subset=["target_date", "rv_target"])
    assert (complete["target_date"] > complete["forecast_origin"]).all()
    assert complete["target_date"].is_monotonic_increasing

