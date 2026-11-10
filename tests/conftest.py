"""Deterministic synthetic fixtures for software tests only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_features
from src.regime_detection import assign_volatility_regimes


@pytest.fixture
def synthetic_daily() -> pd.DataFrame:
    """Create marked synthetic OHLCV data; never used for empirical results."""

    rng = np.random.default_rng(20261110)
    dates = pd.bdate_range("2020-01-02", periods=500)
    shocks = rng.normal(0.0, 0.008, len(dates))
    volatility = 0.008 + 0.7 * np.abs(np.roll(shocks, 1))
    returns = rng.normal(0.0001, volatility)
    close = 4000.0 * np.exp(np.cumsum(returns))
    prior_close = np.r_[4000.0, close[:-1]]
    open_price = prior_close * np.exp(rng.normal(0.0, 0.002, len(dates)))
    high = np.maximum(open_price, close) * np.exp(rng.uniform(0.0005, 0.009, len(dates)))
    low = np.minimum(open_price, close) * np.exp(-rng.uniform(0.0005, 0.009, len(dates)))
    volume = rng.lognormal(18.0, 0.3, len(dates))
    amount = volume * close
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": amount,
            "amplitude": (high - low) / prior_close,
            "change": close - prior_close,
            "pct_change": close / prior_close - 1.0,
            "turnover": rng.uniform(0.3, 1.0, len(dates)),
        }
    )


@pytest.fixture
def synthetic_features(synthetic_daily: pd.DataFrame) -> pd.DataFrame:
    features = build_features(synthetic_daily, intraday_rv=None, vol_window=252)
    return assign_volatility_regimes(
        features,
        window=252,
        low_quantile=0.25,
        high_quantile=0.75,
        confirm_days=3,
    )

