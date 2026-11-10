"""Causal rolling-quantile market state detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_LABELS = {0: "低波动", 1: "正常波动", 2: "高波动"}


def assign_volatility_regimes(
    frame: pd.DataFrame,
    window: int = 252,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
    confirm_days: int = 3,
    signal_column: str = "rv_w",
) -> pd.DataFrame:
    """Assign low/normal/high states without using future observations.

    Thresholds use the prior ``window`` observations, excluding today, while
    today's known volatility signal is classified. A confirmation filter then
    requires ``confirm_days`` consecutive observations before changing state.
    """

    if not 0 < low_quantile < high_quantile < 1:
        raise ValueError("Quantiles must satisfy 0 < low < high < 1")
    if confirm_days < 1:
        raise ValueError("confirm_days must be at least one")
    result = frame.copy()
    signal = result[signal_column]
    prior = signal.shift(1)
    result["regime_low_threshold"] = prior.rolling(
        window, min_periods=window
    ).quantile(low_quantile)
    result["regime_high_threshold"] = prior.rolling(
        window, min_periods=window
    ).quantile(high_quantile)

    raw = pd.Series(1, index=result.index, dtype="int64")
    valid = result["regime_low_threshold"].notna() & result[
        "regime_high_threshold"
    ].notna()
    raw.loc[valid & (signal <= result["regime_low_threshold"])] = 0
    raw.loc[valid & (signal >= result["regime_high_threshold"])] = 2
    raw.loc[~valid] = 1

    filtered = np.ones(len(result), dtype=int)
    if len(result):
        previous = int(raw.iloc[0])
        pending = previous
        pending_count = 0
        for index in range(len(result)):
            candidate = int(raw.iloc[index])
            if candidate == previous:
                pending = previous
                pending_count = 0
            else:
                if candidate == pending:
                    pending_count += 1
                else:
                    pending = candidate
                    pending_count = 1
                if pending_count >= confirm_days:
                    previous = candidate
                    pending_count = 0
            filtered[index] = previous

    result["regime_raw"] = raw
    result["regime"] = filtered
    result["regime_label"] = result["regime"].map(REGIME_LABELS)
    result["regime_transition"] = result["regime"].ne(result["regime"].shift(1)).astype(int)
    return result


def regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize sample counts and switching frequency by regime."""

    grouped = frame.groupby("regime", sort=True)
    summary = grouped.agg(
        days=("date", "size"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        mean_rv_w=("rv_w", "mean"),
        transitions=("regime_transition", "sum"),
    ).reset_index()
    summary["regime_label"] = summary["regime"].map(REGIME_LABELS)
    return summary

