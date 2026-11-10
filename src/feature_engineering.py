"""Causal feature engineering for daily volatility forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


MIN_DAILY_VARIANCE = 1e-12


def garman_klass_variance(frame: pd.DataFrame) -> pd.Series:
    """Garman-Klass daily variance estimator from OHLC prices.

    This is a range-based variance proxy, not high-frequency realized variance.
    Values are in daily decimal-return variance units.
    """

    high_low = np.log(frame["high"] / frame["low"])
    close_open = np.log(frame["close"] / frame["open"])
    estimate = 0.5 * high_low.pow(2) - (2.0 * np.log(2.0) - 1.0) * close_open.pow(2)
    return estimate.clip(lower=MIN_DAILY_VARIANCE)


def parkinson_variance(frame: pd.DataFrame) -> pd.Series:
    """Parkinson high-low daily variance proxy."""

    high_low = np.log(frame["high"] / frame["low"])
    return (high_low.pow(2) / (4.0 * np.log(2.0))).clip(lower=MIN_DAILY_VARIANCE)


def calculate_intraday_realized_variance(intraday: pd.DataFrame) -> pd.DataFrame:
    """Compute daily RV by summing squared intraday log returns.

    Overnight returns are intentionally excluded. A day needs at least 20 bars
    so a sparse or partially downloaded session is not treated as a valid RV.
    """

    required = {"date", "close"}
    missing = sorted(required - set(intraday.columns))
    if missing:
        raise ValueError(f"Missing intraday columns: {missing}")
    frame = intraday.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date")
    frame["trading_day"] = frame["date"].dt.normalize()
    frame["intraday_log_return"] = np.log(frame["close"]).groupby(
        frame["trading_day"]
    ).diff()
    grouped = frame.groupby("trading_day", sort=True)
    result = pd.DataFrame(
        {
            "bar_count": grouped.size(),
            "intraday_rv": grouped["intraday_log_return"].apply(
                lambda values: float(np.nansum(np.square(values)))
            ),
        }
    ).reset_index()
    result = result.loc[result["bar_count"] >= 20].copy()
    result["intraday_rv"] = result["intraday_rv"].clip(lower=MIN_DAILY_VARIANCE)
    return result.rename(columns={"trading_day": "date"})


def _ensure_required_columns(daily: pd.DataFrame) -> None:
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"Missing daily columns: {missing}")


def build_features(
    daily: pd.DataFrame,
    intraday_rv: pd.DataFrame | None = None,
    vol_window: int = 252,
) -> pd.DataFrame:
    """Build only-past-available features and a strictly next-day target.

    At origin t, all features use information through the close of day t.
    The target is the variance proxy on trading day t+1. Rows retain both the
    origin and target dates so leakage checks can be performed downstream.
    """

    _ensure_required_columns(daily)
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    frame["simple_return"] = frame["close"].pct_change()
    frame["log_return"] = np.log(frame["close"]).diff()
    frame["return_variance"] = frame["log_return"].pow(2)
    frame["rv_garman_klass"] = garman_klass_variance(frame)
    frame["rv_parkinson"] = parkinson_variance(frame)

    # GK and close-to-close variance are transparent daily proxies. The system
    # never labels either one as high-frequency realized variance.
    frame["rv_proxy"] = frame["rv_garman_klass"].where(
        frame["rv_garman_klass"] > MIN_DAILY_VARIANCE,
        frame["return_variance"],
    )
    frame["rv_proxy"] = frame["rv_proxy"].clip(lower=MIN_DAILY_VARIANCE)
    frame["rv_d"] = frame["rv_proxy"]
    frame["rv_w"] = frame["rv_proxy"].rolling(5, min_periods=5).mean()
    frame["rv_m"] = frame["rv_proxy"].rolling(22, min_periods=22).mean()
    frame["rv_63"] = frame["rv_proxy"].rolling(63, min_periods=63).mean()
    frame["rv_acceleration"] = frame["rv_d"] / frame["rv_w"].replace(0, np.nan)
    frame["range_ratio"] = np.log(frame["high"] / frame["low"])
    frame["return_5"] = frame["log_return"].rolling(5, min_periods=5).sum()
    frame["return_22"] = frame["log_return"].rolling(22, min_periods=22).sum()

    # Rolling rank is causal: it includes today and earlier observations only.
    frame["historical_vol_percentile"] = frame["rv_w"].rolling(
        vol_window, min_periods=vol_window
    ).rank(pct=True)
    frame["rv_w_percentile"] = frame["rv_w"].rolling(
        vol_window, min_periods=vol_window
    ).apply(lambda values: float(np.mean(values <= values[-1])), raw=True)

    frame["volume_change"] = (
        frame["volume"].replace(0, np.nan).pct_change().replace([np.inf, -np.inf], np.nan)
    ).clip(-1.0, 5.0)
    frame["amount_change"] = (
        frame["amount"].replace(0, np.nan).pct_change().replace([np.inf, -np.inf], np.nan)
    ).clip(-1.0, 5.0)
    frame["volume_ratio_22"] = frame["volume"] / frame["volume"].rolling(
        22, min_periods=22
    ).median()
    frame["amount_ratio_22"] = frame["amount"] / frame["amount"].rolling(
        22, min_periods=22
    ).median()

    # Optional true intraday RV is kept in a distinctly named column.
    if intraday_rv is not None and not intraday_rv.empty:
        rv_lookup = intraday_rv.copy()
        rv_lookup["date"] = pd.to_datetime(rv_lookup["date"], errors="raise")
        rv_lookup = rv_lookup.drop_duplicates("date", keep="last")
        frame = frame.merge(
            rv_lookup.loc[:, ["date", "bar_count", "intraday_rv"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
    else:
        frame["bar_count"] = np.nan
        frame["intraday_rv"] = np.nan

    # The target is known only after the next trading session closes.
    frame["forecast_origin"] = frame["date"]
    frame["target_date"] = frame["date"].shift(-1)
    frame["rv_target"] = frame["rv_proxy"].shift(-1)
    frame["log_rv_target"] = np.log(frame["rv_target"].clip(lower=MIN_DAILY_VARIANCE))

    valid_target = frame["target_date"].notna()
    if (
        frame.loc[valid_target, "target_date"]
        <= frame.loc[valid_target, "forecast_origin"]
    ).any():
        raise AssertionError("Feature engineering produced a non-future target date")
    return frame


FEATURE_COLUMNS = [
    "rv_d",
    "rv_w",
    "rv_m",
    "rv_63",
    "rv_acceleration",
    "range_ratio",
    "historical_vol_percentile",
    "rv_w_percentile",
    "log_return",
    "return_5",
    "return_22",
    "volume_change",
    "amount_change",
    "volume_ratio_22",
    "amount_ratio_22",
]


def complete_case_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Keep rows needed to fit one-step-ahead models and assert time ordering."""

    required = FEATURE_COLUMNS + ["forecast_origin", "target_date", "rv_target"]
    result = features.dropna(subset=required).copy()
    if result.empty:
        raise ValueError("No complete feature rows are available")
    if not result["forecast_origin"].is_monotonic_increasing:
        raise ValueError("Forecast origins are not chronological")
    if not (result["target_date"] > result["forecast_origin"]).all():
        raise ValueError("Target dates must be strictly after forecast origins")
    return result.reset_index(drop=True)

