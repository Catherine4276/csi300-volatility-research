"""Causal regime-aware dynamic ensemble built from prior OOS loss."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_loss_weight(
    history: pd.DataFrame,
    min_rows: int = 30,
    shrinkage: float = 0.20,
    lower: float = 0.05,
    upper: float = 0.95,
) -> tuple[float, int, float, float]:
    """Derive the XGBoost weight from historical QLIKE losses only.

    A lower historical loss receives a higher weight. Shrinkage moves the
    empirical weight toward 0.5; no future observation enters the calculation.
    """

    clean = history.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["qlike_har", "qlike_xgb"]
    )
    if len(clean) < min_rows:
        return 0.5, int(len(clean)), float("nan"), float("nan")
    q_har = float(clean["qlike_har"].mean())
    q_xgb = float(clean["qlike_xgb"].mean())
    if q_har <= 0 and q_xgb <= 0:
        return 0.5, int(len(clean)), q_har, q_xgb
    empirical = q_har / (q_har + q_xgb) if (q_har + q_xgb) > 0 else 0.5
    weight = 0.5 + (1.0 - shrinkage) * (empirical - 0.5)
    return float(np.clip(weight, lower, upper)), int(len(clean)), q_har, q_xgb


def add_dynamic_ensemble(
    forecasts: pd.DataFrame,
    window: int = 252,
    min_rows: int = 30,
    shrinkage: float = 0.20,
    static_weight_xgb: float = 0.5,
    enforce_regime_direction: bool = False,
    xgb_weight_floor_normal: float = 0.50,
    xgb_weight_ceiling_high: float = 0.50,
) -> pd.DataFrame:
    """Build static and dynamic blends for every chronological OOS forecast."""

    required = {
        "pred_har",
        "pred_xgb",
        "rv_target",
        "qlike_har",
        "qlike_xgb",
        "regime",
    }
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise ValueError(f"Missing forecast columns: {missing}")
    result = forecasts.copy().sort_values("forecast_origin").reset_index(drop=True)
    weights: list[float] = []
    source_rows: list[int] = []
    history_q_har: list[float] = []
    history_q_xgb: list[float] = []

    for position, row in result.iterrows():
        recent = result.iloc[max(0, position - window) : position]
        # Match the state known at the current forecast origin. Prior targets
        # have already closed and are therefore admissible at this point.
        matching = recent.loc[recent["regime"].eq(row["regime"])]
        lower, upper = 0.05, 0.95
        if enforce_regime_direction:
            # The learned magnitude remains loss-driven, but the pre-registered
            # hypothesis constrains the direction: defensive HAR weighting is
            # allowed in high volatility, while normal states may lean XGBoost.
            if int(row["regime"]) == 2:
                upper = min(upper, xgb_weight_ceiling_high)
            else:
                lower = max(lower, xgb_weight_floor_normal)
        weight, count, q_har, q_xgb = inverse_loss_weight(
            matching,
            min_rows=min_rows,
            shrinkage=shrinkage,
            lower=lower,
            upper=upper,
        )
        weights.append(weight)
        source_rows.append(count)
        history_q_har.append(q_har)
        history_q_xgb.append(q_xgb)

    result["weight_xgb"] = weights
    result["weight_har"] = 1.0 - result["weight_xgb"]
    result["weight_source_rows"] = source_rows
    result["history_qlike_har"] = history_q_har
    result["history_qlike_xgb"] = history_q_xgb
    result["pred_dynamic"] = (
        result["weight_xgb"] * result["pred_xgb"]
        + result["weight_har"] * result["pred_har"]
    )
    result["pred_static"] = (
        static_weight_xgb * result["pred_xgb"]
        + (1.0 - static_weight_xgb) * result["pred_har"]
    )
    return result

