"""Forecast metrics, Diebold-Mariano tests, and bootstrap model elimination."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .har_model import MIN_VARIANCE


def qlike_loss(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> np.ndarray:
    """Compute QLIKE on variance forecasts; lower is better."""

    actual_array = np.clip(np.asarray(actual, dtype=float), MIN_VARIANCE, None)
    prediction_array = np.clip(np.asarray(prediction, dtype=float), MIN_VARIANCE, None)
    ratio = prediction_array / actual_array
    return ratio - np.log(ratio) - 1.0


def forecast_metrics(
    actual: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    """Calculate comparable out-of-sample variance forecast metrics."""

    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    mask = np.isfinite(actual_array) & np.isfinite(prediction_array)
    actual_array = actual_array[mask]
    prediction_array = prediction_array[mask]
    if len(actual_array) == 0:
        raise ValueError("No finite observations for metric calculation")
    error = prediction_array - actual_array
    denominator = np.sum(np.square(actual_array - actual_array.mean()))
    r2 = (
        1.0 - np.sum(np.square(error)) / denominator
        if denominator > 0
        else float("nan")
    )
    return {
        "n": int(len(actual_array)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "qlike": float(np.mean(qlike_loss(actual_array, prediction_array))),
        "r2_ots": float(r2),
    }


def model_metric_table(
    forecasts: pd.DataFrame,
    actual_column: str = "rv_target",
    model_columns: Iterable[str] = ("pred_har", "pred_xgb", "pred_static", "pred_dynamic"),
) -> pd.DataFrame:
    """Evaluate several prediction columns on exactly the same finite sample."""

    actual = forecasts[actual_column]
    rows = []
    for model_column in model_columns:
        metrics = forecast_metrics(actual, forecasts[model_column])
        rows.append({"model": model_column, **metrics})
    return pd.DataFrame(rows)


def newey_west_dm(
    loss_model: np.ndarray | pd.Series,
    loss_benchmark: np.ndarray | pd.Series,
    max_lags: int = 22,
) -> dict[str, float]:
    """Two-sided DM test with Newey-West long-run variance.

    A positive statistic means the model has higher loss than the benchmark.
    """

    model = np.asarray(loss_model, dtype=float)
    benchmark = np.asarray(loss_benchmark, dtype=float)
    mask = np.isfinite(model) & np.isfinite(benchmark)
    difference = model[mask] - benchmark[mask]
    n = len(difference)
    if n < 10:
        return {"dm_stat": float("nan"), "dm_pvalue": float("nan"), "n": float(n)}
    mean_difference = float(np.mean(difference))
    centered = difference - mean_difference
    lags = min(max_lags, n - 1)
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * weight * autocovariance
    if long_run_variance <= 0 or not np.isfinite(long_run_variance):
        return {
            "mean_loss_diff": mean_difference,
            "dm_stat": float("nan"),
            "dm_pvalue": float("nan"),
            "n": float(n),
        }
    statistic = mean_difference / np.sqrt(long_run_variance / n)
    p_value = 2.0 * stats.t.sf(abs(statistic), df=n - 1)
    return {
        "mean_loss_diff": mean_difference,
        "dm_stat": float(statistic),
        "dm_pvalue": float(p_value),
        "n": float(n),
    }


def dm_comparison_table(
    forecasts: pd.DataFrame,
    loss_columns: dict[str, str],
    benchmark: str,
    max_lags: int = 22,
) -> pd.DataFrame:
    """Compare named forecast losses against a common benchmark."""

    rows = []
    for model, loss_column in loss_columns.items():
        if model == benchmark:
            continue
        result = newey_west_dm(
            forecasts[loss_column],
            forecasts[loss_columns[benchmark]],
            max_lags=max_lags,
        )
        rows.append({"model": model, "benchmark": benchmark, **result})
    return pd.DataFrame(rows)


def giacomini_white_test(
    loss_model: np.ndarray | pd.Series,
    loss_benchmark: np.ndarray | pd.Series,
    instruments: pd.DataFrame | np.ndarray,
    max_lags: int = 22,
) -> dict[str, float]:
    """Conditional predictive ability Wald test using moment conditions.

    Instruments must be known at the forecast origin. The test evaluates
    ``E[(L_model-L_benchmark) * X] = 0`` with HAC covariance, including the
    intercept in ``X``.
    """

    model = np.asarray(loss_model, dtype=float)
    benchmark = np.asarray(loss_benchmark, dtype=float)
    instrument_array = np.asarray(instruments, dtype=float)
    if instrument_array.ndim == 1:
        instrument_array = instrument_array.reshape(-1, 1)
    if not (len(model) == len(benchmark) == len(instrument_array)):
        raise ValueError("Losses and instruments must have the same row count")
    mask = (
        np.isfinite(model)
        & np.isfinite(benchmark)
        & np.isfinite(instrument_array).all(axis=1)
    )
    difference = model[mask] - benchmark[mask]
    valid_instruments = instrument_array[mask]
    n = len(difference)
    if n < 30:
        return {
            "gw_stat": float("nan"),
            "gw_pvalue": float("nan"),
            "gw_df": float("nan"),
            "n": float(n),
        }
    design = np.column_stack([np.ones(n), valid_instruments])
    moment_contributions = design * difference[:, None]
    sample_moment = np.mean(moment_contributions, axis=0)
    centered = moment_contributions - sample_moment
    lags = min(max_lags, n - 1)
    covariance = centered.T @ centered / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = centered[lag:].T @ centered[:-lag] / n
        covariance += weight * (cross + cross.T)
    covariance_inverse = np.linalg.pinv(covariance)
    statistic = float(n * sample_moment.T @ covariance_inverse @ sample_moment)
    degrees_of_freedom = int(design.shape[1])
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
    return {
        "gw_stat": statistic,
        "gw_pvalue": p_value,
        "gw_df": float(degrees_of_freedom),
        "n": float(n),
    }


def giacomini_white_table(
    forecasts: pd.DataFrame,
    loss_columns: dict[str, str],
    benchmark: str,
    instrument_columns: Iterable[str] = ("regime",),
    max_lags: int = 22,
) -> pd.DataFrame:
    """Compare conditional predictive ability against a common benchmark."""

    instrument_frame = forecasts.loc[:, list(instrument_columns)].copy()
    # One-hot state instruments avoid imposing a linear ordering on regimes.
    if "regime" in instrument_frame:
        dummies = pd.get_dummies(
            instrument_frame["regime"].astype(int), prefix="regime", drop_first=True
        ).astype(float)
        instrument_frame = pd.concat(
            [
                instrument_frame.drop(columns=["regime"]).reset_index(drop=True),
                dummies.reset_index(drop=True),
            ],
            axis=1,
        )
    instrument_frame = instrument_frame.apply(pd.to_numeric, errors="coerce")
    rows = []
    for model_name, loss_column in loss_columns.items():
        if model_name == benchmark:
            continue
        result = giacomini_white_test(
            forecasts[loss_column],
            forecasts[loss_columns[benchmark]],
            instrument_frame,
            max_lags=max_lags,
        )
        rows.append({"model": model_name, "benchmark": benchmark, **result})
    return pd.DataFrame(rows)


def _circular_block_indices(
    n: int,
    draws: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Generate shared circular block-bootstrap row indices."""

    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block_length))
    indices = np.empty((draws, n), dtype=np.int32)
    for draw in range(draws):
        starts = rng.integers(0, n, size=blocks)
        sample = np.concatenate(
            [np.arange(start, start + block_length) % n for start in starts]
        )[:n]
        indices[draw] = sample
    return indices


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    draws: int = 2000,
    block_length: int | None = None,
    seed: int = 20261110,
) -> pd.DataFrame:
    """Approximate MCS using a pairwise range and circular block bootstrap.

    This implementation follows the MCS elimination logic: test the range of
    pairwise loss differences, eliminate the worst model, and repeat. It is a
    block-bootstrap operationalization, not a claim of the exact canonical MCS
    implementation, and this nuance is stated in the research report.
    """

    loss_frame = losses.replace([np.inf, -np.inf], np.nan).dropna()
    if loss_frame.shape[1] < 2:
        raise ValueError("MCS requires at least two models")
    n = len(loss_frame)
    if n < 30:
        raise ValueError("MCS requires at least 30 complete rows")
    if block_length is None:
        block_length = max(2, int(round(np.sqrt(n))))
    indices = _circular_block_indices(n, draws, block_length, seed)
    values = loss_frame.to_numpy(dtype=float)
    names = list(loss_frame.columns)
    candidates = list(range(len(names)))
    records: dict[str, dict[str, object]] = {}
    elimination_order: list[str] = []

    while len(candidates) > 1:
        sample = values[:, candidates]
        means = sample.mean(axis=0)
        pair_stats = []
        pair_values = []
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                difference = sample[:, left] - sample[:, right]
                standard_error = np.sqrt(np.var(difference, ddof=1) / n)
                statistic = (
                    abs(float(np.mean(difference) / standard_error))
                    if standard_error > 0
                    else 0.0
                )
                pair_stats.append(statistic)
                pair_values.append((left, right, difference))
        observed_range = max(pair_stats, default=0.0)

        sampled_losses = sample[indices]
        bootstrap_range = np.zeros(draws, dtype=float)
        for left, right, difference in pair_values:
            standard_error = np.sqrt(np.var(difference, ddof=1) / n)
            if standard_error == 0:
                continue
            centered = (
                sampled_losses[:, :, left].mean(axis=1)
                - sampled_losses[:, :, right].mean(axis=1)
                - (means[left] - means[right])
            )
            bootstrap_range = np.maximum(
                bootstrap_range,
                np.abs(centered / standard_error),
            )
        p_value = float((1 + np.sum(bootstrap_range >= observed_range)) / (draws + 1))

        if p_value >= alpha:
            for index in candidates:
                records[names[index]] = {
                    "p_value": 1.0 if len(candidates) == 1 else p_value,
                    "included": True,
                    "elimination_rank": np.nan,
                }
            break

        worst_local = int(np.argmax(means))
        worst_global = candidates.pop(worst_local)
        elimination_order.append(names[worst_global])
        records[names[worst_global]] = {
            "p_value": p_value,
            "included": False,
            "elimination_rank": len(elimination_order),
        }

    for name in names:
        if name not in records:
            records[name] = {
                "p_value": 1.0,
                "included": True,
                "elimination_rank": np.nan,
            }
    return pd.DataFrame(
        [
            {
                "model": name,
                "mean_loss": float(loss_frame[name].mean()),
                **records[name],
            }
            for name in names
        ]
    ).sort_values(["included", "mean_loss"], ascending=[False, True])

