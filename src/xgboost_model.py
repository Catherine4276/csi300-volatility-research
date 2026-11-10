"""Leakage-safe XGBoost volatility model with internal time-series tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .feature_engineering import FEATURE_COLUMNS
from .har_model import MIN_VARIANCE


def qlike(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Element-wise QLIKE loss on variance forecasts."""

    actual = np.clip(np.asarray(actual, dtype=float), MIN_VARIANCE, None)
    prediction = np.clip(np.asarray(prediction, dtype=float), MIN_VARIANCE, None)
    ratio = prediction / actual
    return ratio - np.log(ratio) - 1.0


@dataclass
class XGBoostVolatilityModel:
    """XGBoost regressor trained on log variance and evaluated in variance space."""

    feature_columns: Sequence[str] = tuple(FEATURE_COLUMNS)
    random_state: int = 20261110
    parameter_grid: Sequence[dict[str, Any]] = field(default_factory=tuple)
    n_splits: int = 3
    selected_params_: dict[str, Any] | None = field(default=None, init=False)
    cv_qlike_: float | None = field(default=None, init=False)
    model_: XGBRegressor | None = field(default=None, init=False)
    train_start_: pd.Timestamp | None = field(default=None, init=False)
    train_end_: pd.Timestamp | None = field(default=None, init=False)
    fitted_at_utc_: str | None = field(default=None, init=False)
    training_rows_: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.feature_columns = list(self.feature_columns)
        self.parameter_grid = list(self.parameter_grid)
        if not self.parameter_grid:
            self.parameter_grid = [
                {
                    "n_estimators": 240,
                    "max_depth": 2,
                    "learning_rate": 0.03,
                    "min_child_weight": 8,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 3.0,
                }
            ]

    def _fit_one(self, params: dict[str, Any], frame: pd.DataFrame) -> XGBRegressor:
        features = frame.loc[:, self.feature_columns].astype(float)
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=2,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(
            features,
            np.log(frame["rv_target"].clip(lower=MIN_VARIANCE)),
            verbose=False,
        )
        return model

    def tune(self, frame: pd.DataFrame) -> tuple[dict[str, Any], float]:
        """Tune only on chronological folds inside the supplied training window."""

        if len(frame) < 120:
            return self.parameter_grid[0], float("inf")
        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        results: list[tuple[dict[str, Any], float]] = []
        for params in self.parameter_grid:
            fold_losses: list[float] = []
            for train_index, validation_index in splitter.split(frame):
                train_fold = frame.iloc[train_index]
                valid_fold = frame.iloc[validation_index]
                candidate = self._fit_one(params, train_fold)
                prediction = np.exp(
                    candidate.predict(valid_fold.loc[:, self.feature_columns])
                )
                fold_losses.append(
                    float(np.mean(qlike(valid_fold["rv_target"], prediction)))
                )
            results.append((dict(params), float(np.mean(fold_losses))))
        return min(results, key=lambda item: item[1])

    def fit(
        self,
        frame: pd.DataFrame,
        tune: bool = True,
    ) -> "XGBoostVolatilityModel":
        """Fit on data strictly available at the current refit date."""

        if frame.empty:
            raise ValueError("Cannot fit XGBoost on an empty frame")
        if not frame["forecast_origin"].is_monotonic_increasing:
            raise ValueError("XGBoost training rows must be chronological")
        if tune or self.selected_params_ is None:
            self.selected_params_, self.cv_qlike_ = self.tune(frame)
        self.model_ = self._fit_one(self.selected_params_, frame)
        self.train_start_ = pd.Timestamp(frame["forecast_origin"].min())
        self.train_end_ = pd.Timestamp(frame["forecast_origin"].max())
        self.fitted_at_utc_ = datetime.now(timezone.utc).isoformat()
        self.training_rows_ = int(len(frame))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict positive next-day variance."""

        if self.model_ is None:
            raise RuntimeError("XGBoost model has not been fitted")
        features = frame.loc[:, self.feature_columns].astype(float)
        log_prediction = self.model_.predict(features)
        return np.clip(np.exp(log_prediction), MIN_VARIANCE, None)

    def training_record(self) -> dict[str, Any]:
        """Return reproducibility metadata for experiment logs."""

        return {
            "model": "xgboost",
            "parameters": self.selected_params_,
            "cv_qlike": self.cv_qlike_,
            "train_start": self.train_start_.isoformat() if self.train_start_ else None,
            "train_end": self.train_end_.isoformat() if self.train_end_ else None,
            "training_rows": self.training_rows_,
            "fitted_at_utc": self.fitted_at_utc_,
        }

