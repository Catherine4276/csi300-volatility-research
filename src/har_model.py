"""Interpretable HAR-RV and optional HAR-J implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd


MIN_VARIANCE = 1e-12


@dataclass
class HARModel:
    """Ordinary least-squares HAR model with explicit time metadata."""

    feature_columns: Sequence[str] = ("rv_d", "rv_w", "rv_m")
    intercept: bool = True
    coefficients_: np.ndarray | None = field(default=None, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    train_start_: pd.Timestamp | None = field(default=None, init=False)
    train_end_: pd.Timestamp | None = field(default=None, init=False)
    fitted_at_utc_: str | None = field(default=None, init=False)
    training_rows_: int = field(default=0, init=False)

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        columns = [frame[column].to_numpy(dtype=float) for column in self.feature_columns]
        design = np.column_stack(columns)
        if self.intercept:
            design = np.column_stack([np.ones(len(frame)), design])
        return design

    def fit(self, frame: pd.DataFrame, target_column: str = "rv_target") -> "HARModel":
        """Fit OLS and record the exact information window."""

        if frame.empty:
            raise ValueError("Cannot fit HAR on an empty frame")
        if not frame["forecast_origin"].is_monotonic_increasing:
            raise ValueError("HAR training rows must be chronological")
        design = self._design(frame)
        target = frame[target_column].to_numpy(dtype=float)
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < design.shape[1]:
            raise ValueError("HAR design matrix is rank deficient")
        self.coefficients_ = coefficients
        prefix = ["intercept"] if self.intercept else []
        self.feature_names_ = prefix + list(self.feature_columns)
        self.train_start_ = pd.Timestamp(frame["forecast_origin"].min())
        self.train_end_ = pd.Timestamp(frame["forecast_origin"].max())
        self.fitted_at_utc_ = datetime.now(timezone.utc).isoformat()
        self.training_rows_ = int(len(frame))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict next-day variance and clip to a positive support."""

        if self.coefficients_ is None:
            raise RuntimeError("HAR model has not been fitted")
        prediction = self._design(frame) @ self.coefficients_
        return np.clip(prediction, MIN_VARIANCE, None)

    def parameter_table(self) -> pd.DataFrame:
        """Return coefficients in an inspectable table."""

        if self.coefficients_ is None:
            raise RuntimeError("HAR model has not been fitted")
        return pd.DataFrame(
            {"parameter": self.feature_names_, "coefficient": self.coefficients_}
        )


@dataclass
class HARJModel(HARModel):
    """HAR-J extension with a caller-supplied non-negative jump component.

    The jump column must be known at the forecast origin. A project should not
    invent a jump measure when only daily OHLC data are available.
    """

    feature_columns: Sequence[str] = ("rv_d", "rv_w", "rv_m", "jump_component")
    intercept: bool = True

    def fit(self, frame: pd.DataFrame, target_column: str = "rv_target") -> "HARJModel":
        if "jump_component" not in frame:
            raise ValueError("HAR-J requires a causal 'jump_component' column")
        if (frame["jump_component"].dropna() < 0).any():
            raise ValueError("Jump component must be non-negative")
        super().fit(frame=frame, target_column=target_column)
        return self

