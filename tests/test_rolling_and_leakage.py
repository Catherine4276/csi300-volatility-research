from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.backtest import _assert_training_clean, run_walk_forward
from src.config import load_config
from src.feature_engineering import complete_case_frame


def test_training_gate_rejects_unrealized_target(synthetic_features):
    data = complete_case_frame(synthetic_features).iloc[:100].copy()
    origin = pd.Timestamp(data["forecast_origin"].iloc[-1])
    training = data.iloc[:-1].copy()
    training.loc[training.index[-1], "target_date"] = origin + pd.Timedelta(days=1)
    with pytest.raises(AssertionError, match="after forecast origin"):
        _assert_training_clean(training, origin)


def test_short_walk_forward_is_strictly_out_of_sample(synthetic_features):
    data = complete_case_frame(synthetic_features)
    config = copy.deepcopy(load_config())
    config["validation"]["oos_start"] = data["forecast_origin"].iloc[150].isoformat()
    config["validation"]["minimum_training_rows"] = 40
    config["validation"]["refit_every"] = 25
    config["validation"]["xgb_tune_every"] = 60
    config["xgboost"]["parameter_grid"] = [
        {
            "n_estimators": 12,
            "max_depth": 2,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
        }
    ]
    forecasts, har_log, xgb_log = run_walk_forward(synthetic_features, config)
    assert len(forecasts) > 0
    assert (forecasts["target_date"] > forecasts["forecast_origin"]).all()
    assert forecasts["weight_xgb"].between(0.05, 0.95).all()
    assert har_log and xgb_log
    for record in har_log:
        assert pd.Timestamp(record["train_end"]) < pd.Timestamp(
            record["refit_forecast_origin"]
        )

