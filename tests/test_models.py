from __future__ import annotations

import numpy as np
import pandas as pd

from src.dynamic_ensemble import add_dynamic_ensemble
from src.feature_engineering import complete_case_frame
from src.har_model import HARModel
from src.xgboost_model import XGBoostVolatilityModel


def test_har_training_and_positive_prediction(synthetic_features):
    data = complete_case_frame(synthetic_features).iloc[:300]
    model = HARModel().fit(data)
    prediction = model.predict(data.iloc[-5:])
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()
    assert len(model.parameter_table()) == 4


def test_xgboost_training_and_prediction(synthetic_features):
    data = complete_case_frame(synthetic_features).iloc[:220]
    model = XGBoostVolatilityModel(
        parameter_grid=[
            {
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.05,
                "min_child_weight": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 2.0,
            }
        ],
        random_state=7,
    ).fit(data, tune=True)
    prediction = model.predict(data.iloc[-4:])
    transposed_row = data.iloc[-1].to_frame().T
    transposed_prediction = model.predict(transposed_row)
    assert prediction.shape == (4,)
    assert transposed_prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()
    assert model.training_record()["parameters"]["n_estimators"] == 20


def test_dynamic_weight_uses_prior_loss_and_favors_better_model():
    rows = 120
    forecasts = pd.DataFrame(
        {
            "forecast_origin": pd.bdate_range("2022-01-03", periods=rows),
            "rv_target": np.ones(rows),
            "regime": np.ones(rows, dtype=int),
            "pred_har": np.ones(rows),
            "pred_xgb": np.full(rows, 1.5),
        }
    )
    forecasts["qlike_har"] = forecasts["pred_har"] / forecasts["rv_target"] - np.log(
        forecasts["pred_har"] / forecasts["rv_target"]
    ) - 1
    forecasts["qlike_xgb"] = forecasts["pred_xgb"] / forecasts["rv_target"] - np.log(
        forecasts["pred_xgb"] / forecasts["rv_target"]
    ) - 1
    result = add_dynamic_ensemble(forecasts, window=60, min_rows=20, shrinkage=0.2)
    assert result["weight_xgb"].iloc[-1] < 0.20
    assert result["pred_dynamic"].iloc[-1] < result["pred_xgb"].iloc[-1]


def test_dynamic_regime_direction_bounds_are_enforced():
    rows = 160
    forecasts = pd.DataFrame(
        {
            "forecast_origin": pd.bdate_range("2022-01-03", periods=rows),
            "rv_target": np.ones(rows),
            "regime": np.where(np.arange(rows) < 80, 1, 2),
            "pred_har": np.ones(rows),
            "pred_xgb": np.full(rows, 1.3),
        }
    )
    for model in ("har", "xgb"):
        ratio = forecasts[f"pred_{model}"] / forecasts["rv_target"]
        forecasts[f"qlike_{model}"] = ratio - np.log(ratio) - 1.0
    result = add_dynamic_ensemble(
        forecasts,
        window=160,
        min_rows=20,
        enforce_regime_direction=True,
    )
    normalized = result.loc[result["weight_source_rows"] >= 20]
    assert (normalized.loc[normalized["regime"] != 2, "weight_xgb"] >= 0.5).all()
    assert (normalized.loc[normalized["regime"] == 2, "weight_xgb"] <= 0.5).all()

