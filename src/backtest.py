"""Walk-forward experiment orchestration and artifact generation."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .config import load_config, project_path
from .dynamic_ensemble import add_dynamic_ensemble
from .evaluation import (
    dm_comparison_table,
    forecast_metrics,
    giacomini_white_table,
    model_confidence_set,
    model_metric_table,
    qlike_loss,
)
from .feature_engineering import FEATURE_COLUMNS, complete_case_frame
from .har_model import HARModel
from .regime_detection import assign_volatility_regimes, regime_summary
from .xgboost_model import XGBoostVolatilityModel


MODEL_COLUMNS = ("pred_har", "pred_xgb", "pred_static", "pred_dynamic")
LOSS_COLUMNS = {
    "HAR-RV": "qlike_har",
    "XGBoost": "qlike_xgb",
    "Static Ensemble": "qlike_static",
    "Dynamic Ensemble": "qlike_dynamic",
}


def _sha256_files(paths: Iterable[Path]) -> str:
    """Hash source or configuration files with stable relative names."""

    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_reproducibility_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Collect code, dependency, and configuration identifiers for a run."""

    package_names = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "statsmodels",
        "streamlit",
    )
    versions: dict[str, str] = {}
    for package in package_names:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    config_path = Path(config["_config_path"])
    return {
        "run_id": datetime.now(timezone.utc).strftime("csi300-vol-%Y%m%dT%H%M%SZ"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _sha256_files(project_path("src").glob("*.py")),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "package_versions": versions,
        "random_seed": int(config["xgboost"]["random_state"]),
        "command": "python scripts/prepare_data.py && python -m pytest && python scripts/run_research.py",
    }


def _assert_training_clean(training: pd.DataFrame, origin: pd.Timestamp) -> None:
    """Hard gate against using an unresolved or future target while fitting."""

    if training.empty:
        raise ValueError(f"No training data before {origin.date()}")
    if (training["forecast_origin"] >= origin).any():
        raise AssertionError("Training origin is not strictly before forecast origin")
    if (training["target_date"] > origin).any():
        raise AssertionError("Training target closes after forecast origin")


def run_walk_forward(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate strict one-day-ahead forecasts with scheduled model refits."""

    data = complete_case_frame(features)
    oos_start = pd.Timestamp(config["validation"]["oos_start"])
    first_candidates = np.flatnonzero(data["forecast_origin"] >= oos_start)
    if not len(first_candidates):
        raise ValueError("OOS start is after the available complete sample")
    first_oos_index = int(first_candidates[0])
    minimum_rows = int(config["validation"]["minimum_training_rows"])
    if first_oos_index < minimum_rows:
        raise ValueError(
            f"Need at least {minimum_rows} prior rows; only {first_oos_index} available"
        )

    refit_every = int(config["validation"]["refit_every"])
    tune_every = int(config["validation"]["xgb_tune_every"])
    har = HARModel()
    xgb = XGBoostVolatilityModel(
        parameter_grid=config["xgboost"]["parameter_grid"],
        random_state=int(config["xgboost"]["random_state"]),
    )
    last_fit_position = -10_000
    last_tune_position = -10_000
    har_log: list[dict[str, Any]] = []
    xgb_log: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for position in range(first_oos_index, len(data)):
        origin = pd.Timestamp(data.iloc[position]["forecast_origin"])
        training = data.loc[
            (data["forecast_origin"] < origin) & (data["target_date"] <= origin)
        ]
        _assert_training_clean(training, origin)

        if position - last_fit_position >= refit_every:
            tune_xgb = position - last_tune_position >= tune_every
            har.fit(training)
            xgb.fit(training, tune=tune_xgb)
            last_fit_position = position
            if tune_xgb:
                last_tune_position = position
            har_log.append(
                {
                    "refit_forecast_origin": origin.isoformat(),
                    "train_start": har.train_start_.isoformat(),
                    "train_end": har.train_end_.isoformat(),
                    "training_rows": har.training_rows_,
                    "fitted_at_utc": har.fitted_at_utc_,
                    "parameters": dict(zip(har.feature_names_, har.coefficients_.tolist())),
                }
            )
            xgb_log.append(
                {
                    **xgb.training_record(),
                    "refit_forecast_origin": origin.isoformat(),
                    "tuned_this_refit": tune_xgb,
                }
            )

        row = data.iloc[[position]]
        pred_har = float(har.predict(row)[0])
        pred_xgb = float(xgb.predict(row)[0])
        predictions.append(
            {
                "forecast_origin": origin,
                "target_date": pd.Timestamp(row["target_date"].iloc[0]),
                "close": float(row["close"].iloc[0]),
                "rv_target": float(row["rv_target"].iloc[0]),
                "regime": int(row["regime"].iloc[0]),
                "regime_label": str(row["regime_label"].iloc[0]),
                "regime_low_threshold": float(row["regime_low_threshold"].iloc[0]),
                "regime_high_threshold": float(row["regime_high_threshold"].iloc[0]),
                "pred_har": pred_har,
                "pred_xgb": pred_xgb,
                "model_refit_origin": har_log[-1]["refit_forecast_origin"],
                "train_start": har_log[-1]["train_start"],
                "train_end": har_log[-1]["train_end"],
                "training_rows": har_log[-1]["training_rows"],
            }
        )

    forecasts = pd.DataFrame(predictions)
    if not (forecasts["target_date"] > forecasts["forecast_origin"]).all():
        raise AssertionError("A forecast target is not strictly in the future")
    forecasts["qlike_har"] = qlike_loss(forecasts["rv_target"], forecasts["pred_har"])
    forecasts["qlike_xgb"] = qlike_loss(forecasts["rv_target"], forecasts["pred_xgb"])
    forecasts = add_dynamic_ensemble(
        forecasts,
        window=int(config["validation"]["weight_window"]),
        min_rows=int(config["validation"]["weight_minimum_rows"]),
        shrinkage=float(config["validation"]["weight_shrinkage"]),
        enforce_regime_direction=bool(
            config["validation"].get("enforce_regime_direction", False)
        ),
        xgb_weight_floor_normal=float(
            config["validation"].get("xgb_weight_floor_normal", 0.50)
        ),
        xgb_weight_ceiling_high=float(
            config["validation"].get("xgb_weight_ceiling_high", 0.50)
        ),
    )
    forecasts["qlike_static"] = qlike_loss(
        forecasts["rv_target"], forecasts["pred_static"]
    )
    forecasts["qlike_dynamic"] = qlike_loss(
        forecasts["rv_target"], forecasts["pred_dynamic"]
    )
    forecasts["actual_annualized_vol"] = np.sqrt(forecasts["rv_target"] * 252.0)
    for column in MODEL_COLUMNS:
        forecasts[column.replace("pred_", "annualized_vol_")] = np.sqrt(
            forecasts[column] * 252.0
        )
    return forecasts, har_log, xgb_log


def _metrics_for_range(
    forecasts: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    subset = forecasts.loc[
        forecasts["forecast_origin"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    if subset.empty:
        return pd.DataFrame()
    table = model_metric_table(subset)
    table.insert(0, "sample_start", start)
    table.insert(1, "sample_end", end)
    return table


def build_experiments(
    forecasts: pd.DataFrame,
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Create the seven requested forecast comparison experiments."""

    tables: dict[str, pd.DataFrame] = {}
    overview: dict[str, Any] = {}
    overview["forecast_rows"] = int(len(forecasts))
    overview["forecast_start"] = forecasts["forecast_origin"].min().isoformat()
    overview["forecast_end"] = forecasts["forecast_origin"].max().isoformat()

    overall = model_metric_table(forecasts)
    overall.insert(0, "sample", "overall")
    tables["experiment_1_overall"] = overall

    regime_rows = []
    for regime, subset in forecasts.groupby("regime"):
        table = model_metric_table(subset)
        table.insert(0, "regime", int(regime))
        table.insert(1, "regime_label", str(subset["regime_label"].iloc[0]))
        regime_rows.append(table)
    tables["experiment_2_regime"] = pd.concat(regime_rows, ignore_index=True)

    tables["experiment_3_2015"] = _metrics_for_range(
        forecasts,
        config["events"]["crisis_2015"]["start"],
        config["events"]["crisis_2015"]["end"],
    )
    tables["experiment_4_2020"] = _metrics_for_range(
        forecasts,
        config["events"]["pandemic_2020"]["start"],
        config["events"]["pandemic_2020"]["end"],
    )

    dynamic_sample = forecasts.copy()
    dynamic_sample["sample"] = "full_oos"
    tables["experiment_5_dynamic"] = model_metric_table(dynamic_sample)

    sensitivity_rows = []
    for weight in np.round(np.linspace(0.0, 1.0, 21), 2):
        prediction = weight * forecasts["pred_xgb"] + (1.0 - weight) * forecasts["pred_har"]
        metrics = forecast_metrics(forecasts["rv_target"], prediction)
        sensitivity_rows.append({"weight_xgb": float(weight), **metrics})
    tables["experiment_6_weight_sensitivity"] = pd.DataFrame(sensitivity_rows)

    threshold_rows = []
    base = complete_case_frame(features)
    # The dynamic algorithm is rerun with labels generated causally for each
    # threshold pair. Base forecasts stay fixed, so this isolates state sensitivity.
    for low_q in config["tests"]["regime_low_quantiles"]:
        for high_q in config["tests"]["regime_high_quantiles"]:
            if low_q >= high_q:
                continue
            labeled = assign_volatility_regimes(
                base,
                window=int(config["features"]["vol_window"]),
                low_quantile=float(low_q),
                high_quantile=float(high_q),
                confirm_days=int(config["features"]["regime_confirm_days"]),
            )
            lookup = labeled.set_index("forecast_origin")["regime"]
            scenario = forecasts.copy()
            scenario["regime"] = scenario["forecast_origin"].map(lookup)
            scenario = scenario.dropna(subset=["regime"]).copy()
            scenario["regime"] = scenario["regime"].astype(int)
            scenario = add_dynamic_ensemble(
                scenario,
                window=int(config["validation"]["weight_window"]),
                min_rows=int(config["validation"]["weight_minimum_rows"]),
                shrinkage=float(config["validation"]["weight_shrinkage"]),
                enforce_regime_direction=bool(
                    config["validation"].get("enforce_regime_direction", False)
                ),
                xgb_weight_floor_normal=float(
                    config["validation"].get("xgb_weight_floor_normal", 0.50)
                ),
                xgb_weight_ceiling_high=float(
                    config["validation"].get("xgb_weight_ceiling_high", 0.50)
                ),
            )
            metrics = forecast_metrics(scenario["rv_target"], scenario["pred_dynamic"])
            threshold_rows.append(
                {
                    "low_quantile": float(low_q),
                    "high_quantile": float(high_q),
                    "high_regime_share": float((scenario["regime"] == 2).mean()),
                    "regime_switches": int(
                        labeled["regime"].ne(labeled["regime"].shift(1)).sum()
                    ),
                    **metrics,
                }
            )
    tables["experiment_7_threshold_sensitivity"] = pd.DataFrame(threshold_rows)

    return tables, overview


def run_tests_and_save(
    forecasts: pd.DataFrame,
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Run experiments and inferential tests, returning all report tables."""

    experiments, overview = build_experiments(forecasts, features, config)
    tests: dict[str, pd.DataFrame] = {}
    for benchmark_label, benchmark_key in (
        ("HAR-RV", "HAR-RV"),
        ("Static Ensemble", "Static Ensemble"),
    ):
        table = dm_comparison_table(
            forecasts,
            LOSS_COLUMNS,
            benchmark=benchmark_key,
            max_lags=int(config["tests"]["dm_max_lag"]),
        )
        table.insert(0, "comparison_block", benchmark_label)
        tests[f"dm_vs_{benchmark_label}"] = table
        gw_table = giacomini_white_table(
            forecasts,
            LOSS_COLUMNS,
            benchmark=benchmark_key,
            instrument_columns=("regime",),
            max_lags=int(config["tests"]["dm_max_lag"]),
        )
        gw_table.insert(0, "comparison_block", benchmark_label)
        tests[f"gw_vs_{benchmark_label}"] = gw_table
    loss_matrix = forecasts.loc[:, list(LOSS_COLUMNS.values())].copy()
    loss_matrix.columns = list(LOSS_COLUMNS)
    tests["model_confidence_set"] = model_confidence_set(
        loss_matrix,
        alpha=0.10,
        draws=int(config["tests"]["mcs_bootstrap_draws"]),
        block_length=config["tests"].get("bootstrap_block_length"),
        seed=int(config["tests"]["random_seed"]),
    )
    return {**experiments, **tests}, overview


def latest_forecast(
    features: pd.DataFrame,
    config: dict[str, Any],
    forecasts: pd.DataFrame,
) -> dict[str, Any]:
    """Fit final models and produce the latest origin-to-next-day forecast."""

    complete = complete_case_frame(features)
    usable = features.dropna(subset=FEATURE_COLUMNS).copy()
    if usable.empty:
        raise ValueError("No complete feature row for the latest forecast")
    latest = usable.sort_values("forecast_origin").iloc[-1]
    origin = pd.Timestamp(latest["forecast_origin"])
    training = complete.loc[complete["target_date"] <= origin]
    _assert_training_clean(training, origin)

    har = HARModel().fit(training)
    xgb = XGBoostVolatilityModel(
        parameter_grid=config["xgboost"]["parameter_grid"],
        random_state=int(config["xgboost"]["random_state"]),
    ).fit(training, tune=True)
    models_dir = project_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(har, models_dir / "har_latest_final.joblib")
    joblib.dump(xgb, models_dir / "xgb_latest_final.joblib")
    latest_frame = latest.to_frame().T
    pred_har = float(har.predict(latest_frame)[0])
    pred_xgb = float(xgb.predict(latest_frame)[0])
    latest_resolved = forecasts.sort_values("forecast_origin").iloc[-1]
    weight = float(latest_resolved["weight_xgb"])
    pred_dynamic = weight * pred_xgb + (1.0 - weight) * pred_har
    high_threshold = float(latest["regime_high_threshold"])
    alert = bool(latest["regime"] == 2 or latest["rv_w"] >= high_threshold)
    return {
        "forecast_origin": origin.isoformat(),
        "next_trading_target_expected": latest["target_date"].isoformat()
        if pd.notna(latest["target_date"])
        else None,
        "regime": int(latest["regime"]),
        "regime_label": str(latest["regime_label"]),
        "high_risk_alert": alert,
        "pred_har": pred_har,
        "pred_xgb": pred_xgb,
        "pred_dynamic": pred_dynamic,
        "weight_xgb": weight,
        "weight_har": 1.0 - weight,
        "annualized_vol_har": float(np.sqrt(pred_har * 252.0)),
        "annualized_vol_xgb": float(np.sqrt(pred_xgb * 252.0)),
        "annualized_vol_dynamic": float(np.sqrt(pred_dynamic * 252.0)),
        "latest_close": float(latest["close"]),
        "har_parameters": dict(zip(har.feature_names_, har.coefficients_.tolist())),
        "xgboost_training": xgb.training_record(),
        "training_rows": int(len(training)),
    }


def run_research(config_path: str | Path | None = None) -> dict[str, Any]:
    """Execute backtests, tests, experiments, and persist reproducible artifacts."""

    config = load_config(config_path)
    dataset_path = project_path("data", "processed", "modeling_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_path}. Run scripts/prepare_data.py first."
        )
    features = pd.read_csv(dataset_path)
    for column in ("date", "forecast_origin", "target_date"):
        features[column] = pd.to_datetime(features[column], errors="coerce")
    features = assign_volatility_regimes(
        features,
        window=int(config["features"]["vol_window"]),
        low_quantile=float(config["features"]["regime_low_quantile"]),
        high_quantile=float(config["features"]["regime_high_quantile"]),
        confirm_days=int(config["features"]["regime_confirm_days"]),
    )

    forecasts, har_log, xgb_log = run_walk_forward(features, config)
    tables, overview = run_tests_and_save(forecasts, features, config)
    final_forecast = latest_forecast(features, config, forecasts)
    state_summary = regime_summary(features)

    reports_dir = project_path("reports")
    experiments_dir = project_path("experiments")
    models_dir = project_path("models")
    for directory in (reports_dir, experiments_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    forecasts.to_csv(reports_dir / "forecasts.csv", index=False, encoding="utf-8-sig")
    state_summary.to_csv(reports_dir / "regime_summary.csv", index=False, encoding="utf-8-sig")
    for name, table in tables.items():
        table.to_csv(experiments_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (experiments_dir / "har_refits.json").write_text(
        json.dumps(har_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (experiments_dir / "xgb_refits.json").write_text(
        json.dumps(xgb_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "latest_forecast.json").write_text(
        json.dumps(final_forecast, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (experiments_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                **overview,
                **_run_reproducibility_metadata(config),
                "config": {
                    key: value for key, value in config.items() if not key.startswith("_")
                },
                "model_refits": len(har_log),
                "xgb_tuning_events": sum(bool(item["tuned_this_refit"]) for item in xgb_log),
                "target_definition": "next-trading-day Garman-Klass OHLC variance proxy",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "config": config,
        "forecasts": forecasts,
        "tables": tables,
        "overview": overview,
        "latest_forecast": final_forecast,
        "regime_summary": state_summary,
        "har_log": har_log,
        "xgb_log": xgb_log,
    }

