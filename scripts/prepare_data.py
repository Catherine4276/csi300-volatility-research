"""Download CSI 300 data and build the processed research dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, project_path  # noqa: E402
from src.data_loader import (  # noqa: E402
    fetch_eastmoney_daily,
    fetch_eastmoney_intraday,
    save_csv_with_manifest,
)
from src.feature_engineering import build_features, calculate_intraday_realized_variance  # noqa: E402


def main() -> None:
    config = load_config()
    data_config = config["data"]
    start = pd.Timestamp(data_config["start_date"])

    daily = fetch_eastmoney_daily(
        secid=data_config["secid"],
        limit=10000,
    )
    daily = daily.loc[daily["date"] >= start].reset_index(drop=True)
    save_csv_with_manifest(
        daily,
        project_path("data", "raw", "csi300_daily.csv"),
        project_path("data", "raw", "csi300_daily_manifest.json"),
        {
            "provider": "Eastmoney public quote API",
            "endpoint": "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            "symbol": data_config["symbol"],
            "secid": data_config["secid"],
            "frequency": "1d",
            "adjusted": False,
            "purpose": "long-history baseline OHLCV",
        },
    )

    intraday_error: str | None = None
    try:
        intraday = fetch_eastmoney_intraday(
            secid=data_config["secid"],
            period=int(data_config["intraday_period"]),
            limit=10000,
        )
        save_csv_with_manifest(
            intraday,
            project_path("data", "raw", "csi300_5m.csv"),
            project_path("data", "raw", "csi300_5m_manifest.json"),
            {
                "provider": "Eastmoney public quote API",
                "endpoint": "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                "symbol": data_config["symbol"],
                "secid": data_config["secid"],
                "frequency": f"{data_config['intraday_period']}m",
                "adjusted": False,
                "coverage_warning": "Rolling short public history; not a multi-year sample.",
            },
        )
        intraday_rv = calculate_intraday_realized_variance(intraday)
        save_csv_with_manifest(
            intraday_rv,
            project_path("data", "processed", "intraday_rv_recent.csv"),
            project_path("data", "processed", "intraday_rv_recent_manifest.json"),
            {
                "provider": "Derived from Eastmoney 5-minute bars",
                "method": "sum of within-day squared 5-minute log returns",
                "minimum_bars_per_day": 20,
                "overnight_returns_included": False,
            },
        )
    except Exception as exc:  # Preserve the daily baseline if the short feed is unavailable.
        intraday_error = f"{type(exc).__name__}: {exc}"
        intraday_rv = None

    features = build_features(
        daily,
        intraday_rv=intraday_rv,
        vol_window=int(config["features"]["vol_window"]),
    )
    feature_manifest = save_csv_with_manifest(
        features,
        project_path("data", "processed", "modeling_dataset.csv"),
        project_path("data", "processed", "modeling_dataset_manifest.json"),
        {
            "provider": "Derived from csi300_daily.csv and optional csi300_5m.csv",
            "target": "next trading day Garman-Klass variance proxy",
            "target_is_high_frequency_rv": False,
            "feature_columns": [
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
            ],
            "intraday_error": intraday_error,
            "time_direction": "forecast_origin < target_date",
        },
    )
    print(
        json.dumps(
            {
                "daily_rows": len(daily),
                "daily_range": [daily["date"].min().date().isoformat(), daily["date"].max().date().isoformat()],
                "feature_rows": len(features),
                "feature_sha256": feature_manifest["sha256"],
                "intraday_error": intraday_error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

