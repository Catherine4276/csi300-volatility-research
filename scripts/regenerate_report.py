"""Regenerate report artifacts from persisted backtest outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, project_path  # noqa: E402
from src.diagnostics import build_diagnostic_report  # noqa: E402
from src.reporting import generate_report  # noqa: E402


def main() -> None:
    config = load_config()
    forecasts = pd.read_csv(
        project_path("reports", "forecasts.csv"),
        parse_dates=["forecast_origin", "target_date"],
    )
    tables = {}
    for path in project_path("experiments").glob("*.csv"):
        tables[path.stem] = pd.read_csv(path)
    latest = json.loads(
        project_path("reports", "latest_forecast.json").read_text(encoding="utf-8")
    )
    run_metadata = json.loads(
        project_path("experiments", "run_metadata.json").read_text(encoding="utf-8")
    )
    dataset = pd.read_csv(project_path("data", "processed", "modeling_dataset.csv"))
    diagnostics = build_diagnostic_report(dataset)
    report = generate_report(
        {
            "config": config,
            "forecasts": forecasts,
            "tables": tables,
            "overview": run_metadata,
            "latest_forecast": latest,
        },
        diagnostics,
    )
    print(report)


if __name__ == "__main__":
    main()

