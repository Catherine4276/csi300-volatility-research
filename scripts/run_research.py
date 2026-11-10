"""Run the full walk-forward study and generate the Markdown report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_research  # noqa: E402
from src.config import project_path  # noqa: E402
from src.diagnostics import build_diagnostic_report  # noqa: E402
from src.reporting import generate_report  # noqa: E402


def main() -> None:
    result = run_research()
    dataset = pd.read_csv(project_path("data", "processed", "modeling_dataset.csv"))
    diagnostics = build_diagnostic_report(dataset)
    report_path = generate_report(result, diagnostics)
    summary = {
        "report": str(report_path),
        "forecasts": len(result["forecasts"]),
        "latest_regime": result["latest_forecast"]["regime_label"],
        "latest_dynamic_annualized_vol": result["latest_forecast"][
            "annualized_vol_dynamic"
        ],
        "latest_high_risk_alert": result["latest_forecast"]["high_risk_alert"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

