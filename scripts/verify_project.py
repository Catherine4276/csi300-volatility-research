"""Verify temporal integrity, artifacts, and Skill structure after a research run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_skill_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md is missing YAML frontmatter")
    _, frontmatter, _ = text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    if metadata.get("name") != "quant-research":
        raise ValueError("Skill name must be quant-research")
    if not metadata.get("description"):
        raise ValueError("Skill description is empty")
    if "TODO" in text or "[TODO" in text:
        raise ValueError("Skill contains unresolved TODO placeholders")
    return metadata


def main() -> None:
    required = [
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "config" / "default.yaml",
        PROJECT_ROOT / ".agents" / "skills" / "quant-research" / "SKILL.md",
        PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv",
        PROJECT_ROOT / "reports" / "forecasts.csv",
        PROJECT_ROOT / "reports" / "research_report.md",
        PROJECT_ROOT / "experiments" / "run_metadata.json",
        PROJECT_ROOT / "dashboard" / "app.py",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    forecasts = pd.read_csv(
        PROJECT_ROOT / "reports" / "forecasts.csv",
        parse_dates=["forecast_origin", "target_date"],
    )
    if forecasts.empty:
        raise ValueError("Forecast artifact is empty")
    if not forecasts["forecast_origin"].is_monotonic_increasing:
        raise ValueError("Forecast origins are not chronological")
    if not (forecasts["target_date"] > forecasts["forecast_origin"]).all():
        raise ValueError("A forecast target is not strictly in the future")
    if (pd.to_datetime(forecasts["train_end"]) >= forecasts["forecast_origin"]).any():
        raise ValueError("Training data overlaps a forecast origin")
    model_columns = ["pred_har", "pred_xgb", "pred_static", "pred_dynamic"]
    if not forecasts[model_columns].apply(lambda column: column.notna().all()).all():
        raise ValueError("A model prediction is missing")
    if not (forecasts[model_columns] > 0).all().all():
        raise ValueError("A variance prediction is non-positive")
    if not forecasts["weight_xgb"].between(0.05, 0.95).all():
        raise ValueError("Dynamic weight violates the convex bounded support")
    direction = forecasts.loc[forecasts["weight_source_rows"] > 0]
    if (direction.loc[direction["regime"] != 2, "weight_xgb"] < 0.5).any():
        raise ValueError("Normal-state XGBoost direction constraint failed")
    if (direction.loc[direction["regime"] == 2, "weight_xgb"] > 0.5).any():
        raise ValueError("High-state HAR direction constraint failed")

    skill_metadata = read_skill_frontmatter(required[4])
    run_metadata = json.loads(required[8].read_text(encoding="utf-8"))
    result = {
        "status": "pass",
        "forecast_rows": int(len(forecasts)),
        "forecast_start": forecasts["forecast_origin"].min().date().isoformat(),
        "forecast_end": forecasts["forecast_origin"].max().date().isoformat(),
        "strict_future_targets": True,
        "training_origin_overlap_rows": 0,
        "bounded_dynamic_weights": True,
        "regime_direction_constraints": True,
        "skill": skill_metadata,
        "run_id": run_metadata.get("run_id"),
        "source_sha256": run_metadata.get("source_sha256"),
        "package_versions": run_metadata.get("package_versions"),
    }
    output_path = PROJECT_ROOT / "reports" / "verification.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

