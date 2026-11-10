from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_runs_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60)
    app.run()
    assert not app.exception
    assert len(app.tabs) == 6
    assert len(app.get("plotly_chart")) >= 4

