"""Data acquisition, normalization, validation, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DAILY_FIELDS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "change",
    "pct_change",
    "turnover",
]
INTRADAY_FIELDS = [
    "datetime",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "change",
    "pct_change",
    "turnover",
]
NUMERIC_DAILY_COLUMNS = DAILY_FIELDS[1:]


def _session() -> requests.Session:
    """Create a retrying HTTP session for a public data endpoint."""

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; CSI300-Volatility-Research/1.0)",
            "Referer": "https://quote.eastmoney.com/",
        }
    )
    return session


def _eastmoney_request(secid: str, klt: int, limit: int) -> dict[str, Any]:
    """Fetch OHLC bars from Eastmoney's public historical quote endpoint."""

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "klt": klt,
        "fqt": 0,
        "lmt": limit,
        "end": "20500101",
        "iscca": 1,
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    response = _session().get(url, params=params, timeout=(10, 60))
    response.raise_for_status()
    payload = response.json()
    if payload.get("rc") != 0 or not payload.get("data"):
        raise RuntimeError(f"Unexpected Eastmoney response: {payload.get('rc')}")
    return payload


def _normalize_frame(raw: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Normalize provider columns without changing prices or volumes."""

    missing = sorted(set(fields) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required provider columns: {missing}")
    frame = raw.loc[:, fields].copy()
    date_column = fields[0]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    for column in fields[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates(subset=[date_column], keep="last")
    frame = frame.sort_values(date_column).reset_index(drop=True)
    return frame.rename(columns={date_column: "date"})


def fetch_eastmoney_daily(
    secid: str = "1.000300",
    limit: int = 10000,
) -> pd.DataFrame:
    """Download CSI 300 daily OHLCV history from a legally accessible public source."""

    payload = _eastmoney_request(secid=secid, klt=101, limit=limit)
    rows = [line.split(",") for line in payload["data"]["klines"]]
    raw = pd.DataFrame(rows, columns=DAILY_FIELDS)
    frame = _normalize_frame(raw, DAILY_FIELDS)
    validate_daily_frame(frame)
    return frame


def fetch_eastmoney_intraday(
    secid: str = "1.000300",
    period: int = 5,
    limit: int = 10000,
) -> pd.DataFrame:
    """Download the available public intraday bars.

    The provider currently exposes only a short rolling window of index
    intraday history. Callers must not assume this covers multi-year samples.
    """

    if period not in {1, 5, 15, 30, 60}:
        raise ValueError(f"Unsupported intraday period: {period}")
    payload = _eastmoney_request(secid=secid, klt=period, limit=limit)
    rows = [line.split(",") for line in payload["data"]["klines"]]
    raw = pd.DataFrame(rows, columns=INTRADAY_FIELDS)
    frame = _normalize_frame(raw, INTRADAY_FIELDS)
    validate_ohlc(frame)
    return frame


def validate_ohlc(frame: pd.DataFrame) -> None:
    """Reject structurally invalid OHLC rows and non-positive prices."""

    required = {"date", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if frame.empty:
        raise ValueError("OHLC frame is empty")
    prices = frame[["open", "high", "low", "close"]]
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("OHLC contains missing or non-finite values")
    if (prices <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    invalid = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame["high"] < frame["low"])
    )
    if invalid.any():
        first_bad = frame.loc[invalid, "date"].iloc[0]
        raise ValueError(f"Invalid OHLC ordering at {first_bad}")


def validate_daily_frame(frame: pd.DataFrame) -> None:
    """Run completeness, ordering, OHLC, and volume checks."""

    validate_ohlc(frame)
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate trading dates detected")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Trading dates are not strictly increasing")
    for column in ("volume", "amount"):
        if column in frame and (frame[column].dropna() < 0).any():
            raise ValueError(f"Negative values detected in {column}")
    numeric = frame.loc[:, NUMERIC_DAILY_COLUMNS]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Daily data contains non-finite numeric values")


def load_daily_csv(path: str | Path) -> pd.DataFrame:
    """Load an external daily CSI 300 CSV using the project schema."""

    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError("CSV must contain a 'date' column")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in NUMERIC_DAILY_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    validate_daily_frame(frame)
    return frame


def file_sha256(path: str | Path) -> str:
    """Calculate a deterministic data checksum."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv_with_manifest(
    frame: pd.DataFrame,
    output_path: str | Path,
    metadata_path: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Save a data artifact and a sidecar manifest recording its provenance."""

    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    manifest = {
        **metadata,
        "output_file": output_path.name,
        "rows": int(len(frame)),
        "start_date": frame["date"].min().isoformat() if "date" in frame else None,
        "end_date": frame["date"].max().isoformat() if "date" in frame else None,
        "sha256": file_sha256(output_path),
        "retrieved_or_generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest

