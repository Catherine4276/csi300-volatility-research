#!/usr/bin/env python3
"""Audit basic temporal integrity in a CSV dataset or forecast export."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_timestamp(value: str, column: str) -> datetime:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"Empty timestamp in column {column!r}")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Unparseable timestamp {text!r} in {column!r}") from exc


def audit(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no data rows")

    origin_column = args.origin_column
    target_column = args.target_column
    date_column = args.date_column
    active_date_column = (
        origin_column if origin_column and origin_column in rows[0] else date_column
    )
    if active_date_column not in rows[0]:
        raise ValueError(f"Missing date column {active_date_column!r}")

    timestamps = [
        parse_timestamp(row[active_date_column], active_date_column) for row in rows
    ]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(
            f"{active_date_column!r} must be strictly increasing without duplicates"
        )

    checked_target = False
    if target_column and target_column in rows[0]:
        for index, row in enumerate(rows):
            origin = parse_timestamp(row[active_date_column], active_date_column)
            target_text = (row[target_column] or "").strip()
            if not target_text:
                continue
            target = parse_timestamp(target_text, target_column)
            if target <= origin:
                raise ValueError(
                    f"Row {index + 2}: target {target.isoformat()} is not after origin "
                    f"{origin.isoformat()}"
                )
        checked_target = True

    split_summary = None
    if args.split_column:
        if args.split_column not in rows[0]:
            raise ValueError(f"Missing split column {args.split_column!r}")
        seen_validation = False
        for index, row in enumerate(rows):
            split = (row[args.split_column] or "").strip().lower()
            if split in {"validation", "valid", "test", "oos"}:
                seen_validation = True
            elif split in {"train", "training"} and seen_validation:
                raise ValueError(
                    f"Row {index + 2}: training observation appears after validation/test"
                )
        split_summary = sorted({(row[args.split_column] or "").strip() for row in rows})

    if args.train_end and args.test_start:
        train_end = parse_timestamp(args.train_end, "--train-end")
        test_start = parse_timestamp(args.test_start, "--test-start")
        if test_start <= train_end:
            raise ValueError("--test-start must be strictly after --train-end")

    return {
        "status": "pass",
        "path": str(path.resolve()),
        "rows": len(rows),
        "date_column": active_date_column,
        "start": timestamps[0].isoformat(),
        "end": timestamps[-1].isoformat(),
        "strictly_increasing_unique": True,
        "target_checked": checked_target,
        "target_column": target_column if checked_target else None,
        "split_column": args.split_column,
        "split_values": split_summary,
        "train_end": args.train_end,
        "test_start": args.test_start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="CSV dataset or forecast export")
    parser.add_argument("--origin-column", default="forecast_origin")
    parser.add_argument("--target-column", default="target_date")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--split-column")
    parser.add_argument("--train-end")
    parser.add_argument("--test-start")
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "path": str(Path(args.path).resolve()),
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

