#!/usr/bin/env python3
"""Identify malformed numeric values and normalize decimal separators in a CSV.

Default behavior:
- scans likely numeric columns
- reports all rows with non-standard numeric strings
- converts comma decimal separators to dot when safe (e.g. 0,123 -> 0.123)
- writes a cleaned CSV to a new file
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS = [
    "batch_size",
    "input_size",
    "learning_rate",
    "beta1",
    "beta2",
    "epsilon",
    "time_ms",
    "batch_index",
    "loss",
]

VALID_NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")


def is_valid_numeric_string(s: str) -> bool:
    return bool(VALID_NUMERIC_RE.fullmatch(s))


def normalize_numeric_string(s: str) -> tuple[str, bool]:
    """Return (normalized_value, changed)."""
    raw = s.strip()
    if raw == "":
        return raw, False

    if is_valid_numeric_string(raw):
        return raw, False

    # Safe decimal-comma replacement (e.g. 0,12345 -> 0.12345)
    if "," in raw and "." not in raw and raw.count(",") == 1:
        candidate = raw.replace(",", ".")
        if is_valid_numeric_string(candidate):
            return candidate, True

    # Could add locale-aware thousands-separator handling here if needed.
    return raw, False


def scan_and_fix(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[dict], int]:
    fixed = df.copy()
    issues: list[dict] = []
    changes = 0

    for col in columns:
        if col not in fixed.columns:
            continue

        series = fixed[col].astype(str)
        new_vals = []
        for idx, val in series.items():
            original = val.strip()
            normalized, changed = normalize_numeric_string(original)

            if original != "" and not is_valid_numeric_string(original):
                issues.append(
                    {
                        "row_index": idx,
                        "column": col,
                        "original": original,
                        "normalized": normalized if changed else "",
                        "auto_fixed": changed,
                    }
                )

            if changed:
                changes += 1
                new_vals.append(normalized)
            else:
                new_vals.append(val)

        fixed[col] = new_vals

    return fixed, issues, changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and fix malformed numeric values in benchmark CSV.")
    parser.add_argument("--input", type=Path, default=Path("../data/logs/benchmarks-logs.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmarks-logs.cleaned.csv"),
        help="Path for cleaned CSV output.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("./numeric_issues_report.csv"),
        help="Path for issues report CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, dtype=str)
    fixed, issues, changes = scan_and_fix(df, NUMERIC_COLUMNS)

    issues_df = pd.DataFrame(issues)
    if not issues_df.empty:
        issues_df.to_csv(args.report, index=False)
    else:
        # Keep behavior explicit for downstream automation
        pd.DataFrame(columns=["row_index", "column", "original", "normalized", "auto_fixed"]).to_csv(
            args.report, index=False
        )

    fixed.to_csv(args.output, index=False)

    unresolved = sum(1 for issue in issues if not issue["auto_fixed"])
    print(f"Input:      {args.input}")
    print(f"Cleaned:    {args.output}")
    print(f"Report:     {args.report}")
    print(f"Issues:     {len(issues)}")
    print(f"Auto-fixed: {changes}")
    print(f"Unresolved: {unresolved}")


if __name__ == "__main__":
    main()
