#!/usr/bin/env python3
"""Compute batch-index stats grouped by device_name and input_size.

Outputs a CSV with count, mean, variance, and std for time_ms.

Usage:
  python3 batch_index_stats_by_device_and_input_size.py \
    --input benchmarks.cleaned.csv \
    --output batch_index_stats.csv
"""

from __future__ import annotations

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute batch-index stats grouped by device_name and input_size."
    )
    parser.add_argument(
        "--input",
        default="benchmarks.cleaned.csv",
        help="Path to cleaned benchmark CSV",
    )
    parser.add_argument(
        "--output",
        default="batch_index_stats.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Ensure needed columns exist
    required = ["device_name", "input_size", "batch_index", "time_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in input: {missing}")

    df = df.copy()
    for col in ["input_size", "batch_index", "time_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df[df["input_size"] >= 0]

    stats = (
        df.groupby(["device_name", "input_size", "batch_index"], as_index=False)["time_ms"]
        .agg(
            count="count",
            mean_ms="mean",
            var_ms="var",
            std_ms="std",
        )
        .sort_values(["device_name", "input_size", "batch_index"])
    )

    stats.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
