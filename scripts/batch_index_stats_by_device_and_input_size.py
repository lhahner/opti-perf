#!/usr/bin/env python3
"""Compute batch-index stats grouped by framework, device_name, and input_size.

Outputs a CSV with separate compute and data-transfer stats per batch index.

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
        description="Compute batch-index stats grouped by framework, device_name, and input_size."
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
    required = [
        "framework",
        "device_name",
        "input_size",
        "batch_index",
        "workload_type",
        "time_ms",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in input: {missing}")

    df = df.copy()
    df["framework"] = df["framework"].astype(str).str.strip()
    df["workload_type"] = df["workload_type"].astype(str).str.strip()
    df = df[df["framework"].isin(["OpenCL", "CUDA"])]
    df = df[df["workload_type"].isin(["compute", "data_transfer"])]

    for col in ["input_size", "batch_index", "time_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df[df["input_size"] >= 0]
    # Exclude non-batch entries. The logs contain many batch_index=0 rows that
    # represent separate one-off measurements rather than the comparable batch sequence.
    df = df[df["batch_index"] > 0]

    stats_long = (
        df.groupby(
            ["framework", "device_name", "input_size", "batch_index", "workload_type"],
            as_index=False,
        )["time_ms"]
        .agg(
            count="count",
            mean_ms="mean",
            var_ms="var",
            std_ms="std",
        )
        .sort_values(
            ["framework", "device_name", "input_size", "batch_index", "workload_type"]
        )
    )

    stats = (
        stats_long.set_index(
            ["framework", "device_name", "input_size", "batch_index", "workload_type"]
        )
        .unstack("workload_type")
        .sort_index()
    )
    stats.columns = [
        f"{workload_type}_{metric}" for metric, workload_type in stats.columns.to_flat_index()
    ]
    stats = stats.reset_index()

    stats.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
