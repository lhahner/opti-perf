#!/usr/bin/env python3
"""Plot batch-index distributions for compute and data-transfer times by device.

Outputs one plot per device per workload type, coloring points by workload size.

Usage:
  python3 plot_batch_index_distributions.py \
    --input benchmarks-logs.cleaned.csv \
    --outdir plots/batch_index_distributions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEVICE_LABELS = {
    "gtx_970": "GTX 970",
    "a100": "A100",
    "rtx_5000": "RTX 5000",
}

DEVICE_MATCHERS = {
    "gtx_970": "gtx 970",
    "a100": "a100",
    "rtx_5000": "rtx 5000",
}

WORKLOAD_TYPES = ["data_transfer", "compute"]


def normalize_device_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def pick_device_key(device_name: str) -> str | None:
    name = normalize_device_name(device_name)
    for key, needle in DEVICE_MATCHERS.items():
        if needle in name:
            return key
    return None


def save_fig(outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_device_type(df: pd.DataFrame, device_key: str, workload_type: str, outdir: Path) -> None:
    subset = df[(df["device_key"] == device_key) & (df["workload_type"] == workload_type)]
    if subset.empty:
        return

    sizes = sorted(subset["input_size"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(1, len(sizes)))

    plt.figure(figsize=(8.5, 5))
    rng = np.random.default_rng(42)
    jitter_scale = 0.08
    for idx, size in enumerate(sizes):
        size_rows = subset[subset["input_size"] == size]
        jitter = rng.uniform(-jitter_scale, jitter_scale, size=len(size_rows))
        plt.scatter(
            size_rows["batch_index"] + jitter,
            size_rows["time_ms"],
            s=5,
            alpha=0.45,
            color=cmap(idx),
            label=f"input_size={int(size)}",
        )

    # Average time per batch index (across all sizes)
    avg_by_batch = (
        subset.groupby("batch_index", as_index=False)["time_ms"]
        .mean()
        .sort_values("batch_index")
    )
    if not avg_by_batch.empty:
        plt.plot(
            avg_by_batch["batch_index"],
            avg_by_batch["time_ms"],
            color="#1f1f1f",
            linewidth=2.0,
            label="average",
        )

    plt.xlabel("Batch index")
    ylabel = "Time (ms)"
    plt.ylabel(ylabel)
    device_label = DEVICE_LABELS[device_key]
    title = f"{device_label}: {workload_type.replace('_', ' ').title()} Time by Batch Index"
    plt.title(title)
    plt.legend(title="Workload size", fontsize=8)

    filename = f"{device_key}_{workload_type}_batch_index.png"
    save_fig(outdir, filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot batch-index distributions for compute and data-transfer times by device."
    )
    parser.add_argument(
        "--input",
        default="benchmarks-logs.cleaned.csv",
        help="Path to cleaned benchmark CSV",
    )
    parser.add_argument(
        "--outdir",
        default="plots/batch_index_distributions",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df = df.copy()
    df["device_name"] = df["device_name"].fillna("")
    df["device_key"] = df["device_name"].apply(pick_device_key)
    df = df[df["device_key"].notna()]

    # Ensure numeric columns are numeric
    for col in ["batch_index", "time_ms", "input_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["batch_index", "time_ms", "input_size", "workload_type"])
    df = df[df["input_size"] >= 0]

    outdir = Path(args.outdir)

    for device_key in DEVICE_LABELS:
        for workload_type in WORKLOAD_TYPES:
            plot_device_type(df, device_key, workload_type, outdir)

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
