#!/usr/bin/env python3
"""Plot variance and standard deviation of time per batch index.

Grouped by workload size (input_size) for each device.

Usage:
  python3 plot_batch_index_variance_std.py \
    --input benchmarks.cleaned.csv \
    --outdir plots/batch_index_stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_stat(
    df: pd.DataFrame,
    device_key: str,
    stat_col: str,
    stat_label: str,
    outdir: Path,
) -> None:
    subset = df[df["device_key"] == device_key]
    if subset.empty:
        return

    sizes = sorted(subset["input_size"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(1, len(sizes)))

    plt.figure(figsize=(8.5, 5))
    for idx, size in enumerate(sizes):
        size_rows = subset[subset["input_size"] == size]
        if size_rows.empty:
            continue
        plt.plot(
            size_rows["batch_index"],
            size_rows[stat_col],
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            color=cmap(idx),
            label=f"input_size={int(size)}",
        )

    plt.xlabel("Batch index")
    plt.ylabel(stat_label)
    device_label = DEVICE_LABELS[device_key]
    plt.title(f"{device_label}: {stat_label} by Batch Index")
    plt.legend(title="Workload size", fontsize=8)

    filename = f"{device_key}_{stat_col}_by_batch_index.png"
    save_fig(outdir, filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot variance and standard deviation per batch index, grouped by workload size for each device."
    )
    parser.add_argument(
        "--input",
        default="benchmarks.cleaned.csv",
        help="Path to cleaned benchmark CSV",
    )
    parser.add_argument(
        "--outdir",
        default="plots/batch_index_stats",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df = df.copy()
    df["device_name"] = df.get("device_name", "").fillna("")
    df["device_key"] = df["device_name"].apply(pick_device_key)
    df = df[df["device_key"].notna()]

    # Ensure numeric columns are numeric
    for col in ["batch_index", "time_ms", "input_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["batch_index", "time_ms", "input_size"])
    df = df[df["input_size"] >= 0]

    stats = (
        df.groupby(["device_key", "input_size", "batch_index"], as_index=False)["time_ms"]
        .agg(var_ms="var", std_ms="std")
        .sort_values(["device_key", "input_size", "batch_index"])
    )

    outdir = Path(args.outdir)

    for device_key in DEVICE_LABELS:
        device_rows = stats[stats["device_key"] == device_key]
        if device_rows.empty:
            continue
        plot_stat(device_rows, device_key, "var_ms", "Variance (ms^2)", outdir)
        plot_stat(device_rows, device_key, "std_ms", "Std Dev (ms)", outdir)

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
