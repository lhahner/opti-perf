#!/usr/bin/env python3
"""Plot per-batch measurement spread for OpenCL and CUDA.

Creates one figure per device and framework. Each figure contains separate
compute and data-transfer panels with raw measurement points and a mean +/- std
overlay for every input_size.

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


TARGET_FRAMEWORKS = ["OpenCL", "CUDA"]
TARGET_WORKLOADS = ["compute", "data_transfer"]

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


def save_fig(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_framework_device(
    raw_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    device_key: str,
    framework: str,
    outdir: Path,
) -> None:
    subset = raw_df[
        (raw_df["device_key"] == device_key) & (raw_df["framework"] == framework)
    ]
    if subset.empty:
        return

    stat_subset = stats_df[
        (stats_df["device_key"] == device_key) & (stats_df["framework"] == framework)
    ]
    if stat_subset.empty:
        return

    sizes = sorted(subset["input_size"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(1, len(sizes)))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    for ax, workload_type in zip(axes, TARGET_WORKLOADS):
        workload_raw = subset[subset["workload_type"] == workload_type]
        workload_stats = stat_subset[stat_subset["workload_type"] == workload_type]

        if workload_raw.empty or workload_stats.empty:
            ax.set_visible(False)
            continue

        for idx, size in enumerate(sizes):
            size_raw = workload_raw[workload_raw["input_size"] == size]
            size_stats = workload_stats[workload_stats["input_size"] == size]
            if size_raw.empty or size_stats.empty:
                continue

            color = cmap(idx)
            ax.scatter(
                size_raw["batch_index"],
                size_raw["time_ms"],
                s=16,
                alpha=0.18,
                color=color,
                edgecolors="none",
            )
            ax.plot(
                size_stats["batch_index"],
                size_stats["mean_ms"],
                color=color,
                linewidth=1.8,
                label=f"input_size={int(size)}",
            )
            lower = size_stats["mean_ms"] - size_stats["std_ms"]
            upper = size_stats["mean_ms"] + size_stats["std_ms"]
            ax.fill_between(
                size_stats["batch_index"],
                lower,
                upper,
                color=color,
                alpha=0.16,
            )

        ax.set_ylabel("Time (ms)")
        ax.set_title(
            "Compute" if workload_type == "compute" else "Data Transfer",
            fontsize=11,
        )
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Batch index")
    device_label = DEVICE_LABELS[device_key]
    fig.suptitle(f"{device_label}: {framework} Batch-Index Spread", fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(title="Workload size", fontsize=8)

    framework_slug = framework.lower()
    filename = f"{device_key}_{framework_slug}_batch_index_spread.png"
    save_fig(fig, outdir, filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-batch measurement spread grouped by framework, workload size, "
            "and workload type for each device."
        )
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
    df["framework"] = df.get("framework", "").fillna("").astype(str).str.strip()
    df["workload_type"] = df.get("workload_type", "").fillna("").astype(str).str.strip()
    df["device_key"] = df["device_name"].apply(pick_device_key)

    for col in ["batch_index", "time_ms", "input_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["device_key"].notna()
        & df["framework"].isin(TARGET_FRAMEWORKS)
        & df["workload_type"].isin(TARGET_WORKLOADS)
    ]
    df = df.dropna(subset=["batch_index", "time_ms", "input_size"])
    df = df[(df["input_size"] >= 0) & (df["batch_index"] > 0)]

    stats = (
        df.groupby(
            ["device_key", "framework", "workload_type", "input_size", "batch_index"],
            as_index=False,
        )["time_ms"]
        .agg(mean_ms="mean", std_ms="std")
        .sort_values(
            ["device_key", "framework", "workload_type", "input_size", "batch_index"]
        )
    )

    outdir = Path(args.outdir)

    for device_key in DEVICE_LABELS:
        for framework in TARGET_FRAMEWORKS:
            plot_framework_device(df, stats, device_key, framework, outdir)

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
