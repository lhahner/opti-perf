#!/usr/bin/env python3
"""Plot device-wise OpenCL vs CUDA compute-time comparisons.

Usage:
    python plot_opencl_cuda_devicewise.py \
        --input ../data/logs/benchmarks-logs.csv \
        --outdir ./plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"   # or "Times New Roman", "Arial", etc.
plt.rcParams["font.size"] = 11


TARGET_DEVICES = [
    "NVIDIA GeForce GTX 970",
    "NVIDIA A100-SXM4-80GB",
    "Quadro RTX 5000",
]
TARGET_FRAMEWORKS = ["OpenCL", "CUDA"]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep the hardware model column. The CSV contains two "device" columns,
    # where the second one is usually GPU/CPU.
    model_col = "device"
    if "device.1" in df.columns and df["device"].isin(["GPU", "CPU"]).all():
        model_col = "device.1"

    work = df.copy()
    work["device_model"] = work[model_col].astype(str).str.strip()
    work["framework"] = work["framework"].astype(str).str.strip()

    filtered = work[
        work["device_model"].isin(TARGET_DEVICES)
        & work["framework"].isin(TARGET_FRAMEWORKS)
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No matching rows for target devices/frameworks were found in the CSV."
        )

    return filtered


def plot_devicewise_lines(df: pd.DataFrame, outdir: Path) -> None:
    """Create one normalized line chart per device for avg time vs batch index."""
    selected = df[df["workload_type"].isin(["compute", "data_transfer"])].copy()
    if selected.empty:
        return

    trend = (
        selected.groupby(
            ["device_model", "framework", "workload_type", "batch_index"], as_index=False
        )["time_ms"]
        .mean()
        .sort_values(["device_model", "framework", "workload_type", "batch_index"])
    )

    colors = {"OpenCL": "#fc2803", "CUDA": "#20fc03"}
    styles = {"compute": "-", "data_transfer": "--"}

    for device in TARGET_DEVICES:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax_loss = ax.twinx()
        dsub = trend[trend["device_model"] == device]
        for fw in TARGET_FRAMEWORKS:
            for workload_type in ["compute", "data_transfer"]:
                fsub = dsub[
                    (dsub["framework"] == fw) & (dsub["workload_type"] == workload_type)
                ]
                if fsub.empty:
                    continue
                # Normalize each framework/workload series over its batch-index range.
                # This keeps shape differences while removing absolute scale.
                max_val = fsub["time_ms"].max()
                if pd.isna(max_val) or max_val <= 0:
                    continue
                norm_time = fsub["time_ms"] / max_val
                
                average_workload_type = fsub["time_ms"].mean()
                print(f"Average {workload_type} of device: {device}, framework: {fw}, workload_type: {workload_type} is {average_workload_type}")

                ax.plot(
                    fsub["batch_index"],
                    fsub["time_ms"],
                    marker="o",
                    linewidth=2,
                    markersize=3,
                    linestyle=styles[workload_type],
                    label=f"{fw} {workload_type}",
                    color=colors[fw],
                )

        loss_sub = (
            df[(df["device_model"] == device) & df["loss"].notna()]
            .groupby("batch_index", as_index=False)["loss"]
            .mean()
            .sort_values("batch_index")
        )
        if not loss_sub.empty:
            # Seed the first batch loss if it is zero/non-positive so the curve
            # does not start from an artificial flat baseline.
            first_idx = loss_sub.index[0]
            first_loss = float(loss_sub.loc[first_idx, "loss"])
            if first_loss <= 0:
                positive_losses = loss_sub.loc[loss_sub["loss"] > 0, "loss"]
                if not positive_losses.empty:
                    loss_sub.loc[first_idx, "loss"] = float(positive_losses.iloc[0]) * 1.05

            baseline = loss_sub["loss"].iloc[0]
            if pd.notna(baseline) and baseline != 0:
                loss_for_plot = loss_sub["loss"] / baseline
            else:
                max_loss = loss_sub["loss"].max()
                loss_for_plot = loss_sub["loss"] / max_loss if pd.notna(max_loss) and max_loss > 0 else loss_sub["loss"]

            ax_loss.plot(
                loss_sub["batch_index"],
                loss_for_plot,
                marker="s",
                linewidth=2,
                markersize=3,
                linestyle=":",
                label="avg loss (normalized)",
                color="#1f77b4",
            )
            ymin = max(0.01, float(loss_for_plot.min()) * 0.95)
            ymax = float(loss_for_plot.max()) * 1.05
            ax_loss.set_ylim(ymin, ymax)
            ax_loss.grid(False)
            ax_loss.tick_params(axis="y", labelcolor="#1f77b4")

        ax.set_title(f"{device}: CUDA vs OpenCL")
        ax.set_xlabel("Batch Index")
        ax.set_ylabel("Average Time in ms")
        ax.grid(True, linestyle="--", alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_loss.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="best",
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
        )
        fig.tight_layout()
        filename = (
            f"{device.lower().replace(' ', '_').replace('-', '_')}"
            "_compute_and_transfer_batch_line_normalized.png"
        )
        fig.savefig(outdir / filename, dpi=160)
        plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OpenCL vs CUDA device-wise benchmark graphics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data/logs/benchmarks-logs.csv"),
        help="Path to benchmarks CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("../data/logs/plots"),
        help="Output directory for generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)

    plot_devicewise_lines(df, args.outdir)

    print(f"Plots generated in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
