#!/usr/bin/env python3
"""Generate benchmark plots from CSV logs.

Usage:
  python3 scripts/plot_benchmarks.py --input data/benchmarks/benchmarks-logs.csv --outdir data/benchmarks/plots
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def _to_int(value: str) -> int:
    return int(value) if value != "" else 0


def _to_float(value: str) -> float:
    return float(value) if value != "" else 0.0


def read_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "framework": row["framework"],
                    "workload_name": row["workload_name"],
                    "workload_type": row["workload_type"],
                    "device": row["device"],
                    "batch_size": _to_int(row["batch_size"]),
                    "input_size": _to_int(row["input_size"]),
                    "optimizer": row["optimizer"],
                    "learning_rate": _to_float(row["learning_rate"]),
                    "beta1": _to_float(row["beta1"]),
                    "beta2": _to_float(row["beta2"]),
                    "epsilon": _to_float(row["epsilon"]),
                    "time_ms": _to_float(row["time_ms"]),
                    "batch_index": _to_int(row["batch_index"]),
                    "loss": _to_float(row["loss"]),
                }
            )
    return rows


def group_values(rows: Iterable[Dict[str, object]], key_fn) -> Dict[Tuple[object, ...], List[float]]:
    grouped: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for r in rows:
        grouped[key_fn(r)].append(float(r["time_ms"]))
    return grouped


def save_fig(outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_mean_time_by_framework_device(rows: List[Dict[str, object]], outdir: Path) -> None:
    frameworks = sorted({r["framework"] for r in rows})
    devices = sorted({r["device"] for r in rows})

    grouped = group_values(rows, lambda r: (r["framework"], r["device"]))

    width = 0.8 / max(1, len(devices))
    x = list(range(len(frameworks)))

    plt.figure(figsize=(8, 4.5))
    for i, device in enumerate(devices):
        values = []
        for fw in frameworks:
            key = (fw, device)
            values.append(mean(grouped[key]) if key in grouped else 0.0)
        offset = [xi + (i - (len(devices) - 1) / 2) * width for xi in x]
        plt.bar(offset, values, width=width, label=str(device))

    plt.xticks(x, frameworks, rotation=0)
    plt.ylabel("Mean time (ms)")
    plt.title("Mean Time by Framework and Device")
    plt.legend()

    save_fig(outdir, "mean_time_by_framework_device.png")


def plot_mean_time_by_framework_workload_gpu(rows: List[Dict[str, object]], outdir: Path) -> None:
    gpu_rows = [r for r in rows if r["device"] == "GPU"]
    if not gpu_rows:
        return

    frameworks = sorted({r["framework"] for r in gpu_rows})
    workload_types = sorted({r["workload_type"] for r in gpu_rows})

    grouped = group_values(gpu_rows, lambda r: (r["framework"], r["workload_type"]))

    width = 0.8 / max(1, len(workload_types))
    x = list(range(len(frameworks)))

    plt.figure(figsize=(8, 4.5))
    for i, wtype in enumerate(workload_types):
        values = []
        for fw in frameworks:
            key = (fw, wtype)
            values.append(mean(grouped[key]) if key in grouped else 0.0)
        offset = [xi + (i - (len(workload_types) - 1) / 2) * width for xi in x]
        plt.bar(offset, values, width=width, label=str(wtype))

    plt.xticks(x, frameworks, rotation=0)
    plt.ylabel("Mean time (ms)")
    plt.title("GPU Mean Time by Framework and Workload Type")
    plt.legend()

    save_fig(outdir, "gpu_mean_time_by_framework_workload_type.png")


def plot_boxplot_time_by_framework_gpu_compute(rows: List[Dict[str, object]], outdir: Path) -> None:
    filtered = [
        r for r in rows if r["device"] == "GPU" and r["workload_type"] == "compute"
    ]
    if not filtered:
        return

    frameworks = sorted({r["framework"] for r in filtered})
    series = [
        [float(r["time_ms"]) for r in filtered if r["framework"] == fw]
        for fw in frameworks
    ]

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(series, tick_labels=frameworks, showfliers=False)
    plt.ylabel("Time (ms)")
    plt.title("GPU Compute Time Distribution by Framework")

    save_fig(outdir, "gpu_compute_time_boxplot.png")


def plot_time_over_batch_index_gpu_compute(rows: List[Dict[str, object]], outdir: Path) -> None:
    filtered = [
        r for r in rows if r["device"] == "GPU" and r["workload_type"] == "compute"
    ]
    if not filtered:
        return

    frameworks = sorted({r["framework"] for r in filtered})

    plt.figure(figsize=(8, 4.5))
    for fw in frameworks:
        fw_rows = [r for r in filtered if r["framework"] == fw]
        by_batch: Dict[int, List[float]] = defaultdict(list)
        for r in fw_rows:
            by_batch[int(r["batch_index"])].append(float(r["time_ms"]))
        batch_indices = sorted(by_batch)
        means = [mean(by_batch[i]) for i in batch_indices]
        plt.plot(batch_indices, means, marker="o", linewidth=1.5, label=str(fw))

    plt.xlabel("Batch index")
    plt.ylabel("Mean time (ms)")
    plt.title("GPU Compute Time by Batch Index")
    plt.legend()

    save_fig(outdir, "gpu_compute_time_over_batch.png")


def plot_counts_by_category(rows: List[Dict[str, object]], outdir: Path) -> None:
    categories = [
        ("framework", "Framework"),
        ("device", "Device"),
        ("workload_name", "Workload Name"),
        ("workload_type", "Workload Type"),
        ("optimizer", "Optimizer"),
    ]
    for key, label in categories:
        counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            counts[str(r[key])] += 1
        names = sorted(counts)
        values = [counts[n] for n in names]
        plt.figure(figsize=(8, 4.5))
        plt.bar(names, values)
        plt.ylabel("Count")
        plt.title(f"Record Count by {label}")
        plt.xticks(rotation=20, ha="right")
        save_fig(outdir, f"count_by_{key}.png")


def plot_numeric_distributions(rows: List[Dict[str, object]], outdir: Path) -> None:
    numeric_keys = [
        ("time_ms", "Time (ms)"),
        ("loss", "Loss"),
        ("input_size", "Input Size"),
        ("batch_size", "Batch Size"),
        ("learning_rate", "Learning Rate"),
        ("beta1", "Beta1"),
        ("beta2", "Beta2"),
        ("epsilon", "Epsilon"),
    ]
    for key, label in numeric_keys:
        values = [float(r[key]) for r in rows]
        plt.figure(figsize=(7.5, 4.5))
        plt.hist(values, bins=30, color="#4C78A8", alpha=0.9)
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {label}")
        save_fig(outdir, f"hist_{key}.png")


def plot_time_vs_input_size(rows: List[Dict[str, object]], outdir: Path) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for key in sorted({(r["framework"], r["device"]) for r in rows}):
        fw, device = key
        xs = [float(r["input_size"]) for r in rows if r["framework"] == fw and r["device"] == device]
        ys = [float(r["time_ms"]) for r in rows if r["framework"] == fw and r["device"] == device]
        if xs:
            plt.scatter(xs, ys, s=18, alpha=0.6, label=f"{fw}-{device}")
    plt.xlabel("Input size")
    plt.ylabel("Time (ms)")
    plt.title("Time vs Input Size by Framework/Device")
    plt.legend()
    save_fig(outdir, "time_vs_input_size.png")


def plot_time_vs_batch_size(rows: List[Dict[str, object]], outdir: Path) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for key in sorted({(r["framework"], r["device"]) for r in rows}):
        fw, device = key
        xs = [float(r["batch_size"]) for r in rows if r["framework"] == fw and r["device"] == device]
        ys = [float(r["time_ms"]) for r in rows if r["framework"] == fw and r["device"] == device]
        if xs:
            plt.scatter(xs, ys, s=18, alpha=0.6, label=f"{fw}-{device}")
    plt.xlabel("Batch size")
    plt.ylabel("Time (ms)")
    plt.title("Time vs Batch Size by Framework/Device")
    plt.legend()
    save_fig(outdir, "time_vs_batch_size.png")


def plot_loss_over_batch_index(rows: List[Dict[str, object]], outdir: Path) -> None:
    filtered = [r for r in rows if float(r["loss"]) != 0.0]
    if not filtered:
        return
    plt.figure(figsize=(8, 4.5))
    for key in sorted({(r["framework"], r["device"]) for r in filtered}):
        fw, device = key
        fw_rows = [r for r in filtered if r["framework"] == fw and r["device"] == device]
        by_batch: Dict[int, List[float]] = defaultdict(list)
        for r in fw_rows:
            by_batch[int(r["batch_index"])].append(float(r["loss"]))
        batch_indices = sorted(by_batch)
        means = [mean(by_batch[i]) for i in batch_indices]
        plt.plot(batch_indices, means, marker="o", linewidth=1.5, label=f"{fw}-{device}")
    plt.xlabel("Batch index")
    plt.ylabel("Mean loss")
    plt.title("Loss over Batch Index by Framework/Device")
    plt.legend()
    save_fig(outdir, "loss_over_batch.png")


def plot_loss_vs_time(rows: List[Dict[str, object]], outdir: Path) -> None:
    filtered = [r for r in rows if float(r["loss"]) != 0.0]
    if not filtered:
        return
    plt.figure(figsize=(7.5, 4.5))
    for key in sorted({(r["framework"], r["device"]) for r in filtered}):
        fw, device = key
        xs = [float(r["time_ms"]) for r in filtered if r["framework"] == fw and r["device"] == device]
        ys = [float(r["loss"]) for r in filtered if r["framework"] == fw and r["device"] == device]
        if xs:
            plt.scatter(xs, ys, s=18, alpha=0.6, label=f"{fw}-{device}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Loss")
    plt.title("Loss vs Time by Framework/Device")
    plt.legend()
    save_fig(outdir, "loss_vs_time.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark CSV metrics.")
    parser.add_argument(
        "--input",
        default="data/benchmarks/benchmarks-logs.csv",
        help="Path to benchmark CSV",
    )
    parser.add_argument(
        "--outdir",
        default="data/benchmarks/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    rows = read_rows(input_path)

    plot_mean_time_by_framework_device(rows, outdir)
    plot_mean_time_by_framework_workload_gpu(rows, outdir)
    plot_boxplot_time_by_framework_gpu_compute(rows, outdir)
    plot_time_over_batch_index_gpu_compute(rows, outdir)
    plot_counts_by_category(rows, outdir)
    plot_numeric_distributions(rows, outdir)
    plot_time_vs_input_size(rows, outdir)
    plot_time_vs_batch_size(rows, outdir)
    plot_loss_over_batch_index(rows, outdir)
    plot_loss_vs_time(rows, outdir)

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
