#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot average compute and data transfer times by batch index "
            "for each workload_name."
        )
    )
    p.add_argument(
        "--input",
        default="data/logs/benchmarks-logs.csv",
        help="Path to benchmarks CSV (default: data/logs/benchmarks-logs.csv)",
    )
    p.add_argument(
        "--output",
        default="plots/benchmarks_times_by_batch.png",
        help="Output image path (default: plots/benchmarks_times_by_batch.png)",
    )
    p.add_argument(
        "--title",
        default="Average Time by Batch Index",
        help="Figure title",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150)",
    )
    return p.parse_args()


def read_averages(path):
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            workload = row.get("workload_name", "").strip()
            wtype = row.get("workload_type", "").strip()
            batch = row.get("batch_index", "").strip()
            time_ms = row.get("time_ms", "").strip()

            if not workload or not wtype or not batch or not time_ms:
                continue

            try:
                batch_i = int(batch)
                time_val = float(time_ms)
            except ValueError:
                continue

            sums[workload][batch_i][wtype] += time_val
            counts[workload][batch_i][wtype] += 1

    avgs = defaultdict(dict)
    for workload, batch_map in sums.items():
        for batch_i, type_map in batch_map.items():
            avgs[workload][batch_i] = {}
            for wtype, total in type_map.items():
                count = counts[workload][batch_i][wtype]
                if count:
                    avgs[workload][batch_i][wtype] = total / count

    return avgs


def plot(avgs, title, output_path, dpi):
    workload_names = sorted(avgs.keys())
    if not workload_names:
        raise SystemExit("No data found to plot.")

    n = len(workload_names)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.6 * rows), sharex=False)
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, workload in zip(axes, workload_names):
        batch_map = avgs[workload]
        batches = sorted(batch_map.keys())

        compute = [batch_map[b].get("compute") for b in batches]
        transfer = [batch_map[b].get("data_transfer") for b in batches]

        ax.plot(batches, compute, marker="o", label="compute")
        ax.plot(batches, transfer, marker="s", label="data_transfer")
        ax.set_title(workload)
        ax.set_xlabel("batch_index")
        ax.set_ylabel("avg time_ms")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[len(workload_names) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi)


def main():
    args = parse_args()
    avgs = read_averages(args.input)
    plot(avgs, args.title, args.output, args.dpi)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
