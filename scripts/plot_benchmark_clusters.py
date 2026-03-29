#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PHASES = ("compute", "d2h_transfer")
PHASE_COLORS = {
    "compute": "#d95f02",
    "d2h_transfer": "#1b9e77",
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def load_rows(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create per-device GEMM scatter plots for compute and d2h_transfer times by batch index."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="../data/logs/benchmarks-logs.csv",
        help="Path to GEMM benchmark CSV",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="scripts/plots/benchmark_clusters",
        help="Directory for generated plots",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in load_rows(input_path):
        framework = (row.get("framework") or "").strip()
        device_name = (row.get("device_name") or "").strip()
        phase = (row.get("workload_type") or "").strip()
        workload_name = (row.get("workload_name") or "").strip()
        if framework not in {"CUDA", "OpenCL"}:
            continue
        if phase not in PHASES:
            continue
        if "GEMM" not in workload_name:
            continue
        batch_index = parse_int((row.get("batch_index") or "").strip())
        time_ms = parse_float((row.get("time_ms") or "").strip())
        if batch_index is None or time_ms is None:
            continue
        if not device_name:
            device_name = "unknown-device"
        grouped[(device_name, framework)][phase].append((batch_index, time_ms))

    if not grouped:
        raise SystemExit("No matching CUDA/OpenCL GEMM compute or d2h_transfer rows found in input CSV.")

    devices = sorted({device for device, _ in grouped.keys()})

    for device in devices:
        frameworks_present = [fw for fw in ("CUDA", "OpenCL") if (device, fw) in grouped]
        if not frameworks_present:
            continue

        fig, axes = plt.subplots(
            1,
            len(frameworks_present),
            figsize=(7 * len(frameworks_present), 5),
            squeeze=False,
        )
        fig.suptitle(device)

        for index, framework in enumerate(frameworks_present):
            ax = axes[0][index]
            framework_rows = grouped[(device, framework)]

            for phase in PHASES:
                points = sorted(framework_rows.get(phase, []))
                if not points:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                ax.scatter(
                    xs,
                    ys,
                    s=18,
                    alpha=0.7,
                    color=PHASE_COLORS[phase],
                    label=phase,
                )

            ax.set_title(framework)
            ax.set_xlabel("Batch index")
            ax.set_ylabel("time_ms")
            ax.grid(True, alpha=0.25)
            ax.legend()

        fig.tight_layout()
        output_path = outdir / f"{slugify(device)}_benchmark_clusters.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
