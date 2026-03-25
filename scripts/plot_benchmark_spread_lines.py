#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


FRAMEWORK_COLORS = {
    "CUDA": "#1b9e77",
    "OpenCL": "#d95f02",
}

PHASE_LABELS = {
    "compute": "compute",
    "data_transfer": "transfer",
    "h2d_transfer": "h2d",
    "d2h_transfer": "d2h",
}

PHASE_STYLES = {
    "compute": "-",
    "data_transfer": "--",
    "h2d_transfer": "--",
    "d2h_transfer": ":",
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


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-device GEMM mean lines with spread bands using all gathered data."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="scripts/benchmarks-logs.cleaned.csv",
        help="Path to cleaned GEMM benchmark CSV",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="scripts/plots/benchmark_spread_lines",
        help="Directory for generated plots",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # device -> framework -> phase -> batch_index -> list[time_ms]
    grouped: dict[str, dict[str, dict[str, dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    observed_phases: set[str] = set()

    for row in load_rows(input_path):
        framework = (row.get("framework") or "").strip()
        device_name = (row.get("device_name") or "").strip() or "unknown-device"
        phase = (row.get("workload_type") or "").strip()
        workload_name = (row.get("workload_name") or "").strip()
        if framework not in {"CUDA", "OpenCL"}:
            continue
        if "GEMM" not in workload_name:
            continue
        if phase not in PHASE_LABELS:
            continue

        batch_index = parse_int((row.get("batch_index") or "").strip())
        time_ms = parse_float((row.get("time_ms") or "").strip())
        if batch_index is None or time_ms is None:
            continue

        grouped[device_name][framework][phase][batch_index].append(time_ms)
        observed_phases.add(phase)

    if not grouped:
        raise SystemExit("No matching GEMM rows found in input CSV.")

    phase_order = [phase for phase in ("compute", "data_transfer", "h2d_transfer", "d2h_transfer") if phase in observed_phases]

    for device_name, device_data in grouped.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for framework in ("CUDA", "OpenCL"):
            framework_data = device_data.get(framework, {})
            if not framework_data:
                continue

            for phase in phase_order:
                phase_data = framework_data.get(phase, {})
                if not phase_data:
                    continue

                batch_indices = sorted(phase_data.keys())
                means = []
                lowers = []
                uppers = []
                for batch_index in batch_indices:
                    mean, std = mean_std(phase_data[batch_index])
                    means.append(mean)
                    lowers.append(max(0.0, mean - std))
                    uppers.append(mean + std)

                color = FRAMEWORK_COLORS[framework]
                style = PHASE_STYLES[phase]
                label = f"{framework} {PHASE_LABELS[phase]}"
                ax.plot(batch_indices, means, linestyle=style, color=color, linewidth=2, label=label)
                ax.fill_between(batch_indices, lowers, uppers, color=color, alpha=0.12)

        ax.set_title(device_name)
        ax.set_xlabel("Batch index")
        ax.set_ylabel("time_ms")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)

        output_path = outdir / f"{slugify(device_name)}_spread_lines.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
