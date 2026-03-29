#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

FRAMEWORK_COLORS = {
    "CUDA": "#008000",
    "OpenCL": "#FF0000",
}

PHASE_LABELS = {
    "compute": "compute"
}

PHASE_STYLES = {
    "compute": "-"
}

PHASE_MARKERS = {
    "compute": "o"
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


def median(values: list[float]) -> float:
    ordered = sorted(values)
    size = len(ordered)
    if size == 0:
        return 0.0
    mid = size // 2
    if size % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def trimmed_limits(series_list: list[tuple[list[float], list[float], list[float]]]) -> tuple[float, float]:
    lower_candidates: list[float] = []
    upper_candidates: list[float] = []

    for means, lowers, uppers in series_list:
        include = [True] * len(means)
        if len(means) > 1:
            tail = [value for value in means[1:] if value > 0]
            tail_median = median(tail)
            if tail_median > 0 and means[0] > tail_median * 5.0:
                include[0] = False

        for keep, lower, upper in zip(include, lowers, uppers):
            if not keep:
                continue
            lower_candidates.append(lower)
            upper_candidates.append(upper)

    if not upper_candidates:
        return 0.0, 1.0

    ymin = min(lower_candidates) if lower_candidates else 0.0
    ymax = max(upper_candidates)
    span = max(ymax - ymin, ymax * 0.05, 1e-6)
    return max(0.0, ymin - 0.08 * span), ymax + 0.12 * span


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-device GEMM mean lines with spread bands using all gathered data."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="data/logs/benchmarks-logs.postprocessed.csv",
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
        input_size = (row.get("input_size") or "").strip()

        if framework not in {"CUDA", "OpenCL"}:
            continue
        if "GEMM" not in workload_name:
            continue
        if phase not in PHASE_LABELS:
            continue
        if "2341011456" not in input_size:
            continue

        batch_index = parse_int((row.get("batch_index") or "").strip())
        time_ms = parse_float((row.get("time_ms") or "").strip())
        if batch_index is None or time_ms is None:
            continue

        grouped[device_name][framework][phase][batch_index].append(time_ms)
        observed_phases.add(phase)

    if not grouped:
        raise SystemExit("No matching GEMM rows found in input CSV.")

    phase_order = [phase for phase in ("compute", "d2h_transfer") if phase in observed_phases]

    for device_name, device_data in grouped.items():
        if not phase_order:
            continue
        fig, axes = plt.subplots(1, len(phase_order), figsize=(6.5 * len(phase_order), 5.5), squeeze=False)

        for axis_index, phase in enumerate(phase_order):
            ax = axes[0][axis_index]
            phase_series: list[tuple[list[float], list[float], list[float]]] = []

            for framework in ("CUDA", "OpenCL"):
                framework_data = device_data.get(framework, {})
                if not framework_data:
                    continue

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
                phase_series.append((means, lowers, uppers))

                color = FRAMEWORK_COLORS[framework]
                style = PHASE_STYLES[phase]
                marker = PHASE_MARKERS[phase]
                label = f"{framework} {PHASE_LABELS[phase]}"
                ax.plot(
                    batch_indices,
                    means,
                    linestyle=style,
                    marker=marker,
                    color=color,
                    linewidth=2,
                    markersize=4,
                    label=label,
                )
                ax.fill_between(batch_indices, lowers, uppers, color=color, alpha=0.08)

            ax.set_xlabel("Batch index")
            ax.set_ylabel("time_ms")
            ax.set_xticks(range(1, 11))
            ymin, ymax = trimmed_limits(phase_series)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

        output_path = outdir / f"{slugify(device_name)}_spread_lines.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
