#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PHASES = ("compute", "data_transfer", "d2h_transfer", "h2d_transfer")
PHASE_COLORS = {
    "compute": "#d95f02",
    "data_transfer": "#1b9e77",
    "d2h_transfer": "#1b9e77",
    "h2d_transfer": "#7570b3",
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


def jittered_x(batch_index: int, framework: str, phase: str, ordinal: int) -> float:
    seed = f"{framework}|{phase}|{batch_index}|{ordinal}".encode("utf-8")
    digest = hashlib.md5(seed).digest()
    fraction = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return batch_index + (fraction - 0.5) * 0.5


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create per-device GEMM scatter plots from benchmarks-logs.cleaned.csv with jittered x positions."
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
        default="scripts/plots/benchmark_clusters_cleaned",
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
        raise SystemExit("No matching CUDA/OpenCL GEMM rows found in input CSV.")

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
                xs = [
                    jittered_x(point[0], framework, phase, ordinal)
                    for ordinal, point in enumerate(points)
                ]
                ys = [point[1] for point in points]
                ax.scatter(
                    xs,
                    ys,
                    s=20,
                    alpha=0.55,
                    color=PHASE_COLORS[phase],
                    label=phase,
                )

            ax.set_title(framework)
            ax.set_xlabel("Batch index")
            ax.set_ylabel("time_ms")
            ax.set_xlim(left=0.5)
            ax.grid(True, alpha=0.25)
            ax.legend()

        fig.tight_layout()
        output_path = outdir / f"{slugify(device)}_benchmark_clusters_cleaned.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
