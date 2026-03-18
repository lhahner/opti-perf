#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean compute and data transfer times grouped by framework, "
            "device name, and workload size."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks-logs.cleaned.csv"),
        help="Path to the cleaned benchmark CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV path. If omitted, results are printed.",
    )
    return parser.parse_args()


def normalize_device(row: dict[str, str]) -> str:
    device_name = (row.get("device_name") or "").strip()
    if device_name:
        return device_name
    return (row.get("device") or "").strip() or "UNKNOWN"


def main() -> None:
    args = parse_args()

    sums: dict[tuple[str, str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            workload_type = (row.get("workload_type") or "").strip()
            if workload_type not in {"compute", "data_transfer"}:
                continue

            input_size = (row.get("input_size") or "").strip()
            if not input_size:
                continue

            framework = (row.get("framework") or "").strip()
            if not framework:
                framework = "UNKNOWN"

            time_raw = (row.get("time_ms") or "").strip()
            if not time_raw:
                continue

            try:
                time_ms = float(time_raw)
            except ValueError:
                continue

            key = (framework, normalize_device(row), input_size, workload_type)
            sums[key] += time_ms
            counts[key] += 1

    rows = []
    grouped_keys = sorted(
        {(framework, device, input_size) for framework, device, input_size, _ in sums}
    )
    for framework, device, input_size in grouped_keys:
        compute_key = (framework, device, input_size, "compute")
        transfer_key = (framework, device, input_size, "data_transfer")

        compute_mean = (
            sums[compute_key] / counts[compute_key] if counts[compute_key] else ""
        )
        transfer_mean = (
            sums[transfer_key] / counts[transfer_key] if counts[transfer_key] else ""
        )

        rows.append(
            {
                "framework": framework,
                "device_name": device,
                "input_size": input_size,
                "mean_compute_ms": compute_mean,
                "mean_data_transfer_ms": transfer_mean,
            }
        )

    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "framework",
                    "device_name",
                    "input_size",
                    "mean_compute_ms",
                    "mean_data_transfer_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}")
        return

    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=[
            "framework",
            "device_name",
            "input_size",
            "mean_compute_ms",
            "mean_data_transfer_ms",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
