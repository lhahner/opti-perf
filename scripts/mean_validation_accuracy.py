#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def normalize_device(row: dict[str, str]) -> str:
    device_name = (row.get("device_name") or "").strip()
    return device_name if device_name else "unknown-device"


def workload_size(row: dict[str, str]) -> str:
    batch_size = (row.get("batch_size") or "").strip()
    input_size = (row.get("input_size") or "").strip()
    return f"batch={batch_size},input={input_size}"


def iter_input_files(path: Path):
    if path.is_file():
        yield path
        return

    if not path.exists():
        raise SystemExit(f"Input path not found: {path}")

    for csv_path in sorted(path.glob("validation-benchmark-logs*.csv")):
        if csv_path.is_file():
            yield csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute mean validation accuracy from validation benchmark CSV logs."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="data/logs",
        help="Path to a validation CSV file or a directory containing validation benchmark logs",
    )
    parser.add_argument(
        "--workload-type",
        default="evaluation",
        help="Only include rows with this workload type; use '*' to include all rows",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Only include rows whose device_name contains this value (case-insensitive)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    input_files = list(iter_input_files(input_path))
    if not input_files:
        raise SystemExit(f"No validation benchmark CSV files found in: {input_path}")

    sums: dict[tuple[str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for csv_path in input_files:
        for row in load_rows(csv_path):
            workload_type = (row.get("workload_type") or "").strip()
            if args.workload_type != "*" and workload_type != args.workload_type:
                continue

            framework = (row.get("framework") or "").strip() or "unknown-framework"
            device = normalize_device(row)
            size = workload_size(row)
            if args.device and args.device.casefold() not in device.casefold():
                continue

            try:
                accuracy = float((row.get("accuracy") or "").strip())
            except ValueError:
                continue

            key = (framework, device, size)
            sums[key] += accuracy
            counts[key] += 1

    if not counts:
        raise SystemExit("No matching accuracy rows found in the provided validation logs.")

    print("framework,device_name,workload_size,mean_accuracy,samples")
    for key in sorted(counts.keys()):
        mean_accuracy = sums[key] / counts[key]
        framework, device, size = key
        print(f"{framework},{device},{size},{mean_accuracy:.6f},{counts[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
