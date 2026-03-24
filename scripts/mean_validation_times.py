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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute mean validation benchmark times grouped by device, framework, workload size, and workload type."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="data/logs/validation-benchmark-logs.csv",
        help="Path to validation benchmark CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    sums: dict[tuple[str, str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)

    for row in load_rows(input_path):
        workload_type = (row.get("workload_type") or "").strip()
        if workload_type not in {"compute", "data_transfer"}:
            continue

        framework = (row.get("framework") or "").strip()
        device = normalize_device(row)
        size = workload_size(row)

        try:
            time_ms = float((row.get("time_ms") or "").strip())
        except ValueError:
            continue

        key = (device, framework, size, workload_type)
        sums[key] += time_ms
        counts[key] += 1

    print("device,framework,workload_size,workload_type,mean_time_ms,samples")
    for key in sorted(sums.keys()):
        total = sums[key]
        count = counts[key]
        mean = total / count if count else 0.0
        device, framework, size, workload_type = key
        print(f"{device},{framework},{size},{workload_type},{mean:.6f},{count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
