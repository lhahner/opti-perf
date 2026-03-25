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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute mean full_step time grouped by framework, device, and batch size."
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

    sums: dict[tuple[str, str, int], float] = defaultdict(float)
    counts: dict[tuple[str, str, int], int] = defaultdict(int)

    for row in load_rows(input_path):
        if (row.get("workload_type") or "").strip() != "full_step":
            continue

        framework = (row.get("framework") or "").strip()
        device_name = (row.get("device_name") or "").strip() or "unknown-device"
        try:
            batch_size = int((row.get("batch_size") or "").strip())
            time_ms = float((row.get("time_ms") or "").strip())
        except ValueError:
            continue

        key = (framework, device_name, batch_size)
        sums[key] += time_ms
        counts[key] += 1

    print("framework,device_name,batch_size,mean_full_step_ms,samples")
    for key in sorted(sums.keys(), key=lambda item: (item[0], item[1], item[2])):
        mean = sums[key] / counts[key]
        framework, device_name, batch_size = key
        print(f"{framework},{device_name},{batch_size},{mean:.6f},{counts[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
