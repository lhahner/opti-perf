#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


CANONICAL_HEADER = [
    "timestamp",
    "device_name",
    "framework",
    "workload_name",
    "workload_type",
    "device",
    "batch_size",
    "input_size",
    "optimizer",
    "learning_rate",
    "beta1",
    "beta2",
    "epsilon",
    "time_ms",
    "batch_index",
    "loss",
    "accuracy",
]

TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")
VALID_NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")

OLD_HEADER = CANONICAL_HEADER[:-1]
NUMERIC_COLUMNS = {
    "batch_size",
    "input_size",
    "learning_rate",
    "beta1",
    "beta2",
    "epsilon",
    "time_ms",
    "batch_index",
    "loss",
    "accuracy",
}


@dataclass
class NormalizedRow:
    cells: list[str]
    source_len: int


def normalize_numeric_string(value: str) -> str:
    raw = value.strip()
    if not raw:
        return raw
    if VALID_NUMERIC_RE.fullmatch(raw):
        return raw
    if "," in raw and "." not in raw and raw.count(",") == 1:
        candidate = raw.replace(",", ".")
        if VALID_NUMERIC_RE.fullmatch(candidate):
            return candidate
    return raw


def split_concatenated_records(line: str) -> list[str]:
    starts = [match.start() for match in TIMESTAMP_RE.finditer(line)]
    if len(starts) <= 1:
        return [line]

    parts: list[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(line)
        part = line[start:end].strip()
        if part:
            parts.append(part)
    return parts or [line]


def clean_cells(header: list[str], row: list[str]) -> list[str]:
    cleaned: list[str] = []
    for key, value in zip(header, row):
        cell = value.strip()
        if key in NUMERIC_COLUMNS:
            cell = normalize_numeric_string(cell)
        cleaned.append(cell)
    return cleaned


def normalize_row(row: list[str]) -> NormalizedRow | None:
    if not row:
        return None
    if len(row) == 1 and row[0].strip("=") == "":
        return None
    if row == CANONICAL_HEADER or row == OLD_HEADER:
        return None

    if len(row) == 16:
        return NormalizedRow(clean_cells(OLD_HEADER, row) + [""], 16)
    if len(row) == 17:
        return NormalizedRow(clean_cells(CANONICAL_HEADER, row), 17)
    return None


def detect_phase_cycle_len(rows: list[NormalizedRow]) -> int:
    phases: list[str] = []
    for row in rows:
        phase = row.cells[4]
        if phase in phases:
            break
        phases.append(phase)
    return max(1, len(phases))


def experiment_key(row: NormalizedRow) -> tuple[str, ...]:
    return tuple(row.cells[pos] for pos in (1, 2, 3, 6, 7, 8, 9, 10, 11, 12))


def parse_batch_index(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


def rewrite_broken_batch_indices(rows: list[NormalizedRow]) -> int:
    fixed = 0
    index = 0
    while index < len(rows):
        key = experiment_key(rows[index])
        end = index + 1
        while end < len(rows):
            if experiment_key(rows[end]) != key:
                break
            end += 1

        group = rows[index:end]
        batch_indices = [parse_batch_index(item.cells[14]) for item in group]
        rewrite_group = any(value is None or value < 1 or value > 10 for value in batch_indices)

        if not rewrite_group and any(item.source_len == 17 for item in group):
            distinct = {value for value in batch_indices if value is not None}
            rewrite_group = len(distinct) <= 1

        if not rewrite_group:
            index = end
            continue

        cycle_len = detect_phase_cycle_len(group)
        for offset, item in enumerate(group):
            step_number = offset // cycle_len
            item.cells[14] = str((step_number % 10) + 1)
            fixed += 1
        index = end

    return fixed


def load_rows(input_path: Path) -> tuple[list[NormalizedRow], int, int]:
    rows: list[NormalizedRow] = []
    skipped = 0
    splits = 0

    with input_path.open("r", encoding="utf-8", newline="") as src:
        next(src)
        for raw_line in src:
            chunks = split_concatenated_records(raw_line.rstrip("\n"))
            if len(chunks) > 1:
                splits += len(chunks) - 1
            for chunk in chunks:
                parsed = next(csv.reader([chunk]))
                normalized = normalize_row(parsed)
                if normalized is None:
                    skipped += 1
                    continue
                rows.append(normalized)

    return rows, skipped, splits


def write_rows(rows: list[NormalizedRow], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(CANONICAL_HEADER)
        for row in rows:
            writer.writerow(row.cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair benchmark logs and reconstruct batch_index as a 1..10 cycle for broken new-format blocks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/logs/benchmarks-logs.csv"),
        help="Input benchmark CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/logs/benchmarks-logs.postprocessed.csv"),
        help="Output path for the repaired CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows, skipped, splits = load_rows(args.input)
    rewritten = rewrite_broken_batch_indices(rows)
    write_rows(rows, args.output)

    print(f"Input:     {args.input}")
    print(f"Output:    {args.output}")
    print(f"Rows:      {len(rows)}")
    print(f"Skipped:   {skipped}")
    print(f"Splits:    {splits}")
    print(f"Rewritten: {rewritten}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
