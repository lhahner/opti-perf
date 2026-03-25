#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import struct
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def read_idx_images(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic in {path}: {magic}")
        raw = handle.read()
    tensor = torch.tensor(list(raw), dtype=torch.float32)
    tensor = tensor.view(count, rows * cols) / 255.0
    return tensor


def read_idx_labels(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic in {path}: {magic}")
        raw = handle.read()
    tensor = torch.tensor(list(raw), dtype=torch.long)
    if tensor.numel() != count:
        raise ValueError(f"Label count mismatch in {path}: expected {count}, got {tensor.numel()}")
    return tensor


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * batch_x.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == batch_y).sum().item())
            total_examples += batch_x.size(0)
    return total_loss / total_examples, total_correct / total_examples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PyTorch reference training script for the MNIST linear classifier benchmark."
    )
    parser.add_argument("--dataset-dir", default="data/mnist", help="Directory containing raw MNIST IDX files")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max-samples", type=int, default=1024, help="Maximum number of training samples to use")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Number of initial steps excluded from timing stats")
    parser.add_argument(
        "--output",
        default="data/logs/pytorch-mnist-reference.csv",
        help="CSV file for per-step timing and final summary",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_images = read_idx_images(dataset_dir / "train-images-idx3-ubyte")
    train_labels = read_idx_labels(dataset_dir / "train-labels-idx1-ubyte")
    test_images = read_idx_images(dataset_dir / "t10k-images-idx3-ubyte")
    test_labels = read_idx_labels(dataset_dir / "t10k-labels-idx1-ubyte")

    if args.max_samples > 0:
        train_images = train_images[: args.max_samples]
        train_labels = train_labels[: args.max_samples]

    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(28 * 28, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    step_times_ms: list[float] = []
    global_step = 0

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "phase",
                "device",
                "epoch",
                "batch_index",
                "global_step",
                "batch_size",
                "max_samples",
                "time_ms",
                "loss",
                "accuracy",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch_index, (batch_x, batch_y) in enumerate(train_loader, start=1):
                global_step += 1
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

                step_ms = (end - start) * 1000.0
                if global_step > args.warmup_steps:
                    step_times_ms.append(step_ms)

                writer.writerow(
                    [
                        "train_step",
                        device.type,
                        epoch,
                        batch_index,
                        global_step,
                        batch_x.size(0),
                        train_images.size(0),
                        f"{step_ms:.6f}",
                        f"{loss.item():.6f}",
                        "",
                    ]
                )

            test_loss, test_accuracy = evaluate(model, test_loader, device)
            writer.writerow(
                [
                    "evaluation",
                    device.type,
                    epoch,
                    "",
                    global_step,
                    args.batch_size,
                    train_images.size(0),
                    "",
                    f"{test_loss:.6f}",
                    f"{test_accuracy:.6f}",
                ]
            )

        mean_step_ms, std_step_ms = mean_std(step_times_ms)
        final_test_loss, final_test_accuracy = evaluate(model, test_loader, device)
        writer.writerow(
            [
                "summary",
                device.type,
                args.epochs,
                "",
                global_step,
                args.batch_size,
                train_images.size(0),
                f"{mean_step_ms:.6f}",
                f"{final_test_loss:.6f}",
                f"{final_test_accuracy:.6f}",
            ]
        )

    print(f"device={device.type}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"max_samples={train_images.size(0)}")
    print(f"warmup_steps={args.warmup_steps}")
    print(f"mean_train_step_ms={mean_step_ms:.6f}")
    print(f"std_train_step_ms={std_step_ms:.6f}")
    print(f"final_test_loss={final_test_loss:.6f}")
    print(f"final_test_accuracy={final_test_accuracy:.6f}")
    print(f"csv={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
