from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(size: int, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator)
    if size < 2:
        return indices, indices

    val_size = max(1, int(size * val_ratio))
    train_size = max(1, size - val_size)
    if train_size + val_size > size:
        val_size = size - train_size
    if val_size == 0:
        return indices, indices[:0]
    return indices[:train_size], indices[train_size:]


def json_ready_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def take_rows(dataset: Dataset, max_rows: int | None) -> list[dict[str, Any]]:
    total = len(dataset) if max_rows is None else min(len(dataset), max_rows)
    return [dataset[idx] for idx in range(total)]
