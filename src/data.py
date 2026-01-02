# src/data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 100
    val_batch_size: int = 5000
    val_ratio: float = 0.1667  # ~10k out of 60k (similar to your train/val split)
    num_workers: int = 0
    seed: int = 12


def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train_loader and val_loader.

    Uses torchvision MNIST; downloads automatically into cfg.data_dir.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # outputs [0,1], shape (1,28,28)
    ])

    full_train = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)

    val_size = int(len(full_train) * cfg.val_ratio)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
