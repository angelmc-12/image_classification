# src/utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def set_seed(seed: int = 12) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model: torch.nn.Module, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    torch.save(model.state_dict(), path)


def load_model_state(model: torch.nn.Module, path: str | Path, map_location: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


def plot_training_curves(costs: List[float], accs: List[float], save_path: str | Path) -> None:
    ensure_dir(Path(save_path).parent)
    fig, ax1 = plt.subplots()

    ax1.plot(costs)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cost")

    ax2 = ax1.twinx()
    ax2.plot(accs)
    ax2.set_ylabel("accuracy")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path) -> None:
    ensure_dir(Path(save_path).parent)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap=None, colorbar=False)  # no explicit colors requirement; matplotlib default
    ax.set_title("Confusion Matrix (Validation)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, val_loader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct = 0
    total = 0

    all_true = []
    all_pred = []

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.numel()

        all_true.append(y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    acc = correct / max(total, 1)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return acc, y_true, y_pred
