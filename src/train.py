# src/train.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from src.model import CNN
from src.data import DataConfig, get_dataloaders
from src.utils import (
    set_seed,
    get_device,
    save_model,
    plot_training_curves,
    evaluate_accuracy,
    compute_confusion_matrix,
    ensure_dir,
)


def train(
    epochs: int,
    lr: float,
    out_1: int,
    out_2: int,
    batch_size: int,
    val_batch_size: int,
    seed: int,
    model_path: str,
    assets_dir: str,
) -> None:
    set_seed(seed)
    device = get_device(prefer_cuda=True)

    cfg = DataConfig(
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        seed=seed,
    )
    train_loader, val_loader = get_dataloaders(cfg)

    model = CNN(out_1=out_1, out_2=out_2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    costs: List[float] = []
    accs: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_cost = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_cost += float(loss.detach().cpu().item())

        # Validation accuracy
        acc, _, _ = evaluate_accuracy(model, val_loader, device)

        costs.append(epoch_cost)
        accs.append(acc)

        print(f"Epoch {epoch:02d}/{epochs} | Cost: {epoch_cost:.4f} | Val Acc: {acc:.4f}")

    # Save model weights
    save_model(model, model_path)
    print(f"\nSaved model to: {model_path}")

    # Save training curves
    ensure_dir(assets_dir)
    curves_path = str(Path(assets_dir) / "training_curves.png")
    plot_training_curves(costs, accs, curves_path)
    print(f"Saved training curves to: {curves_path}")

    # Confusion matrix
    _, y_true, y_pred = evaluate_accuracy(model, val_loader, device)
    cm_path = str(Path(assets_dir) / "confusion_matrix.png")
    compute_confusion_matrix(y_true, y_pred, cm_path)
    print(f"Saved confusion matrix to: {cm_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST CNN from scratch (PyTorch).")
    p.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate (SGD).")
    p.add_argument("--out1", type=int, default=16, help="Conv1 channels.")
    p.add_argument("--out2", type=int, default=32, help="Conv2 channels.")
    p.add_argument("--batch-size", type=int, default=100, help="Train batch size.")
    p.add_argument("--val-batch-size", type=int, default=5000, help="Validation batch size.")
    p.add_argument("--seed", type=int, default=12, help="Random seed.")
    p.add_argument("--model-path", type=str, default="models/mnist_cnn.pt", help="Where to save weights.")
    p.add_argument("--assets-dir", type=str, default="assets", help="Where to save plots/images.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        out_1=args.out1,
        out_2=args.out2,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        seed=args.seed,
        model_path=args.model_path,
        assets_dir=args.assets_dir,
    )
