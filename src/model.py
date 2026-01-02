# src/model.py
from __future__ import annotations
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Simple CNN for MNIST:
      Conv(1 -> out_1, 5x5, pad=2) + ReLU + MaxPool(2)
      Conv(out_1 -> out_2, 5x5, pad=2) + ReLU + MaxPool(2)
      Flatten
      FC(out_2*7*7 -> 10)

    Matches the spirit of your notebook.
    """

    def __init__(self, out_1: int = 16, out_2: int = 32):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(out_2 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
