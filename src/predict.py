# src/predict.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import CNN
from src.utils import get_device, load_model_state


def _to_28x28_grayscale(img: Image.Image) -> Image.Image:
    """
    Ensures a 28x28 grayscale image (MNIST-like).
    """
    img = img.convert("L")               # grayscale
    img = img.resize((28, 28))           # simple resize (ok for demo)
    return img


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """
    Converts PIL to torch tensor shaped [1,1,28,28] with values in [0,1].
    """
    img = _to_28x28_grayscale(img)
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    return x


def load_trained_model(weights_path: str = "models/mnist_cnn.pt", out_1: int = 16, out_2: int = 32) -> Tuple[CNN, torch.device]:
    device = get_device(prefer_cuda=True)
    model = CNN(out_1=out_1, out_2=out_2).to(device)
    model = load_model_state(model, weights_path, map_location=device)
    model.eval()
    return model, device


@torch.no_grad()
def predict_tensor(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> Tuple[int, np.ndarray]:
    """
    x: [1,1,28,28] float tensor in [0,1]
    returns: predicted class and probabilities (numpy shape [10])
    """
    x = x.to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, probs
