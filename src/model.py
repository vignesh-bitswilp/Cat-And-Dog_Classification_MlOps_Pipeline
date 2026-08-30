"""
model.py
--------
Baseline CNN for binary Cats vs Dogs classification, plus a shared
inference utility function used by both the training script and the
FastAPI service.

Keeping `predict_from_array` here (rather than duplicated in app/main.py)
means the same code path is exercised by:
  * src/train.py            (post-training evaluation)
  * app/main.py              (REST /predict endpoint)
  * tests/test_model_utils.py (unit test, M3 requirement)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

IMG_SIZE = 224
CLASS_NAMES = ["cat", "dog"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SimpleCNN(nn.Module):
    """A small, fast-to-train baseline CNN (not meant to be SOTA)."""

    def __init__(self, num_classes: int = 2, img_size: int = IMG_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(weights_path: Path, device: str = "cpu") -> SimpleCNN:
    """Load a trained SimpleCNN from a serialized .pt state_dict file."""
    model = SimpleCNN()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Apply ImageNet-style normalization to an HxWx3 float32 [0,1] array."""
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def predict_from_array(model: nn.Module, image_array: np.ndarray, device: str = "cpu") -> Dict:
    """
    Run inference on a single preprocessed image.

    Args:
        model: a loaded SimpleCNN in eval() mode.
        image_array: HxWx3 float32 numpy array with values in [0, 1],
            e.g. the output of src.data_preprocessing.preprocess_image.
        device: torch device string.

    Returns:
        dict with keys: label (str), class_index (int), probabilities (dict)
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 array, got shape {image_array.shape}")

    normalized = _normalize_array(image_array)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    class_index = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[class_index],
        "class_index": class_index,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }
