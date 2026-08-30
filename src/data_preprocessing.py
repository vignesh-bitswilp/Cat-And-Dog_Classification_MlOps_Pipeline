"""
data_preprocessing.py
----------------------
Preprocessing utilities for the Cats vs Dogs classification pipeline.

Responsibilities (Assignment M1):
  * Read raw images from data/raw/<class>/*.jpg
  * Resize/normalize to 224x224 RGB tensors suitable for a CNN
  * Split into train/val/test (80/10/10) with a reproducible seed
  * Provide torchvision-based augmentation transforms for training
  * Provide PyTorch DataLoaders for train/val/test splits

This module is imported both by src/train.py and by tests/test_data_preprocessing.py.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_SIZE = 224
CLASSES = ["cats", "dogs"]

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Core preprocessing function (unit-tested)
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Convert a PIL image into a normalized 224x224x3 float32 numpy array.

    - Converts to RGB (handles grayscale / RGBA inputs safely)
    - Resizes to (img_size, img_size)
    - Scales pixel values to [0, 1]

    This is the single-responsibility function covered by the unit test
    required in M3 ("unit tests for at least one data pre-processing function").
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((img_size, img_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.shape != (img_size, img_size, 3):
        raise ValueError(f"Unexpected preprocessed shape: {arr.shape}")
    return arr


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Augmentation pipeline used only for the training split."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic pipeline used for validation/test/inference."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------
def split_dataset(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> None:
    """
    Split data/raw/<class>/*.jpg into data/processed/{train,val,test}/<class>/*.jpg
    using the given ratios. Deterministic given `seed`.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    random.seed(seed)

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        files = sorted((raw_dir / cls).glob("*.jpg"))
        random.shuffle(files)
        n = len(files)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        split_map = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }
        for split, split_files in split_map.items():
            for f in split_files:
                shutil.copy(f, processed_dir / split / cls / f.name)

    print(f"Split complete. Files written under {processed_dir}")


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoaders
# ---------------------------------------------------------------------------
class CatsDogsDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.samples = []
        for label, cls in enumerate(CLASSES):
            for f in sorted((split_dir / cls).glob("*.jpg")):
                self.samples.append((f, label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(
    processed_dir: Path = PROCESSED_DIR, batch_size: int = 16, num_workers: int = 0
):
    train_ds = CatsDogsDataset(processed_dir / "train", transform=get_train_transforms())
    val_ds = CatsDogsDataset(processed_dir / "val", transform=get_eval_transforms())
    test_ds = CatsDogsDataset(processed_dir / "test", transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    split_dataset()
