"""
data_preprocessing.py
----------------------
Preprocessing utilities for the Cats vs Dogs classification pipeline.

Responsibilities:
  * Read raw images from data/raw/<class>/*.jpg
  * Resize/normalize to 224x224 RGB tensors suitable for a CNN
  * Split into train/val/test (80/10/10) with a reproducible seed
  * Use a configurable subset of images for faster experimentation
  * Provide torchvision-based augmentation transforms for training
  * Provide PyTorch DataLoaders for train/val/test splits
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
# Core preprocessing function
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Convert a PIL image into a normalized img_size x img_size x 3 float32
    numpy array.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((img_size, img_size))

    arr = np.asarray(image, dtype=np.float32) / 255.0

    if arr.shape != (img_size, img_size, 3):
        raise ValueError(f"Unexpected preprocessed shape: {arr.shape}")

    return arr


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Augmentation pipeline used only for the training split."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic pipeline used for validation/test/inference."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
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
    max_per_class: int = 1500,
) -> None:
    """
    Split a subset of data/raw/<class>/*.jpg into:

        data/processed/train/<class>
        data/processed/val/<class>
        data/processed/test/<class>

    max_per_class controls how many images are used from each class.

    With max_per_class=1500 and an 80/10/10 split:

        1200 train images per class
         150 val images per class
         150 test images per class

    Total:
        2400 train
         300 val
         300 test
        3000 images
    """

    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    assert max_per_class > 0, "max_per_class must be greater than 0"

    random.seed(seed)

    # Remove previous processed dataset so the split is rebuilt cleanly.
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    # Create directory structure.
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (processed_dir / split / cls).mkdir(
                parents=True,
                exist_ok=True,
            )

    print(f"Using at most {max_per_class} images per class.")

    for cls in CLASSES:
        files = sorted((raw_dir / cls).glob("*.jpg"))

        if not files:
            raise FileNotFoundError(
                f"No JPG images found in {raw_dir / cls}"
            )

        # Shuffle reproducibly.
        random.shuffle(files)

        # Limit number of images used from this class.
        files = files[:max_per_class]

        n = len(files)

        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        split_map = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        print(
            f"{cls}: "
            f"{len(split_map['train'])} train, "
            f"{len(split_map['val'])} val, "
            f"{len(split_map['test'])} test"
        )

        for split, split_files in split_map.items():
            for f in split_files:
                shutil.copy(
                    f,
                    processed_dir / split / cls / f.name,
                )

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
    processed_dir: Path = PROCESSED_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_ds = CatsDogsDataset(
        processed_dir / "train",
        transform=get_train_transforms(),
    )

    val_ds = CatsDogsDataset(
        processed_dir / "val",
        transform=get_eval_transforms(),
    )

    test_ds = CatsDogsDataset(
        processed_dir / "test",
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"DataLoaders created: "
        f"train={len(train_ds)}, "
        f"val={len(val_ds)}, "
        f"test={len(test_ds)}"
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    split_dataset(
        max_per_class=1500,
    )
