"""
generate_synthetic_data.py
---------------------------
Generates a small synthetic "cats vs dogs" image dataset so the full
MLOps pipeline (preprocessing -> training -> packaging -> CI/CD) can be
run end-to-end in any environment, including one without internet/Kaggle
API access.

USING THE REAL KAGGLE DATASET
------------------------------
To use the real dataset instead of the synthetic one:

    pip install kaggle
    kaggle datasets download -d salader/dogs-vs-cats -p data/raw --unzip

Then arrange images as:

    data/raw/cats/*.jpg
    data/raw/dogs/*.jpg

and re-run:

    python src/data_preprocessing.py

The rest of the pipeline (training, packaging, CI/CD) is agnostic to
where the images came from -- it only cares about the
data/raw/<class>/*.jpg folder layout produced here.
"""
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
IMAGES_PER_CLASS = int(os.environ.get("SYNTH_IMAGES_PER_CLASS", 60))
IMG_SIZE = 256  # generated larger than 224 so preprocessing resize is exercised


def _random_color():
    return tuple(random.randint(30, 225) for _ in range(3))


def make_cat_image(size=IMG_SIZE):
    """Cats are drawn as a circle-headed shape with triangular ears."""
    img = Image.new("RGB", (size, size), color=_random_color())
    draw = ImageDraw.Draw(img)
    body_color = _random_color()
    cx, cy = size // 2, size // 2
    r = random.randint(size // 4, size // 3)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=body_color)
    # ears (triangles) -> distinguishing "cat" feature
    ear_color = _random_color()
    draw.polygon([(cx - r, cy - r), (cx - r // 2, cy - r - r // 2), (cx, cy - r)], fill=ear_color)
    draw.polygon([(cx + r, cy - r), (cx + r // 2, cy - r - r // 2), (cx, cy - r)], fill=ear_color)
    return img


def make_dog_image(size=IMG_SIZE):
    """Dogs are drawn as a rectangle-bodied shape with floppy oval ears."""
    img = Image.new("RGB", (size, size), color=_random_color())
    draw = ImageDraw.Draw(img)
    body_color = _random_color()
    w = random.randint(size // 3, size // 2)
    h = random.randint(size // 3, size // 2)
    x0, y0 = (size - w) // 2, (size - h) // 2
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=body_color)
    # floppy ears (ellipses) -> distinguishing "dog" feature
    ear_color = _random_color()
    draw.ellipse([x0 - 20, y0 - 5, x0 + 10, y0 + 40], fill=ear_color)
    draw.ellipse([x0 + w - 10, y0 - 5, x0 + w + 20, y0 + 40], fill=ear_color)
    return img


def generate(images_per_class=IMAGES_PER_CLASS, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    for cls, fn in [("cats", make_cat_image), ("dogs", make_dog_image)]:
        out_dir = RAW_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = fn()
            img.save(out_dir / f"{cls[:-1]}_{i:04d}.jpg", quality=90)
    print(f"Generated {images_per_class} synthetic images per class in {RAW_DIR}")


if __name__ == "__main__":
    generate()
