"""
Unit tests for src/data_preprocessing.py (Assignment M3, Task 1:
"unit tests for at least one data pre-processing function").
"""
import numpy as np
import pytest
from PIL import Image

from src.data_preprocessing import preprocess_image, IMG_SIZE


def test_preprocess_image_output_shape():
    img = Image.new("RGB", (500, 300), color=(255, 0, 0))
    result = preprocess_image(img)
    assert result.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_preprocess_image_value_range():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    result = preprocess_image(img)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    # a pure white image should normalize to (close to) 1.0 everywhere
    assert np.allclose(result, 1.0, atol=1e-6)


def test_preprocess_image_handles_grayscale():
    """Grayscale ('L' mode) images must be converted to 3-channel RGB."""
    img = Image.new("L", (64, 64), color=128)
    result = preprocess_image(img)
    assert result.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_preprocess_image_handles_rgba():
    """RGBA images (with alpha channel) must be converted to RGB."""
    img = Image.new("RGBA", (64, 64), color=(10, 20, 30, 128))
    result = preprocess_image(img)
    assert result.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_preprocess_image_custom_size():
    img = Image.new("RGB", (50, 50), color=(0, 128, 255))
    result = preprocess_image(img, img_size=64)
    assert result.shape == (64, 64, 3)
