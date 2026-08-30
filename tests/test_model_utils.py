"""
Unit tests for src/model.py (Assignment M3, Task 1:
"one model utility/inference function").
"""
import numpy as np
import pytest
import torch

from src.model import SimpleCNN, predict_from_array, CLASS_NAMES, IMG_SIZE


@pytest.fixture(scope="module")
def untrained_model():
    """
    We don't need a trained model to unit-test the *inference plumbing*
    (shapes, output schema, determinism-in-eval-mode). Training correctness
    is validated separately via the MLflow-tracked training run.
    """
    model = SimpleCNN()
    model.eval()
    return model


def test_predict_from_array_output_schema(untrained_model):
    dummy_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    result = predict_from_array(untrained_model, dummy_image)

    assert "label" in result
    assert "class_index" in result
    assert "probabilities" in result
    assert result["label"] in CLASS_NAMES
    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)


def test_predict_from_array_probabilities_sum_to_one(untrained_model):
    dummy_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    result = predict_from_array(untrained_model, dummy_image)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-4


def test_predict_from_array_rejects_bad_shape(untrained_model):
    bad_image = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)  # missing channel dim
    with pytest.raises(ValueError):
        predict_from_array(untrained_model, bad_image)


def test_predict_from_array_is_deterministic_in_eval_mode(untrained_model):
    dummy_image = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    result1 = predict_from_array(untrained_model, dummy_image)
    result2 = predict_from_array(untrained_model, dummy_image)
    assert result1["probabilities"] == result2["probabilities"]
