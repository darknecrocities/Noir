"""Unit tests for uncertainty quantification."""

import numpy as np
import torch
from noir.mind.uncertainty import calculate_entropy, calculate_prediction_confidence, calculate_variance


def test_calculate_entropy_uniform_vs_certain():
    # Perfectly uniform distribution over 4 classes -> Maximum normalized entropy (~1.0)
    uniform_p = np.array([0.25, 0.25, 0.25, 0.25])
    h_uniform = calculate_entropy(uniform_p)
    assert abs(h_uniform - 1.0) < 0.05

    # Certain distribution over 4 classes -> Zero entropy (~0.0)
    certain_p = np.array([1.0, 0.0, 0.0, 0.0])
    h_certain = calculate_entropy(certain_p)
    assert h_certain < 0.01


def test_calculate_prediction_confidence():
    p = torch.tensor([[0.8, 0.1, 0.1], [0.9, 0.05, 0.05]])
    conf = calculate_prediction_confidence(p)
    assert 0.84 < conf < 0.86


def test_calculate_variance():
    preds = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])
    var = calculate_variance(preds)
    assert var > 0.0
