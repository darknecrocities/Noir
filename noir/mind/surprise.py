"""Mathematical surprise engine and anomaly detection."""

from typing import Optional, Tuple, Union
import numpy as np
import torch


def calculate_surprise(
    actual: Union[np.ndarray, torch.Tensor],
    predicted: Union[np.ndarray, torch.Tensor],
) -> float:
    """Calculates squared prediction error surprise: S_t = ||s_{t+1} - s^_t+1||^2."""
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()

    diff = actual - predicted
    mse = float(np.mean(np.square(diff)))
    return float(mse)


def calculate_event_surprise(probability: float, eps: float = 1e-7) -> float:
    """Calculates self-information surprise: I(event) = -log(P(event))."""
    p = np.clip(probability, eps, 1.0)
    surprise = float(-np.log(p))
    # Normalized between 0.0 and 1.0 using tanh scaling
    return float(np.tanh(surprise / 3.0))


class SurpriseDetector:
    """Tracks surprise values and detects significant perceptual shocks."""

    def __init__(self, threshold: float = 0.70, alpha: float = 0.05):
        self.threshold = threshold
        self.alpha = alpha  # EMA smoothing factor
        self.running_mean = 0.1
        self.running_var = 0.05

    def update(self, surprise_value: float) -> Tuple[bool, float, float]:
        """Update tracker with new surprise observation.

        Returns:
            (is_surprised, raw_surprise, normalized_surprise)
        """
        # Update running statistics
        diff = surprise_value - self.running_mean
        self.running_mean += self.alpha * diff
        self.running_var = (1 - self.alpha) * self.running_var + self.alpha * (diff ** 2)
        std = max(1e-5, np.sqrt(self.running_var))

        # Normalized z-score surprise mapped via sigmoid
        z = (surprise_value - self.running_mean) / std
        normalized_surprise = float(1.0 / (1.0 + np.exp(-z)))

        is_surprised = normalized_surprise >= self.threshold
        return is_surprised, float(surprise_value), normalized_surprise
