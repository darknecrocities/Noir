"""Mathematical uncertainty quantification module."""

from typing import Union
import numpy as np
import torch


def calculate_entropy(probabilities: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> float:
    """Calculate Shannon entropy: H(p) = -sum(p_i * log(p_i)).

    Higher entropy reflects higher predictive uncertainty.
    """
    if isinstance(probabilities, torch.Tensor):
        p = probabilities.detach().cpu().numpy()
    else:
        p = np.array(probabilities)

    p = np.clip(p, eps, 1.0)
    # Normalize if necessary
    if p.ndim > 1:
        p = p / np.sum(p, axis=-1, keepdims=True)
        entropy = -np.sum(p * np.log(p), axis=-1)
        mean_entropy = float(np.mean(entropy))
        num_classes = p.shape[-1]
    else:
        p = p / np.sum(p)
        mean_entropy = float(-np.sum(p * np.log(p)))
        num_classes = len(p)

    # Normalize entropy between 0.0 and 1.0 (dividing by log(K))
    max_entropy = np.log(max(2, num_classes))
    normalized_entropy = float(np.clip(mean_entropy / max_entropy, 0.0, 1.0))
    return normalized_entropy


def calculate_variance(predictions: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute empirical variance across ensemble or multi-head predictions."""
    if isinstance(predictions, torch.Tensor):
        preds = predictions.detach().cpu().numpy()
    else:
        preds = np.array(predictions)

    var = float(np.mean(np.var(preds, axis=0)))
    return float(np.clip(var, 0.0, 1.0))


def calculate_prediction_confidence(probabilities: Union[np.ndarray, torch.Tensor]) -> float:
    """Calculate maximum softmax probability / prediction certainty."""
    if isinstance(probabilities, torch.Tensor):
        p = probabilities.detach().cpu().numpy()
    else:
        p = np.array(probabilities)

    if p.ndim > 1:
        max_p = np.mean(np.max(p, axis=-1))
    else:
        max_p = np.max(p)

    return float(np.clip(max_p, 0.0, 1.0))
