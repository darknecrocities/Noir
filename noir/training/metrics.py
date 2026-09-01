"""Numerical metric calculation utilities for real ML training."""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


class MetricTracker:
    """Tracks running statistics and computing windowed moving averages."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._history: Dict[str, List[float]] = {}

    def update(self, name: str, value: float) -> None:
        if name not in self._history:
            self._history[name] = []
        self._history[name].append(float(value))
        if len(self._history[name]) > self.window_size:
            self._history[name].pop(0)

    def update_dict(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.update(k, v)

    def get_average(self, name: str) -> float:
        vals = self._history.get(name, [])
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def get_latest(self, name: str) -> Optional[float]:
        vals = self._history.get(name, [])
        return vals[-1] if vals else None

    def get_all_averages(self) -> Dict[str, float]:
        return {k: self.get_average(k) for k in self._history}


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate top-1 classification accuracy percentage."""
    with torch.no_grad():
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            preds = torch.argmax(predictions, dim=1)
        else:
            preds = (torch.sigmoid(predictions) > 0.5).long()
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return float(correct / max(1, total))


def calculate_gradient_norm(model: nn.Module) -> float:
    """Compute overall L2 norm of parameter gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return float(total_norm ** 0.5)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Extract current learning rate from first parameter group."""
    for param_group in optimizer.param_groups:
        return float(param_group.get("lr", 0.0))
    return 0.0
