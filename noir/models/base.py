"""Base PyTorch model abstraction with hooks for activation and gradient extraction."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings
import torch
import torch.nn as nn

# Suppress PyTorch backward hook notification on leaf layers
warnings.filterwarnings("ignore", message=".*Full backward hook is firing.*")

from noir.core.logging import get_logger

logger = get_logger("models.base")


class NoirBaseModel(nn.Module, ABC):
    """Base neural network model instrumented for real-time visualization."""

    def __init__(self):
        super().__init__()
        self._layer_activations: Dict[str, torch.Tensor] = {}
        self._layer_gradients: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[Any] = []
        self._enable_hooks = True

    def register_visualization_hooks(self) -> None:
        """Register forward and backward hooks across all sub-layers."""
        self.remove_hooks()
        for name, module in self.named_modules():
            # Only attach to leaf modules (Linear, Conv2d, etc.)
            if len(list(module.children())) == 0 and not isinstance(module, (nn.Dropout, nn.Identity)):
                h_fwd = module.register_forward_hook(self._create_forward_hook(name))
                self._hook_handles.append(h_fwd)

                h_bwd = module.register_full_backward_hook(self._create_backward_hook(name))
                self._hook_handles.append(h_bwd)

        logger.debug("Registered %d visualization hooks on model", len(self._hook_handles))

    def _create_forward_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            if self._enable_hooks:
                # Store detached tensor on CPU
                if isinstance(output, torch.Tensor):
                    self._layer_activations[name] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    self._layer_activations[name] = output[0].detach().cpu()
        return hook

    def _create_backward_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
            if self._enable_hooks:
                if len(grad_output) > 0 and isinstance(grad_output[0], torch.Tensor):
                    self._layer_gradients[name] = grad_output[0].detach().cpu()
        return hook

    def remove_hooks(self) -> None:
        """Remove all registered PyTorch hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def get_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Retrieve recent layer forward activations."""
        return dict(self._layer_activations)

    def get_layer_gradients(self) -> Dict[str, torch.Tensor]:
        """Retrieve recent layer backprop gradients."""
        return dict(self._layer_gradients)

    def get_layer_weights(self) -> Dict[str, torch.Tensor]:
        """Retrieve current weight tensors for all parameterized layers."""
        weights = {}
        for name, param in self.named_parameters():
            if "weight" in name and param.requires_grad:
                layer_name = name.rsplit(".", 1)[0]
                weights[layer_name] = param.detach().cpu()
        return weights

    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute mean, std, min, max, and L2 norm for each weight tensor."""
        stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad and param.data is not None:
                p = param.detach()
                stats[name] = {
                    "mean": float(p.mean().item()),
                    "std": float(p.std().item()) if p.numel() > 1 else 0.0,
                    "min": float(p.min().item()),
                    "max": float(p.max().item()),
                    "norm": float(torch.norm(p).item()),
                    "num_elements": p.numel(),
                }
        return stats

    def get_total_gradient_norm(self) -> float:
        """Compute global L2 gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return float(total_norm ** 0.5)

    def get_architecture_summary(self) -> List[Dict[str, Any]]:
        """Returns structured architectural blueprint of model layers."""
        summary = []
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:
                layer_type = module.__class__.__name__
                params_count = sum(p.numel() for p in module.parameters())
                summary.append({
                    "name": name,
                    "type": layer_type,
                    "params": params_count,
                })
        return summary
