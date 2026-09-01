"""Multi-Layer Perceptron architecture for Project NOIR."""

from typing import List, Optional
import torch
import torch.nn as nn
from noir.models.base import NoirBaseModel


class NoirMLP(NoirBaseModel):
    """Configurable Multi-Layer Perceptron supporting classification and regression."""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, h_dim)
            # Xavier/Kaiming normal initialization
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h_dim

        # Final output layer
        out_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_normal_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        self.network = nn.Sequential(*layers)
        self.register_visualization_hooks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
