"""Convolutional neural network architecture for visual spatial representation."""

from typing import List, Optional
import torch
import torch.nn as nn
from noir.models.base import NoirBaseModel


class NoirConvNet(NoirBaseModel):
    """Convolutional neural network for grid and image observations."""

    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 8,
        num_classes: int = 4,
        channels: Optional[List[int]] = None,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32]

        conv_layers: List[nn.Module] = []
        prev_c = in_channels

        for c in channels:
            conv = nn.Conv2d(prev_c, c, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
            conv_layers.append(conv)
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_c = c

        self.conv_trunk = nn.Sequential(*conv_layers)

        # Compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            flat_dim = self.conv_trunk(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        self.register_visualization_hooks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_trunk(x)
        flattened = features.view(features.size(0), -1)
        return self.fc(flattened)
