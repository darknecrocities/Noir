"""Actor-Critic architecture for Reinforcement Learning (PPO)."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical

from noir.models.base import NoirBaseModel


class ActorCriticNetwork(NoirBaseModel):
    """Dual-headed Actor-Critic network for policy optimization."""

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        feature_layers: List[nn.Module] = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            nn.init.orthogonal_(linear.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(linear.bias)
            feature_layers.append(linear)
            feature_layers.append(nn.ReLU())
            prev_dim = h_dim

        self.shared_trunk = nn.Sequential(*feature_layers)

        # Actor head (policy)
        self.actor_head = nn.Linear(prev_dim, action_dim)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        # Critic head (value function)
        self.critic_head = nn.Linear(prev_dim, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

        self.register_visualization_hooks()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing action logits and state value.

        Args:
            state: Tensor of shape (B, state_dim)

        Returns:
            (logits, value)
        """
        features = self.shared_trunk(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value.squeeze(-1)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, calculate log probability, entropy, and state value.

        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value = self.forward(state)
        distribution = Categorical(logits=logits)

        if action is None:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, log_prob, entropy, value
