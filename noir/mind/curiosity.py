"""Curiosity and forward-dynamics intrinsic motivation model."""

from typing import Tuple
import torch
import torch.nn as nn


class ForwardDynamicsModel(nn.Module):
    """Predicts next environment state s_{t+1} given current state s_t and discrete action a_t."""

    def __init__(self, state_dim: int = 16, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.action_embed = nn.Embedding(action_dim, 8)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_embed(action)
        x = torch.cat([state, action_emb], dim=-1)
        return self.net(x)


class CuriosityEngine:
    """Manages forward-dynamics learning and computes intrinsic curiosity rewards."""

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 4,
        lr: float = 0.001,
        intrinsic_weight: float = 0.2,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = ForwardDynamicsModel(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction="none")
        self.intrinsic_weight = intrinsic_weight

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        train_model: bool = True,
    ) -> Tuple[float, float]:
        """Compute intrinsic curiosity reward: eta * ||f(s, a) - s'||^2.

        Returns:
            (intrinsic_reward, prediction_error)
        """
        self.model.eval()
        with torch.no_grad():
            s = state.unsqueeze(0).to(self.device) if state.ndim == 1 else state.to(self.device)
            a = action.unsqueeze(0).to(self.device) if action.ndim == 0 else action.to(self.device)
            ns = next_state.unsqueeze(0).to(self.device) if next_state.ndim == 1 else next_state.to(self.device)

            pred_ns = self.model(s, a)
            error = self.criterion(pred_ns, ns).mean(dim=-1)
            raw_error = float(error.mean().item())

        if train_model:
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(s, a)
            loss = nn.functional.mse_loss(pred, ns)
            loss.backward()
            self.optimizer.step()

        # Scaled curiosity reward
        intrinsic_reward = float(self.intrinsic_weight * min(1.0, raw_error))
        return intrinsic_reward, raw_error
