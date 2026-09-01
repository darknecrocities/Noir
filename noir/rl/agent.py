"""PPO reinforcement learning agent implementation."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical

from noir.core.logging import get_logger
from noir.models.actor_critic import ActorCriticNetwork
from noir.rl.trajectory_memory import TrajectoryBuffer

logger = get_logger("rl.agent")


class PPOAgent:
    """Proximal Policy Optimization (PPO) agent executing real mathematical policy gradients."""

    def __init__(
        self,
        network: ActorCriticNetwork,
        learning_rate: float = 0.0003,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action from policy given observation state.

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            s = state.to(self.device)
            if s.ndim == 1:
                s = s.unsqueeze(0)
            action, log_prob, _, value = self.network.get_action_and_value(s)
            return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def train_step(
        self,
        buffer: TrajectoryBuffer,
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Perform PPO gradient updates across collected trajectory minibatches."""
        self.network.train()

        total_losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []

        for _ in range(n_epochs):
            for b_states, b_actions, b_old_log_probs, b_advantages, b_returns in buffer.get_batches(batch_size, self.device):
                # 1. Forward pass on current policy
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(b_states, b_actions)

                # 2. Probability ratio
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)

                # Track clipping fraction
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                    clip_fractions.append(clip_frac)

                # 3. Clipped Surrogate Policy Loss
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 4. Value Loss (Squared Error)
                value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                # 5. Entropy Loss
                entropy_loss = -entropy.mean()

                # 6. Total Objective
                total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # 7. Backward propagation & Gradient step
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_losses.append(float(total_loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.mean().item()))

        return {
            "loss": float(torch.tensor(total_losses).mean().item()) if total_losses else 0.0,
            "policy_loss": float(torch.tensor(policy_losses).mean().item()) if policy_losses else 0.0,
            "value_loss": float(torch.tensor(value_losses).mean().item()) if value_losses else 0.0,
            "entropy": float(torch.tensor(entropies).mean().item()) if entropies else 0.0,
            "clip_fraction": float(torch.tensor(clip_fractions).mean().item()) if clip_fractions else 0.0,
        }
