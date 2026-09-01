"""Trajectory rollout buffer with Generalized Advantage Estimation (GAE)."""

from typing import Dict, Generator, List, Tuple
import numpy as np
import torch


class TrajectoryBuffer:
    """Stores on-policy experience rollouts and computes GAE advantages."""

    def __init__(self, capacity: int = 256, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

        self.advantages: np.ndarray = np.array([])
        self.returns: np.ndarray = np.array([])

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def is_full(self) -> bool:
        return len(self.states) >= self.capacity

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        """Compute GAE advantages and discounted returns."""
        num_steps = len(self.rewards)
        advantages = np.zeros(num_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t + 1])
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        self.advantages = norm_advantages
        self.returns = returns

    def get_batches(
        self, batch_size: int, device: torch.device
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Yield minibatches of experience."""
        total_samples = len(self.states)
        indices = np.random.permutation(total_samples)

        states_tensor = torch.stack(self.states).to(device)
        actions_tensor = torch.stack(self.actions).to(device)
        log_probs_tensor = torch.stack(self.log_probs).to(device)
        advantages_tensor = torch.from_numpy(self.advantages).float().to(device)
        returns_tensor = torch.from_numpy(self.returns).float().to(device)

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield (
                states_tensor[batch_indices],
                actions_tensor[batch_indices],
                log_probs_tensor[batch_indices],
                advantages_tensor[batch_indices],
                returns_tensor[batch_indices],
            )

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = np.array([])
        self.returns = np.array([])
