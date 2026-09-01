"""Reinforcement learning environment wrappers."""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from noir.datasets.grid_world import GridWorldEnv


class NoirRLWrapper:
    """Standardized wrapper for GridWorld and Gymnasium environments."""

    def __init__(self, env_id: str = "GridWorld-v0", grid_size: int = 8, max_steps: int = 150):
        self.env_id = env_id

        if "gridworld" in env_id.lower():
            self.env = GridWorldEnv(grid_size=grid_size, max_steps=max_steps)
            self.state_dim = self.env.state_dim
            self.action_dim = self.env.action_dim
        else:
            # Fallback / Gymnasium integration
            try:
                import gymnasium as gym
                self.env = gym.make(env_id)
                self.state_dim = self.env.observation_space.shape[0]
                self.action_dim = self.env.action_space.n
            except Exception:
                self.env = GridWorldEnv(grid_size=grid_size, max_steps=max_steps)
                self.state_dim = self.env.state_dim
                self.action_dim = self.env.action_dim

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)
