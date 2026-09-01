"""GridWorld environment for reinforcement learning."""

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class GridWorldEnv:
    """Configurable 2D GridWorld navigation environment."""

    ACTIONS = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1),  # LEFT
    }

    def __init__(
        self,
        grid_size: int = 8,
        num_obstacles: int = 4,
        max_steps: int = 150,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.seed(seed)

        self.agent_pos = [0, 0]
        self.goal_pos = [grid_size - 1, grid_size - 1]
        self.obstacles: List[Tuple[int, int]] = []
        self.current_step = 0
        self.prev_dist = 0.0

        # State feature dimension = 16
        self.state_dim = 16
        self.action_dim = 4

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]

        # Generate obstacles
        self.obstacles = []
        possible_coords = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) != (0, 0) and (r, c) != tuple(self.goal_pos)
        ]
        if possible_coords and self.num_obstacles > 0:
            sample_count = min(self.num_obstacles, len(possible_coords))
            indices = np.random.choice(len(possible_coords), size=sample_count, replace=False)
            self.obstacles = [possible_coords[i] for i in indices]

        self.prev_dist = self._manhattan_distance(self.agent_pos, self.goal_pos)
        obs = self._get_observation()
        info = {
            "agent_pos": list(self.agent_pos),
            "goal_pos": list(self.goal_pos),
            "distance": self.prev_dist,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one action step in the environment.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        dr, dc = self.ACTIONS.get(action, (0, 0))

        new_r = max(0, min(self.grid_size - 1, self.agent_pos[0] + dr))
        new_c = max(0, min(self.grid_size - 1, self.agent_pos[1] + dc))

        hit_obstacle = (new_r, new_c) in self.obstacles
        if not hit_obstacle:
            self.agent_pos = [new_r, new_c]

        curr_dist = self._manhattan_distance(self.agent_pos, self.goal_pos)

        # Rewards calculation
        step_penalty = -0.02
        dist_shaping = (self.prev_dist - curr_dist) * 0.2
        obstacle_penalty = -0.3 if hit_obstacle else 0.0

        reward = step_penalty + dist_shaping + obstacle_penalty

        reached_goal = (self.agent_pos == self.goal_pos)
        if reached_goal:
            reward += 10.0

        self.prev_dist = curr_dist

        terminated = reached_goal
        truncated = self.current_step >= self.max_steps

        info = {
            "agent_pos": list(self.agent_pos),
            "goal_pos": list(self.goal_pos),
            "reached_goal": reached_goal,
            "hit_obstacle": hit_obstacle,
            "step": self.current_step,
            "distance": curr_dist,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _manhattan_distance(self, pos1: List[int], pos2: List[int]) -> float:
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def _get_observation(self) -> np.ndarray:
        """Construct a 16-dimensional continuous normalized feature vector."""
        obs = np.zeros(self.state_dim, dtype=np.float32)

        # Agent normalized coords
        obs[0] = self.agent_pos[0] / max(1, self.grid_size - 1)
        obs[1] = self.agent_pos[1] / max(1, self.grid_size - 1)

        # Goal normalized coords
        obs[2] = self.goal_pos[0] / max(1, self.grid_size - 1)
        obs[3] = self.goal_pos[1] / max(1, self.grid_size - 1)

        # Relative delta
        obs[4] = (self.goal_pos[0] - self.agent_pos[0]) / max(1, self.grid_size - 1)
        obs[5] = (self.goal_pos[1] - self.agent_pos[1]) / max(1, self.grid_size - 1)

        # Distance
        obs[6] = self.prev_dist / (2 * self.grid_size)

        # Proximity sensors (4 directions)
        r, c = self.agent_pos
        obs[7] = 1.0 if r == 0 else 0.0  # Top boundary
        obs[8] = 1.0 if c == self.grid_size - 1 else 0.0  # Right boundary
        obs[9] = 1.0 if r == self.grid_size - 1 else 0.0  # Bottom boundary
        obs[10] = 1.0 if c == 0 else 0.0  # Left boundary

        # Obstacle proximity
        obs[11] = 1.0 if (r - 1, c) in self.obstacles else 0.0
        obs[12] = 1.0 if (r, c + 1) in self.obstacles else 0.0
        obs[13] = 1.0 if (r + 1, c) in self.obstacles else 0.0
        obs[14] = 1.0 if (r, c - 1) in self.obstacles else 0.0

        # Step progress
        obs[15] = self.current_step / self.max_steps

        return obs
