"""Reward shaping and multi-objective motivation aggregator."""

from typing import Dict


class RewardEngine:
    """Aggregates extrinsic environment rewards, intrinsic curiosity, and penalty weights."""

    def __init__(self, step_penalty: float = -0.01, goal_multiplier: float = 1.0):
        self.step_penalty = step_penalty
        self.goal_multiplier = goal_multiplier

    def shape_reward(self, raw_reward: float, intrinsic_reward: float = 0.0) -> Dict[str, float]:
        """Combine reward streams."""
        total = (raw_reward * self.goal_multiplier) + intrinsic_reward
        return {
            "total_reward": float(total),
            "extrinsic_reward": float(raw_reward),
            "intrinsic_reward": float(intrinsic_reward),
        }
