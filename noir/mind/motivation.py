"""Motivation and goal dynamics modeling."""

import numpy as np


class MotivationEngine:
    """Computes motivational drivers, novelty, and exploration-exploitation trade-offs."""

    def __init__(self, novelty_decay: float = 0.98):
        self.novelty_decay = novelty_decay
        self.visited_state_hashes = set()
        self.goal_progress_history = []

    def compute_novelty(self, state_vector: np.ndarray) -> float:
        """Estimate state novelty based on state space discretization density."""
        # Discretize state into spatial hash
        quantized = tuple(np.round(state_vector[:4] * 10).astype(int))
        if quantized not in self.visited_state_hashes:
            self.visited_state_hashes.add(quantized)
            novelty = 1.0
        else:
            novelty = 0.2

        return float(novelty)

    def compute_goal_progress(self, current_distance: float, initial_distance: float) -> float:
        """Compute relative goal progress metric between 0.0 and 1.0."""
        if initial_distance <= 1e-5:
            return 1.0
        progress = max(0.0, 1.0 - (current_distance / initial_distance))
        return float(np.clip(progress, 0.0, 1.0))
