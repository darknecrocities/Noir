"""Episodic memory system recording salient experiences and discoveries."""

import time
import uuid
from typing import Any, Dict, List, Optional


class EpisodicMemory:
    """Stores high-salience experiences: surprises, breakthroughs, and catastrophic failures."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes: List[Dict[str, Any]] = []

    def record_experience(
        self,
        event_type: str,
        description: str,
        state_summary: Dict[str, Any],
        importance: float = 1.0,
        step: int = 0,
    ) -> str:
        """Store a high-salience episodic event."""
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        record = {
            "id": episode_id,
            "event_type": event_type,
            "description": description,
            "state_summary": state_summary,
            "importance": float(importance),
            "step": step,
            "timestamp": time.time(),
        }

        self.episodes.append(record)
        # Sort by importance and enforce capacity limit
        if len(self.episodes) > self.capacity:
            self.episodes.sort(key=lambda x: x["importance"], reverse=True)
            self.episodes = self.episodes[: self.capacity]

        return episode_id

    def get_salient_experiences(self, min_importance: float = 0.5, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve most salient experiences matching or exceeding importance threshold."""
        matches = [e for e in self.episodes if e["importance"] >= min_importance]
        matches.sort(key=lambda x: x["timestamp"], reverse=True)
        return matches[:limit]

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.episodes)

    def load_from_list(self, records: List[Dict[str, Any]]) -> None:
        self.episodes = list(records)
