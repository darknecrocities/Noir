"""Semantic memory for persistent facts, behavioral rules, and patterns."""

from typing import Any, Dict, List, Optional
import time


class SemanticMemory:
    """Stores persistent hypotheses, hyperparameter strategies, and structural facts."""

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.knowledge: Dict[str, Dict[str, Any]] = {}

    def store_concept(self, key: str, value: Any, confidence: float = 1.0) -> None:
        """Add or update a learned rule or semantic concept."""
        self.knowledge[key] = {
            "key": key,
            "value": value,
            "confidence": float(confidence),
            "updated_at": time.time(),
        }

    def get_concept(self, key: str) -> Optional[Dict[str, Any]]:
        return self.knowledge.get(key)

    def get_all_concepts(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.knowledge)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.knowledge)

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        self.knowledge = dict(data)
