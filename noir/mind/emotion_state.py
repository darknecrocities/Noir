"""Mathematical representation of internal affective state."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EmotionState:
    """Affective vector E_t = [C, F, A, S, U, X, Ca, P].

    Components:
        confidence (C): 0.0 - 1.0 (Certainty in mastery / performance)
        frustration (F): 0.0 - 1.0 (Response to stagnation or high error)
        anticipation (A): 0.0 - 1.0 (Expectancy of impending reward or outcome)
        satisfaction (S): 0.0 - 1.0 (Reward / positive outcome fulfillment)
        uncertainty (U): 0.0 - 1.0 (Entropy / ambiguity in predictions)
        curiosity (X): 0.0 - 1.0 (Drive for exploration and novelty)
        caution (Ca): 0.0 - 1.0 (Risk aversion / sensitivity to penalties)
        persistence (P): 0.0 - 1.0 (Drive to maintain pursuit despite difficulties)
    """
    confidence: float = 0.5
    frustration: float = 0.1
    anticipation: float = 0.5
    satisfaction: float = 0.5
    uncertainty: float = 0.5
    curiosity: float = 0.7
    caution: float = 0.3
    persistence: float = 0.8

    def __post_init__(self):
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.frustration = float(np.clip(self.frustration, 0.0, 1.0))
        self.anticipation = float(np.clip(self.anticipation, 0.0, 1.0))
        self.satisfaction = float(np.clip(self.satisfaction, 0.0, 1.0))
        self.uncertainty = float(np.clip(self.uncertainty, 0.0, 1.0))
        self.curiosity = float(np.clip(self.curiosity, 0.0, 1.0))
        self.caution = float(np.clip(self.caution, 0.0, 1.0))
        self.persistence = float(np.clip(self.persistence, 0.0, 1.0))

    def to_vector(self) -> np.ndarray:
        """Convert state to 8-dimensional numpy float array."""
        return np.array([
            self.confidence,
            self.frustration,
            self.anticipation,
            self.satisfaction,
            self.uncertainty,
            self.curiosity,
            self.caution,
            self.persistence,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Convert state to dictionary with rounded floats."""
        return {
            "confidence": round(float(self.confidence), 4),
            "frustration": round(float(self.frustration), 4),
            "anticipation": round(float(self.anticipation), 4),
            "satisfaction": round(float(self.satisfaction), 4),
            "uncertainty": round(float(self.uncertainty), 4),
            "curiosity": round(float(self.curiosity), 4),
            "caution": round(float(self.caution), 4),
            "persistence": round(float(self.persistence), 4),
        }

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "EmotionState":
        v_clipped = np.clip(v, 0.0, 1.0)
        return cls(
            confidence=float(v_clipped[0]),
            frustration=float(v_clipped[1]),
            anticipation=float(v_clipped[2]),
            satisfaction=float(v_clipped[3]),
            uncertainty=float(v_clipped[4]),
            curiosity=float(v_clipped[5]),
            caution=float(v_clipped[6]),
            persistence=float(v_clipped[7]),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "EmotionState":
        return cls(
            confidence=float(np.clip(d.get("confidence", 0.5), 0.0, 1.0)),
            frustration=float(np.clip(d.get("frustration", 0.1), 0.0, 1.0)),
            anticipation=float(np.clip(d.get("anticipation", 0.5), 0.0, 1.0)),
            satisfaction=float(np.clip(d.get("satisfaction", 0.5), 0.0, 1.0)),
            uncertainty=float(np.clip(d.get("uncertainty", 0.5), 0.0, 1.0)),
            curiosity=float(np.clip(d.get("curiosity", 0.7), 0.0, 1.0)),
            caution=float(np.clip(d.get("caution", 0.3), 0.0, 1.0)),
            persistence=float(np.clip(d.get("persistence", 0.8), 0.0, 1.0)),
        )
