"""Event representation model for Project NOIR."""

import time
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from noir.events.event_types import EventType


class NoirEvent(BaseModel):
    """Immutable event record representing a concrete occurrence in the system."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    event_type: EventType
    experiment_id: str = "default"
    training_step: int = 0
    epoch: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to serialized dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "experiment_id": self.experiment_id,
            "training_step": self.training_step,
            "epoch": self.epoch,
            "payload": self.payload,
        }

    @classmethod
    def create(
        cls,
        event_type: EventType,
        experiment_id: str = "default",
        training_step: int = 0,
        epoch: int = 0,
        **payload: Any,
    ) -> "NoirEvent":
        """Convenience factory method for creating typed events."""
        return cls(
            event_type=event_type,
            experiment_id=experiment_id,
            training_step=training_step,
            epoch=epoch,
            payload=payload,
        )
