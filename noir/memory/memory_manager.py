"""Unified Memory Manager coordinating memory tiers and persistence."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_bus import EventBus, get_event_bus
from noir.events.event_types import EventType
from noir.memory.episodic import EpisodicMemory
from noir.memory.replay_buffer import ReplayBuffer
from noir.memory.semantic import SemanticMemory
from noir.memory.short_term import ShortTermMemory
from noir.storage.database import DatabaseManager, MemoryRecordModel

logger = get_logger("memory.manager")


class MemoryManager:
    """Unified coordinator for cognitive memory architecture."""

    def __init__(
        self,
        experiment_id: str = "default",
        db_manager: Optional[DatabaseManager] = None,
        memory_dir: str | Path = "memory",
        event_bus: Optional[EventBus] = None,
    ):
        self.experiment_id = experiment_id
        self.db = db_manager
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.short_term = ShortTermMemory(capacity=100)
        self.episodic = EpisodicMemory(capacity=1000)
        self.semantic = SemanticMemory(capacity=500)
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.event_bus = event_bus or get_event_bus()
        self._register_event_handlers()

    def _register_event_handlers(self) -> None:
        """Listen to event bus for salience triggers."""
        self.event_bus.subscribe(EventType.SURPRISE_DETECTED, self._on_surprise)
        self.event_bus.subscribe(EventType.EPISODE_COMPLETED, self._on_episode_completed)

    def _on_surprise(self, event: NoirEvent) -> None:
        """Store surprise occurrence in episodic memory."""
        surprise_val = event.payload.get("surprise_value", 0.0)
        self.episodic.record_experience(
            event_type="SURPRISE_EVENT",
            description=f"Perceptual surprise shock detected: value {surprise_val:.3f}",
            state_summary=event.payload,
            importance=min(1.0, 0.5 + surprise_val * 0.5),
            step=event.training_step,
        )

    def _on_episode_completed(self, event: NoirEvent) -> None:
        """Store notable episode outcome."""
        reward = event.payload.get("episode_reward", 0.0)
        reached_goal = event.payload.get("reached_goal", False)
        importance = 1.0 if reached_goal else (0.7 if reward > 0 else 0.3)

        self.episodic.record_experience(
            event_type="EPISODE_RESULT",
            description=f"Episode {event.epoch} completed with reward {reward:.2f} (Goal: {reached_goal})",
            state_summary=event.payload,
            importance=importance,
            step=event.training_step,
        )

    def save_to_disk(self) -> None:
        """Save memory snapshots to JSON files."""
        exp_mem_dir = self.memory_dir / self.experiment_id
        exp_mem_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_mem_dir / "episodic.json", "w", encoding="utf-8") as f:
            json.dump(self.episodic.to_list(), f, indent=2)

        with open(exp_mem_dir / "semantic.json", "w", encoding="utf-8") as f:
            json.dump(self.semantic.to_dict(), f, indent=2)

        logger.debug("Persisted memory snapshot for experiment %s", self.experiment_id)

    def load_from_disk(self) -> None:
        """Load memory state from disk if available."""
        exp_mem_dir = self.memory_dir / self.experiment_id
        ep_file = exp_mem_dir / "episodic.json"
        sem_file = exp_mem_dir / "semantic.json"

        if ep_file.exists():
            with open(ep_file, "r", encoding="utf-8") as f:
                self.episodic.load_from_list(json.load(f))

        if sem_file.exists():
            with open(sem_file, "r", encoding="utf-8") as f:
                self.semantic.load_from_dict(json.load(f))
