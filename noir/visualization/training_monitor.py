"""Training monitor extracting hook outputs and updating the visualizer."""

from typing import Any, Optional
from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_bus import EventBus, get_event_bus
from noir.events.event_types import EventType
from noir.visualization.neural_graph import NeuralGraph
from noir.visualization.visualizer_3d import NeuralVisualizer3D

logger = get_logger("visualization.monitor")


class TrainingMonitor:
    """Subscribes to training events and pushes state updates into the 3D visualizer."""

    def __init__(
        self,
        visualizer: NeuralVisualizer3D,
        engine: Any,
        event_bus: Optional[EventBus] = None,
    ):
        self.visualizer = visualizer
        self.engine = engine
        self.event_bus = event_bus or get_event_bus()
        self.graph = NeuralGraph()
        self._subscribe_events()

    def _subscribe_events(self) -> None:
        self.event_bus.subscribe(EventType.WEIGHTS_UPDATED, self._on_weights_updated)
        self.event_bus.subscribe(EventType.SURPRISE_DETECTED, self._on_surprise)
        self.event_bus.subscribe(EventType.REWARD_RECEIVED, self._on_reward)

    def _on_weights_updated(self, event: NoirEvent) -> None:
        if hasattr(self.engine, "model") and self.engine.model:
            self.graph.update_from_model(self.engine.model)
            self.visualizer.set_graph(self.graph)

    def _on_surprise(self, event: NoirEvent) -> None:
        val = event.payload.get("surprise_value", 0.8)
        self.visualizer.trigger_surprise_shock(intensity=val)

    def _on_reward(self, event: NoirEvent) -> None:
        reward = event.payload.get("reward", 1.0)
        if reward > 0:
            self.visualizer.trigger_reward_pulse(amount=reward)
