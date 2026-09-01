"""Unit tests for EventBus subsystem."""

import time
import pytest
from noir.events.event import NoirEvent
from noir.events.event_bus import EventBus
from noir.events.event_types import EventType


def test_event_bus_subscribe_and_publish():
    bus = EventBus(async_workers=1)
    received = []

    def handler(event: NoirEvent):
        received.append(event)

    bus.subscribe(EventType.LOSS_CALCULATED, handler)

    event = NoirEvent.create(
        EventType.LOSS_CALCULATED,
        experiment_id="test_exp",
        training_step=42,
        loss=0.314,
    )
    bus.publish(event, asynchronous=False)

    assert len(received) == 1
    assert received[0].event_type == EventType.LOSS_CALCULATED
    assert received[0].payload["loss"] == 0.314
    assert received[0].training_step == 42

    bus.shutdown()


def test_event_bus_unsubscribe():
    bus = EventBus(async_workers=1)
    received = []

    def handler(event: NoirEvent):
        received.append(event)

    bus.subscribe(EventType.EPOCH_STARTED, handler)
    bus.unsubscribe(EventType.EPOCH_STARTED, handler)

    event = NoirEvent.create(EventType.EPOCH_STARTED, epoch=1)
    bus.publish(event, asynchronous=False)

    assert len(received) == 0
    bus.shutdown()


def test_event_bus_global_subscriber():
    bus = EventBus(async_workers=1)
    received = []

    def global_handler(event: NoirEvent):
        received.append(event)

    bus.subscribe(None, global_handler)

    e1 = NoirEvent.create(EventType.TRAINING_STARTED)
    e2 = NoirEvent.create(EventType.REWARD_RECEIVED, reward=10.0)

    bus.publish(e1, asynchronous=False)
    bus.publish(e2, asynchronous=False)

    assert len(received) == 2
    assert bus.total_published == 2
    bus.shutdown()
