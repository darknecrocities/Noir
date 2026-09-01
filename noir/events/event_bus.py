"""Thread-safe, multi-subscriber event bus for real-time streaming."""

import collections
import inspect
import queue
import threading
from typing import Callable, DefaultDict, Dict, List, Optional, Set, Union
from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_types import EventType

logger = get_logger("event_bus")

EventHandler = Callable[[NoirEvent], None]


class EventBus:
    """Central event distribution hub facilitating decoupled communication."""

    def __init__(self, history_size: int = 1000, async_workers: int = 2):
        self._lock = threading.RLock()
        self._subscribers: DefaultDict[Optional[EventType], List[EventHandler]] = collections.defaultdict(list)
        self._history: collections.deque[NoirEvent] = collections.deque(maxlen=history_size)
        self._total_published = 0

        # Async queue and worker thread
        self._async_queue: queue.Queue[NoirEvent] = queue.Queue(maxsize=10000)
        self._running = True
        self._worker_threads: List[threading.Thread] = []

        for i in range(async_workers):
            t = threading.Thread(target=self._worker_loop, name=f"NoirEventWorker-{i}", daemon=True)
            t.start()
            self._worker_threads.append(t)

    def subscribe(self, event_type: Optional[Union[EventType, str]], handler: EventHandler) -> None:
        """Subscribe a handler callback to a specific event type or ALL events (if None)."""
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                pass

        with self._lock:
            handlers = self._subscribers[event_type]
            if handler not in handlers:
                handlers.append(handler)
                logger.debug("Subscribed handler %s to event %s", handler, event_type)

    def unsubscribe(self, event_type: Optional[Union[EventType, str]], handler: EventHandler) -> None:
        """Unsubscribe a handler callback."""
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                pass

        with self._lock:
            handlers = self._subscribers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)
                logger.debug("Unsubscribed handler %s from event %s", handler, event_type)

    def publish(self, event: NoirEvent, asynchronous: bool = True) -> None:
        """Publish an event to all matched subscribers.

        Args:
            event: The NoirEvent instance to publish.
            asynchronous: If True, queues for background processing. If False, executes immediately.
        """
        with self._lock:
            self._history.append(event)
            self._total_published += 1

        if asynchronous:
            try:
                self._async_queue.put_nowait(event)
            except queue.Full:
                logger.warning("Event bus async queue is full. Dropping event: %s", event.event_type)
        else:
            self._dispatch(event)

    def _dispatch(self, event: NoirEvent) -> None:
        """Internal dispatch logic executing matched handlers."""
        handlers_to_call: List[EventHandler] = []
        with self._lock:
            # Type-specific handlers
            handlers_to_call.extend(self._subscribers.get(event.event_type, []))
            # Global handlers
            handlers_to_call.extend(self._subscribers.get(None, []))

        for handler in handlers_to_call:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    "Error executing event handler %s for event %s: %s",
                    handler,
                    event.event_type.value,
                    e,
                    exc_info=True,
                )

    def _worker_loop(self) -> None:
        """Background thread worker pulling from queue."""
        while self._running:
            try:
                event = self._async_queue.get(timeout=0.1)
                self._dispatch(event)
                self._async_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Unexpected error in event worker loop: %s", e, exc_info=True)

    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[NoirEvent]:
        """Retrieve recent events from in-memory history buffer."""
        with self._lock:
            events = list(self._history)

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    @property
    def total_published(self) -> int:
        with self._lock:
            return self._total_published

    def clear(self) -> None:
        """Clear all subscribers and history."""
        with self._lock:
            self._subscribers.clear()
            self._history.clear()
            self._total_published = 0

    def shutdown(self) -> None:
        """Gracefully stop background worker threads."""
        self._running = False
        for t in self._worker_threads:
            t.join(timeout=1.0)


# Global singleton instance for easy cross-module access
_global_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get or create the global singleton EventBus."""
    global _global_bus
    with _bus_lock:
        if _global_bus is None:
            _global_bus = EventBus()
        return _global_bus
