"""Lifecycle state management for the Project NOIR application."""

from enum import Enum
import threading
import time
from typing import Callable, Dict, List, Optional
from noir.core.exceptions import EngineStateError
from noir.core.logging import get_logger

logger = get_logger("lifecycle")


class LifecycleState(str, Enum):
    """Execution lifecycle states."""
    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


VALID_TRANSITIONS: Dict[LifecycleState, List[LifecycleState]] = {
    LifecycleState.UNINITIALIZED: [LifecycleState.INITIALIZING, LifecycleState.ERROR],
    LifecycleState.INITIALIZING: [LifecycleState.READY, LifecycleState.ERROR],
    LifecycleState.READY: [LifecycleState.STARTING, LifecycleState.RUNNING, LifecycleState.STOPPED, LifecycleState.ERROR],
    LifecycleState.STARTING: [LifecycleState.RUNNING, LifecycleState.ERROR, LifecycleState.STOPPED],
    LifecycleState.RUNNING: [LifecycleState.PAUSED, LifecycleState.STOPPING, LifecycleState.STOPPED, LifecycleState.ERROR],
    LifecycleState.PAUSED: [LifecycleState.RUNNING, LifecycleState.STOPPING, LifecycleState.STOPPED, LifecycleState.ERROR],
    LifecycleState.STOPPING: [LifecycleState.STOPPED, LifecycleState.ERROR],
    LifecycleState.STOPPED: [LifecycleState.INITIALIZING, LifecycleState.STARTING, LifecycleState.RUNNING, LifecycleState.READY],
    LifecycleState.ERROR: [LifecycleState.INITIALIZING, LifecycleState.STOPPED, LifecycleState.READY],
}


class LifecycleManager:
    """Manages application and training execution state transitions thread-safely."""

    def __init__(self, initial_state: LifecycleState = LifecycleState.UNINITIALIZED):
        self._state = initial_state
        self._lock = threading.RLock()
        self._listeners: List[Callable[[LifecycleState, LifecycleState], None]] = []
        self._state_start_time = time.time()

    @property
    def current_state(self) -> LifecycleState:
        with self._lock:
            return self._state

    def transition_to(self, target_state: LifecycleState) -> LifecycleState:
        """Transition to a new state if valid.

        Args:
            target_state: Desired next state.

        Returns:
            The newly assigned state.

        Raises:
            EngineStateError: If the transition is illegal.
        """
        with self._lock:
            current = self._state
            if current == target_state:
                return current

            allowed = VALID_TRANSITIONS.get(current, [])
            if target_state not in allowed:
                msg = f"Illegal lifecycle transition from {current.value} to {target_state.value}"
                logger.error(msg)
                raise EngineStateError(msg)

            self._state = target_state
            self._state_start_time = time.time()
            logger.info("Lifecycle state changed: %s -> %s", current.value, target_state.value)

            listeners_copy = list(self._listeners)

        for listener in listeners_copy:
            try:
                listener(current, target_state)
            except Exception as e:
                logger.error("Error in lifecycle listener callback: %s", e, exc_info=True)

        return target_state

    def add_listener(self, callback: Callable[[LifecycleState, LifecycleState], None]) -> None:
        """Register a callback for state transition events."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[LifecycleState, LifecycleState], None]) -> None:
        """Unregister a state transition callback."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def is_running(self) -> bool:
        with self._lock:
            return self._state == LifecycleState.RUNNING

    def is_paused(self) -> bool:
        with self._lock:
            return self._state == LifecycleState.PAUSED

    def is_stopped(self) -> bool:
        with self._lock:
            return self._state in (LifecycleState.STOPPED, LifecycleState.READY, LifecycleState.UNINITIALIZED)
