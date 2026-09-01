"""Abstract base trainer managing training threads and lifecycle."""

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch

from noir.core.exceptions import TrainingError
from noir.core.lifecycle import LifecycleManager, LifecycleState
from noir.core.logging import get_logger
from noir.training.callbacks import BaseCallback

logger = get_logger("training.base_trainer")


class BaseTrainer(ABC):
    """Base trainer managing lifecycle, execution thread, and device placement."""

    def __init__(
        self,
        experiment_id: str,
        device: str = "auto",
        callbacks: Optional[List[BaseCallback]] = None,
    ):
        self.experiment_id = experiment_id
        self.callbacks = callbacks or []
        self.lifecycle = LifecycleManager(LifecycleState.READY)

        # Device determination
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.global_step = 0
        self.current_epoch = 0
        self.latest_metrics: Dict[str, float] = {}

        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        self._stop_event = threading.Event()

    def start_training(self) -> None:
        """Start the training process asynchronously in a dedicated worker thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Trainer thread is already running.")
            return

        self._stop_event.clear()
        self._pause_event.set()
        self.lifecycle.transition_to(LifecycleState.RUNNING)

        self._thread = threading.Thread(
            target=self._run_training_loop,
            name=f"NoirTrainer-{self.experiment_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Training thread started for experiment %s on device %s", self.experiment_id, self.device)

    def pause_training(self) -> None:
        """Pause the ongoing training loop."""
        if self.lifecycle.current_state == LifecycleState.RUNNING:
            self._pause_event.clear()
            self.lifecycle.transition_to(LifecycleState.PAUSED)
            logger.info("Training paused at step %d", self.global_step)

    def resume_training(self) -> None:
        """Resume paused training."""
        if self.lifecycle.current_state == LifecycleState.PAUSED:
            self._pause_event.set()
            self.lifecycle.transition_to(LifecycleState.RUNNING)
            logger.info("Training resumed at step %d", self.global_step)

    def stop_training(self, wait: bool = True) -> None:
        """Signal training loop to terminate cleanly."""
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused
        if self.lifecycle.current_state in (LifecycleState.RUNNING, LifecycleState.PAUSED):
            self.lifecycle.transition_to(LifecycleState.STOPPING)

        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self.lifecycle.transition_to(LifecycleState.STOPPED)
        logger.info("Training stopped at step %d", self.global_step)

    def _run_training_loop(self) -> None:
        """Thread worker invoking the concrete training procedure."""
        try:
            self._on_train_start()
            self._train_loop()
        except Exception as e:
            logger.error("Error in training loop: %s", e, exc_info=True)
            self.lifecycle.transition_to(LifecycleState.ERROR)
            raise TrainingError(f"Training loop execution failed: {e}") from e
        finally:
            self._on_train_end()
            if self.lifecycle.current_state not in (LifecycleState.ERROR, LifecycleState.STOPPED):
                self.lifecycle.transition_to(LifecycleState.STOPPED)

    @abstractmethod
    def _train_loop(self) -> None:
        """Concrete training loop implementation."""
        pass

    def _check_pause_and_stop(self) -> bool:
        """Check if stopped or block while paused. Returns True if stop requested."""
        if self._stop_event.is_set():
            return True

        if not self._pause_event.is_set():
            logger.debug("Trainer entering wait state (paused)...")
            self._pause_event.wait()
            logger.debug("Trainer woke up from pause.")

        return self._stop_event.is_set()

    # Callback Triggers
    def _on_train_start(self) -> None:
        for cb in self.callbacks:
            cb.on_train_start(self)

    def _on_train_end(self) -> None:
        for cb in self.callbacks:
            cb.on_train_end(self)

    def _on_epoch_start(self, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(self, epoch)

    def _on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(self, epoch, metrics)

    def _on_batch_start(self, batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_start(self, batch_idx)

    def _on_batch_end(self, batch_idx: int, loss: float, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(self, batch_idx, loss, metrics)

    def _on_step_end(self, step: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_step_end(self, step, metrics)
