"""Callback system for training lifecycle hooks and event generation."""

import time
from typing import Any, Dict, List, Optional
import torch

from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_bus import EventBus, get_event_bus
from noir.events.event_types import EventType
from noir.storage.checkpoint_manager import CheckpointManager
from noir.storage.metrics_repository import MetricsRepository

logger = get_logger("training.callbacks")


class BaseCallback:
    """Base interface for training loop callbacks."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        pass

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float, metrics: Dict[str, float]) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, metrics: Dict[str, float]) -> None:
        pass


class EventEmissionCallback(BaseCallback):
    """Publishes training events to the internal EventBus."""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or get_event_bus()

    def on_train_start(self, trainer: Any) -> None:
        self.event_bus.publish(
            NoirEvent.create(
                EventType.TRAINING_STARTED,
                experiment_id=trainer.experiment_id,
                training_step=trainer.global_step,
                epoch=trainer.current_epoch,
                timestamp=time.time(),
            )
        )

    def on_train_end(self, trainer: Any) -> None:
        self.event_bus.publish(
            NoirEvent.create(
                EventType.TRAINING_STOPPED,
                experiment_id=trainer.experiment_id,
                training_step=trainer.global_step,
                epoch=trainer.current_epoch,
            )
        )

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self.event_bus.publish(
            NoirEvent.create(
                EventType.EPOCH_STARTED,
                experiment_id=trainer.experiment_id,
                training_step=trainer.global_step,
                epoch=epoch,
            )
        )

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        self.event_bus.publish(
            NoirEvent.create(
                EventType.EPOCH_COMPLETED,
                experiment_id=trainer.experiment_id,
                training_step=trainer.global_step,
                epoch=epoch,
                metrics=metrics,
            )
        )

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float, metrics: Dict[str, float]) -> None:
        # Emit weights updated and metrics
        self.event_bus.publish(
            NoirEvent.create(
                EventType.WEIGHTS_UPDATED,
                experiment_id=trainer.experiment_id,
                training_step=trainer.global_step,
                epoch=trainer.current_epoch,
                loss=loss,
                metrics=metrics,
                grad_norm=metrics.get("grad_norm", 0.0),
            ),
            asynchronous=True,
        )


class MetricsLoggingCallback(BaseCallback):
    """Persists metrics directly to SQLite MetricsRepository."""

    def __init__(self, metrics_repo: MetricsRepository):
        self.metrics_repo = metrics_repo

    def on_step_end(self, trainer: Any, step: int, metrics: Dict[str, float]) -> None:
        self.metrics_repo.log_metrics_dict(
            experiment_id=trainer.experiment_id,
            step=step,
            metrics=metrics,
            epoch=trainer.current_epoch,
        )

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        self.metrics_repo.log_metrics_dict(
            experiment_id=trainer.experiment_id,
            step=trainer.global_step,
            metrics=metrics,
            epoch=epoch,
            flush_immediately=True,
        )


class CheckpointCallback(BaseCallback):
    """Automatically saves checkpoints periodically and on best performance."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        interval_steps: int = 100,
        autosave_seconds: int = 60,
        best_metric_name: str = "val_acc",
        higher_is_better: bool = True,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.interval_steps = interval_steps
        self.autosave_seconds = autosave_seconds
        self.best_metric_name = best_metric_name
        self.higher_is_better = higher_is_better

        self.last_save_time = time.time()
        self.best_metric_value: Optional[float] = None

    def on_step_end(self, trainer: Any, step: int, metrics: Dict[str, float]) -> None:
        now = time.time()
        step_trigger = (step > 0 and step % self.interval_steps == 0)
        time_trigger = (now - self.last_save_time >= self.autosave_seconds)

        is_best = False
        if self.best_metric_name in metrics:
            val = metrics[self.best_metric_name]
            if self.best_metric_value is None:
                self.best_metric_value = val
                is_best = True
            elif self.higher_is_better and val > self.best_metric_value:
                self.best_metric_value = val
                is_best = True
            elif not self.higher_is_better and val < self.best_metric_value:
                self.best_metric_value = val
                is_best = True

        if step_trigger or time_trigger or is_best:
            self._save(trainer, is_best=is_best)
            self.last_save_time = now

    def _save(self, trainer: Any, is_best: bool = False, tag: Optional[str] = None) -> None:
        try:
            emotion_state = {}
            if hasattr(trainer, "affective_engine") and trainer.affective_engine:
                emotion_state = trainer.affective_engine.current_state.to_dict()

            self.checkpoint_manager.save_checkpoint(
                experiment_id=trainer.experiment_id,
                step=trainer.global_step,
                epoch=trainer.current_epoch,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=getattr(trainer, "scheduler", None),
                emotion_state=emotion_state,
                config=getattr(trainer, "config", {}),
                metrics=trainer.latest_metrics,
                is_best=is_best,
                tag=tag,
            )
        except Exception as e:
            logger.error("Failed to execute automatic checkpoint: %s", e)
