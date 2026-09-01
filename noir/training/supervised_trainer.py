"""Supervised learning trainer executing real PyTorch optimization."""

import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from noir.core.logging import get_logger
from noir.models.base import NoirBaseModel
from noir.training.base_trainer import BaseTrainer
from noir.training.callbacks import BaseCallback
from noir.training.metrics import (
    MetricTracker,
    calculate_accuracy,
    calculate_gradient_norm,
    get_learning_rate,
)

logger = get_logger("training.supervised")


class SupervisedTrainer(BaseTrainer):
    """Supervised trainer for real deep neural network optimization."""

    def __init__(
        self,
        experiment_id: str,
        model: NoirBaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        num_epochs: int = 100,
        gradient_clip_val: float = 0.5,
        device: str = "auto",
        callbacks: Optional[List[BaseCallback]] = None,
        affective_engine: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(experiment_id=experiment_id, device=device, callbacks=callbacks)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.gradient_clip_val = gradient_clip_val
        self.affective_engine = affective_engine
        self.config = config or {}

        self.metric_tracker = MetricTracker(window_size=50)

    def _train_loop(self) -> None:
        """Main multi-epoch supervised optimization loop."""
        logger.info(
            "Starting supervised training: epochs=%d, batches_per_epoch=%d",
            self.num_epochs,
            len(self.train_loader),
        )

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.num_epochs):
            if self._check_pause_and_stop():
                break

            self.current_epoch = epoch
            self._on_epoch_start(epoch)
            self.model.train()

            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = len(self.train_loader)

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if self._check_pause_and_stop():
                    break

                step_start_time = time.time()
                self._on_batch_start(batch_idx)

                # Move tensors to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 1. Zero gradients
                self.optimizer.zero_grad()

                # 2. Forward pass (Hooks record layer activations)
                outputs = self.model(inputs)

                # 3. Calculate Loss
                loss = self.criterion(outputs, targets)

                # 4. Backpropagation (Hooks record layer gradients)
                loss.backward()

                # 5. Extract metrics & gradient norm
                grad_norm = calculate_gradient_norm(self.model)
                if self.gradient_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                # 6. Optimizer step (Real weight update)
                self.optimizer.step()

                # 7. Compute step metrics
                loss_val = float(loss.item())
                acc_val = calculate_accuracy(outputs, targets)
                lr_val = get_learning_rate(self.optimizer)
                step_time = time.time() - step_start_time

                self.global_step += 1
                epoch_loss += loss_val
                epoch_acc += acc_val

                self.metric_tracker.update("train_loss", loss_val)
                self.metric_tracker.update("train_acc", acc_val)
                self.metric_tracker.update("grad_norm", grad_norm)
                self.metric_tracker.update("lr", lr_val)

                step_metrics = {
                    "train_loss": loss_val,
                    "train_acc": acc_val,
                    "grad_norm": grad_norm,
                    "lr": lr_val,
                    "step_time": step_time,
                }
                self.latest_metrics = step_metrics

                # 8. Update Affective Mind Engine if attached
                if self.affective_engine:
                    try:
                        probs = torch.softmax(outputs.detach(), dim=-1).cpu().numpy()
                        self.affective_engine.update_from_supervised_step(
                            loss=loss_val,
                            accuracy=acc_val,
                            probabilities=probs,
                            step=self.global_step,
                        )
                    except Exception as me:
                        logger.debug("Mind update error: %s", me)

                self._on_batch_end(batch_idx, loss_val, step_metrics)
                self._on_step_end(self.global_step, step_metrics)

                # Small yield for responsive UI threading
                time.sleep(0.001)

            # Validation pass
            val_metrics = self._evaluate()
            if self.scheduler:
                self.scheduler.step()

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            avg_epoch_acc = epoch_acc / max(1, num_batches)

            epoch_summary = {
                "epoch_train_loss": avg_epoch_loss,
                "epoch_train_acc": avg_epoch_acc,
                **val_metrics,
            }
            self.latest_metrics.update(epoch_summary)
            self._on_epoch_end(epoch, self.latest_metrics)

            logger.info(
                "Epoch %d/%d completed | Loss: %.4f | Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
                epoch + 1,
                self.num_epochs,
                avg_epoch_loss,
                avg_epoch_acc * 100,
                val_metrics.get("val_loss", 0.0),
                val_metrics.get("val_acc", 0.0) * 100,
            )

    def _evaluate(self) -> Dict[str, float]:
        """Perform evaluation pass on validation loader."""
        if not self.val_loader:
            return {}

        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += float(loss.item())
                val_acc += calculate_accuracy(outputs, targets)

        return {
            "val_loss": val_loss / max(1, num_batches),
            "val_acc": val_acc / max(1, num_batches),
        }
