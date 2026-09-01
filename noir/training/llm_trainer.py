"""Real-Time Open Web Language Model Trainer executing continuous internet learning on GPU."""

import math
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from noir.core.logging import get_logger
from noir.datasets.open_web import OpenWebStreamer, WebTextTokenizer
from noir.events.event import NoirEvent
from noir.events.event_types import EventType
from noir.models.transformer import NoirTransformerLM
from noir.training.base_trainer import BaseTrainer
from noir.training.callbacks import BaseCallback
from noir.training.metrics import MetricTracker, calculate_gradient_norm, get_learning_rate

logger = get_logger("training.open_web_llm")


class OpenWebLLMTrainer(BaseTrainer):
    """Continuously trains a Causal Transformer directly on live open web internet data."""

    def __init__(
        self,
        experiment_id: str,
        model: NoirTransformerLM,
        learning_rate: float = 0.0005,
        batch_size: int = 16,
        block_size: int = 64,
        gradient_clip_val: float = 1.0,
        max_steps: int = 10000,
        device: str = "auto",
        callbacks: Optional[List[BaseCallback]] = None,
        affective_engine: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(experiment_id=experiment_id, device=device, callbacks=callbacks)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,  # Anti-overfitting regularization
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps,
            eta_min=learning_rate * 0.1,
        )

        self.batch_size = batch_size
        self.block_size = block_size
        self.gradient_clip_val = gradient_clip_val
        self.max_steps = max_steps
        self.affective_engine = affective_engine
        self.config = config or {}

        self.streamer = OpenWebStreamer()
        self.tokenizer = WebTextTokenizer()
        self.metric_tracker = MetricTracker(window_size=50)

        self.latest_generated_text = ""
        self.active_article_title = "Live Internet Stream"
        self.active_article_url = "https://en.wikipedia.org"
        self.latest_val_loss = 0.0
        self.latest_val_perplexity = 0.0

        logger.info(
            "OpenWebLLMTrainer initialized on device: %s (%s)",
            self.device,
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
        )

    def _train_loop(self) -> None:
        """Main continuous open web training loop with anti-overfitting validation."""
        logger.info("Commencing live Open Web learning loop on GPU (Max steps: %d)...", self.max_steps)

        for step in range(self.global_step, self.max_steps):
            if self._check_pause_and_stop():
                break

            step_start_time = time.time()
            self._on_batch_start(step)

            # 1. Fetch live internet training batch
            inputs, targets, article_title, article_url = self.streamer.create_batch(
                batch_size=self.batch_size,
                block_size=self.block_size,
                is_validation=False,
            )
            self.active_article_title = article_title
            self.active_article_url = article_url

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 2. Forward pass with causal self-attention
            self.model.train()
            self.optimizer.zero_grad()

            logits, loss = self.model(inputs, targets=targets)
            loss_val = float(loss.item())

            # 3. Backpropagation & Weight Update on GPU
            loss.backward()

            grad_norm = calculate_gradient_norm(self.model)
            if self.gradient_clip_val > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            self.optimizer.step()
            self.scheduler.step()

            # 4. Compute Train Perplexity: exp(loss)
            perplexity = math.exp(min(loss_val, 15.0))

            # 5. Out-of-sample Validation step (Anti-overfitting guard every 20 steps)
            if step % 20 == 0:
                self._evaluate_validation_loss()

            # 6. Affective Mind Engine Update
            if self.affective_engine:
                with torch.no_grad():
                    probs = torch.softmax(logits[:, -1, :], dim=-1).cpu().numpy()
                    self.affective_engine.update_from_supervised_step(
                        loss=loss_val,
                        accuracy=1.0 / max(1.0, math.log(perplexity + 1.0)),
                        probabilities=probs,
                        step=step,
                    )

            # 7. Periodically generate text sample from live model weights
            if step % 25 == 0 or step == 1:
                self._generate_sample_text()

            # 8. Step Metrics & Event Emission
            step_duration = time.time() - step_start_time
            step_metrics = {
                "train_loss": loss_val,
                "perplexity": perplexity,
                "val_loss": self.latest_val_loss or loss_val,
                "val_perplexity": self.latest_val_perplexity or perplexity,
                "grad_norm": grad_norm,
                "lr": get_learning_rate(self.optimizer),
                "step_time_ms": step_duration * 1000.0,
                "article": self.active_article_title,
                "url": self.active_article_url,
                "generated_sample": self.latest_generated_text,
            }

            self.global_step = step + 1
            self.latest_metrics = step_metrics
            self.metric_tracker.update(step_metrics)

            self._on_batch_end(step, loss_val, step_metrics)

            if step % 10 == 0:
                logger.info(
                    "Step %d | [LIVE WEB: %s] | URL: %s | Loss: %.4f | PPL: %.2f | Val PPL: %.2f | GPU: %.1fMB",
                    step,
                    self.active_article_title[:28],
                    self.active_article_url,
                    loss_val,
                    perplexity,
                    self.latest_val_perplexity or perplexity,
                    torch.cuda.memory_allocated() / (1024 * 1024) if self.device.type == "cuda" else 0.0,
                )

            # 9. Periodic GPU memory optimization and clean single-stream pacing
            if self.device.type == "cuda" and step % 50 == 0:
                torch.cuda.empty_cache()

            # Pacing for single-stream stability
            time.sleep(0.01)

        logger.info("Open Web LLM Training loop concluded.")

    def _evaluate_validation_loss(self) -> None:
        """Evaluate next-token prediction loss on unseen out-of-sample internet text."""
        self.model.eval()
        try:
            with torch.no_grad():
                val_x, val_y, _, _ = self.streamer.create_batch(
                    batch_size=self.batch_size,
                    block_size=self.block_size,
                    is_validation=True,
                )
                val_x = val_x.to(self.device)
                val_y = val_y.to(self.device)

                _, v_loss = self.model(val_x, targets=val_y)
                self.latest_val_loss = float(v_loss.item())
                self.latest_val_perplexity = math.exp(min(self.latest_val_loss, 15.0))
        except Exception as e:
            logger.debug("Validation eval error: %s", e)
        finally:
            self.model.train()

    def _generate_sample_text(self) -> None:
        """Autoregressively generate sample completion using live trained weights."""
        self.model.eval()
        try:
            prompt = "The future of intelligence "
            prompt_tokens = self.tokenizer.encode(prompt)
            idx_cond = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

            with torch.no_grad():
                generated_idx = self.model.generate(
                    idx_cond,
                    max_new_tokens=40,
                    temperature=0.8,
                    top_k=40,
                )
                generated_tokens = generated_idx[0].cpu().tolist()
                self.latest_generated_text = self.tokenizer.decode(generated_tokens)
                logger.info("[LLM COMPLETION] '%s'", self.latest_generated_text.replace("\n", " ")[:80])
        except Exception as e:
            logger.debug("Text generation error: %s", e)
        finally:
            self.model.train()
