"""Real-Time Open Web Language Model Trainer executing continuous internet learning."""

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
            weight_decay=0.01,
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

        logger.info(
            "OpenWebLLMTrainer initialized on device: %s (%s)",
            self.device,
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
        )

    def _train_loop(self) -> None:
        """Main continuous open web training loop."""
        logger.info("Commencing live Open Web learning loop (Max steps: %d)...", self.max_steps)

        for step in range(self.global_step, self.max_steps):
            if self._check_pause_and_stop():
                break

            step_start_time = time.time()
            self._on_batch_start(step)

            # 1. Fetch live internet tokens batch
            inputs, targets, article_title = self.streamer.create_batch_stream(
                batch_size=self.batch_size,
                block_size=self.block_size,
            )
            self.active_article_title = article_title

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

            # 4. Compute Perplexity: exp(cross_entropy_loss)
            perplexity = math.exp(min(loss_val, 20.0))

            # 5. Affective Mind Engine Update
            if self.affective_engine:
                with torch.no_grad():
                    # Calculate output probability distribution entropy
                    probs = torch.softmax(logits[:, -1, :], dim=-1).cpu().numpy()
                    self.affective_engine.update_from_supervised_step(
                        loss=loss_val,
                        accuracy=1.0 / max(1.0, math.log(perplexity + 1.0)),
                        probabilities=probs,
                        step=step,
                    )

            # 6. Periodically generate text sample from model weights
            if step % 25 == 0 or step == 1:
                self._generate_sample_text()

            # 7. Metrics & Event Tracking
            step_duration = time.time() - step_start_time
            step_metrics = {
                "train_loss": loss_val,
                "perplexity": perplexity,
                "grad_norm": grad_norm,
                "lr": get_learning_rate(self.optimizer),
                "step_time_ms": step_duration * 1000.0,
                "article": self.active_article_title,
                "generated_sample": self.latest_generated_text,
            }

            self.global_step = step
            self.latest_metrics = step_metrics
            self.metric_tracker.update(step_metrics)

            self._on_batch_end(step, loss_val, step_metrics)

            if step % 10 == 0:
                logger.info(
                    "Step %d | Web Article: '%s' | Loss: %.4f | Perplexity: %.2f | GPU Mem: %.1fMB",
                    step,
                    self.active_article_title[:30],
                    loss_val,
                    perplexity,
                    torch.cuda.memory_allocated() / (1024 * 1024) if self.device.type == "cuda" else 0.0,
                )

            # High-performance pacing
            time.sleep(0.01)

        logger.info("Open Web LLM Training loop concluded.")

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
                    max_new_tokens=30,
                    temperature=0.8,
                    top_k=40,
                )
                generated_tokens = generated_idx[0].cpu().tolist()
                self.latest_generated_text = self.tokenizer.decode(generated_tokens)
                logger.info("Sample text generated by LLM: '%s'", self.latest_generated_text[:60])
        except Exception as e:
            logger.debug("Text generation sample error: %s", e)
        finally:
            self.model.train()
