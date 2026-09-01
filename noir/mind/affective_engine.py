"""Core Affective Engine governing mathematical internal state transitions."""

import threading
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_bus import EventBus, get_event_bus
from noir.events.event_types import EventType
from noir.mind.curiosity import CuriosityEngine
from noir.mind.emotion_state import EmotionState
from noir.mind.motivation import MotivationEngine
from noir.mind.surprise import SurpriseDetector, calculate_surprise
from noir.mind.uncertainty import calculate_entropy, calculate_prediction_confidence

logger = get_logger("mind.affective_engine")


class AffectiveEngine:
    """Computes, balances, and updates the mathematical affective state vector."""

    def __init__(
        self,
        experiment_id: str = "default",
        surprise_threshold: float = 0.70,
        curiosity_weight: float = 0.20,
        frustration_decay: float = 0.95,
        confidence_decay: float = 0.99,
        event_bus: Optional[EventBus] = None,
        device: str = "cpu",
    ):
        self.experiment_id = experiment_id
        self.state = EmotionState()
        self.surprise_detector = SurpriseDetector(threshold=surprise_threshold)
        self.curiosity_engine = CuriosityEngine(intrinsic_weight=curiosity_weight, device=device)
        self.motivation_engine = MotivationEngine()
        self.event_bus = event_bus or get_event_bus()

        self.frustration_decay = frustration_decay
        self.confidence_decay = confidence_decay

        self._lock = threading.RLock()
        self._prev_loss: Optional[float] = None
        self._loss_stagnation_count = 0
        self._consecutive_rewards = 0

    @property
    def current_state(self) -> EmotionState:
        with self._lock:
            return EmotionState.from_dict(self.state.to_dict())

    def update_from_supervised_step(
        self,
        loss: float,
        accuracy: float,
        probabilities: np.ndarray,
        step: int,
    ) -> EmotionState:
        """Update emotional state from a supervised optimization batch."""
        with self._lock:
            # 1. Uncertainty: Normalized Shannon entropy
            u_t = calculate_entropy(probabilities)
            conf_pred = calculate_prediction_confidence(probabilities)

            # 2. Loss delta and stagnation
            loss_delta = 0.0
            if self._prev_loss is not None:
                loss_delta = self._prev_loss - loss
                if loss_delta < 1e-4:
                    self._loss_stagnation_count += 1
                else:
                    self._loss_stagnation_count = max(0, self._loss_stagnation_count - 1)
            self._prev_loss = loss

            # 3. Surprise calculation based on sudden prediction shocks or high loss
            is_surprised, raw_surprise, norm_surprise = self.surprise_detector.update(loss)

            # 4. Mathematical state updates
            # Grounded Confidence: converges smoothly with loss reduction, low entropy, and prediction certainty
            loss_factor = float(np.exp(-min(loss, 6.0) / 2.5))
            target_confidence = 0.45 * loss_factor + 0.35 * (1.0 - u_t) + 0.20 * max(accuracy, conf_pred)
            c_t = 0.85 * self.state.confidence + 0.15 * target_confidence

            # Frustration grows when loss is high and stagnating, but decays when model is in high-confidence convergence (loss < 1.8)
            stagnation_weight = float(np.clip((loss - 1.5) / 2.0, 0.0, 1.0))
            f_t = self.state.frustration * self.frustration_decay + (0.08 * stagnation_weight * min(5, self._loss_stagnation_count) / 5.0)

            # Satisfaction tracks steady loss reductions and high accuracy
            s_t = 0.85 * self.state.satisfaction + 0.15 * (0.5 * loss_factor + 0.5 * max(0.0, min(1.0, loss_delta * 4.0)))

            # Anticipation scales with positive learning velocity
            a_t = 0.80 * self.state.anticipation + 0.20 * (0.5 + 0.3 * (accuracy - 0.5) + 0.2 * (1.0 - f_t))

            # Curiosity rises with information entropy and novel surprises
            x_t = 0.80 * self.state.curiosity + 0.20 * (0.4 * u_t + 0.4 * norm_surprise + 0.2 * (1.0 - c_t))

            # Caution scales with high entropy and frustration
            ca_t = 0.80 * self.state.caution + 0.20 * (0.5 * u_t + 0.3 * f_t + 0.2 * (1.0 - c_t))

            # Persistence remains highly resilient (0.80 - 1.00)
            p_t = 0.85 + 0.15 * (1.0 - f_t)

            self._apply_and_emit(
                c=c_t, f=f_t, a=a_t, s=s_t, u=u_t, x=x_t, ca=ca_t, p=p_t,
                step=step, is_surprised=is_surprised, surprise_val=norm_surprise,
            )
            return self.current_state

    def update_from_rl_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        info: Dict[str, Any],
        step: int,
    ) -> Tuple[EmotionState, float]:
        """Update emotional state from RL environment interaction.

        Returns:
            (updated_emotion_state, total_reward_with_intrinsic)
        """
        with self._lock:
            # 1. Compute Curiosity and Forward Dynamics Prediction Error
            int_reward, pred_error = self.curiosity_engine.compute_intrinsic_reward(
                state, action, next_state, train_model=True
            )
            total_reward = reward + int_reward

            # 2. Surprise detection
            is_surprised, raw_surprise, norm_surprise = self.surprise_detector.update(pred_error)

            # 3. Novelty & Goal Progress
            s_np = state.detach().cpu().numpy().flatten()
            novelty = self.motivation_engine.compute_novelty(s_np)
            goal_dist = info.get("distance", 1.0)
            goal_prog = self.motivation_engine.compute_goal_progress(goal_dist, 8.0)

            # 4. State updates
            if reward > 0:
                self._consecutive_rewards += 1
            else:
                self._consecutive_rewards = 0

            # Confidence
            c_t = self.state.confidence * self.confidence_decay + 0.1 * min(1.0, self._consecutive_rewards / 5.0)
            # Frustration increases on negative reward or collision
            penalty = 0.2 if info.get("hit_obstacle", False) else 0.0
            f_t = self.state.frustration * self.frustration_decay + (0.15 if reward < 0 else 0.0) + penalty
            # Satisfaction spikes on goal completion (+10)
            s_t = self.state.satisfaction * 0.85 + (0.4 if info.get("reached_goal", False) else 0.05 * max(0.0, reward))
            # Anticipation grows as agent approaches goal
            a_t = 0.3 + 0.6 * goal_prog
            # Uncertainty reflects surprise and novelty
            u_t = 0.2 + 0.4 * norm_surprise + 0.4 * novelty
            # Curiosity stimulated by novelty and prediction errors
            x_t = 0.2 + 0.5 * novelty + 0.3 * norm_surprise
            # Caution rises near obstacles or after penalties
            ca_t = 0.2 + 0.6 * penalty + 0.2 * u_t
            # Persistence
            p_t = 0.80 + 0.20 * goal_prog

            self._apply_and_emit(
                c=c_t, f=f_t, a=a_t, s=s_t, u=u_t, x=x_t, ca=ca_t, p=p_t,
                step=step, is_surprised=is_surprised, surprise_val=norm_surprise,
            )
            return self.current_state, total_reward

    def _apply_and_emit(
        self,
        c: float, f: float, a: float, s: float, u: float, x: float, ca: float, p: float,
        step: int, is_surprised: bool, surprise_val: float,
    ) -> None:
        """Clip values, store state, and publish events."""
        self.state = EmotionState(
            confidence=float(np.clip(c, 0.0, 1.0)),
            frustration=float(np.clip(f, 0.0, 1.0)),
            anticipation=float(np.clip(a, 0.0, 1.0)),
            satisfaction=float(np.clip(s, 0.0, 1.0)),
            uncertainty=float(np.clip(u, 0.0, 1.0)),
            curiosity=float(np.clip(x, 0.0, 1.0)),
            caution=float(np.clip(ca, 0.0, 1.0)),
            persistence=float(np.clip(p, 0.0, 1.0)),
        )

        state_dict = self.state.to_dict()

        self.event_bus.publish(
            NoirEvent.create(
                EventType.EMOTION_UPDATED,
                experiment_id=self.experiment_id,
                training_step=step,
                emotion_state=state_dict,
            ),
            asynchronous=True,
        )

        self.event_bus.publish(
            NoirEvent.create(
                EventType.UNCERTAINTY_UPDATED,
                experiment_id=self.experiment_id,
                training_step=step,
                uncertainty=self.state.uncertainty,
                confidence=self.state.confidence,
            ),
            asynchronous=True,
        )

        if is_surprised:
            logger.info("Surprise event triggered at step %d (Surprise: %.3f)", step, surprise_val)
            self.event_bus.publish(
                NoirEvent.create(
                    EventType.SURPRISE_DETECTED,
                    experiment_id=self.experiment_id,
                    training_step=step,
                    surprise_value=surprise_val,
                    curiosity=self.state.curiosity,
                ),
                asynchronous=True,
            )
