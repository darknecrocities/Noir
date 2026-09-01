"""Reinforcement Learning Trainer executing real PPO optimization."""

import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_bus import EventBus, get_event_bus
from noir.events.event_types import EventType
from noir.models.actor_critic import ActorCriticNetwork
from noir.rl.agent import PPOAgent
from noir.rl.environment import NoirRLWrapper
from noir.rl.trajectory_memory import TrajectoryBuffer
from noir.training.base_trainer import BaseTrainer
from noir.training.callbacks import BaseCallback
from noir.training.metrics import MetricTracker

logger = get_logger("training.rl")


class RLTrainer(BaseTrainer):
    """Reinforcement learning trainer driving PPO policy learning and affective feedback."""

    def __init__(
        self,
        experiment_id: str,
        network: ActorCriticNetwork,
        env_id: str = "GridWorld-v0",
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        n_steps: int = 256,
        batch_size: int = 64,
        n_epochs: int = 4,
        max_episodes: int = 500,
        device: str = "auto",
        callbacks: Optional[List[BaseCallback]] = None,
        affective_engine: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(experiment_id=experiment_id, device=device, callbacks=callbacks)

        self.network = network.to(self.device)
        self.model = self.network  # Alias for BaseTrainer/Callbacks
        self.agent = PPOAgent(
            network=self.network,
            learning_rate=learning_rate,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            device=str(self.device),
        )
        self.optimizer = self.agent.optimizer  # Alias for BaseTrainer/Callbacks
        self.env = NoirRLWrapper(env_id=env_id)
        self.buffer = TrajectoryBuffer(capacity=n_steps, gamma=gamma, gae_lambda=gae_lambda)

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_episodes = max_episodes
        self.affective_engine = affective_engine
        self.config = config or {}
        self.event_bus = event_bus or get_event_bus()

        self.metric_tracker = MetricTracker(window_size=50)

    def _train_loop(self) -> None:
        """Main on-policy rollout and PPO update loop."""
        logger.info(
            "Starting PPO reinforcement learning: env=%s, rollout_steps=%d, epochs=%d",
            self.env.env_id,
            self.n_steps,
            self.max_episodes,
        )

        obs, info = self.env.reset()
        state = torch.from_numpy(obs).float()

        episode_reward = 0.0
        episode_length = 0
        total_episodes = 0

        while total_episodes < self.max_episodes:
            if self._check_pause_and_stop():
                break

            self.buffer.clear()

            # 1. Rollout Phase: Collect n_steps of environment interaction
            for _ in range(self.n_steps):
                if self._check_pause_and_stop():
                    break

                self.global_step += 1
                action, log_prob, value = self.agent.select_action(state)
                action_item = int(action.item())

                next_obs, reward, terminated, truncated, step_info = self.env.step(action_item)
                done = terminated or truncated
                next_state = torch.from_numpy(next_obs).float()

                # 2. Update Affective Mind Engine (Computes curiosity reward & emotion vector)
                if self.affective_engine:
                    try:
                        _, shaped_reward = self.affective_engine.update_from_rl_step(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state,
                            done=done,
                            info=step_info,
                            step=self.global_step,
                        )
                    except Exception as me:
                        logger.debug("Mind update error: %s", me)
                        shaped_reward = reward
                else:
                    shaped_reward = reward

                # 3. Store in Trajectory Rollout Buffer
                self.buffer.add(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    reward=shaped_reward,
                    value=float(value.item()),
                    done=done,
                )

                episode_reward += reward
                episode_length += 1

                # Emit step telemetry
                self.event_bus.publish(
                    NoirEvent.create(
                        EventType.REWARD_RECEIVED,
                        experiment_id=self.experiment_id,
                        training_step=self.global_step,
                        reward=reward,
                        shaped_reward=shaped_reward,
                        action=action_item,
                    ),
                    asynchronous=True,
                )

                if done:
                    total_episodes += 1
                    self.current_epoch = total_episodes
                    self.metric_tracker.update("episode_reward", episode_reward)
                    self.metric_tracker.update("episode_length", episode_length)

                    self.event_bus.publish(
                        NoirEvent.create(
                            EventType.EPISODE_COMPLETED,
                            experiment_id=self.experiment_id,
                            training_step=self.global_step,
                            epoch=total_episodes,
                            episode_reward=episode_reward,
                            episode_length=episode_length,
                            reached_goal=step_info.get("reached_goal", False),
                        )
                    )

                    obs, info = self.env.reset()
                    state = torch.from_numpy(obs).float()
                    episode_reward = 0.0
                    episode_length = 0
                else:
                    state = next_state

                # Small yield for responsive GUI thread
                time.sleep(0.001)

            if self._stop_event.is_set():
                break

            # 4. GAE Calculation
            with torch.no_grad():
                _, _, last_val = self.agent.select_action(state)
                last_value_float = float(last_val.item())
            self.buffer.compute_returns_and_advantages(last_value=last_value_float, last_done=False)

            # 5. PPO Optimization Phase
            update_metrics = self.agent.train_step(
                buffer=self.buffer,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
            )

            # Combine telemetry metrics
            combined_metrics = {
                "loss": update_metrics["loss"],
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "entropy": update_metrics["entropy"],
                "avg_reward": self.metric_tracker.get_average("episode_reward"),
                "avg_length": self.metric_tracker.get_average("episode_length"),
            }
            self.latest_metrics = combined_metrics

            self._on_epoch_end(total_episodes, combined_metrics)
            self._on_step_end(self.global_step, combined_metrics)

            logger.info(
                "Episode %d | Steps: %d | Loss: %.4f | Policy: %.4f | Value: %.4f | Avg Reward: %.2f",
                total_episodes,
                self.global_step,
                update_metrics["loss"],
                update_metrics["policy_loss"],
                update_metrics["value_loss"],
                self.metric_tracker.get_average("episode_reward"),
            )
