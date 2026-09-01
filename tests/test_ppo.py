"""Unit tests for PPO Reinforcement Learning optimization."""

import pytest
import torch

from noir.models.actor_critic import ActorCriticNetwork
from noir.rl.agent import PPOAgent
from noir.rl.trajectory_memory import TrajectoryBuffer


def test_ppo_actor_critic_network():
    net = ActorCriticNetwork(state_dim=16, action_dim=4, hidden_dims=[32, 16])
    dummy_state = torch.randn(2, 16)

    logits, value = net(dummy_state)
    assert logits.shape == (2, 4)
    assert value.shape == (2,)

    action, log_prob, entropy, val = net.get_action_and_value(dummy_state)
    assert action.shape == (2,)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert val.shape == (2,)


def test_ppo_policy_update_step():
    net = ActorCriticNetwork(state_dim=16, action_dim=4, hidden_dims=[32, 16])
    agent = PPOAgent(network=net, learning_rate=0.001, device="cpu")
    buffer = TrajectoryBuffer(capacity=32)

    # Populate buffer with dummy transitions
    for _ in range(32):
        s = torch.randn(16)
        a, lp, val = agent.select_action(s)
        buffer.add(
            state=s,
            action=a,
            log_prob=lp,
            reward=1.0,
            value=float(val.item()),
            done=False,
        )

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=False)

    metrics = agent.train_step(buffer, n_epochs=2, batch_size=16)

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert isinstance(metrics["loss"], float)
