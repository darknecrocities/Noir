"""End-to-End integration tests for Project NOIR."""

import time
import pytest
import torch

from noir.core.engine import NoirEngine
from noir.events.event_types import EventType


def test_supervised_e2e(tmp_path):
    # Set custom db and checkpoint dirs in temp
    engine = NoirEngine()
    engine.checkpoint_manager.base_dir = tmp_path / "checkpoints"
    engine.experiment_repo.experiments_dir = tmp_path / "experiments"
    engine.memory_manager.memory_dir = tmp_path / "memory"

    # Start supervised experiment
    exp_id = engine.start_supervised_experiment(
        name="Integration Supervised Run",
        num_epochs=2,
        batch_size=32,
    )

    # Let trainer run briefly
    time.sleep(1.5)

    assert engine.trainer is not None
    assert engine.trainer.global_step > 0
    assert engine.trainer.latest_metrics.get("train_loss") is not None

    # Pause and resume
    engine.pause_training()
    assert engine.lifecycle.is_paused()

    engine.resume_training()
    assert engine.lifecycle.is_running()

    # Save checkpoint
    ckpt_path = engine.save_checkpoint(tag="integration_test")
    assert ckpt_path.exists()
    assert (ckpt_path / "meta.json").exists()

    engine.stop_training()
    engine.shutdown()


def test_rl_ppo_e2e(tmp_path):
    engine = NoirEngine()
    engine.checkpoint_manager.base_dir = tmp_path / "checkpoints"
    engine.experiment_repo.experiments_dir = tmp_path / "experiments"
    engine.memory_manager.memory_dir = tmp_path / "memory"

    # Start RL experiment
    exp_id = engine.start_rl_experiment(
        name="Integration PPO Run",
        n_steps=32,
        max_episodes=5,
    )

    time.sleep(1.5)

    assert engine.trainer is not None
    assert engine.trainer.global_step > 0
    assert engine.affective_engine.current_state is not None

    engine.stop_training()
    engine.shutdown()


def test_mcp_tools_dispatch():
    engine = NoirEngine()
    from noir.mcp.tools import MCPToolRegistry

    registry = MCPToolRegistry(engine)
    tools = registry.list_tools()
    tool_names = [t["name"] for t in tools]

    assert "get_training_status" in tool_names
    assert "get_emotion_state" in tool_names
    assert "inspect_model" in tool_names
    assert "create_checkpoint" in tool_names

    status_res = registry.execute("get_training_status", {})
    assert status_res["success"] is True
    assert "state" in status_res["result"]

    engine.shutdown()
