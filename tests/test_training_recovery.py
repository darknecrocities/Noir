"""Unit tests for recovery manager."""

import shutil
import pytest
import torch

from noir.models.mlp import NoirMLP
from noir.storage.checkpoint_manager import CheckpointManager
from noir.storage.database import DatabaseManager
from noir.storage.experiment_repository import ExperimentRepository
from noir.storage.recovery import RecoveryManager


@pytest.fixture
def recovery_env(tmp_path):
    db_file = tmp_path / "recovery.db"
    ckpt_dir = tmp_path / "checkpoints"
    exp_dir = tmp_path / "experiments"

    db = DatabaseManager(db_file)
    ckpts = CheckpointManager(base_dir=ckpt_dir)
    repo = ExperimentRepository(db_manager=db, experiments_dir=exp_dir)
    rec = RecoveryManager(db_manager=db, checkpoint_manager=ckpts, experiments_dir=exp_dir)

    yield repo, ckpts, rec
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_recovery_detection(recovery_env):
    repo, ckpts, rec = recovery_env

    # Setup experiment and save a checkpoint
    repo.create_experiment("Crash Test Exp", config={"mode": "supervised"}, experiment_id="exp_crash")
    model = NoirMLP(input_dim=4, hidden_dims=[8], output_dim=2)

    ckpts.save_checkpoint(
        experiment_id="exp_crash",
        step=2500,
        epoch=12,
        model=model,
        metrics={"loss": 0.185},
    )

    # Check recovery discovery
    recovery_info = rec.check_for_recovery()
    assert recovery_info is not None
    assert recovery_info["experiment_id"] == "exp_crash"
    assert recovery_info["step"] == 2500
    assert recovery_info["epoch"] == 12
    assert recovery_info["metrics"]["loss"] == 0.185
