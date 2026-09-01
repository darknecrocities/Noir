"""Unit tests for atomic CheckpointManager."""

import shutil
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from noir.models.mlp import NoirMLP
from noir.storage.checkpoint_manager import CheckpointManager


@pytest.fixture
def temp_ckpt_dir(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_atomic_checkpoint_save_and_load(temp_ckpt_dir):
    manager = CheckpointManager(base_dir=temp_ckpt_dir, retention=5)
    model = NoirMLP(input_dim=8, hidden_dims=[16], output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    emotion_state = {"confidence": 0.85, "frustration": 0.12, "curiosity": 0.90}
    metrics = {"loss": 0.245, "accuracy": 0.92}

    # Save checkpoint
    saved_path = manager.save_checkpoint(
        experiment_id="test_exp",
        step=150,
        epoch=3,
        model=model,
        optimizer=optimizer,
        emotion_state=emotion_state,
        metrics=metrics,
        is_best=True,
    )

    assert saved_path.exists()
    assert (saved_path / "meta.json").exists()

    # Create fresh model instance to test restoration
    new_model = NoirMLP(input_dim=8, hidden_dims=[16], output_dim=2)
    new_opt = torch.optim.Adam(new_model.parameters(), lr=0.001)

    meta = manager.load_checkpoint(
        checkpoint_path=saved_path,
        model=new_model,
        optimizer=new_opt,
    )

    assert meta["step"] == 150
    assert meta["epoch"] == 3
    assert meta["emotion_state"]["confidence"] == 0.85

    # Verify model weights match exactly
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
        assert torch.equal(p1, p2)


def test_checkpoint_latest_and_best_pointer(temp_ckpt_dir):
    manager = CheckpointManager(base_dir=temp_ckpt_dir, retention=5)
    model = NoirMLP(input_dim=4, hidden_dims=[8], output_dim=2)

    manager.save_checkpoint(
        experiment_id="exp_ptr",
        step=50,
        epoch=1,
        model=model,
        is_best=False,
    )
    manager.save_checkpoint(
        experiment_id="exp_ptr",
        step=100,
        epoch=2,
        model=model,
        is_best=True,
    )

    latest = manager.get_latest_checkpoint("exp_ptr")
    best = manager.get_best_checkpoint("exp_ptr")

    assert latest is not None and latest.exists()
    assert best is not None and best.exists()
