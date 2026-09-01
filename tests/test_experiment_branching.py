"""Unit tests for experiment creation and branching."""

import shutil
import pytest
from noir.storage.database import DatabaseManager
from noir.storage.experiment_repository import ExperimentRepository


@pytest.fixture
def temp_exp_env(tmp_path):
    db_file = tmp_path / "test_noir.db"
    exp_dir = tmp_path / "experiments"
    db_man = DatabaseManager(db_file)
    repo = ExperimentRepository(db_manager=db_man, experiments_dir=exp_dir)
    yield repo
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_create_and_branch_experiment(temp_exp_env):
    repo = temp_exp_env

    # 1. Create Parent Experiment
    parent_config = {
        "training": {"learning_rate": 0.001, "batch_size": 64},
        "emotion": {"curiosity_weight": 0.2},
    }
    parent = repo.create_experiment("Parent Run", config=parent_config, experiment_id="exp_parent")
    assert parent.id == "exp_parent"

    # 2. Branch Experiment with modified LR
    overrides = {"training": {"learning_rate": 0.0001}}
    branch_id = repo.branch_experiment(
        parent_id="exp_parent",
        new_name="Branched Run",
        config_overrides=overrides,
    )

    branched = repo.get_experiment(branch_id)
    assert branched is not None
    assert branched["parent_id"] == "exp_parent"
    assert branched["config"]["training"]["learning_rate"] == 0.0001
    assert branched["config"]["training"]["batch_size"] == 64  # Preserved from parent

    # Verify parent config is unaltered
    parent_loaded = repo.get_experiment("exp_parent")
    assert parent_loaded["config"]["training"]["learning_rate"] == 0.001


def test_compare_experiments(temp_exp_env):
    repo = temp_exp_env

    repo.create_experiment("Exp 1", config={"lr": 0.01, "batch": 32}, experiment_id="e1")
    repo.create_experiment("Exp 2", config={"lr": 0.001, "batch": 32}, experiment_id="e2")

    comparison = repo.compare_experiments("e1", "e2")
    assert "lr" in comparison["config_differences"]
    assert comparison["config_differences"]["lr"]["exp1"] == 0.01
    assert comparison["config_differences"]["lr"]["exp2"] == 0.001
