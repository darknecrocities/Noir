"""Unit tests for Memory Manager."""

import shutil
import pytest
from noir.events.event import NoirEvent
from noir.events.event_types import EventType
from noir.memory.memory_manager import MemoryManager


@pytest.fixture
def temp_mem_dir(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    yield mem_dir
    shutil.rmtree(mem_dir, ignore_errors=True)


def test_memory_manager_persistence(temp_mem_dir):
    manager = MemoryManager(experiment_id="exp_mem", memory_dir=temp_mem_dir)

    manager.episodic.record_experience(
        event_type="DISCOVERY",
        description="Found optimal policy trajectory",
        state_summary={"reward": 9.8},
        importance=0.95,
        step=500,
    )
    manager.semantic.store_concept("learning_rate_optimal", 0.0003, confidence=0.9)

    # Save to disk
    manager.save_to_disk()

    # Create new manager and load
    new_manager = MemoryManager(experiment_id="exp_mem", memory_dir=temp_mem_dir)
    new_manager.load_from_disk()

    salient = new_manager.episodic.get_salient_experiences(min_importance=0.8)
    assert len(salient) == 1
    assert salient[0]["description"] == "Found optimal policy trajectory"

    concept = new_manager.semantic.get_concept("learning_rate_optimal")
    assert concept is not None
    assert concept["value"] == 0.0003
