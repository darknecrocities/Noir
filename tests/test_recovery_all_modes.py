"""Test recovery and resumption across all experiment modes."""

import time
import pytest
import torch

from noir.core.engine import NoirEngine
from noir.models.transformer import NoirTransformerLM


def test_open_web_llm_recovery(tmp_path):
    engine = NoirEngine()
    exp_id = engine.start_open_web_llm_experiment(
        name="Test Open Web Recovery",
        max_steps=5,
    )
    time.sleep(1.0)
    engine.save_checkpoint(tag="test_rec")
    engine.stop_training()

    # Test recovery of the open web LLM experiment
    rec_exp_id = engine.recover_from_previous_session(action="resume")
    assert rec_exp_id is not None
    assert isinstance(engine.model, NoirTransformerLM)

    engine.stop_training()
    engine.shutdown()
