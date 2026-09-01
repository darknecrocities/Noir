"""Unit tests for Affective / Emotion Engine."""

import numpy as np
import pytest
import torch

from noir.mind.affective_engine import AffectiveEngine
from noir.mind.emotion_state import EmotionState


def test_emotion_state_bounds():
    state = EmotionState(
        confidence=1.5,
        frustration=-0.5,
        anticipation=0.7,
        satisfaction=0.9,
        uncertainty=0.1,
        curiosity=0.8,
        caution=0.4,
        persistence=0.9,
    )
    vec = state.to_vector()
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.5)  # Raw unclipped constructor

    bounded = EmotionState.from_vector(vec)
    bounded_vec = bounded.to_vector()
    assert np.all(bounded_vec >= 0.0)
    assert np.all(bounded_vec <= 1.0)
    assert bounded.confidence == 1.0
    assert bounded.frustration == 0.0


def test_affective_engine_supervised_step():
    engine = AffectiveEngine(experiment_id="test_exp")

    # High accuracy, low entropy step
    probs = np.array([[0.95, 0.02, 0.02, 0.01]])
    state = engine.update_from_supervised_step(
        loss=0.08,
        accuracy=0.98,
        probabilities=probs,
        step=10,
    )

    assert 0.0 <= state.confidence <= 1.0
    assert 0.0 <= state.frustration <= 1.0
    assert 0.0 <= state.uncertainty <= 1.0
    assert state.confidence > 0.4
    assert state.uncertainty < 0.4


def test_affective_engine_rl_step():
    engine = AffectiveEngine(experiment_id="test_exp")

    s = torch.zeros(16)
    a = torch.tensor(1)
    ns = torch.ones(16) * 0.1

    state, shaped_reward = engine.update_from_rl_step(
        state=s,
        action=a,
        reward=10.0,
        next_state=ns,
        done=True,
        info={"reached_goal": True, "distance": 0.0},
        step=1,
    )

    assert state.satisfaction > 0.5
    assert shaped_reward >= 10.0
