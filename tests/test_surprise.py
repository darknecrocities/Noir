"""Unit tests for Surprise Engine."""

import numpy as np
import torch
from noir.mind.surprise import SurpriseDetector, calculate_event_surprise, calculate_surprise


def test_calculate_surprise():
    actual = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0])

    s = calculate_surprise(actual, pred)
    assert abs(s - (1.0 / 3.0)) < 1e-4


def test_calculate_event_surprise():
    # Rare event -> high surprise
    s_rare = calculate_event_surprise(0.01)
    # Common event -> low surprise
    s_common = calculate_event_surprise(0.99)

    assert s_rare > s_common
    assert 0.0 <= s_rare <= 1.0
    assert 0.0 <= s_common <= 1.0


def test_surprise_detector_trigger():
    detector = SurpriseDetector(threshold=0.75)

    # Establish baseline with low error
    for _ in range(30):
        detector.update(0.05)

    # Sudden high shock
    is_surprised, raw, norm = detector.update(5.0)
    assert is_surprised is True
    assert norm >= 0.75
