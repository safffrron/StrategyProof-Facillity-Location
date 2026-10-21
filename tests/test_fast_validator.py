"""Unit tests for src/fast_validator.py: known strategyproof rules pass, known manipulable rules fail."""

import numpy as np
import pytest

from src.environment import Environment1D, Interval
from src.fast_validator import check_strategyproofness, enumerate_profiles
from src.learning_augmented import (
    CenterMinMaxP,
    DictatorRule,
    HullNearestFeasible,
    IntervalClamp,
    LeftmostRule,
    MeanRule,
    MedianRule,
    MinMaxP,
    PredictionOrMedian,
    SnapTowardPrediction,
)

GRID = np.linspace(0, 1, 11)
POINTS = list(np.linspace(-0.2, 1.2, 8))
INTERVALS = [(a, b) for a in (0.0, 0.4, 0.8) for b in (0.1, 0.5, 1.0) if a <= b]


@pytest.fixture
def env():
    return Environment1D(length=1.0, num_grid_points=101, blocked_intervals=[Interval(0.3, 0.7)])


def test_enumerate_profiles_counts():
    assert len(enumerate_profiles(GRID, 3, anonymous=True)) == 286  # C(13, 3)
    assert len(enumerate_profiles(GRID, 3, anonymous=False)) == 11 ** 3


@pytest.mark.parametrize("mech, preds", [
    (MedianRule(), (None,)),
    (LeftmostRule(), (None,)),
    (DictatorRule(1), (None,)),
    (MinMaxP(), POINTS),
    (CenterMinMaxP(), INTERVALS),
    (IntervalClamp(), INTERVALS),
])
def test_snapped_generalised_medians_are_strategyproof(env, mech, preds):
    assert check_strategyproofness(mech, 3, GRID, env, predictions=preds).is_strategyproof


@pytest.mark.parametrize("mech, preds, use_env", [
    (MeanRule(snap=False), (None,), False),
    (PredictionOrMedian(snap=False), POINTS, False),
    (HullNearestFeasible(), POINTS, True),
    (SnapTowardPrediction(), POINTS, True),
])
def test_manipulable_rules_are_caught(env, mech, preds, use_env):
    report = check_strategyproofness(mech, 3, GRID, env if use_env else None, predictions=preds)
    assert not report.is_strategyproof
    assert report.counterexample is not None and report.counterexample.gain > 0


def test_snapping_needs_symmetric_costs(env):
    # plain median stays SP for asymmetric single-peaked costs; the snapped median does not
    assert check_strategyproofness(MedianRule(snap=False), 3, GRID, env, alpha=3.0).is_strategyproof
    assert not check_strategyproofness(MedianRule(), 3, GRID, env, alpha=3.0).is_strategyproof
