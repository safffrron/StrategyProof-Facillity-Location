"""Unit tests for src/fast_validator.py: known strategyproof rules pass, known manipulable rules fail."""

import numpy as np
import pytest

from src.environment import Environment1D, Interval
from src.fast_validator import check_strategyproofness, enumerate_profiles
from src.learning_augmented import (CenterMinMaxP, DictatorRule, LeftmostRule, MeanRule, MedianRule, MinMaxP,
                                    PhantomMedian, PredictionOrMedian, RandomizedPredictionSnap)

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
    (PhantomMedian(1), POINTS),
    (MinMaxP(), POINTS),
    (CenterMinMaxP(), INTERVALS),
    (RandomizedPredictionSnap(MinMaxP()), POINTS),
    (RandomizedPredictionSnap(MinMaxP(), use_center=True), INTERVALS),
])
def test_snapped_mechanisms_are_strategyproof(env, mech, preds):
    assert check_strategyproofness(mech, 3, GRID, env, predictions=preds).is_strategyproof


def test_the_briefs_example_rule_is_manipulable():
    # peaks (0, 0, 0.95), prediction 1: the agent at 0.95 reports 1 and moves the facility from 0 to 1
    report = check_strategyproofness(PredictionOrMedian(snap=False), 3, GRID, None, predictions=POINTS + [1.0])
    assert not report.is_strategyproof


def test_the_mean_is_manipulable():
    # residents at 0, 4 and 8: the one at 8 reports 14 and moves the mean from 4 to 6 (Section 1)
    report = check_strategyproofness(MeanRule(snap=False), 3, np.arange(0, 15, 2.0), None)
    assert not report.is_strategyproof
    assert report.counterexample is not None and report.counterexample.gain > 0
