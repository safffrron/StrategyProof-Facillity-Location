"""Tests for randomised snapping (Definition 3 and Theorems 9-10 of the report)."""

import numpy as np
import pytest

from src.environment import Environment1D, Interval
from src.experiment_utils import Scenario, hill_climb
from src.learning_augmented import MinMaxP, RandomizedPredictionSnap, mean_preserving_rounding


@pytest.fixture
def env():
    return Environment1D(length=1.0, num_grid_points=11, blocked_intervals=[Interval(0.3, 0.7)])


def test_mean_preserving_rounding_keeps_mean_inside_zones(env):
    p = np.linspace(0.31, 0.69, 39)
    locs, probs = mean_preserving_rounding(p, env)
    assert np.allclose((locs * probs).sum(axis=1), p)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((locs == 0.3) | (locs == 0.7))


def test_allowed_predictions_are_not_randomised(env):
    p = np.array([0.1, 0.3, 0.7, 0.9])
    locs, probs = mean_preserving_rounding(p, env)
    assert np.allclose(locs[:, 0], p) and np.allclose(probs[:, 0], 1.0)


def test_figure_1_example():
    # agents at 0 and 2, zone (1, 1.67), prediction 1.35: probabilities 0.48 / 0.52, expected max cost 1.35
    F = (np.array([-10.0, 1.67]), np.array([1.0, 10.0]))
    locs, probs = RandomizedPredictionSnap(MinMaxP()).distribution(np.array([[0.0, 2.0]]), F, 1.35)
    assert np.allclose(locs[0], [1.0, 1.67])
    assert probs[0, 0] == pytest.approx(0.32 / 0.67)
    expected_mc = (probs[0] * np.maximum(np.abs(locs[0] - 0.0), np.abs(locs[0] - 2.0))).sum()
    assert expected_mc == pytest.approx(1.35)


def test_randomised_snapping_halves_the_error_amplification():
    det = hill_climb(Scenario(MinMaxP(), "mc", 2, 2, "error", eta=0.25), restarts=256, steps=400).best_ratio
    rnd = hill_climb(Scenario(RandomizedPredictionSnap(MinMaxP()), "mc", 2, 2, "error", eta=0.25),
                     restarts=256, steps=400).best_ratio
    assert det == pytest.approx(1.5, abs=0.01)
    assert rnd == pytest.approx(1.25, abs=0.01)
