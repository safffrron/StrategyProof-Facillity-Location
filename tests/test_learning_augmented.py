"""Unit tests for src/learning_augmented.py."""

import numpy as np
import pytest

from src.environment import Environment1D, Interval
from src.mechanisms import optimal_max_cost, optimal_social_cost
from src.learning_augmented import (
    CenterMinMaxP,
    HullNearestFeasible,
    IntervalClamp,
    IntervalPhantoms,
    MedianRule,
    MinMaxP,
    PhantomMedian,
    PredictionOrMedian,
    max_cost_batch,
    optimal_max_cost_batch,
    optimal_social_cost_batch,
    snap_to_feasible,
)

rng = np.random.default_rng(0)


@pytest.fixture
def blocked_env():
    return Environment1D(length=1.0, num_grid_points=101, blocked_intervals=[Interval(0.3, 0.7)])


def test_snap_matches_environment_projection(blocked_env):
    ys = np.linspace(-0.2, 1.2, 301)
    expected = [blocked_env.project_nearest(y, tie_break="left") for y in ys]
    assert np.allclose(snap_to_feasible(ys, blocked_env), expected)


def test_snap_tie_breaks_left(blocked_env):
    assert snap_to_feasible(np.array([0.5]), blocked_env)[0] == pytest.approx(0.3)


def test_batch_optima_match_week1_solvers():
    env = Environment1D.random_blocked(length=1.0, num_blocks=3, blocked_density=0.4, seed=3)
    X = rng.uniform(0, 1, (200, 5))
    _, sc = optimal_social_cost_batch(X, env)
    _, mc = optimal_max_cost_batch(X, env)
    assert np.allclose(sc, [optimal_social_cost(x, env)[1] for x in X])
    assert np.allclose(mc, [optimal_max_cost(x, env)[1] for x in X])


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_minmaxp_is_phantom_median_with_n_minus_1_phantoms(n):
    X = rng.uniform(0, 1, (500, n))
    p = rng.uniform(-0.3, 1.3, 500)
    assert np.allclose(MinMaxP(snap=False).locate(X, None, p), PhantomMedian(n - 1, snap=False).locate(X, None, p))
    assert np.allclose(PhantomMedian(0, snap=False).locate(X, None, p), MedianRule(snap=False).locate(X))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_interval_clamp_is_a_generalised_median_and_interpolates(n):
    X = rng.uniform(0, 1, (500, n))
    lo = rng.uniform(-0.3, 1.3, 500)
    I = np.stack([lo, lo + rng.exponential(0.3, 500)], axis=1)
    p = rng.uniform(-0.3, 1.3, 500)
    clamp = IntervalClamp(snap=False)
    assert np.allclose(clamp.locate(X, None, I), IntervalPhantoms(snap=False).locate(X, None, I))
    assert np.allclose(clamp.locate(X, None, np.stack([p, p], axis=1)), MinMaxP(snap=False).locate(X, None, p))
    assert np.allclose(clamp.locate(X, None, (-1e9, 1e9)), MedianRule(snap=False).locate(X))


def test_minmaxp_examples():
    X = np.array([[2.0, 5.0, 9.0]])
    m = MinMaxP(snap=False)
    assert m.locate(X, None, 7.0)[0] == 7.0
    assert m.locate(X, None, 100.0)[0] == 9.0
    assert m.locate(X, None, 0.0)[0] == 2.0


def test_center_minmaxp_uses_interval_midpoint():
    X = np.array([[0.0, 1.0]])
    assert CenterMinMaxP(snap=False).locate(X, None, (0.2, 0.6))[0] == pytest.approx(0.4)


def test_minmaxp_star_tight_error_instance():
    # agents at 0 and 2, blocked zone (1, 1 + 2*eta - eps), prediction 1 + eta: ratio -> 1 + 2*eta
    for eta in (0.1, 0.3, 0.5, 0.9):
        F = (np.array([0.0, 1 + 2 * eta - 1e-9]), np.array([1.0, 3.0]))
        X = np.array([[0.0, 2.0]])
        f = MinMaxP().locate(X, F, 1 + eta)
        _, opt = optimal_max_cost_batch(X, F)
        assert max_cost_batch(f, X)[0] / opt[0] == pytest.approx(1 + 2 * eta, abs=1e-6)


def test_prediction_or_median_falls_back():
    X = np.array([[0.0, 0.0, 0.95]])
    assert PredictionOrMedian(snap=False).locate(X, None, 1.0)[0] == 0.0
    assert PredictionOrMedian(snap=False).locate(X, None, 0.5)[0] == 0.5


def test_hull_nearest_feasible_output_is_feasible(blocked_env):
    X = rng.uniform(0, 1, (300, 3))
    p = rng.uniform(0, 1, 300)
    y = HullNearestFeasible().locate(X, blocked_env, p)
    assert np.all((y <= 0.3 + 1e-9) | (y >= 0.7 - 1e-9))
