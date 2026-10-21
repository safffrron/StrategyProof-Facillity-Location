"""Tests for the research extensions: finite-n social-cost frontier (Theorem 9), randomized snapping
(Theorem 13), favourite-feasible-point mechanisms for asymmetric costs, and negative examples."""

import numpy as np
import pytest

from src.environment import Environment1D, Interval
from src.experiment_utils import Scenario, hill_climb
from src.fast_validator import check_strategyproofness
from src.learning_augmented import (
    LRM,
    MedianRule,
    MinMaxP,
    PhantomMedian,
    RandomizedPredictionSnap,
    RoundedOutput,
    SpikeMechanism,
    TopsGeneralizedMedian,
    feasible_tops,
    mean_preserving_rounding,
    optimal_social_cost_batch,
    social_cost_batch,
    snap_to_feasible,
)

GRID = np.linspace(0, 1, 11)
PREDS = list(np.round(np.linspace(-0.2, 1.2, 15), 4))


@pytest.fixture
def env():
    return Environment1D(length=1.0, num_grid_points=11, blocked_intervals=[Interval(0.3, 0.7)])


@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_theorem9_tight_instances(n):
    F = (np.array([-50.0, 1.0]), np.array([0.0, 50.0]))  # blocked zone (0, 1)
    for k in range(0, n, 2):
        X = np.array([[0.5 - 1e-6] * ((n + k + 1) // 2) + [1.0] * ((n - k - 1) // 2)])
        y = PhantomMedian(k).locate(X, F, 1.0)
        _, opt = optimal_social_cost_batch(X, F)
        assert social_cost_batch(y, X)[0] / opt[0] == pytest.approx((3 * n - 1 - k) / (n + 1 + k), rel=1e-4)
        X = np.array([[0.5 - 1e-6] * ((n - k + 1) // 2) + [1.0] * ((n + k - 1) // 2)])
        y = PhantomMedian(k).locate(X, F, -40.0)
        _, opt = optimal_social_cost_batch(X, F)
        assert social_cost_batch(y, X)[0] / opt[0] == pytest.approx((3 * n + k - 1) / (n - k + 1), rel=1e-4)


def test_mean_preserving_rounding_keeps_mean_inside_zones(env):
    p = np.linspace(0.31, 0.69, 39)
    locs, probs = mean_preserving_rounding(p, env)
    assert np.allclose((locs * probs).sum(axis=1), p)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_tops_mechanisms_coincide_with_snapped_ones_for_symmetric_costs(env):
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (2000, 5))
    p = rng.uniform(-0.3, 1.3, 2000)
    assert np.allclose(TopsGeneralizedMedian(0).locate(X, env), MedianRule().locate(X, env))
    assert np.allclose(TopsGeneralizedMedian(4).locate(X, env, p), MinMaxP().locate(X, env, p))


@pytest.mark.parametrize("mech", [RandomizedPredictionSnap(MinMaxP()), RandomizedPredictionSnap(PhantomMedian(2)),
                                  SpikeMechanism()])
def test_randomized_mechanisms_are_truthful_in_expectation(env, mech):
    preds = PREDS if mech.prediction_type else [None]
    assert check_strategyproofness(mech, 3, GRID, env, predictions=preds).is_strategyproof


@pytest.mark.parametrize("mech, preds", [(RoundedOutput(MinMaxP(snap=False)), PREDS), (LRM(), [None]),
                                         (LRM(tops=True), [None])])
def test_naive_randomized_adaptations_are_manipulable(env, mech, preds):
    assert not check_strategyproofness(mech, 3, GRID, env, predictions=preds).is_strategyproof


@pytest.mark.parametrize("alpha", [0.25, 3.0])
def test_tops_mechanisms_are_strategyproof_for_asymmetric_costs(env, alpha):
    assert check_strategyproofness(TopsGeneralizedMedian(2, alpha=alpha), 3, GRID, env, predictions=PREDS,
                                   alpha=alpha).is_strategyproof
    assert not check_strategyproofness(MedianRule(), 3, GRID, env, alpha=alpha).is_strategyproof


def test_voting_for_favourite_feasible_point_is_strategyproof_with_private_asymmetry(env):
    for a in (0.25, 3.0):
        rep = check_strategyproofness(TopsGeneralizedMedian(2, alpha=1.0), 3, GRID, env, predictions=PREDS, alpha=a,
                                      truthful_action=lambda X, a=a: feasible_tops(X, env, a))
        assert rep.is_strategyproof


def test_randomized_snapping_halves_the_error_amplification():
    det = hill_climb(Scenario(MinMaxP(), "mc", 2, 2, "error", eta=0.25), restarts=256, steps=400).best_ratio
    rnd = hill_climb(Scenario(RandomizedPredictionSnap(MinMaxP()), "mc", 2, 2, "error", eta=0.25),
                     restarts=256, steps=400).best_ratio
    assert det == pytest.approx(1.5, abs=0.01)
    assert rnd == pytest.approx(1.25, abs=0.01)
