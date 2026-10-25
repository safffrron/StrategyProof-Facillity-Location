"""Proposition 27 and Theorem 28: social cost with predictions of the unconstrained optimum (the median)."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e15_social_cost_target import best_deterministic_consistency, tight_ratio  # noqa: E402
from src.learning_augmented import MedianRule, MinMaxP, lower_median  # noqa: E402


def test_exact_median_prediction_is_median_star():
    rng = np.random.default_rng(4)
    for _ in range(30):
        n = int(rng.choice([3, 5, 7]))
        X = rng.uniform(0, 1, (4000, n))
        a, w = rng.uniform(0, 1), rng.uniform(0, 0.5)
        F = (np.array([-10.0, a + w]), np.array([a, 10.0]))
        assert np.allclose(MinMaxP().locate(X, F, lower_median(X)), MedianRule().locate(X, F))


def test_tight_instance_matches_formula():
    for n in (3, 5, 7, 11):
        assert np.isclose(tight_ratio(n), (3 * n - 1) / (n + 1), atol=1e-4)


def test_milp_small_grids():
    for s in (0.5, 0.25):
        grid = np.round(np.arange(-1.5, 1.5 + 1e-9, s), 9)
        assert np.isclose(best_deterministic_consistency(grid), 2 - 3 * s / (2 + s), atol=1e-6)


def test_theorem28_instances():
    from src.learning_augmented import social_cost_batch
    for n in (3, 5, 9):
        k, eps = (n - 1) // 2, 1e-7
        X = np.array([[-1.0] * k + [0.0] + [eps] * k])
        sc = {y: float(social_cost_batch(np.array([y]), X)[0]) for y in (-1.0, 1.0)}
        assert np.isclose(sc[1.0] / sc[-1.0], (3 * n - 1) / (n + 1), atol=1e-5)
