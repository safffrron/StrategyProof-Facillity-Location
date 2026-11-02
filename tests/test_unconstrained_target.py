"""Theorem 6 of the report: predictions of the unconstrained optimum m."""

import numpy as np

from src.learning_augmented import (
    CenterMinMaxP, MinMaxP, max_cost_batch, optimal_max_cost_batch,
)


def _random_layout(rng):
    k = int(rng.integers(1, 4))
    cuts = np.sort(rng.uniform(0, 1, 2 * k))
    starts = np.concatenate([[-10.0], cuts[1::2]])
    ends = np.concatenate([cuts[0::2], [10.0]])
    return starts, ends


def _mid(X):
    return 0.5 * (X.min(axis=1) + X.max(axis=1))


def test_bound_holds_on_random_instances():
    rng = np.random.default_rng(1)
    for _ in range(40):
        F = _random_layout(rng)
        X = rng.uniform(0, 1, (5000, int(rng.integers(2, 7))))
        o, opt = optimal_max_cost_batch(X, F)
        m = _mid(X)
        pred = m + rng.normal(0, 0.3, X.shape[0])
        r = max_cost_batch(MinMaxP().locate(X, F, pred), X) / opt
        eta_m, eta_o = np.abs(pred - m) / opt, np.abs(pred - o) / opt
        bound = np.minimum.reduce([1 + 2 * eta_m, 1 + 2 * eta_o, np.full_like(r, 3.0)])
        assert np.all(r <= bound + 1e-9)


def test_exact_prediction_of_m_is_optimal():
    rng = np.random.default_rng(2)
    for _ in range(40):
        F = _random_layout(rng)
        X = rng.uniform(0, 1, (5000, 4))
        _, opt = optimal_max_cost_batch(X, F)
        r = max_cost_batch(MinMaxP().locate(X, F, _mid(X)), X) / opt
        assert np.allclose(r, 1.0)


def test_tight_instance():
    # agents 0 and 2 (m = o = 1, OPT = 1), zone (1, 1 + 2 eta - eps), prediction 1 + eta
    eta, eps = 0.3, 1e-4
    F = (np.array([-10.0, 1 + 2 * eta - eps]), np.array([1.0, 10.0]))
    X = np.array([[0.0, 2.0]])
    y = MinMaxP().locate(X, F, 1 + eta)
    assert np.isclose(max_cost_batch(y, X)[0], 1 + 2 * eta - eps)


def test_interval_corollary():
    rng = np.random.default_rng(3)
    for _ in range(40):
        F = _random_layout(rng)
        X = rng.uniform(0, 1, (5000, 5))
        _, opt = optimal_max_cost_batch(X, F)
        m = _mid(X)
        width = rng.uniform(0, 0.5, X.shape[0]) * opt
        lo = m - rng.uniform(0, 1, X.shape[0]) * width      # interval contains m
        pred = np.stack([lo, lo + width], axis=1)
        r = max_cost_batch(CenterMinMaxP().locate(X, F, pred), X) / opt
        assert np.all(r <= np.minimum(1 + width / opt, 3.0) + 1e-9)
