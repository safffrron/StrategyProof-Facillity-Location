"""Facility location with double-peaked preferences (Filos-Ratsikas, Li, Zhang & Zhang, JAAMAS 2017), with
predictions.

Model: an agent at x has peaks x - b and x + b and pays cost(y, x) = c + dist(y, {x - b, x + b}), with b, c > 0.
Mechanisms return lotteries as arrays ``locs`` (P, K) and ``probs`` (P, K) for P sorted profiles.

* ``M1``: the median agent's two peaks with probability 1/2 each (truthful in expectation; Filos-Ratsikas et al.).
* ``Window(k)``: our learning-augmented family. Let g be the median of the reports plus k phantom copies of the
  prediction p. Output p if |g - p| <= b, otherwise g - b or g + b with probability 1/2 each. ``Window(0)`` uses the
  median (``PredMed``, for social cost); ``Window(n - 1)`` uses g = clamp(p, x_1, x_n) (``PredMinMax``, the
  double-peaked analogue of MinMaxP, for max cost). Every member is truthful in expectation (Theorem 13).
"""

from __future__ import annotations
from itertools import combinations_with_replacement
from typing import Optional
import numpy as np


def dp_cost(y, x, b: float, c: float):
    """Cost of an agent at x (broadcasting) for facility y."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return c + np.minimum(np.abs(y - (x - b)), np.abs(y - (x + b)))


def objective(Y, X, b: float, c: float, kind: str):
    """Social or max cost of facility locations Y (P, K) for sorted profiles X (P, n); returns (P, K)."""
    costs = dp_cost(Y[:, :, None], X[:, None, :], b, c)              # (P, K, n)
    return costs.sum(axis=2) if kind == "sc" else costs.max(axis=2)


def candidate_points(X, b: float):
    """Points among which every social- and max-cost optimum lies: peaks, valleys, and midpoints of peaks."""
    X = np.atleast_2d(X)
    P = np.concatenate([X - b, X, X + b], axis=1)
    mids = (P[:, :, None] + P[:, None, :]) / 2
    return np.concatenate([P, mids.reshape(len(X), -1)], axis=1)


def optimum(X, b: float, c: float, kind: str):
    """Exact optimal cost and one optimal location for each sorted profile (the objectives are piecewise linear
    with breakpoints among ``candidate_points``)."""
    cand = candidate_points(X, b)
    vals = objective(cand, np.atleast_2d(X), b, c, kind)
    j = vals.argmin(axis=1)
    rows = np.arange(len(cand))
    return vals[rows, j], cand[rows, j]


class Mechanism:
    name = "mechanism"
    deterministic = False

    def lottery(self, X: np.ndarray, p: Optional[float], b: float):
        """Locations (P, K) and probabilities (P, K) for sorted profiles X (P, n)."""
        raise NotImplementedError


class M1(Mechanism):
    name = "M1"

    def lottery(self, X, p, b):
        g = X[:, (X.shape[1] - 1) // 2]
        return np.stack([g - b, g + b], axis=1), np.full((len(X), 2), 0.5)


class Window(Mechanism):
    """Trust the prediction if it lies within b of the generalised median g; otherwise flip between g - b and g + b."""

    def __init__(self, k: int = 0, name: Optional[str] = None):
        self.k = k
        self.name = name or f"Window[{k}]"

    def lottery(self, X, p, b):
        P, n = X.shape
        Z = np.sort(np.concatenate([X, np.full((P, self.k), p, dtype=float)], axis=1), axis=1)
        g = Z[:, (n + self.k - 1) // 2]
        trust = np.abs(g - p) <= b + 1e-12
        locs = np.where(trust[:, None], np.full((P, 2), p, dtype=float), np.stack([g - b, g + b], axis=1))
        probs = np.where(trust[:, None], np.array([1.0, 0.0]), np.array([0.5, 0.5]))
        return locs, probs


def PredMed() -> Window:
    return Window(0, "PredMed")


def PredMinMax(n: int) -> Window:
    return Window(n - 1, "PredMinMax")


def expected_cost(locs, probs, x, b, c):
    """Expected cost of type x (P,) under lotteries (P, K)."""
    return (probs * dp_cost(locs, np.asarray(x, float)[:, None], b, c)).sum(axis=1)


def expected_objective(mech: Mechanism, X, p, b, c, kind):
    locs, probs = mech.lottery(X, p, b)
    return (probs * objective(locs, X, b, c, kind)).sum(axis=1)


def check_tie(mech: Mechanism, n: int, grid, predictions, b: float, c: float, tol: float = 1e-9):
    """Exhaustive truthfulness-in-expectation check over every sorted profile on ``grid``, every agent, every
    misreport on ``grid`` and every prediction. Returns (max gain, number of deviations checked, example)."""
    grid = np.asarray(sorted(set(float(g) for g in grid)))
    profiles = np.array(list(combinations_with_replacement(grid, n)))
    worst, example, checked = 0.0, None, 0
    for p in predictions:
        locs, probs = mech.lottery(profiles, p, b)
        for i in range(n):
            if i > 0:
                fresh = profiles[:, i] != profiles[:, i - 1]          # identical agents: check once
            else:
                fresh = np.ones(len(profiles), bool)
            base = expected_cost(locs, probs, profiles[:, i], b, c)
            for d in grid:
                Y = profiles.copy(); Y[:, i] = d
                Y.sort(axis=1)
                l2, p2 = mech.lottery(Y, p, b)
                gain = base - expected_cost(l2, p2, profiles[:, i], b, c)
                gain[~fresh] = -np.inf
                checked += int(fresh.sum())
                j = int(np.argmax(gain))
                if gain[j] > worst + tol:
                    worst = float(gain[j])
                    example = dict(prediction=p, profile=tuple(profiles[j]), agent=i, misreport=float(d), gain=worst)
    return worst, checked, example


def ratios(mech: Mechanism, X, predictions, b, c, kind):
    """Ratios (len(predictions), P) of the mechanism on sorted profiles X for each prediction."""
    opt, _ = optimum(X, b, c, kind)
    return np.array([expected_objective(mech, X, p, b, c, kind) / opt for p in predictions])
