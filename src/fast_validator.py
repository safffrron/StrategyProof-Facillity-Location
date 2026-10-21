"""Vectorised exhaustive strategyproofness checker.

Complements :mod:`src.validator` (which checks one profile at a time) with a checker that, for a
discretised line, enumerates *every* profile, *every* agent, *every* misreport and *every*
prediction, evaluating whole batches of profiles per mechanism call.

For anonymous mechanisms it is enough to enumerate sorted profiles (multisets): any other profile
is a permutation of one of them, and permuting agents permutes the deviations one-to-one.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product
from typing import Callable, Iterable, List, Optional, Sequence
import time
import numpy as np

from src.learning_augmented import BatchMechanism, Feasible, agent_cost, expected_agent_cost


@dataclass
class Counterexample:
    """A profitable unilateral deviation."""
    profile: tuple
    agent: int
    true_peak: float
    misreport: float
    prediction: object
    facility_truthful: float
    facility_misreport: float
    cost_truthful: float
    cost_misreport: float
    gain: float

    def describe(self) -> str:
        pred = "" if self.prediction is None else f", prediction={_fmt(self.prediction)}"
        return (
            f"peaks={_fmt(self.profile)}{pred}: agent at {self.true_peak:.3g} reports "
            f"{self.misreport:.3g}; facility {self.facility_truthful:.3g} -> {self.facility_misreport:.3g}, "
            f"cost {self.cost_truthful:.3g} -> {self.cost_misreport:.3g}"
        )


@dataclass
class SPReport:
    """Summary of an exhaustive strategyproofness check."""
    mechanism: str
    setting: str
    n_agents: int
    grid_size: int
    n_predictions: int
    n_profiles: int
    n_deviations: int
    n_violating: int
    max_gain: float
    seconds: float
    alpha: float = 1.0
    counterexample: Optional[Counterexample] = None

    @property
    def is_strategyproof(self) -> bool:
        return self.n_violating == 0

    @property
    def violating_fraction(self) -> float:
        total = self.n_profiles * self.n_predictions
        return self.n_violating / total if total else 0.0

    def as_row(self) -> dict:
        return {
            "mechanism": self.mechanism,
            "setting": self.setting,
            "n": self.n_agents,
            "grid": self.grid_size,
            "predictions": self.n_predictions,
            "profiles": self.n_profiles,
            "deviations_checked": self.n_deviations,
            "cost_alpha": self.alpha,
            "strategyproof": self.is_strategyproof,
            "violating_fraction": round(self.violating_fraction, 6),
            "max_gain": round(self.max_gain, 6),
            "counterexample": self.counterexample.describe() if self.counterexample else "",
        }


def _fmt(v) -> str:
    if isinstance(v, (tuple, list, np.ndarray)):
        return "(" + ", ".join(f"{float(x):.3g}" for x in v) + ")"
    return f"{float(v):.3g}"


def enumerate_profiles(grid: Sequence[float], n_agents: int, anonymous: bool = True) -> np.ndarray:
    """All profiles over ``grid`` (sorted multisets if ``anonymous``), shape ``(P, n)``."""
    grid = np.asarray(grid, dtype=float)
    idx_iter: Iterable = (
        combinations_with_replacement(range(len(grid)), n_agents) if anonymous
        else product(range(len(grid)), repeat=n_agents)
    )
    idx = np.array(list(idx_iter), dtype=int)
    return grid[idx]


def check_strategyproofness(
    mechanism: BatchMechanism,
    n_agents: int,
    grid: Sequence[float],
    feasible: Feasible = None,
    predictions: Sequence = (None,),
    misreports: Optional[Sequence[float]] = None,
    alpha: float = 1.0,
    tol: float = 1e-9,
    setting: str = "",
    truthful_action: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> SPReport:
    """Exhaustively search for a profitable unilateral misreport.

    Randomized mechanisms (``mechanism.randomized``) are checked for truthfulness in expectation:
    expected costs are compared.

    Args:
        mechanism: Mechanism to check.
        n_agents: Number of agents.
        grid: Possible true peaks (every profile over this grid is checked).
        feasible: Feasible set (``None`` = unconstrained line).
        predictions: Predictions to check (each is fixed before agents report).
        misreports: Possible misreports; defaults to ``grid`` plus two points outside it.
        alpha: Cost asymmetry (1 = symmetric distance), see :func:`agent_cost`.
        tol: A deviation counts only if it lowers the agent's cost by more than ``tol``.
        setting: Label for reports.
        truthful_action: Maps true peaks to truthful reports (default: report the peak). For
            mechanisms where agents vote for their favourite feasible point, pass that map.
    """
    start = time.time()
    grid = np.asarray(grid, dtype=float)
    if misreports is None:
        span = grid.max() - grid.min()
        misreports = np.concatenate([grid, [grid.min() - 0.5 * span, grid.max() + 0.5 * span]])
    misreports = np.asarray(misreports, dtype=float)

    X = enumerate_profiles(grid, n_agents, anonymous=mechanism.anonymous)
    R = X.copy() if truthful_action is None else truthful_action(X)
    P = X.shape[0]
    n_violating = 0
    max_gain = 0.0
    best: Optional[Counterexample] = None

    for pred in predictions:
        y_true = mechanism.locate(R, feasible, pred)
        violating = np.zeros(P, dtype=bool)
        for i in range(n_agents):
            peaks = X[:, i]
            cost_true = expected_agent_cost(mechanism, R, peaks, feasible, pred, alpha)
            X_dev = R.copy()
            for r in misreports:
                X_dev[:, i] = r
                y_dev = mechanism.locate(X_dev, feasible, pred)
                gain = cost_true - expected_agent_cost(mechanism, X_dev, peaks, feasible, pred, alpha)
                hit = gain > tol
                if hit.any():
                    violating |= hit
                    j = int(np.argmax(np.where(hit, gain, -np.inf)))
                    if gain[j] > max_gain:
                        max_gain = float(gain[j])
                        best = Counterexample(
                            profile=tuple(np.round(X[j], 6).tolist()),
                            agent=i,
                            true_peak=float(peaks[j]),
                            misreport=float(r),
                            prediction=pred,
                            facility_truthful=float(y_true[j]),
                            facility_misreport=float(y_dev[j]),
                            cost_truthful=float(cost_true[j]),
                            cost_misreport=float(cost_true[j] - gain[j]),
                            gain=float(gain[j]),
                        )
        n_violating += int(violating.sum())

    return SPReport(
        mechanism=mechanism.name,
        setting=setting,
        n_agents=n_agents,
        grid_size=len(grid),
        n_predictions=len(predictions),
        n_profiles=P,
        n_deviations=P * n_agents * len(misreports) * len(predictions),
        n_violating=n_violating,
        max_gain=max_gain,
        seconds=time.time() - start,
        alpha=alpha,
        counterexample=best,
    )
