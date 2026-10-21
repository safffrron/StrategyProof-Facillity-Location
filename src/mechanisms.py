"""Facility Location Mechanisms, Cost Metrics, and Optimal Solvers.

This module provides implementations of classical and constrained facility location mechanisms,
exact optimal solvers for Social Cost (utilitarian) and Maximum Cost (egalitarian), and
functions for calculating empirical approximation ratios.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Optional, Tuple, List, Union
import numpy as np

from src.environment import Environment1D, Interval


# ============================================================================
# Cost Metrics and Optimal Benchmark Solvers
# ============================================================================

def social_cost(facility: float, agent_locations: Sequence[float]) -> float:
    """Calculate the utilitarian Social Cost: sum of distances from agents to facility.

    SC(y, x) = sum_{i=1}^n |y - x_i|
    """
    agents = np.asarray(agent_locations, dtype=float)
    return float(np.sum(np.abs(agents - facility)))


def max_cost(facility: float, agent_locations: Sequence[float]) -> float:
    """Calculate the egalitarian Maximum Cost: maximum distance from any agent to facility.

    MC(y, x) = max_{i=1}^n |y - x_i|
    """
    agents = np.asarray(agent_locations, dtype=float)
    if len(agents) == 0:
        return 0.0
    return float(np.max(np.abs(agents - facility)))


def optimal_social_cost(
    agent_locations: Sequence[float],
    env: Environment1D,
) -> Tuple[float, float]:
    """Find the optimal facility location y* in the feasible set F minimizing Social Cost.

    Since SC(y, x) is convex and piecewise linear with subgradient kinks at agent locations x_i,
    the constrained minimum over a union of closed intervals must occur at either:
    1. An agent location x_i that lies within a feasible interval.
    2. An endpoint of a feasible interval (start or end).

    Returns:
        Tuple[float, float]: (optimal_location, min_social_cost)
    """
    agents = np.asarray(agent_locations, dtype=float)
    if len(agents) == 0:
        return 0.0, 0.0

    candidate_points: List[float] = []

    # Add endpoints of all feasible intervals
    for iv in env.feasible_intervals:
        candidate_points.append(iv.start)
        candidate_points.append(iv.end)

    # Add agent points that fall within feasible intervals
    for a in agents:
        if env.is_feasible(a):
            candidate_points.append(float(a))

    # Evaluate SC at all candidates
    best_y = candidate_points[0]
    best_sc = social_cost(best_y, agents)

    for cand in candidate_points[1:]:
        cost = social_cost(cand, agents)
        if cost < best_sc - 1e-9:
            best_sc = cost
            best_y = cand

    return best_y, best_sc


def optimal_max_cost(
    agent_locations: Sequence[float],
    env: Environment1D,
) -> Tuple[float, float]:
    """Find the optimal facility location y* in the feasible set F minimizing Maximum Cost.

    For any profile x, MC(y, x) = max(y - min(x), max(x) - y) is strictly convex with an
    unconstrained minimum at the midpoint mid = (min(x) + max(x)) / 2.
    Over a union of closed intervals, the minimum on interval [a, b] is simply the projection
    of 'mid' onto [a, b]. The global optimum is the minimum across all feasible intervals.

    Returns:
        Tuple[float, float]: (optimal_location, min_max_cost)
    """
    agents = np.asarray(agent_locations, dtype=float)
    if len(agents) == 0:
        return 0.0, 0.0

    min_a = float(np.min(agents))
    max_a = float(np.max(agents))
    unconstrained_mid = (min_a + max_a) / 2.0

    best_y: Optional[float] = None
    best_mc = float("inf")

    for iv in env.feasible_intervals:
        # Project unconstrained midpoint onto interval
        cand = iv.project(unconstrained_mid)
        cost = max_cost(cand, agents)
        if cost < best_mc - 1e-9:
            best_mc = cost
            best_y = cand

    assert best_y is not None
    return best_y, best_mc


def social_cost_approximation_ratio(
    facility: float,
    agent_locations: Sequence[float],
    env: Environment1D,
) -> float:
    """Compute the approximation ratio for Social Cost: SC(y, x) / SC*(x, F)."""
    _, opt_sc = optimal_social_cost(agent_locations, env)
    actual_sc = social_cost(facility, agent_locations)
    if opt_sc < 1e-9:
        return 1.0 if actual_sc < 1e-9 else float("inf")
    return actual_sc / opt_sc


def max_cost_approximation_ratio(
    facility: float,
    agent_locations: Sequence[float],
    env: Environment1D,
) -> float:
    """Compute the approximation ratio for Maximum Cost: MC(y, x) / MC*(x, F)."""
    _, opt_mc = optimal_max_cost(agent_locations, env)
    actual_mc = max_cost(facility, agent_locations)
    if opt_mc < 1e-9:
        return 1.0 if actual_mc < 1e-9 else float("inf")
    return actual_mc / opt_mc


# ============================================================================
# Facility Location Mechanisms
# ============================================================================

class FacilityLocationMechanism(ABC):
    """Abstract base class for deterministic 1D facility location mechanisms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for the mechanism."""
        pass

    @abstractmethod
    def __call__(
        self,
        reports: Sequence[float],
        env: Optional[Environment1D] = None,
    ) -> float:
        """Compute the facility location given reported agent locations and environment."""
        pass


class MedianMechanism(FacilityLocationMechanism):
    """Standard unconstrained Median Mechanism (Moulin 1980).

    For sorted reports x_(1) <= ... <= x_(n):
    - Odd n: selects the unique median x_((n+1)/2).
    - Even n: selects the left-median x_(n/2) (1-indexed) by default to guarantee strategyproofness.
    """

    def __init__(self, tie_break: str = "left") -> None:
        if tie_break not in ("left", "right", "midpoint"):
            raise ValueError(f"Unknown tie_break rule: {tie_break}")
        self.tie_break = tie_break

    @property
    def name(self) -> str:
        return f"MedianMechanism(tie_break={self.tie_break})"

    def __call__(
        self,
        reports: Sequence[float],
        env: Optional[Environment1D] = None,
    ) -> float:
        arr = np.sort(np.asarray(reports, dtype=float))
        n = len(arr)
        if n == 0:
            return 0.0

        if n % 2 == 1:
            return float(arr[n // 2])
        else:
            if self.tie_break == "left":
                return float(arr[n // 2 - 1])
            elif self.tie_break == "right":
                return float(arr[n // 2])
            else:
                return float((arr[n // 2 - 1] + arr[n // 2]) / 2.0)


class NaiveProjectedMedianMechanism(FacilityLocationMechanism):
    """Naive Projected Median Mechanism.

    Computes the unconstrained median, then snaps to the nearest point in the feasible set F.
    The project proposal conjectured that the discontinuities introduced by snapping would be
    exploitable. They are not: with a fixed tie-break and symmetric distance costs, snapping any
    generalised median keeps it strategyproof (Walsh 2021; see docs/RESULTS.md, Lemma 1, and
    tests/test_validator.py). Snapping does break strategyproofness for asymmetric costs.
    """

    def __init__(self, tie_break: str = "left") -> None:
        self.median_rule = MedianMechanism(tie_break=tie_break)
        self.tie_break = tie_break

    @property
    def name(self) -> str:
        return f"NaiveProjectedMedian(tie_break={self.tie_break})"

    def __call__(
        self,
        reports: Sequence[float],
        env: Optional[Environment1D] = None,
    ) -> float:
        unconstrained_loc = self.median_rule(reports, env)
        if env is None or env.blocked_density < 1e-9:
            return unconstrained_loc
        return env.project_nearest(unconstrained_loc, tie_break=self.tie_break)


class MeanMechanism(FacilityLocationMechanism):
    """Mean / Average Mechanism (Non-Strategyproof benchmark).

    Computes the arithmetic mean of agent reports, optionally snapped to feasible set.
    Included as a benchmark to validate that the strategyproofness validator correctly
    identifies manipulable mechanisms with counterexamples.
    """

    def __init__(self, snap_to_feasible: bool = False) -> None:
        self.snap_to_feasible = snap_to_feasible

    @property
    def name(self) -> str:
        return f"MeanMechanism(snap={self.snap_to_feasible})"

    def __call__(
        self,
        reports: Sequence[float],
        env: Optional[Environment1D] = None,
    ) -> float:
        arr = np.asarray(reports, dtype=float)
        if len(arr) == 0:
            return 0.0
        avg = float(np.mean(arr))
        if self.snap_to_feasible and env is not None:
            return env.project_nearest(avg)
        return avg
