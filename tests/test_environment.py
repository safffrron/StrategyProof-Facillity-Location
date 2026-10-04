"""Unit tests for src/environment.py."""

import pytest
import numpy as np
from src.environment import (
    Interval,
    merge_intervals,
    compute_complement_intervals,
    Environment1D,
)


def test_interval_basics():
    iv = Interval(0.2, 0.6)
    assert iv.length == pytest.approx(0.4)
    assert iv.contains(0.2)
    assert iv.contains(0.4)
    assert iv.contains(0.6)
    assert not iv.contains(0.1)
    assert not iv.contains(0.7)

    assert iv.distance_to(0.1) == pytest.approx(0.1)
    assert iv.distance_to(0.8) == pytest.approx(0.2)
    assert iv.distance_to(0.3) == pytest.approx(0.0)

    assert iv.project(0.1) == pytest.approx(0.2)
    assert iv.project(0.8) == pytest.approx(0.6)
    assert iv.project(0.3) == pytest.approx(0.3)

    with pytest.raises(ValueError):
        Interval(0.8, 0.2)


def test_merge_intervals():
    intervals = [
        Interval(0.1, 0.3),
        Interval(0.25, 0.5),
        Interval(0.7, 0.9),
    ]
    merged = merge_intervals(intervals)
    assert len(merged) == 2
    assert merged[0] == Interval(0.1, 0.5)
    assert merged[1] == Interval(0.7, 0.9)


def test_compute_complement_intervals():
    blocked = [
        Interval(0.2, 0.4),
        Interval(0.6, 0.8),
    ]
    feasible = compute_complement_intervals(blocked, domain_start=0.0, domain_end=1.0)
    assert len(feasible) == 3
    assert feasible[0] == Interval(0.0, 0.2)
    assert feasible[1] == Interval(0.4, 0.6)
    assert feasible[2] == Interval(0.8, 1.0)


def test_environment_unconstrained():
    env = Environment1D.unconstrained(length=1.0, num_grid_points=101)
    assert env.blocked_density == pytest.approx(0.0)
    assert env.feasible_density == pytest.approx(1.0)
    assert len(env.feasible_intervals) == 1
    assert env.feasible_intervals[0] == Interval(0.0, 1.0)
    assert env.is_feasible(0.5)
    assert env.project_nearest(0.5) == pytest.approx(0.5)
    assert env.project_nearest(-0.2) == pytest.approx(0.0)
    assert env.project_nearest(1.5) == pytest.approx(1.0)


def test_environment_with_blocked_zones():
    blocked = [Interval(0.3, 0.7)]
    env = Environment1D(length=1.0, blocked_intervals=blocked, num_grid_points=1001)

    assert env.blocked_density == pytest.approx(0.4)
    assert len(env.feasible_intervals) == 2
    assert env.feasible_intervals[0] == Interval(0.0, 0.3)
    assert env.feasible_intervals[1] == Interval(0.7, 1.0)

    # Feasibility check
    assert env.is_feasible(0.2)
    assert not env.is_feasible(0.5)
    assert env.is_feasible(0.8)

    # Vectorized feasibility check
    pts = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(env.is_feasible(pts), np.array([True, False, True]))

    # Nearest projection
    assert env.project_nearest(0.2) == pytest.approx(0.2)
    assert env.project_nearest(0.35) == pytest.approx(0.3)
    assert env.project_nearest(0.65) == pytest.approx(0.7)

    # Equidistant tie-breaking at midpoint 0.5
    assert env.project_nearest(0.5, tie_break="left") == pytest.approx(0.3)
    assert env.project_nearest(0.5, tie_break="right") == pytest.approx(0.7)


def test_random_blocked_environment():
    env = Environment1D.random_blocked(
        length=10.0,
        num_blocks=3,
        blocked_density=0.4,
        seed=42,
    )
    assert env.length == pytest.approx(10.0)
    assert env.blocked_density == pytest.approx(0.4, abs=1e-3)
    assert len(env.blocked_intervals) <= 3
    assert len(env.feasible_intervals) >= 1
    # Check that grid has feasible points
    assert len(env.feasible_grid) > 0


def test_snap_to_grid():
    env = Environment1D.unconstrained(length=1.0, num_grid_points=11)
    # Grid is 0.0, 0.1, 0.2, ..., 1.0
    assert env.snap_to_grid(0.14) == pytest.approx(0.1)
    assert env.snap_to_grid(0.16) == pytest.approx(0.2)
