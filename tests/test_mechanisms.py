"""Unit tests for src/mechanisms.py."""

import pytest
import numpy as np
from src.environment import Environment1D, Interval
from src.mechanisms import (
    MedianMechanism,
    NaiveProjectedMedianMechanism,
    MeanMechanism,
    social_cost,
    max_cost,
    optimal_social_cost,
    optimal_max_cost,
    social_cost_approximation_ratio,
    max_cost_approximation_ratio,
)


def test_median_mechanism_odd():
    med = MedianMechanism()
    reports = [0.2, 0.9, 0.5]
    assert med(reports) == pytest.approx(0.5)


def test_median_mechanism_even():
    reports = [0.1, 0.4, 0.6, 0.9]
    med_left = MedianMechanism(tie_break="left")
    med_right = MedianMechanism(tie_break="right")
    med_mid = MedianMechanism(tie_break="midpoint")

    assert med_left(reports) == pytest.approx(0.4)
    assert med_right(reports) == pytest.approx(0.6)
    assert med_mid(reports) == pytest.approx(0.5)


def test_naive_projected_median():
    # Feasible: [0.0, 0.3] and [0.7, 1.0], blocked [0.3, 0.7]
    env = Environment1D(length=1.0, blocked_intervals=[Interval(0.3, 0.7)])
    proj_med = NaiveProjectedMedianMechanism(tie_break="left")

    # Reports whose unconstrained median is 0.4 (inside blocked zone)
    # 0.4 is closer to 0.3 than to 0.7, so it snaps to 0.3
    reports = [0.1, 0.4, 0.9]
    assert proj_med(reports, env) == pytest.approx(0.3)

    # Reports whose unconstrained median is 0.6 (inside blocked zone)
    # 0.6 is closer to 0.7, so it snaps to 0.7
    reports2 = [0.1, 0.6, 0.9]
    assert proj_med(reports2, env) == pytest.approx(0.7)

    # Reports whose unconstrained median is 0.2 (already feasible)
    reports3 = [0.1, 0.2, 0.9]
    assert proj_med(reports3, env) == pytest.approx(0.2)


def test_social_and_max_cost():
    agents = [1.0, 3.0, 8.0]
    facility = 4.0

    # SC: |4-1| + |4-3| + |4-8| = 3 + 1 + 4 = 8
    assert social_cost(facility, agents) == pytest.approx(8.0)

    # MC: max(|4-1|, |4-3|, |4-8|) = max(3, 1, 4) = 4
    assert max_cost(facility, agents) == pytest.approx(4.0)


def test_optimal_solvers_consistency_with_grid():
    # Test that optimal_social_cost and optimal_max_cost match exhaustive fine grid search
    env = Environment1D(
        length=10.0,
        blocked_intervals=[Interval(3.0, 5.0), Interval(7.0, 8.5)],
        num_grid_points=10001,
    )
    agents = [1.5, 4.2, 6.0, 9.1]

    # Grid brute-force
    grid_sc = [social_cost(g, agents) for g in env.feasible_grid]
    grid_best_sc = min(grid_sc)

    grid_mc = [max_cost(g, agents) for g in env.feasible_grid]
    grid_best_mc = min(grid_mc)

    # Analytical solvers
    opt_y_sc, opt_sc = optimal_social_cost(agents, env)
    opt_y_mc, opt_mc = optimal_max_cost(agents, env)

    assert env.is_feasible(opt_y_sc)
    assert env.is_feasible(opt_y_mc)
    assert opt_sc == pytest.approx(grid_best_sc, abs=1e-3)
    assert opt_mc == pytest.approx(grid_best_mc, abs=1e-3)


def test_approximation_ratios():
    env = Environment1D.unconstrained(length=1.0)
    agents = [0.0, 0.5, 1.0]

    # Median places at 0.5
    # Optimal SC on line for [0, 0.5, 1.0] is at median 0.5: SC = 0.5 + 0 + 0.5 = 1.0
    # Median achieves SC = 1.0 -> ratio = 1.0
    med = MedianMechanism()
    loc = med(agents, env)
    sc_ratio = social_cost_approximation_ratio(loc, agents, env)
    mc_ratio = max_cost_approximation_ratio(loc, agents, env)

    assert sc_ratio == pytest.approx(1.0)
    assert mc_ratio == pytest.approx(1.0)
