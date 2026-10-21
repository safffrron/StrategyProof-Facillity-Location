"""Unit tests for src/validator.py."""

import pytest
import numpy as np
from src.environment import Environment1D, Interval
from src.mechanisms import (
    MedianMechanism,
    MeanMechanism,
    NaiveProjectedMedianMechanism,
)
from src.validator import StrategyproofnessValidator, Deviation


def test_validator_detects_unconstrained_median_is_strategyproof():
    # Standard Median mechanism on unconstrained line is provably strategyproof (Moulin 1980)
    env = Environment1D.unconstrained(length=1.0, num_grid_points=101)
    validator = StrategyproofnessValidator(env=env, epsilon=1e-6)
    med = MedianMechanism(tie_break="left")

    # Verify on synthetic uniform distribution
    res = validator.verify_distribution(
        mechanism=med,
        n_agents=5,
        num_profiles=30,
        distribution="uniform",
        seed=123,
    )
    assert res.is_strategyproof
    assert res.num_violations_found == 0
    assert res.max_gain == pytest.approx(0.0)
    assert res.worst_violation is None


def test_validator_catches_mean_mechanism_manipulation():
    # The arithmetic mean mechanism is famously NOT strategyproof
    env = Environment1D.unconstrained(length=1.0, num_grid_points=101)
    validator = StrategyproofnessValidator(env=env, epsilon=1e-6)
    mean_mech = MeanMechanism()

    res = validator.verify_distribution(
        mechanism=mean_mech,
        n_agents=3,
        num_profiles=20,
        distribution="uniform",
        seed=42,
    )
    assert not res.is_strategyproof
    assert res.num_violations_found > 0
    assert res.max_gain > 1e-4
    assert res.worst_violation is not None
    assert isinstance(res.worst_violation, Deviation)
    assert res.worst_violation.gain > 0


def test_validator_exhaustive_grid():
    # Exhaustive verification on a 15-point grid for n=2 agents
    env = Environment1D.unconstrained(length=1.0, num_grid_points=15)
    validator = StrategyproofnessValidator(env=env, epsilon=1e-6)
    med = MedianMechanism(tie_break="left")

    res = validator.verify_exhaustive_grid(mechanism=med, n_agents=2, grid_coarseness=15)
    assert res.is_strategyproof
    assert res.num_profiles_tested == 15 * 15
    assert res.num_violations_found == 0


def test_validator_check_single_profile():
    env = Environment1D.unconstrained(length=1.0, num_grid_points=51)
    validator = StrategyproofnessValidator(env=env, epsilon=1e-6)
    med = MedianMechanism(tie_break="left")

    prof = [0.1, 0.4, 0.9]
    res = validator.check_profile(med, prof)
    assert res.is_strategyproof
    assert len(res.violations) == 0
    assert res.max_gain == pytest.approx(0.0)


def test_validator_confirms_projected_median_is_strategyproof_with_blocked_zones():
    # The proposal conjectured that snapping the median to the nearest feasible point is
    # manipulable; exhaustive checking with this validator shows it is not (fixed tie-break).
    env = Environment1D(length=1.0, num_grid_points=11, blocked_intervals=[Interval(0.3, 0.7)])
    validator = StrategyproofnessValidator(env=env, epsilon=1e-9)
    for tie_break in ("left", "right"):
        res = validator.verify_exhaustive_grid(
            mechanism=NaiveProjectedMedianMechanism(tie_break=tie_break), n_agents=3, grid_coarseness=11
        )
        assert res.is_strategyproof
        assert res.num_violations_found == 0
