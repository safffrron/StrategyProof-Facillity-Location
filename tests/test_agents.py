"""Unit tests for src/agents.py."""

import pytest
import numpy as np
from src.environment import Environment1D, Interval
from src.agents import AgentDistribution, generate_agents


@pytest.fixture
def env_constrained():
    return Environment1D(
        length=10.0,
        blocked_intervals=[Interval(2.0, 4.0), Interval(6.0, 8.0)],
        num_grid_points=1001,
    )


def test_generate_agents_uniform(env_constrained):
    agents = generate_agents(n_agents=50, env=env_constrained, distribution=AgentDistribution.UNIFORM, seed=123)
    assert len(agents) == 50
    assert np.all(agents >= 0.0)
    assert np.all(agents <= 10.0)


def test_generate_agents_uniform_feasible(env_constrained):
    agents = generate_agents(
        n_agents=100,
        env=env_constrained,
        distribution=AgentDistribution.UNIFORM_FEASIBLE,
        seed=123,
    )
    assert len(agents) == 100
    for a in agents:
        assert env_constrained.is_feasible(a)


def test_generate_agents_gaussian(env_constrained):
    agents = generate_agents(
        n_agents=100,
        env=env_constrained,
        distribution=AgentDistribution.GAUSSIAN,
        params={"mu": 5.0, "sigma": 1.0},
        seed=42,
    )
    assert len(agents) == 100
    assert np.all(agents >= 0.0)
    assert np.all(agents <= 10.0)
    assert 4.0 < np.mean(agents) < 6.0


def test_generate_agents_bimodal(env_constrained):
    agents = generate_agents(
        n_agents=200,
        env=env_constrained,
        distribution=AgentDistribution.BIMODAL,
        params={"mu1": 1.0, "mu2": 9.0},
        seed=42,
    )
    assert len(agents) == 200
    # Should have samples near 1.0 and samples near 9.0
    assert np.any(agents < 3.0)
    assert np.any(agents > 7.0)


def test_generate_agents_grid_snapping(env_constrained):
    agents = generate_agents(
        n_agents=20,
        env=env_constrained,
        distribution=AgentDistribution.UNIFORM,
        snap_to_grid=True,
        seed=77,
    )
    # Check that each agent is exactly on the grid
    for a in agents:
        dist_to_grid = np.min(np.abs(env_constrained.grid - a))
        assert dist_to_grid < 1e-9


def test_generate_agents_reproducibility(env_constrained):
    a1 = generate_agents(n_agents=10, env=env_constrained, distribution="uniform", seed=99)
    a2 = generate_agents(n_agents=10, env=env_constrained, distribution="uniform", seed=99)
    np.testing.assert_array_almost_equal(a1, a2)
