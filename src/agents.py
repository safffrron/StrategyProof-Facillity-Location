"""Synthetic Agent Generation with Controlled Distributions.

This module models agent populations drawing their true peak locations on a 1D line [0, L]
from various parametric probability distributions (Uniform, Gaussian, Bimodal, Clustered, Polarized).
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any, Union, Sequence
import numpy as np
from scipy import stats

from src.environment import Environment1D


class AgentDistribution(str, Enum):
    """Supported distribution types for generating agent true peaks."""
    UNIFORM = "uniform"
    UNIFORM_FEASIBLE = "uniform_feasible"
    GAUSSIAN = "gaussian"
    BIMODAL = "bimodal"
    CLUSTERED = "clustered"
    POLARIZED = "polarized"


def generate_agents(
    n_agents: int,
    env: Environment1D,
    distribution: Union[AgentDistribution, str] = AgentDistribution.UNIFORM,
    params: Optional[Dict[str, Any]] = None,
    snap_to_grid: bool = False,
    feasible_only: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic agent true peak locations on a 1D environment.

    Args:
        n_agents (int): Number of agents to generate (must be >= 1).
        env (Environment1D): The 1D environment domain [0, L].
        distribution (AgentDistribution | str): Name of the distribution.
        params (Optional[Dict[str, Any]]): Distribution hyperparameters.
        snap_to_grid (bool): If True, snap generated continuous peaks to env.grid.
        feasible_only (bool): If True, reject or project samples outside feasible regions.
        seed (Optional[int]): Random seed for reproducible generation.

    Returns:
        np.ndarray: 1D array of shape (n_agents,) representing agent locations.
    """
    if n_agents <= 0:
        raise ValueError(f"n_agents must be positive, got {n_agents}")

    dist_type = AgentDistribution(distribution)
    rng = np.random.default_rng(seed)
    params = params or {}
    L = env.length

    if dist_type == AgentDistribution.UNIFORM:
        raw = rng.uniform(0.0, L, size=n_agents)

    elif dist_type == AgentDistribution.UNIFORM_FEASIBLE:
        # Uniformly sample across the union of feasible intervals weighted by length
        intervals = env.feasible_intervals
        lengths = np.array([iv.length for iv in intervals], dtype=float)
        probs = lengths / lengths.sum()
        chosen_indices = rng.choice(len(intervals), size=n_agents, p=probs)
        raw = np.array([
            rng.uniform(intervals[idx].start, intervals[idx].end)
            for idx in chosen_indices
        ])

    elif dist_type == AgentDistribution.GAUSSIAN:
        mu = float(params.get("mu", L / 2.0))
        sigma = float(params.get("sigma", L / 6.0))
        # Truncated normal on [0, L]
        a_clip = (0.0 - mu) / sigma
        b_clip = (L - mu) / sigma
        raw = stats.truncnorm.rvs(a_clip, b_clip, loc=mu, scale=sigma, size=n_agents, random_state=rng)

    elif dist_type == AgentDistribution.BIMODAL:
        # Mixture of two Gaussians (e.g. two population centers)
        mu1 = float(params.get("mu1", 0.25 * L))
        sigma1 = float(params.get("sigma1", 0.08 * L))
        mu2 = float(params.get("mu2", 0.75 * L))
        sigma2 = float(params.get("sigma2", 0.08 * L))
        weight1 = float(params.get("weight1", 0.5))

        # Sample cluster assignment
        cluster = rng.binomial(1, 1.0 - weight1, size=n_agents)
        mus = np.where(cluster == 0, mu1, mu2)
        sigmas = np.where(cluster == 0, sigma1, sigma2)

        # Truncate to [0, L]
        samples = rng.normal(mus, sigmas)
        raw = np.clip(samples, 0.0, L)

    elif dist_type == AgentDistribution.CLUSTERED:
        # Mixture of k clusters with random or specified centers
        k = int(params.get("k", 3))
        centers = params.get("centers", None)
        if centers is None:
            centers = np.linspace(0.15 * L, 0.85 * L, k)
        else:
            centers = np.asarray(centers, dtype=float)
        cluster_sigma = float(params.get("sigma", 0.05 * L))

        chosen_centers = rng.choice(centers, size=n_agents)
        samples = rng.normal(chosen_centers, cluster_sigma)
        raw = np.clip(samples, 0.0, L)

    elif dist_type == AgentDistribution.POLARIZED:
        # Heavy density near endpoints 0 and L
        beta_param = float(params.get("beta", 0.5))  # U-shaped Beta(beta, beta)
        raw = rng.beta(beta_param, beta_param, size=n_agents) * L

    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

    # If feasible_only is requested, ensure all locations fall into feasible zones
    if feasible_only:
        raw = np.array([env.project_nearest(x) for x in raw])

    # Discretization snapping
    if snap_to_grid:
        raw = np.array([env.snap_to_grid(x, feasible_only=feasible_only) for x in raw])

    return np.asarray(raw, dtype=float)
