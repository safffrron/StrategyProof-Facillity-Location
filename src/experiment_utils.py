"""Shared machinery for the experiments: instance sampling, controlled-error predictions and a
batched adversarial (hill-climbing) search for worst-case instances.

Conventions (following Agrawal et al. 2022):
* ``OPT`` is the optimal cost over the *feasible* set.
* Prediction error and interval width are normalised by the optimal cost: by ``OPT`` for the
  max-cost objective, and by the optimal *average* cost ``OPT / n`` for the social-cost objective.
* A point prediction has error ``eta = |o_hat - o*| / scale``.
* An interval prediction [lo, hi] has normalised width ``omega = (hi - lo) / scale`` and error
  ``eta = dist(o*, [lo, hi]) / scale``; ``eta = 0`` means the interval contains the optimum.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.environment import Environment1D
from src.agents import generate_agents
from src.learning_augmented import (
    BatchMechanism,
    is_feasible_batch,
    mechanism_distribution,
    Feasible,
    max_cost_batch,
    optimal_max_cost_batch,
    optimal_social_cost_batch,
    social_cost_batch,
)


# ============================================================================
# Objectives
# ============================================================================

def objective_cost(Y: np.ndarray, X: np.ndarray, objective: str) -> np.ndarray:
    return max_cost_batch(Y, X) if objective == "mc" else social_cost_batch(Y, X)


def expected_objective(mech: BatchMechanism, X: np.ndarray, feasible: Feasible, prediction, objective: str) -> np.ndarray:
    """Objective value of a (possibly randomized) mechanism, in expectation over its randomness."""
    locs, probs = mechanism_distribution(mech, X, feasible, prediction)
    total = np.zeros(X.shape[0])
    for j in range(locs.shape[1]):
        total += probs[:, j] * objective_cost(locs[:, j], X, objective)
    return total


def optimum(X: np.ndarray, feasible: Feasible, objective: str) -> Tuple[np.ndarray, np.ndarray]:
    return optimal_max_cost_batch(X, feasible) if objective == "mc" else optimal_social_cost_batch(X, feasible)


def error_scale(opt: np.ndarray, n: int, objective: str) -> np.ndarray:
    """Normaliser for prediction error / interval width (OPT for max cost, OPT/n for social cost)."""
    return opt if objective == "mc" else opt / n


def ratio(cost: np.ndarray, opt: np.ndarray) -> np.ndarray:
    safe = np.where(opt > 1e-12, opt, 1.0)
    return np.where(opt > 1e-12, cost / safe, 1.0)


# ============================================================================
# Predictions with controlled error
# ============================================================================

def point_predictions(o_star: np.ndarray, scale: np.ndarray, eta: float, rng: np.random.Generator) -> np.ndarray:
    """Point predictions at distance exactly ``eta * scale`` from the optimum, random side."""
    sign = rng.choice([-1.0, 1.0], size=o_star.shape)
    return o_star + sign * eta * scale


def interval_predictions(
    o_star: np.ndarray,
    scale: np.ndarray,
    omega: float,
    eta: float,
    rng: np.random.Generator,
    position: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Interval predictions of normalised width ``omega``.

    If ``eta == 0`` the optimum lies inside the interval at relative position ``position``
    (uniform in [0, 1] by default; 0 = left end). Otherwise the interval lies at normalised
    distance ``eta`` from the optimum, on a random side.
    """
    width = omega * scale
    if eta == 0:
        t = rng.uniform(0, 1, size=o_star.shape) if position is None else position
        lo = o_star - t * width
    else:
        sign = rng.choice([-1.0, 1.0], size=o_star.shape)
        gap = eta * scale
        lo = np.where(sign > 0, o_star + gap, o_star - gap - width)
    return np.stack([lo, lo + width], axis=1)


# ============================================================================
# Typical-instance sampling (uses the Week 1-2 environment and agent generators)
# ============================================================================

def sample_typical_instances(
    n_agents: int,
    blocked_density: float,
    n_envs: int,
    profiles_per_env: int,
    distribution: str = "uniform",
    num_blocks: int = 3,
    seed: int = 0,
    params: Optional[dict] = None,
) -> List[Tuple[Environment1D, np.ndarray]]:
    """Random environments (``Environment1D.random_blocked``) with random agent profiles."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_envs):
        env_seed = int(rng.integers(0, 2**31 - 1))
        if blocked_density <= 0:
            env = Environment1D.unconstrained(length=1.0, num_grid_points=11)
        else:
            env = Environment1D.random_blocked(
                length=1.0, num_blocks=num_blocks, blocked_density=blocked_density,
                min_block_size=0.02, num_grid_points=11, seed=env_seed,
            )
        X = np.stack([
            generate_agents(n_agents, env, distribution=distribution, params=params,
                            seed=int(rng.integers(0, 2**31 - 1)))
            for _ in range(profiles_per_env)
        ])
        out.append((env, X))
    return out


# ============================================================================
# Adversarial search
# ============================================================================

@dataclass
class Scenario:
    """A worst-case question: how bad can ``mechanism`` be on ``objective`` under these conditions?

    Attributes:
        mode: ``"none"`` (no prediction), ``"robust"`` (arbitrary point prediction),
            ``"error"`` (point prediction with error <= eta), ``"error_feasible"`` (as ``"error"``,
            but the prediction must be a feasible point, as in Fang et al. 2025), ``"interval"``
            (interval prediction of normalised width omega at error eta; eta = 0 means it contains
            the optimum).
        n_gaps: Number of blocked zones (0 = unconstrained real line).
    """
    mechanism: BatchMechanism
    objective: str
    n: int
    n_gaps: int = 0
    mode: str = "none"
    eta: float = 0.0
    omega: float = 0.0
    label: str = ""

    @property
    def dim(self) -> int:
        extra = {"none": 0, "robust": 1, "error": 1, "error_feasible": 1, "interval": 2}[self.mode]
        return self.n + 2 * self.n_gaps + extra


def layouts_from_params(params: np.ndarray, n_gaps: int) -> Feasible:
    """Turn ``(R, 2*n_gaps)`` parameters (centre, half-length per gap) into feasible intervals in [0, 1]."""
    if n_gaps == 0:
        return None
    R = params.shape[0]
    centres = params[:, 0::2]
    halves = np.abs(params[:, 1::2])
    g_start = np.clip(centres - halves, 0.01, 0.99)
    g_end = np.clip(centres + halves, 0.01, 0.99)
    order = np.argsort(g_start, axis=1)
    g_start = np.take_along_axis(g_start, order, axis=1)
    g_end = np.take_along_axis(g_end, order, axis=1)
    for j in range(1, n_gaps):
        g_start[:, j] = np.maximum(g_start[:, j], g_end[:, j - 1])
    g_end = np.maximum(g_end, g_start)
    f_start = np.concatenate([np.zeros((R, 1)), g_end], axis=1)
    f_end = np.concatenate([g_start, np.ones((R, 1))], axis=1)
    return f_start, f_end


def evaluate(theta: np.ndarray, sc: Scenario) -> Tuple[np.ndarray, dict]:
    """Approximation ratio of ``sc.mechanism`` on each parameter row of ``theta``."""
    n = sc.n
    X = np.clip(theta[:, :n], 0.0, 1.0)
    feasible = layouts_from_params(theta[:, n:n + 2 * sc.n_gaps], sc.n_gaps)
    p = theta[:, n + 2 * sc.n_gaps:]
    o_star, opt = optimum(X, feasible, sc.objective)
    scale = error_scale(opt, n, sc.objective)
    if sc.mode == "none":
        pred = None
    elif sc.mode == "robust":
        pred = np.clip(p[:, 0], -1.0, 2.0)
    elif sc.mode in ("error", "error_feasible"):
        pred = o_star + np.clip(p[:, 0], -1.0, 1.0) * sc.eta * scale
    else:
        width = sc.omega * scale
        if sc.eta == 0:
            lo = o_star - np.clip(p[:, 0], 0.0, 1.0) * width
        else:
            gap = sc.eta * scale
            lo = np.where(p[:, 1] >= 0, o_star + gap, o_star - gap - width)
        pred = np.stack([lo, lo + width], axis=1)
    Y = sc.mechanism.locate(X, feasible, pred)
    r = ratio(expected_objective(sc.mechanism, X, feasible, pred, sc.objective), opt)
    if sc.mode == "error_feasible":
        r = np.where(is_feasible_batch(pred, feasible), r, 0.0)  # infeasible predictions do not count
    return r, {"X": X, "feasible": feasible, "prediction": pred, "facility": Y, "opt_location": o_star, "opt": opt}


@dataclass
class SearchResult:
    scenario: Scenario
    best_ratio: float
    instance: dict = field(default_factory=dict)


def _initial_theta(sc: Scenario, restarts: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 1.0, size=(restarts, sc.dim))
    # Half of the restarts start *clustered*: agents sit on 1-3 common points. Worst cases of
    # median-type rules put many agents on the same spot (cf. Agrawal et al., Lemma 4.2), which
    # independent per-agent perturbations rarely discover.
    half = restarts // 2
    n_clusters = rng.integers(1, 4, size=half)
    centres = rng.uniform(0.0, 1.0, size=(half, 3))
    assign = (rng.random((half, sc.n)) * n_clusters[:, None]).astype(int)
    theta[:half, :sc.n] = np.take_along_axis(centres, assign, axis=1)
    if sc.n_gaps:
        theta[:, sc.n + 1:sc.n + 2 * sc.n_gaps:2] *= 0.25  # moderate gap half-lengths
    if sc.mode in ("error", "interval"):
        theta[:, -1] = rng.uniform(-1, 1, size=restarts)
    if sc.mode == "robust":
        theta[:, -1] = rng.uniform(-0.5, 1.5, size=restarts)
    return theta


def _propose(theta: np.ndarray, sc: Scenario, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """One proposal per row, mixing three move types:
    independent Gaussian jitter, moving a whole cluster of coincident agents, and merging two agents."""
    R, n = theta.shape[0], sc.n
    prop = theta + (rng.random(theta.shape) < 0.5) * rng.normal(0.0, sigma, size=theta.shape)
    move = rng.random(R)
    rows = np.arange(R)
    # cluster move: shift every agent within 1e-6 of a random agent by the same amount
    cl = move < 0.25
    if cl.any():
        j = rng.integers(0, n, size=R)
        X = theta[:, :n]
        same = np.abs(X - X[rows, j][:, None]) < 1e-6
        delta = rng.normal(0.0, sigma, size=R)
        prop[cl, :n] = (X + same * delta[:, None])[cl]
        prop[cl, n:] = theta[cl, n:]
    # merge: put a random agent exactly on another agent's position
    mg = (move >= 0.25) & (move < 0.4)
    if mg.any():
        i = rng.integers(0, n, size=R)
        j = rng.integers(0, n, size=R)
        merged = theta.copy()
        merged[rows, i] = theta[rows, j]
        prop[mg] = merged[mg]
    return prop


def hill_climb(sc: Scenario, restarts: int = 256, steps: int = 400, seed: int = 0,
               sigma_start: float = 0.15, sigma_end: float = 0.002) -> SearchResult:
    """Batched random-restart hill climbing over agent positions, blocked zones and the prediction.

    The value found is a *lower bound* on the true worst case (it is attained by an explicit
    instance); it matches a proved upper bound only when the bound is tight.
    """
    rng = np.random.default_rng(seed)
    theta = _initial_theta(sc, restarts, rng)
    cur, _ = evaluate(theta, sc)
    sigmas = np.geomspace(sigma_start, sigma_end, steps)
    for s in sigmas:
        prop = _propose(theta, sc, s, rng)
        val, _ = evaluate(prop, sc)
        better = val >= cur
        theta[better] = prop[better]
        cur[better] = val[better]
    j = int(np.argmax(cur))
    _, info = evaluate(theta[j:j + 1], sc)
    inst = {k: (v[0] if isinstance(v, np.ndarray) else v) for k, v in info.items() if k != "feasible"}
    if info["feasible"] is not None:
        inst["feasible"] = list(zip(np.round(info["feasible"][0][0], 4), np.round(info["feasible"][1][0], 4)))
    return SearchResult(scenario=sc, best_ratio=float(cur[j]), instance=inst)
