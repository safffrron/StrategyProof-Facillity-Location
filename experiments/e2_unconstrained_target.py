"""E2 - Predictions of the unconstrained optimum m (report: Section 4.2, Theorem 6).

A model that ignores the blocked zones predicts m, the midpoint of the extreme agents, instead of the
best allowed location o = sigma(m). Theorem 6: MinMaxP* has max-cost ratio at most
min{1 + 2 eta_m, 1 + 2 eta_o, 3}, where eta_m = |o_hat - m| / OPT and eta_o = |o_hat - o| / OPT.

(a) Worst case: random and local search for the largest ratio at each eta_m (two agents, one zone).
(b) Typical case: 108,000 random instances (seven agents, 20/40/60% of the line blocked, predictions
    aimed at m); no instance may exceed the bound, and exact predictions of m must give ratio 1.

Outputs: results/e2_worst_case.md, results/e2_typical.md
"""

from __future__ import annotations
import os

import numpy as np

from common import RES_DIR, write_markdown_table
from src.experiment_utils import sample_typical_instances
from src.learning_augmented import MinMaxP, max_cost_batch, optimal_max_cost_batch

ETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]
DENSITIES = [0.2, 0.4, 0.6]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def midpoint(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X.min(axis=1) + X.max(axis=1))


def evaluate(X: np.ndarray, F, pred: np.ndarray):
    """Ratio of MinMaxP*, the constrained optimum, OPT and both normalised errors."""
    o, opt = optimal_max_cost_batch(X, F)
    y = MinMaxP().locate(X, F, pred)
    r = max_cost_batch(y, X) / opt
    m = midpoint(X)
    return r, np.abs(pred - m) / opt, np.abs(pred - o) / opt


def bound_m(eta_m, eta_o):
    """Theorem 6: min{1 + 2 eta_m, 1 + 2 eta_o, 3}."""
    return np.minimum(np.minimum(1 + 2 * eta_m, 1 + 2 * eta_o), 3.0)


def bound_o(eta_o):
    """Theorem 3, which only sees the error measured from o: min{1 + 2 eta_o, 3}."""
    return np.minimum(1 + 2 * eta_o, 3.0)


# ----------------------------------------------------------------------------
# (a) Worst case at fixed eta_m
# ----------------------------------------------------------------------------

def _instances(theta: np.ndarray, eta: float, sign: np.ndarray):
    """theta rows: (x1, x2, a, w) -> agents (x1, x2), one blocked zone (a, a + w).

    Each row is its own instance, so the feasible set is passed row by row as (P, K) bounds.
    The prediction sits at m + sign * eta * OPT.
    """
    X = np.sort(theta[:, :2], axis=1)
    a, w = theta[:, 2], np.abs(theta[:, 3])
    starts = np.stack([np.full_like(a, -1e6), a + w], axis=1)
    ends = np.stack([a, np.full_like(a, 1e6)], axis=1)
    F = (starts, ends)
    _, opt = optimal_max_cost_batch(X, F)
    pred = midpoint(X) + sign * eta * opt
    return X, F, pred


def worst_case(eta: float, seed: int = 0, batch: int = 200_000, rounds: int = 40) -> float:
    rng = np.random.default_rng(seed)
    theta = np.column_stack([rng.uniform(0, 2, batch), rng.uniform(0, 2, batch),
                             rng.uniform(-0.5, 2.5, batch), rng.uniform(0, 2, batch)])
    sign = rng.choice([-1.0, 1.0], batch)
    keep = np.isfinite(theta).all(axis=1)
    best_val, best_theta, best_sign = 0.0, None, 1.0
    scale = 0.3
    for _ in range(rounds):
        X, F, pred = _instances(theta[keep], eta, sign[keep])
        r, eta_m, _ = evaluate(X, F, pred)
        r = np.where(np.isfinite(r) & (X[:, 1] - X[:, 0] > 1e-6), r, 0.0)
        j = int(np.argmax(r))
        if r[j] > best_val:
            best_val, best_theta, best_sign = float(r[j]), theta[keep][j].copy(), float(sign[keep][j])
        # local refinement around the incumbent
        theta = best_theta + rng.normal(0, scale, (batch, 4))
        sign = np.full(batch, best_sign)
        keep = np.ones(batch, dtype=bool)
        scale *= 0.85
    return best_val


# ----------------------------------------------------------------------------
# (b, c) Typical instances, predictions aimed at m
# ----------------------------------------------------------------------------

def typical(eta: float, density: float, rng):
    vals, go, gm = [], [], []
    for env, X in sample_typical_instances(7, density, n_envs=60, profiles_per_env=50, seed=11):
        _, opt = optimal_max_cost_batch(X, env)
        pred = midpoint(X) + rng.choice([-1.0, 1.0], X.shape[0]) * eta * opt
        r, eta_m, eta_o = evaluate(X, env, pred)
        vals.append(r)
        go.append(bound_o(eta_o))
        gm.append(bound_m(eta_m, eta_o))
    v, bo, bm = map(np.concatenate, (vals, go, gm))
    return {"instances": len(v), "mean ratio": round(float(v.mean()), 4), "max ratio": round(float(v.max()), 4),
            "mean bound in eta_o (Thm 3)": round(float(bo.mean()), 4),
            "mean bound in eta_m (Thm 6)": round(float(bm.mean()), 4),
            "violations": int((v > bm + 1e-9).sum())}


# ----------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(0)
    wc = [{"eta_m": e, "worst ratio found": round(worst_case(e, seed=i), 4), "bound min{1 + 2 eta_m, 3}":
           min(1 + 2 * e, 3.0)} for i, e in enumerate(ETAS)]
    write_markdown_table(wc, list(wc[0]), os.path.join(RES_DIR, "e2_worst_case.md"),
                         header="E2a - Worst max-cost ratio of MinMaxP* found by search for predictions of m "
                                "(two agents, one blocked zone), against Theorem 6.")
    ty = [{"blocked": f"{int(d * 100)}%", "eta_m": e, **typical(e, d, rng)} for d in DENSITIES for e in ETAS]
    write_markdown_table(ty, list(ty[0]), os.path.join(RES_DIR, "e2_typical.md"),
                         header="E2b - MinMaxP* on random instances (seven agents, three random zones) with "
                                "predictions aimed at m.")
    for r in wc:
        print("  ", r)
    total = sum(r["instances"] for r in ty)
    assert all(abs(r["worst ratio found"] - r["bound min{1 + 2 eta_m, 3}"]) < 1e-3 for r in wc), "bound not attained"
    assert all(r["violations"] == 0 for r in ty), "Theorem 6 violated on a random instance"
    assert all(r["max ratio"] == 1.0 for r in ty if r["eta_m"] == 0.0), "exact prediction of m not optimal"
    print(f"E2: {total:,} random instances, no violation")


if __name__ == "__main__":
    main()
