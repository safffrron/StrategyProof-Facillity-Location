"""E14 - Predictions of the *unconstrained* optimum (max-cost objective, blocked zones).

A model trained without knowledge of the blocked zones predicts the unconstrained optimum m
(the midpoint of the extreme agents) instead of the constrained optimum o = sigma(m).
Theorem 25 (docs/RESULTS.md): MinMaxP* has ratio <= min{1 + 2 eta_m, 1 + 2 eta_o, 3}, where
eta_m = |o_hat - m| / OPT and eta_o = |o_hat - o| / OPT. In particular it is 1-consistent for
predictions of m, although Theorem 4 alone only guarantees min{1 + 2 d/OPT, 3} there.

(a) Worst case: random + local search for the largest ratio at each eta_m, against 1 + 2 eta_m.
(b) Typical case: mean ratio on random instances, predictions aimed at m, several densities.
(c) Exact predictions of m (eta_m = 0) on the same typical instances: the realised ratio is 1,
    while Theorem 4's guarantee (which only sees eta_o = |m - o|/OPT) ranges up to almost 3.

Outputs: figures/e14_unconstrained_target.{png,pdf}, results/e14_worst_case.csv,
results/e14_typical.csv.
"""

from __future__ import annotations
import csv
import os

import numpy as np

import matplotlib.pyplot as plt
from common import INK_2, MARKERS, RAMP, RES_DIR, SERIES, apply_style, save, theory_line
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


def bound_new(eta_m, eta_o):
    return np.minimum(np.minimum(1 + 2 * eta_m, 1 + 2 * eta_o), 3.0)


def bound_thm4(eta_o):
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
    vals, g4, g25 = [], [], []
    for env, X in sample_typical_instances(7, density, n_envs=60, profiles_per_env=50, seed=11):
        _, opt = optimal_max_cost_batch(X, env)
        pred = midpoint(X) + rng.choice([-1.0, 1.0], X.shape[0]) * eta * opt
        r, eta_m, eta_o = evaluate(X, env, pred)
        vals.append(r)
        g4.append(bound_thm4(eta_o))
        g25.append(bound_new(eta_m, eta_o))
    v, b4, b25 = map(np.concatenate, (vals, g4, g25))
    return {"mean": float(v.mean()), "p95": float(np.quantile(v, 0.95)), "max": float(v.max()),
            "thm4_guarantee_mean": float(b4.mean()), "thm25_guarantee_mean": float(b25.mean()),
            "thm4_guarantee_p95": float(np.quantile(b4, 0.95)), "thm4_guarantee_max": float(b4.max()),
            "violations": int((v > b25 + 1e-9).sum())}


# ----------------------------------------------------------------------------

def main() -> None:
    apply_style()
    rng = np.random.default_rng(0)

    wc = [{"eta_m": e, "worst_ratio_found": round(worst_case(e, seed=i), 4),
           "bound_min_1p2eta_3": min(1 + 2 * e, 3.0)} for i, e in enumerate(ETAS)]
    with open(os.path.join(RES_DIR, "e14_worst_case.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(wc[0]))
        w.writeheader()
        w.writerows(wc)

    ty = []
    for d in DENSITIES:
        for e in ETAS:
            ty.append({"density": d, "eta_m": e, **typical(e, d, rng)})
    with open(os.path.join(RES_DIR, "e14_typical.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ty[0]))
        w.writeheader()
        w.writerows(ty)
    assert all(row["violations"] == 0 for row in ty), "Theorem 25 bound violated on a typical instance"

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    etas = np.array(ETAS)

    ax = axes[0]
    theory_line(ax, etas, np.minimum(1 + 2 * etas, 3), r"bound $\min\{1+2\eta_m,3\}$", text_xy=(0.45, 1.55))
    ax.plot(etas, [r["worst_ratio_found"] for r in wc], color=SERIES[1], marker=MARKERS[1],
            linestyle="none", label="worst found")
    ax.set_title("(a) Worst case, prediction of $m$")
    ax.set_xlabel(r"error $\eta_m=|\hat o-m|/\mathrm{OPT}$")
    ax.set_ylabel("max-cost ratio")
    ax.legend(loc="lower right")

    ax = axes[1]
    for i, d in enumerate(DENSITIES):
        rows = [r for r in ty if r["density"] == d]
        ax.plot(etas, [r["mean"] for r in rows], color=RAMP[i + 1], marker=MARKERS[i],
                label=f"blocked {int(d * 100)}%")
    ax.set_title("(b) Typical instances, mean ratio")
    ax.set_xlabel(r"$\eta_m$")
    ax.legend(loc="upper left")

    ax = axes[2]
    zero = [r for r in ty if r["eta_m"] == 0.0]
    xs = np.arange(len(DENSITIES))
    width = 0.26
    for j, (key, lab) in enumerate([("thm4_guarantee_mean", "mean"), ("thm4_guarantee_p95", "95th pct"),
                                    ("thm4_guarantee_max", "max")]):
        ax.bar(xs + (j - 1) * width, [r[key] for r in zero], width, color=RAMP[j + 1],
               label=f"Thm 4 guarantee, {lab}")
    ax.axhline(1.0, color=SERIES[1], linewidth=1.6, label="realised ratio = Thm 25 guarantee")
    ax.set_xticks(xs, [f"{int(d * 100)}% blocked" for d in DENSITIES])
    ax.set_ylim(0.9, 3.1)
    ax.set_title(r"(c) Exact prediction of $m$ ($\eta_m=0$)")
    ax.legend(loc="upper left", fontsize=7)

    print(save(fig, "e14_unconstrained_target"))
    for r in wc:
        print(r)


if __name__ == "__main__":
    main()
