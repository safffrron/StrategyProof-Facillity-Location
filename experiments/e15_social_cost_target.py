"""E15 - Social cost with predictions of the *unconstrained* optimum (the median).

Proposition 27 and Theorem 28 (docs/RESULTS.md):
(a) Proposition 27. Given an exact prediction of the (lower) median, MinMaxP* outputs exactly Median*, so its
    worst-case ratio is Median*'s, (3n - 1)/(n + 1) for odd n (Theorem 9 with k = 0), whereas an
    exact prediction of the constrained optimum o makes it optimal (Proposition 8a).
(b) Independent check of Theorem 28 (no deterministic SP mechanism beats (3n-1)/(n+1)): for
    n = 3 and F = {-1, 1}, an integer program over ALL deterministic strategyproof mechanisms
    (types and reports on a grid of step s, prediction 0 = true median) returns (4 - s)/(2 + s),
    which is exactly the ratio of the proof's instances with epsilon = s, and tends to 2.

Outputs: figures/e15_social_cost_target.{png,pdf}, results/e15_median_predictions.md.
"""

from __future__ import annotations
import itertools
import os
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

import matplotlib.pyplot as plt
from common import INK_2, MARKERS, RES_DIR, SERIES, apply_style, save, theory_line, write_markdown_table
from src.learning_augmented import (
    MedianRule, MinMaxP, lower_median, optimal_social_cost_batch, social_cost_batch,
)

N_VALUES = [3, 5, 7, 9, 11, 15, 21]
STEPS = [0.5, 0.25, 0.125]


# ----------------------------------------------------------------------------
# (a) MinMaxP* with an exact median prediction
# ----------------------------------------------------------------------------

def tight_ratio(n: int, d: float = 1e-6) -> float:
    """k+1 agents at 0 and k at 1+d (n = 2k+1), zone (-1, 1+d): median 0 snaps to -1."""
    k = (n - 1) // 2
    X = np.array([[0.0] * (k + 1) + [1.0 + d] * k])
    F = (np.array([-1e6, 1.0 + d]), np.array([-1.0, 1e6]))
    _, opt = optimal_social_cost_batch(X, F)
    y = MinMaxP().locate(X, F, lower_median(X))
    return float(social_cost_batch(y, X)[0] / opt[0])


def random_check(n: int, rng, batches: int = 30, P: int = 40_000) -> tuple:
    """Largest ratio on random one-zone instances, and whether MinMaxP*(median) == Median*."""
    worst, same = 0.0, True
    for _ in range(batches):
        X = rng.uniform(0, 1, (P, n))
        a, w = rng.uniform(0, 1, P), rng.uniform(0, 0.8, P)
        F = (np.stack([np.full(P, -1e6), a + w], 1), np.stack([a, np.full(P, 1e6)], 1))
        _, opt = optimal_social_cost_batch(X, F)
        y = MinMaxP().locate(X, F, lower_median(X))
        same &= bool(np.allclose(y, MedianRule().locate(X, F)))
        ok = opt > 1e-9
        worst = max(worst, float((social_cost_batch(y, X)[ok] / opt[ok]).max()))
    return worst, same


# ----------------------------------------------------------------------------
# (b) Integer program over all deterministic SP mechanisms, n = 3
# ----------------------------------------------------------------------------

def best_deterministic_consistency(grid: np.ndarray, candidates=(-1.0, 1.0)) -> float:
    """min over deterministic SP f: grid^3 -> candidates of the worst SC ratio over profiles
    whose lower median is 0 (= the prediction). Anonymity is not assumed."""
    G, C = np.asarray(grid, float), np.asarray(candidates, float)
    g, K = len(G), len(C)
    profs = list(itertools.product(range(g), repeat=3))
    pid = {p: i for i, p in enumerate(profs)}
    nv = len(profs) * K + 1
    idx = lambda p, k: p * K + k  # noqa: E731
    M = lil_matrix((len(profs) * (2 + 3 * (g - 1)), nv))
    lo, hi, r = [], [], 0
    for p in range(len(profs)):                       # one candidate per profile
        for k in range(K):
            M[r, idx(p, k)] = 1
        lo.append(1); hi.append(1); r += 1
    for p, prof in enumerate(profs):                  # strategyproofness, every agent and lie
        for i in range(3):
            t = G[prof[i]]
            for rep in range(g):
                if rep == prof[i]:
                    continue
                q = list(prof); q[i] = rep; q = pid[tuple(q)]
                for k in range(K):
                    dist = abs(C[k] - t)
                    M[r, idx(p, k)] += dist
                    M[r, idx(q, k)] -= dist
                lo.append(-np.inf); hi.append(0); r += 1
    for p, prof in enumerate(profs):                  # ratio <= t where the prediction is right
        x = np.sort(G[list(prof)])
        if abs(x[1]) > 1e-12:
            continue
        sc = np.array([np.abs(x - c).sum() for c in C])
        for k in range(K):
            M[r, idx(p, k)] = sc[k]
        M[r, nv - 1] = -sc.min()
        lo.append(-np.inf); hi.append(0); r += 1
    M = M[:r].tocsr()
    c = np.zeros(nv); c[-1] = 1
    integ = np.ones(nv); integ[-1] = 0
    res = milp(c, constraints=LinearConstraint(M, lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), np.r_[np.ones(nv - 1), np.inf]))
    assert res.status == 0, res.message
    return float(res.fun)


# ----------------------------------------------------------------------------

def main() -> None:
    apply_style()
    rng = np.random.default_rng(0)

    rows_a = []
    for n in N_VALUES:
        worst, same = random_check(n, rng)
        rows_a.append({"n": n, "formula (3n-1)/(n+1)": round((3 * n - 1) / (n + 1), 4),
                       "tight instance": round(tight_ratio(n), 4), "random worst": round(worst, 4),
                       "equals Median*": same})
    assert all(r["equals Median*"] for r in rows_a)
    assert all(r["random worst"] <= r["formula (3n-1)/(n+1)"] + 1e-9 for r in rows_a)

    rows_b = []
    for s in STEPS:
        grid = np.round(np.arange(-1.5, 1.5 + 1e-9, s), 9)
        t0 = time.time()
        v = best_deterministic_consistency(grid)
        rows_b.append({"step s": s, "grid points": len(grid), "best consistency (MILP)": round(v, 4),
                       "(4-s)/(2+s)": round(2 - 3 * s / (2 + s), 4), "seconds": round(time.time() - t0)})

    path = os.path.join(RES_DIR, "e15_median_predictions.md")
    write_markdown_table(rows_a, list(rows_a[0]), path,
                         header="## (a) MinMaxP* with an exact median prediction (social cost)")
    with open(path, "a") as fh:
        fh.write("\n")
    tmp = path + ".b"
    write_markdown_table(rows_b, list(rows_b[0]), tmp,
                         header="## (b) Best deterministic SP consistency, n = 3, F = {-1, 1} (MILP)")
    with open(path, "a") as fh, open(tmp) as fb:
        fh.write(fb.read())
    os.remove(tmp)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax = axes[0]
    ns = np.array(N_VALUES)
    theory_line(ax, ns, (3 * ns - 1) / (ns + 1), r"bound $(3n-1)/(n+1)$", text_xy=(9.5, 2.85))
    ax.plot(ns, [r["tight instance"] for r in rows_a], color=SERIES[1], marker=MARKERS[1],
            linestyle="none", label="prediction of median: worst case")
    ax.plot(ns, np.ones_like(ns, dtype=float), color=SERIES[0], marker=MARKERS[0],
            label="prediction of $o$ (Prop. 8a)")
    ax.set_title("(a) MinMaxP$^*$, exact predictions")
    ax.set_xlabel("number of agents $n$")
    ax.set_ylabel("social-cost ratio")
    ax.set_ylim(0.9, 3.1)
    ax.legend(loc="center right", fontsize=7)

    ax = axes[1]
    inv = np.array([1 / s for s in STEPS])
    fine = np.linspace(1.5, 10, 100)
    theory_line(ax, fine, 2 - 3 * (1 / fine) / (2 + 1 / fine), r"$(4-s)/(2+s)$", text_xy=(5.2, 1.62))
    ax.axhline(2.0, color=SERIES[1], linewidth=1.2, linestyle="--", label="Median$^*$ $=2$")
    ax.plot(inv, [r["best consistency (MILP)"] for r in rows_b], color=SERIES[0], marker=MARKERS[0],
            linestyle="none", label="best over all mechanisms")
    ax.set_title("(b) Theorem 28 check, all mechanisms ($n=3$)")
    ax.set_xlabel("grid resolution $1/s$")
    ax.set_ylim(1.2, 2.1)
    ax.legend(loc="lower right", fontsize=7)

    print(save(fig, "e15_social_cost_target"))
    for r in rows_a + rows_b:
        print(r)


if __name__ == "__main__":
    main()
