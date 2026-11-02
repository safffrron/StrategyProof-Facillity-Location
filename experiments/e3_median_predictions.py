"""E3 - Social cost with predictions of the median (report: Section 4.2, Proposition 7, Theorem 8).

(a) Proposition 7: given an exact prediction of the median, MinMaxP* outputs exactly Median*, whose
    worst-case ratio is (3n - 1)/(n + 1) for odd n. Checked on the tight instances and on random
    one-zone instances.
(b) Theorem 8: no deterministic strategyproof mechanism does better. For n = 3 and allowed sites
    {-1, 1}, an integer program over ALL deterministic strategyproof mechanisms (reports on a grid of
    step s, prediction 0 = the true median) returns (4 - s)/(2 + s), the ratio of the proof's instances
    with epsilon = s.

Output: results/e3_median_predictions.md
"""

from __future__ import annotations
import itertools
import os

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from common import RES_DIR, write_markdown_table
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
        v = best_deterministic_consistency(grid)
        rows_b.append({"step s": s, "grid points": len(grid), "best consistency (MILP)": round(v, 4),
                       "(4-s)/(2+s)": round(2 - 3 * s / (2 + s), 4)})

    path = os.path.join(RES_DIR, "e3_median_predictions.md")
    write_markdown_table(rows_a, list(rows_a[0]), path,
                         header="## E3a - MinMaxP* with an exact median prediction (social cost)")
    with open(path, "a") as fh:
        fh.write("\n")
    tmp = path + ".b"
    write_markdown_table(rows_b, list(rows_b[0]), tmp,
                         header="## E3b - Best consistency of any deterministic strategyproof mechanism, n = 3, "
                                "allowed sites {-1, 1} (integer program)")
    with open(path, "a") as fh, open(tmp) as fb:
        fh.write(fb.read())
    os.remove(tmp)

    for r in rows_a + rows_b:
        print("  ", r)
    assert all(r["best consistency (MILP)"] == r["(4-s)/(2+s)"] for r in rows_b), "integer program disagrees"


if __name__ == "__main__":
    main()
