"""E5 - Optimisation over ALL mechanisms (report: Section 4.3; Theorems 4, 5 and 10).

Two agents, a fixed prediction inside the blocked zone (-1, 1), and ten candidate sites. Agent types and
misreports are restricted to grids (a relaxation, so each optimum is a valid lower bound).

* Point prediction 0: minimise the worst normalised excess t = max_x (MC(f(x)) - OPT(x)) / |o_hat - o(x)|.
* Interval prediction I = [-1, 1] containing the optimum: minimise t = max_x (MC - OPT) / |I|.

An integer program (MILP) optimises over every deterministic strategyproof mechanism (expected t = 2 for
the point, 1 for the interval: ratios 1 + 2 eta and 1 + omega); relaxing integrality gives a linear
program over every truthful-in-expectation mechanism (expected 1 and 1/2: ratios 1 + eta and 1 + omega/2).

Output: results/e5_optimal_mechanisms.md
"""

from __future__ import annotations
import os
import time
from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from common import RES_DIR, write_markdown_table


def solve(D: float, step: float, candidates: np.ndarray, normaliser: str, integral: bool = True) -> dict:
    """min_f max_x excess(x)/norm(x) over deterministic SP mechanisms f: A x B -> candidates
    (``integral=False``: over randomized truthful-in-expectation mechanisms, an LP)."""
    A = np.round(np.arange(-D - 1, -D + 1 + 1e-9, step), 9)   # agent 1 types / reports
    B = np.round(np.arange(D - 1, D + 1 + 1e-9, step), 9)     # agent 2 types / reports
    C = np.asarray(candidates, dtype=float)
    nA, nB, K = len(A), len(B), len(C)
    nv = nA * nB * K + 1                                      # binaries z[a,b,k] + continuous t
    idx = lambda a, b, k: (a * nB + b) * K + k
    lo, hi = [], []
    M = lil_matrix((nA * nB + 2 * nA * nA * nB + 2 * nB * nB * nA + nA * nB, nv))
    r = 0
    # exactly one candidate per profile
    for a, b in product(range(nA), range(nB)):
        for k in range(K):
            M[r, idx(a, b, k)] = 1
        lo.append(1); hi.append(1); r += 1
    # strategyproofness for agent 1 (true type a, report a2) and agent 2
    for a, a2, b in product(range(nA), range(nA), range(nB)):
        if a == a2:
            continue
        for k in range(K):
            d = abs(C[k] - A[a])
            M[r, idx(a, b, k)] += d
            M[r, idx(a2, b, k)] -= d
        lo.append(-np.inf); hi.append(0); r += 1
    for b, b2, a in product(range(nB), range(nB), range(nA)):
        if b == b2:
            continue
        for k in range(K):
            d = abs(C[k] - B[b])
            M[r, idx(a, b, k)] += d
            M[r, idx(a, b2, k)] -= d
        lo.append(-np.inf); hi.append(0); r += 1
    # normalised excess <= t
    for a, b in product(range(nA), range(nB)):
        mc = np.maximum(np.abs(C - A[a]), np.abs(C - B[b]))
        opt_k = int(np.argmin(mc + 1e-12 * np.abs(C)))
        opt = mc[opt_k]
        norm = abs(C[opt_k]) if normaliser == "point" else 2.0  # |o_hat - o| with o_hat = 0, or |I| = 2
        norm = max(norm, 1e-9)
        for k in range(K):
            M[r, idx(a, b, k)] = (mc[k] - opt) / norm
        M[r, nv - 1] = -1
        lo.append(-np.inf); hi.append(0); r += 1
    M = M[:r].tocsr()
    c = np.zeros(nv); c[-1] = 1.0
    integrality = np.ones(nv) if integral else np.zeros(nv); integrality[-1] = 0
    bounds = Bounds(np.zeros(nv), np.concatenate([np.ones(nv - 1), [10.0]]))
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(M, lo, hi), integrality=integrality, bounds=bounds,
               options={"time_limit": 600})
    return {"value": float(res.x[-1]) if res.x is not None else float("nan"), "status": res.message,
            "seconds": time.time() - t0, "profiles": nA * nB, "candidates": K}


def main() -> None:
    rows = []
    cands = np.array([-7.0, -5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0, 7.0])
    cases = [("deterministic strategyproof (MILP), point prediction", "point", True, "2 (ratio 1 + 2 eta)"),
             ("deterministic strategyproof (MILP), interval [-1, 1]", "interval", True, "1 (ratio 1 + omega)"),
             ("truthful in expectation (LP), point prediction", "point", False, "1 (ratio 1 + eta)"),
             ("truthful in expectation (LP), interval [-1, 1]", "interval", False, "0.5 (ratio 1 + omega/2)")]
    for D, step in ((4.0, 0.25), (8.0, 0.125)):
        for label, norm, integral, predicted in cases:
            res = solve(D, step, cands, norm, integral)
            rows.append({"mechanisms": label, "D": D, "grid step": step, "profiles": res["profiles"],
                         "optimal normalised excess": f"{res['value']:.4f}", "predicted": predicted})
            print("  ", rows[-1], f"({res['seconds']:.1f}s)")
    write_markdown_table(rows, list(rows[0].keys()), os.path.join(RES_DIR, "e5_optimal_mechanisms.md"),
                         header="E5 - Best worst-case normalised excess over ALL deterministic strategyproof (MILP) "
                                "or truthful-in-expectation (LP) mechanisms (two agents, candidate sites "
                                + ", ".join(f"{c:g}" for c in cands) + "; prediction inside the blocked zone (-1, 1)).")
    for r in rows:
        assert abs(float(r["optimal normalised excess"]) - float(r["predicted"].split()[0])) < 1e-3


if __name__ == "__main__":
    main()
