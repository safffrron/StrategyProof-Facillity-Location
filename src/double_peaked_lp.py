"""Computer-assisted lower bounds for truthful-in-expectation (TIE) mechanisms with double-peaked preferences.

Reports lie on the grid G = {k h : -K <= k <= K}, with the peak distance b a multiple of h. Every relevant
function (agent costs, social and max cost) is linear between consecutive points of the finer grid T = (h/2) Z,
so any lottery can be replaced by one supported on T with the same expected costs ("spreading"); mass beyond the
ends of T is represented exactly by its probability and the first moment of its overshoot. Restricting reports to
G and symmetrising (agent swaps, reflection x -> -x) are relaxations, so the LP value is a valid lower bound.

The certificate is checked by ``verify_dp_certificate`` in exact arithmetic, independently of the LP. Moments are
capped using an a-priori bound U on the ratio (a mechanism with ratio above U satisfies the claimed bound anyway):
k * moment <= U * OPT, with k = n for social cost and 1 for max cost.
"""

from __future__ import annotations
import time
from collections import defaultdict
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Dict, List

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

Frac = Fraction


def _cost(t, x, b, c):
    return c + min(abs(t - (x - b)), abs(t - (x + b)))


class DPLowerBound:
    def __init__(self, n: int, K: int, h="1/2", b="1", c="1", objective: str = "sc"):
        self.n, self.K, self.obj = n, K, objective
        self.h, self.b, self.c = Frac(h), Frac(b), Frac(c)
        if (self.b / self.h).denominator != 1:
            raise ValueError("b must be a multiple of h")
        self.G = [k * self.h for k in range(-K, K + 1)]
        step = self.h / 2
        lo, hi = self.G[0] - self.b, self.G[-1] + self.b
        self.T = [lo + j * step for j in range(int((hi - lo) / step) + 1)]
        self.m, self.nT = len(self.G), len(self.T)
        self.L = self.nT + 4                                          # p_T, mass_L, moment_L, mass_R, moment_R
        Gf, Tf = np.array([float(g) for g in self.G]), np.array([float(t) for t in self.T])
        bf, cf = float(self.b), float(self.c)
        self.cost = cf + np.minimum(np.abs(Tf[None, :] - (Gf[:, None] - bf)), np.abs(Tf[None, :] - (Gf[:, None] + bf)))
        self.orbit, self.reps = {}, []
        r = self.m - 1
        for prof in combinations_with_replacement(range(self.m), n):
            if prof in self.orbit:
                continue
            oid = len(self.reps)
            self.reps.append(prof)
            self.orbit[prof] = (oid, False)
            self.orbit.setdefault(tuple(sorted(r - a for a in prof)), (oid, True))
        self.nv = len(self.reps) * self.L + 1
        self.k = n if objective == "sc" else 1

    def _cols(self, prof):
        oid, flip = self.orbit[tuple(sorted(prof))]
        base = oid * self.L
        if not flip:
            return list(range(base, base + self.L))
        return [base + self.nT - 1 - j for j in range(self.nT)] + [base + self.nT + 2, base + self.nT + 3,
                                                                     base + self.nT, base + self.nT + 1]

    def _agent(self, x):
        cx = self.cost[x]
        return np.concatenate([cx, [cx[0], 1.0, cx[-1], 1.0]])

    def _objective(self, prof):
        cs = self.cost[list(prof)]
        v = cs.sum(axis=0) if self.obj == "sc" else cs.max(axis=0)
        return np.concatenate([v, [v[0], self.k, v[-1], self.k]]), float(v.min())

    def solve(self, verbose: bool = False):
        rows, cols, vals, self.tie_rows = [], [], [], []
        r = 0
        seen = set()
        for prof in combinations_with_replacement(range(self.m), self.n):
            for i in sorted(set(prof)):
                others = list(prof)
                others.remove(i)
                ct = self._cols(prof)
                coef = self._agent(i)
                for d in range(self.m):
                    if d == i:
                        continue
                    cd = self._cols(others + [d])
                    key = (tuple(ct), tuple(cd), i)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows += [r] * (2 * self.L); cols += ct + cd; vals += list(coef) + list(-coef)
                    self.tie_rows.append((prof, i, d)); r += 1
        n_tie = r
        for prof in self.reps:
            coef, opt = self._objective(prof)
            rows += [r] * self.L + [r]; cols += self._cols(prof) + [self.nv - 1]; vals += list(coef) + [-opt]; r += 1
        A_ub = coo_matrix((vals, (rows, cols)), shape=(r, self.nv)).tocsr()
        er, ec = [], []
        for oid in range(len(self.reps)):
            for j in list(range(self.nT)) + [self.nT, self.nT + 2]:
                er.append(oid); ec.append(oid * self.L + j)
        A_eq = coo_matrix((np.ones(len(er)), (er, ec)), shape=(len(self.reps), self.nv)).tocsr()
        c = np.zeros(self.nv); c[-1] = 1
        t0 = time.time()
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(r), A_eq=A_eq, b_eq=np.ones(len(self.reps)),
                      bounds=[(0, None)] * (self.nv - 1) + [(0, 100)], method="highs")
        if res.status != 0:
            raise RuntimeError(res.message)
        if verbose:
            print(f"    LP {res.fun:.6f} ({len(self.reps)} profile classes, {n_tie} misreport rows, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        self.result, self.n_tie = res, n_tie
        return float(res.fun)

    def certificate(self, U="2", denominator: int = 10 ** 9) -> dict:
        duals = -self.result.ineqlin.marginals
        G = [str(g) for g in self.G]
        tie = [[[G[a] for a in prof], G[i], G[d], int(round(max(duals[j], 0.0) * denominator))]
               for j, (prof, i, d) in enumerate(self.tie_rows)]
        ratio = [[[G[a] for a in prof], int(round(max(duals[self.n_tie + k], 0.0) * denominator))]
                 for k, prof in enumerate(self.reps)]
        cert = {"setting": "double-peaked preferences, truthful in expectation, no prediction",
                "n": self.n, "K": self.K, "h": str(self.h), "b": str(self.b), "c": str(self.c),
                "objective": self.obj, "U": str(U), "denominator": denominator,
                "lp_value": float(self.result.fun),
                "tie": [t for t in tie if t[3] > 0], "ratio": [t for t in ratio if t[1] > 0]}
        bound = verify_dp_certificate(cert)
        cert["bound"] = f"{bound.numerator}/{bound.denominator}"
        cert["bound_float"] = float(bound)
        return cert


def verify_dp_certificate(cert: dict) -> Fraction:
    """Exact lower bound min(U, m / S) implied by a certificate; independent of the LP code above."""
    n, b, c, h = int(cert["n"]), Frac(cert["b"]), Frac(cert["c"]), Frac(cert["h"])
    K, U, den, obj = int(cert["K"]), Frac(cert["U"]), int(cert["denominator"]), cert["objective"]
    k_obj = n if obj == "sc" else 1
    G = [j * h for j in range(-K, K + 1)]
    step = h / 2
    lo, hi = G[0] - b, G[-1] + b
    T = [lo + j * step for j in range(int((hi - lo) / step) + 1)]
    index = {t: j for j, t in enumerate(T)}
    nT = len(T)

    def canon(prof):
        a = tuple(sorted(prof))
        m = tuple(sorted(-x for x in prof))
        return (a, False) if a <= m else (m, True)

    coef: Dict[tuple, List[Fraction]] = defaultdict(lambda: [Frac(0)] * (nT + 4))

    def add(prof, vec):
        rep, flip = canon(prof)
        tgt = coef[rep]
        if not flip:
            for j in range(nT + 4):
                tgt[j] += vec[j]
            return
        for j in range(nT):
            tgt[index[-T[j]]] += vec[j]
        tgt[nT + 2] += vec[nT]; tgt[nT + 3] += vec[nT + 1]; tgt[nT] += vec[nT + 2]; tgt[nT + 1] += vec[nT + 3]

    def agent_vec(x):
        v = [_cost(t, x, b, c) for t in T]
        return v + [v[0], Frac(1), v[-1], Frac(1)]

    for prof, i, d, w in cert["tie"]:
        prof = [Frac(p) for p in prof]; i, d = Frac(i), Frac(d); w = Frac(int(w), den)
        if w < 0 or i not in prof:
            raise ValueError("invalid misreport multiplier")
        dev = list(prof); dev.remove(i); dev.append(d)
        v = agent_vec(i)
        add(prof, [w * x for x in v])
        add(dev, [-w * x for x in v])
    S = Frac(0)
    opts = {}
    for prof, w in cert["ratio"]:
        prof = [Frac(p) for p in prof]; w = Frac(int(w), den)
        if w < 0:
            raise ValueError("invalid ratio multiplier")
        vals = [sum(_cost(t, x, b, c) for x in prof) if obj == "sc" else max(_cost(t, x, b, c) for x in prof)
                for t in T]
        opt = min(vals)
        rep, flip = canon(prof)
        opts[rep] = opt
        add(prof, [w * v for v in vals + [vals[0], Frac(k_obj), vals[-1], Frac(k_obj)]])
        S += w * opt
    if S <= 0:
        raise ValueError("no positive ratio multiplier")
    total = Frac(0)
    for rep, cf in coef.items():
        total += min(cf[:nT] + [cf[nT], cf[nT + 2]])
        if rep not in opts:
            vals = [sum(_cost(t, x, b, c) for x in rep) if obj == "sc" else max(_cost(t, x, b, c) for x in rep) for t in T]
            opts[rep] = min(vals)
        cap = U * opts[rep] / k_obj                                   # k * moment <= U * OPT
        for j in (nT + 1, nT + 3):
            if cf[j] < 0:
                total += cf[j] * cap
    return min(U, total / S)
