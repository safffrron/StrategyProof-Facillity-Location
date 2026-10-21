"""Computer-assisted lower bounds for randomized mechanisms with candidate locations.

Setting (Tang, Wang, Zhang & Zhao, COCOA 2020): two agents on the real line, a single facility that must be
placed at one of finitely many candidate locations ``C``, and the maximum-cost objective. A randomized
mechanism maps the two reports to a lottery over ``C``; it is truthful in expectation (TIE) if no agent can
lower their expected distance to the facility by misreporting. Tang et al. prove that no TIE mechanism has
ratio below 2 (deterministic mechanisms: exactly 3) and leave open whether 2 is attainable.

This module proves larger lower bounds for concrete candidate sets:

1. *Grid relaxation.* A TIE mechanism on the line, restricted to reports from a finite grid ``G``, is TIE
   on the grid and its ratio on grid profiles is at most its ratio on the line. So the optimal ratio over
   grid mechanisms, a linear program, lower-bounds every TIE mechanism on the line.
2. *Symmetrisation.* Averaging a mechanism over swapping the agents, and over the reflection
   ``x -> min(C) + max(C) - x`` when ``C`` is symmetric, keeps it TIE and does not raise its ratio. So the
   LP may be restricted to symmetric mechanisms, which shrinks it about fourfold.
3. *Constraint generation.* Start from local misreports and add every violated misreport until none
   remain; the final LP optimum is then optimal for the full LP.
4. *Exact certificate.* From the LP duals we keep nonnegative rational multipliers ``mu`` (misreport
   constraints) and ``nu`` (ratio constraints). For every symmetric TIE grid mechanism ``p`` with ratio ``r``,
   ``r * S >= sum_x nu_x cost_x(p) + sum_j mu_j g_j(p) >= m``, where ``S = sum_x nu_x OPT(x)`` and ``m`` is the
   minimum of the middle expression over *all* lottery assignments, a sum of per-profile minima. Hence
   ``r >= m / S``, which :func:`verify_certificate` evaluates in exact rational arithmetic, independently of
   the LP solver.
"""

from __future__ import annotations
import json
import time
from collections import defaultdict
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

Frac = Fraction


def refined_grid(candidates: Sequence, offsets: Iterable = ("1/500", "1/50", "1/10"),
                 step: Optional[str] = None, lo=None, hi=None, far: str = "2",
                 mirror: bool = False) -> List[Fraction]:
    """Report grid: candidates and the bisectors between neighbours, each with symmetric offsets; points
    beyond the extreme candidates; an optional uniform step on [lo, hi]; with ``mirror``, the bisector
    points reflected about both neighbouring candidates (an agent as far from a candidate as one just past
    the bisector). Closed under the reflection when the candidate set is symmetric."""
    C = sorted(Frac(c) for c in candidates)
    offsets = [Frac(e) for e in offsets]
    far = Frac(far)
    pts = set(C)
    anchors = C + [(C[i] + C[i + 1]) / 2 for i in range(len(C) - 1)]
    for p in anchors:
        pts.add(p)
        for e in offsets:
            pts.update((p - e, p + e))
    if mirror:
        for i in range(len(C) - 1):
            b = (C[i] + C[i + 1]) / 2
            for q in [b] + [b + e for e in offsets] + [b - e for e in offsets]:
                pts.update((2 * C[i] - q, 2 * C[i + 1] - q))
    pts.update((C[0] - far, C[-1] + far, C[0] - far / 2, C[-1] + far / 2))
    for i in range(len(C) - 1):
        gap = C[i + 1] - C[i]
        pts.update((C[i + 1] + gap / 2, C[i] - gap / 2))
    if step is not None:
        x, s = Frac(lo), Frac(step)
        while x <= Frac(hi):
            pts.add(x)
            x += s
    L = C[0] + C[-1]
    if sorted(L - c for c in C) == C:
        pts |= {L - p for p in pts}
    return sorted(pts)


class CandidateLP:
    """Best randomized TIE mechanism for two agents with reports on ``grid`` and outcomes in ``candidates``
    (max-cost objective), over symmetric mechanisms."""

    def __init__(self, candidates: Sequence, grid: Sequence):
        self.C = sorted(Frac(c) for c in candidates)
        self.G = sorted(set(Frac(g) for g in grid))
        self.K, self.m = len(self.C), len(self.G)
        L = self.C[0] + self.C[-1]
        self.reflect = sorted(L - c for c in self.C) == self.C
        if self.reflect and sorted(L - g for g in self.G) != self.G:
            raise ValueError("for a symmetric candidate set the grid must be closed under reflection")
        Cf = np.array([float(c) for c in self.C])
        Gf = np.array([float(g) for g in self.G])
        self.D = np.abs(Cf[None, :] - Gf[:, None])                    # D[a, k] = |c_k - g_a|
        # orbits of profiles (a, b) under agent swap (and reflection a -> m-1-a, c_k -> c_{K-1-k})
        self.orbit: Dict[Tuple[int, int], Tuple[int, bool]] = {}
        self.reps: List[Tuple[int, int]] = []
        for a in range(self.m):
            for b in range(self.m):
                if (a, b) in self.orbit:
                    continue
                oid = len(self.reps)
                self.reps.append((min(a, b), max(a, b)))
                images = [((a, b), False), ((b, a), False)]
                if self.reflect:
                    r = self.m - 1
                    images += [((r - a, r - b), True), ((r - b, r - a), True)]
                for prof, flip in images:
                    self.orbit.setdefault(prof, (oid, flip))
        self.nv = len(self.reps) * self.K + 1                          # lottery variables + ratio r
        self.rows: List[Tuple[int, int, int]] = []
        self.result = None
        self.converged = False

    def _var(self, a: int, b: int, k: int) -> int:
        oid, flip = self.orbit[(a, b)]
        return oid * self.K + (self.K - 1 - k if flip else k)

    def _solve_once(self):
        K, rows, cols, vals = self.K, [], [], []
        r = 0
        for a, a2, b in self.rows:   # agent truly at G[a] (other at G[b]) must not gain by reporting G[a2]
            for k in range(K):
                rows += [r, r]
                cols += [self._var(a, b, k), self._var(a2, b, k)]
                vals += [self.D[a, k], -self.D[a, k]]
            r += 1
        for oid, (a, b) in enumerate(self.reps):
            cost = np.maximum(self.D[a], self.D[b])
            for k in range(K):
                rows.append(r); cols.append(oid * K + k); vals.append(cost[k])
            rows.append(r); cols.append(self.nv - 1); vals.append(-cost.min())
            r += 1
        n = len(self.reps)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(r, self.nv)).tocsr()
        A_eq = coo_matrix((np.ones(n * K), (np.repeat(np.arange(n), K), np.arange(n * K))),
                          shape=(n, self.nv)).tocsr()
        c = np.zeros(self.nv); c[-1] = 1
        for method in ("highs", "highs-ipm"):          # the interior-point fallback rescues rare HiGHS errors
            res = linprog(c, A_ub=A_ub, b_ub=np.zeros(r), A_eq=A_eq, b_eq=np.ones(n),
                          bounds=[(0, 1)] * (self.nv - 1) + [(0, 100)], method=method)
            if res.status == 0:
                return res
        raise RuntimeError(res.message)

    def lotteries(self, x: np.ndarray) -> np.ndarray:
        """P[a, b, k]: probability of candidate k at profile (G[a], G[b])."""
        Q = x[:-1].reshape(len(self.reps), self.K)
        P = np.empty((self.m, self.m, self.K))
        for (a, b), (oid, flip) in self.orbit.items():
            P[a, b] = Q[oid][::-1] if flip else Q[oid]
        return P

    def _violations(self, x: np.ndarray, tol: float, per_profile: int):
        P = self.lotteries(x)
        new, worst = [], 0.0
        for b in range(self.m):
            costs = self.D @ P[:, b, :].T            # [a, a2]: cost of type G[a] when reporting G[a2]
            gain = np.diag(costs)[:, None] - costs
            worst = max(worst, float(gain.max()))
            for a in np.where(gain.max(axis=1) > tol)[0]:
                for a2 in np.argsort(-gain[a])[:per_profile]:
                    if gain[a, a2] > tol:
                        new.append((int(a), int(a2), int(b)))
        return new, worst

    def solve(self, reach: int = 2, generate: bool = True, tol: float = 1e-9, per_profile: int = 2,
              max_iter: int = 100, verbose: bool = False) -> float:
        """Optimal LP value. With ``generate=False`` only local misreports (and misreports to the candidates and
        the grid ends) are imposed: a further relaxation, so its value and certificate are still valid lower
        bounds, and on large grids it is much faster."""
        special = [self.G.index(c) for c in self.C if c in self.G] + [0, self.m - 1]
        rows = set()
        for b in range(self.m):
            for a in range(self.m):
                for d in range(1, reach + 1):
                    rows.update((a, a2, b) for a2 in (a - d, a + d) if 0 <= a2 < self.m)
                rows.update((a, a2, b) for a2 in special if a2 != a)
        t0 = time.time()
        for it in range(max_iter):
            self.rows = sorted(rows)
            res = self._solve_once()
            if verbose:
                print(f"    iteration {it}: {len(rows)} misreport rows, LP {res.fun:.6f}, "
                      f"{time.time() - t0:.0f}s", flush=True)
            self.result = res
            if not generate:
                self.converged = False
                return float(res.fun)
            new, worst = self._violations(res.x, tol, per_profile)
            if not new:
                self.converged = True
                return float(res.fun)
            rows.update(new)
        raise RuntimeError("constraint generation did not converge")

    def certificate(self, denominator: int = 10 ** 9) -> dict:
        """Rationalised dual multipliers (as exact integers over ``denominator``), with grid coordinates."""
        if self.result is None:
            raise RuntimeError("call solve() first")
        duals = -self.result.ineqlin.marginals
        n_sp = len(self.rows)
        s = lambda q: str(q)
        sp, ratio = [], []
        for j, (a, a2, b) in enumerate(self.rows):
            w = int(round(max(duals[j], 0.0) * denominator))
            if w > 0:
                sp.append([s(self.G[a]), s(self.G[b]), s(self.G[a2]), w])
        for oid, (a, b) in enumerate(self.reps):
            w = int(round(max(duals[n_sp + oid], 0.0) * denominator))
            if w > 0:
                ratio.append([s(self.G[a]), s(self.G[b]), w])
        cert = {
            "setting": "two agents on a line, facility at a candidate location, expected maximum cost, "
                       "truthful in expectation",
            "candidates": [s(c) for c in self.C],
            "symmetry": ["swap", "reflection"] if self.reflect else ["swap"],
            "denominator": denominator,
            "grid_size": self.m,
            "lp_value": float(self.result.fun),
            "all_misreports_imposed": bool(self.converged),
            "sp": sp,
            "ratio": ratio,
        }
        bound = verify_certificate(cert)
        cert["bound"] = f"{bound.numerator}/{bound.denominator}"
        cert["bound_float"] = float(bound)
        return cert


def verify_certificate(cert: dict) -> Fraction:
    """Exact lower bound implied by a certificate (see the module docstring); independent of the LP.

    Each ``sp`` entry ``[x, y, x2, w]`` is the constraint "an agent at x, with the other agent at y, does not
    gain by reporting x2" with multiplier w / denominator; each ``ratio`` entry ``[x, y, w]`` is the ratio
    constraint at profile (x, y)."""
    C = [Frac(c) for c in cert["candidates"]]
    K, L = len(C), C[0] + C[-1]
    reflect = "reflection" in cert["symmetry"]
    if reflect and sorted(L - c for c in C) != C:
        raise ValueError("reflection symmetry needs a symmetric candidate set")
    den = int(cert["denominator"])

    def orbit(x, y):
        a = tuple(sorted((x, y)))
        if not reflect:
            return a, False
        r = tuple(sorted((L - x, L - y)))
        return (a, False) if a <= r else (r, True)

    coef: Dict[tuple, List[Fraction]] = defaultdict(lambda: [Frac(0)] * K)

    def add(x, y, vec):
        rep, flip = orbit(x, y)
        tgt = coef[rep]
        for k in range(K):
            tgt[K - 1 - k if flip else k] += vec[k]

    for x, y, x2, w in cert["sp"]:
        x, y, x2, w = Frac(x), Frac(y), Frac(x2), Frac(int(w), den)
        if w < 0:
            raise ValueError("multipliers must be nonnegative")
        d = [abs(c - x) for c in C]
        add(x, y, [w * dk for dk in d])
        add(x2, y, [-w * dk for dk in d])
    S = Frac(0)
    for x, y, w in cert["ratio"]:
        x, y, w = Frac(x), Frac(y), Frac(int(w), den)
        if w < 0:
            raise ValueError("multipliers must be nonnegative")
        cost = [max(abs(c - x), abs(c - y)) for c in C]
        add(x, y, [w * ck for ck in cost])
        S += w * min(cost)
    if S <= 0:
        raise ValueError("certificate has no positive ratio multiplier")
    total = Frac(0)
    for rep, vec in coef.items():
        cost = [max(abs(c - rep[0]), abs(c - rep[1])) for c in C]
        # both agents on one candidate: any finite-ratio mechanism must choose it
        total += vec[cost.index(0)] if min(cost) == 0 else min(vec)
    return total / S


def save_certificate(cert: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cert, f, indent=0)


def load_certificate(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
