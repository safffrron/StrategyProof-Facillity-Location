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


def _interior_point_primal(c, A_ub, A_eq, col_upper, disp: bool, crossover: bool = False):
    """Interior point method of HiGHS (needs the ``highspy`` package). Without crossover, returns the primal
    solution; with crossover, returns a scipy-like result with vertex duals."""
    import highspy
    from scipy.sparse import vstack
    A = vstack([A_ub, A_eq]).tocsc()
    n_ub, n_eq = A_ub.shape[0], A_eq.shape[0]
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = np.asarray(c, dtype=float)
    lp.col_lower_ = np.zeros(A.shape[1])
    lp.col_upper_ = np.asarray(col_upper, dtype=float)
    lp.row_lower_ = np.concatenate([np.full(n_ub, -highspy.kHighsInf), np.ones(n_eq)])
    lp.row_upper_ = np.concatenate([np.zeros(n_ub), np.ones(n_eq)])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(disp))
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "on" if crossover else "off")
    h.passModel(lp)
    h.run()
    sol = h.getSolution()
    x = np.asarray(sol.col_value)
    if not crossover:
        return x, -np.asarray(sol.row_dual[:n_ub])
    status = h.getModelStatus()
    return type("IpmResult", (), {"fun": float(np.dot(c, x)), "x": x,
                                  "status": 0 if status == highspy.HighsModelStatus.kOptimal else 1,
                                  "message": h.modelStatusToString(status),
                                  "ineqlin": type("M", (), {"marginals": np.asarray(sol.row_dual[:n_ub])})()})()


def _dual_simplex_multipliers(c, A_ub, A_eq, disp: bool):
    """Solve the LP dual directly by simplex (``highspy``): max 1'w s.t. -A_ub'y + A_eq'w <= c, y >= 0, w free.
    Its basis has one row per primal column, far fewer than the primal's rows, and an optimal y is exactly a set
    of certificate multipliers. Returns a scipy-like result (marginals = -y)."""
    import highspy
    from scipy.sparse import hstack
    n_ub, n_eq = A_ub.shape[0], A_eq.shape[0]
    M = hstack([-A_ub.T, A_eq.T]).tocsc()          # rows: primal columns; columns: (y, w)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = M.shape[1], M.shape[0]
    lp.col_cost_ = np.concatenate([np.zeros(n_ub), -np.ones(n_eq)])          # minimise -1'w
    lp.col_lower_ = np.concatenate([np.zeros(n_ub), np.full(n_eq, -highspy.kHighsInf)])
    lp.col_upper_ = np.full(n_ub + n_eq, highspy.kHighsInf)
    lp.row_lower_ = np.full(M.shape[0], -highspy.kHighsInf)
    lp.row_upper_ = np.asarray(c, dtype=float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr
    lp.a_matrix_.index_ = M.indices
    lp.a_matrix_.value_ = M.data
    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(disp))
    h.setOptionValue("solver", "simplex")
    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    sol = h.getSolution()
    y = np.asarray(sol.col_value[:n_ub])
    w = np.asarray(sol.col_value[n_ub:])
    ok = status == highspy.HighsModelStatus.kOptimal
    return type("DualResult", (), {"fun": float(w.sum()), "x": None, "status": 0 if ok else 1,
                                   "message": h.modelStatusToString(status),
                                   "ineqlin": type("M", (), {"marginals": -y})()})()


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

    def _matrices(self):
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
        return c, A_ub, A_eq

    def _solve_once(self, method: str = "highs", disp: bool = False):
        c, A_ub, A_eq = self._matrices()
        if method == "highspy-ipm":                    # newer HiGHS: interior point plus crossover
            res = _interior_point_primal(c, A_ub, A_eq, [1.0] * (self.nv - 1) + [100.0], disp, crossover=True)
            if res.status != 0:
                raise RuntimeError(res.message)
            return res
        r, n = A_ub.shape[0], A_eq.shape[0]
        for m in dict.fromkeys((method, "highs-ipm")):  # the interior-point fallback rescues rare HiGHS errors
            res = linprog(c, A_ub=A_ub, b_ub=np.zeros(r), A_eq=A_eq, b_eq=np.ones(n),
                          bounds=[(0, 1)] * (self.nv - 1) + [(0, 100)], method=m, options={"disp": disp})
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
              max_iter: int = 100, verbose: bool = False, method: str = "highs", disp: bool = False,
              polish_tol: float = 1e-6, polish_dual: float = 0.0) -> float:
        """Optimal LP value. With ``generate=False`` only local misreports (and misreports to the candidates and
        the grid ends) are imposed: a further relaxation, so its value and certificate are still valid lower
        bounds, and on large grids it is much faster. ``method="highs-ipm"`` (interior point) is usually faster on
        large grids; ``"ipm-polish"`` (needs the ``highspy`` package) runs the interior point method without its slow
        crossover, then solves the dual of the LP restricted to the tight misreport rows by simplex (with
        ``polish_dual > 0``, only rows whose interior-point dual exceeds that fraction of the largest are kept); ``disp=True`` prints the solver's log."""
        special = [self.G.index(c) for c in self.C if c in self.G] + [0, self.m - 1]
        rows = set()
        for b in range(self.m):
            for a in range(self.m):
                for d in range(1, reach + 1):
                    rows.update((a, a2, b) for a2 in (a - d, a + d) if 0 <= a2 < self.m)
                rows.update((a, a2, b) for a2 in special if a2 != a)
        t0 = time.time()
        if method == "ipm-polish":
            # Interior point (no crossover) on all rows, then keep only the misreport rows that are tight at its
            # solution and re-solve by simplex for exact vertex duals. Dropping rows is a relaxation, so the value
            # is still a valid lower bound; with every tight row kept it equals the full optimum.
            self.rows = sorted(rows)
            c, A_ub, A_eq = self._matrices()
            x, y = _interior_point_primal(c, A_ub, A_eq, [1.0] * (self.nv - 1) + [100.0], disp)
            slack = -(A_ub[:len(self.rows)] @ x)
            y = y[:len(self.rows)]
            weight = y > polish_dual * max(float(y.max()), 1e-300) if polish_dual > 0 else np.ones(len(y), bool)
            keep = [row for row, sl, ok in zip(self.rows, slack, weight) if sl <= polish_tol and ok]
            if verbose:
                print(f"    interior point: LP {x[-1]:.6f}, {len(keep)} of {len(self.rows)} misreport rows tight, "
                      f"{time.time() - t0:.0f}s", flush=True)
            self.rows = keep
            c, A_ub, A_eq = self._matrices()
            self.result = _dual_simplex_multipliers(c, A_ub, A_eq, disp)
            if self.result.status != 0:
                raise RuntimeError(self.result.message)
            self.converged = False
            if verbose:
                print(f"    simplex on tight rows: LP {self.result.fun:.6f}, {time.time() - t0:.0f}s", flush=True)
            return float(self.result.fun)
        for it in range(max_iter):
            self.rows = sorted(rows)
            res = self._solve_once(method, disp)
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
