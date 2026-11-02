"""Tests for the computer-assisted lower bound with candidate sites (Theorem 11 of the report)."""

import glob
import os
import unittest
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from src.candidate_lower_bound import CandidateLP, refined_grid, verify_certificate, load_certificate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def full_lp_value(C, G):
    """Best ratio over ALL (not necessarily symmetric) TIE grid mechanisms, with every constraint."""
    C, G = np.asarray(C, float), np.asarray(G, float)
    K, m = len(C), len(G)
    D = np.abs(C[None, :] - G[:, None])
    nv = m * m * K + 1
    idx = lambda a, b, k: (a * m + b) * K + k
    rows, cols, vals, r = [], [], [], 0
    for a in range(m):
        for a2 in range(m):
            for b in range(m):
                if a2 == a:
                    continue
                for k in range(K):   # agent 1 at G[a] reporting G[a2]; agent 2 at G[a] reporting G[a2]
                    rows += [r, r, r + 1, r + 1]
                    cols += [idx(a, b, k), idx(a2, b, k), idx(b, a, k), idx(b, a2, k)]
                    vals += [D[a, k], -D[a, k], D[a, k], -D[a, k]]
                r += 2
    for a in range(m):
        for b in range(m):
            cost = np.maximum(D[a], D[b])
            for k in range(K):
                rows.append(r); cols.append(idx(a, b, k)); vals.append(cost[k])
            rows.append(r); cols.append(nv - 1); vals.append(-cost.min()); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    A_eq = coo_matrix((np.ones(m * m * K), (np.repeat(np.arange(m * m), K), np.arange(m * m * K))),
                      shape=(m * m, nv)).tocsr()
    c = np.zeros(nv); c[-1] = 1
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(r), A_eq=A_eq, b_eq=np.ones(m * m),
                  bounds=[(0, 1)] * (nv - 1) + [(0, 100)], method="highs")
    return res.fun


class TestCandidateLowerBound(unittest.TestCase):

    def test_two_candidates_recover_tang_et_al(self):
        lp = CandidateLP([0, 2], refined_grid([0, 2]))
        lp.solve()
        cert = lp.certificate()
        self.assertGreater(cert["bound_float"], 1.99)            # their bound 2 is the eps -> 0 limit
        self.assertLessEqual(len(cert["sp"]), 4)                  # a three-misreport argument, like theirs

    def test_three_candidates_exceed_two(self):
        lp = CandidateLP([0, 2, 4], refined_grid([0, 2, 4]))
        value = lp.solve()
        cert = lp.certificate()
        bound = verify_certificate(cert)
        self.assertGreater(bound, Fraction(208, 100))
        self.assertLessEqual(float(bound), value + 1e-9)          # weak duality

    def test_symmetrisation_loses_nothing(self):
        C = [0, 2, 4]
        G = refined_grid(C, offsets=("1/10",), far="1")
        lp = CandidateLP(C, G)
        self.assertAlmostEqual(lp.solve(), full_lp_value(C, [float(g) for g in G]), places=6)

    def test_verifier_is_not_a_rubber_stamp(self):
        lp = CandidateLP([0, 2, 4], refined_grid([0, 2, 4]))
        lp.solve()
        cert = lp.certificate()
        weaker = dict(cert, sp=cert["sp"][::2])                  # dropping multipliers weakens the bound
        self.assertLess(verify_certificate(weaker), verify_certificate(cert))
        with self.assertRaises(ValueError):
            verify_certificate(dict(cert, sp=[[*cert["sp"][0][:3], -1]] + cert["sp"][1:]))

    def test_stored_certificates_verify(self):
        paths = sorted(glob.glob(os.path.join(ROOT, "results", "certificates", "*.json")))
        self.assertTrue(paths, "run experiments/e6_candidate_lower_bound.py first")
        for path in paths:
            cert = load_certificate(path)
            bound = verify_certificate(cert)
            self.assertEqual(f"{bound.numerator}/{bound.denominator}", cert["bound"], path)
            self.assertGreater(bound, 2 if len(cert["candidates"]) > 2 else Fraction(199, 100), path)


if __name__ == "__main__":
    unittest.main()
