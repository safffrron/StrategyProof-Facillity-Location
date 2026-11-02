"""Tests for predictions with double-peaked preferences (Proposition 12, Theorems 13-14 of the report)."""

import unittest

import numpy as np

from src.double_peaked import (M1, Mechanism, PredMed, PredMinMax, Window, check_tie, dp_cost,
                               expected_objective, optimum)

B, C = 1.0, 1.0
GRID = np.arange(-2, 2.001, 0.25)
PREDS = np.arange(-3, 3.001, 0.5)


class Clamp(Mechanism):
    name = "clamp"
    deterministic = True

    def lottery(self, X, p, b):
        return np.clip(p, X[:, 0] - b, X[:, -1] + b)[:, None], np.ones((len(X), 1))


class TestDoublePeaked(unittest.TestCase):

    def test_cost_has_two_peaks(self):
        self.assertAlmostEqual(float(dp_cost(-1.0, 0.0, B, C)), C)
        self.assertAlmostEqual(float(dp_cost(1.0, 0.0, B, C)), C)
        self.assertAlmostEqual(float(dp_cost(0.0, 0.0, B, C)), C + B)        # the valley at the agent's location

    def test_window_mechanisms_are_truthful_in_expectation(self):
        for n in (2, 3):
            for mech in (M1(), PredMed(), PredMinMax(n), Window(1)):
                gain, _, ex = check_tie(mech, n, GRID, PREDS, B, C)
                self.assertLess(gain, 1e-9, f"{mech.name}, n={n}: {ex}")

    def test_window_truthful_for_other_peak_distances(self):
        for b, c in ((0.5, 1.0), (1.5, 0.5)):
            gain, _, ex = check_tie(PredMinMax(3), 3, GRID, PREDS, b, c)
            self.assertLess(gain, 1e-9, ex)

    def test_literal_minmaxp_analogue_is_manipulable(self):
        gain, _, ex = check_tie(Clamp(), 2, GRID, PREDS, B, C)
        self.assertGreater(gain, 0.5)

    def test_consistency_one(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            X = np.sort(rng.uniform(-3, 3, 3))[None, :]
            for kind, mech in (("sc", PredMed()), ("mc", PredMinMax(3))):
                opt, o = optimum(X, B, C, kind)
                self.assertAlmostEqual(expected_objective(mech, X, o[0], B, C, kind)[0], opt[0], places=9)

    def test_tight_robustness_instances(self):
        # social cost: all agents at one point, prediction at their common valley -> 1 + b/c
        X = np.zeros((1, 3))
        self.assertAlmostEqual(expected_objective(PredMed(), X, 0.0, B, C, "sc")[0] / (3 * C), 1 + B / C)
        # max cost: agents 2b apart, prediction at the left agent's left peak -> 1 + 2b/c
        X = np.array([[0.0, 2 * B]])
        self.assertAlmostEqual(expected_objective(PredMinMax(2), X, -B, B, C, "mc")[0] / C, 1 + 2 * B / C)

    def test_robustness_bounds_hold_on_random_instances(self):
        rng = np.random.default_rng(1)
        for b, c in ((0.5, 1.0), (1.0, 1.0), (2.0, 1.0)):
            for _ in range(300):
                X = np.sort(rng.uniform(-4, 4, 3))[None, :]
                for p in rng.uniform(-6, 6, 4):
                    sc = expected_objective(PredMed(), X, p, b, c, "sc")[0] / optimum(X, b, c, "sc")[0][0]
                    mc = expected_objective(PredMinMax(3), X, p, b, c, "mc")[0] / optimum(X, b, c, "mc")[0][0]
                    self.assertLessEqual(sc, 1 + b / c + 1e-9)
                    self.assertLessEqual(mc, max(1 + 2 * b / c, 2) + 1e-9)


if __name__ == "__main__":
    unittest.main()
