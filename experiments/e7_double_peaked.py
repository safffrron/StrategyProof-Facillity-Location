"""E7 - Predictions with double-peaked preferences (report: Section 4.5, Proposition 12, Theorems 13-14).

Agents have two peaks, x - b and x + b, and pay c + the distance to the nearer one (Filos-Ratsikas et al.
2017). Window rules trust the prediction when it lies within b of a generalised median g, and otherwise
pick g - b or g + b with probability 1/2 each.

(a) Exhaustive truthfulness-in-expectation checks (every profile, agent, misreport and prediction on a
    grid, n = 2..5): M1 and the window rules pass; the clamp rule of Proposition 12 and the "median's peak
    nearest the prediction" rule are manipulable.
(b) Worst consistency and robustness found by search, against Theorem 14.

Outputs: results/e7_truthfulness.md, results/e7_performance.md
"""

from __future__ import annotations
import os
import time

import numpy as np

from common import RES_DIR, write_markdown_table
from src.double_peaked import (M1, Mechanism, PredMed, PredMinMax, Window, candidate_points, check_tie,
                               expected_objective, objective, optimum)

B, C = 1.0, 1.0
GRID = np.arange(-3, 3.001, 0.25)
PREDS = np.arange(-4, 4.001, 0.5)


class ClampAnalogue(Mechanism):
    """The literal MinMaxP analogue of Proposition 12: clamp the prediction to [x_1 - b, x_n + b]."""
    name = "clamp(p, x_1 - b, x_n + b)"
    deterministic = True

    def lottery(self, X, p, b):
        return np.clip(p, X[:, 0] - b, X[:, -1] + b)[:, None], np.ones((len(X), 1))


class NearestMedianPeak(Mechanism):
    """The median agent's peak on the prediction's side (deterministic)."""
    name = "median's peak nearest the prediction"
    deterministic = True

    def lottery(self, X, p, b):
        g = X[:, (X.shape[1] - 1) // 2]
        return np.where(p <= g, g - b, g + b)[:, None], np.ones((len(X), 1))


def part_a():
    rows = []
    for n in (2, 3, 4, 5):
        mechs = [M1(), PredMed(), PredMinMax(n), Window(1, "Window[1]"), ClampAnalogue(), NearestMedianPeak()]
        for mech in mechs:
            t0 = time.time()
            gain, checked, ex = check_tie(mech, n, GRID, PREDS, B, C)
            rows.append({"n": n, "mechanism": mech.name, "deviations checked": f"{checked:,}",
                         "max gain": f"{gain:.4f}", "truthful": "yes" if gain < 1e-9 else "NO",
                         "example": "" if ex is None else
                         f"p={ex['prediction']:g}, peaks {tuple(round(v, 2) for v in ex['profile'])}, "
                         f"agent {ex['agent'] + 1} reports {ex['misreport']:g}"})
            print(f"  A n={n} {mech.name}: max gain {gain:.4f} ({time.time() - t0:.1f}s)")
    write_markdown_table(rows, list(rows[0]), os.path.join(RES_DIR, "e7_truthfulness.md"),
                         header="E7a - Exhaustive truthfulness-in-expectation checks, double-peaked costs (b = c = 1), "
                                "peaks and misreports on a 0.25-grid of [-3, 3], predictions on a 0.5-grid of [-4, 4].")
    total = sum(int(r["deviations checked"].replace(",", "")) for r in rows)
    print(f"  E7a: {total:,} deviations")
    expected = {"clamp(p, x_1 - b, x_n + b)": "NO", "median's peak nearest the prediction": "NO"}
    assert all(r["truthful"] == expected.get(r["mechanism"], "yes") for r in rows), "unexpected outcome"


def worst_case(mech, n, b, c, kind, rng, trials):
    cons = rob = 1.0
    for t in range(trials):
        base = rng.choice(np.arange(-6, 6.001, 0.25), n) if t % 2 else rng.uniform(-6, 6, n)
        X = np.sort(base)[None, :]
        opt, _ = optimum(X, b, c, kind)
        cand = candidate_points(X, b)
        vals = objective(cand, X, b, c, kind)[0]
        for p in cand[0][vals <= vals.min() + 1e-9]:                       # every optimal location
            cons = max(cons, expected_objective(mech, X, p, b, c, kind)[0] / opt[0])
        preds = np.concatenate([rng.uniform(-9, 9, 6), X[0] - b, X[0] + b, X[0]])
        for p in preds:
            rob = max(rob, expected_objective(mech, X, p, b, c, kind)[0] / opt[0])
    return cons, rob


def tight_instances(b, c):
    """The instances behind the tightness claims of Theorem 14."""
    sc = expected_objective(PredMed(), np.array([[0.0, 0.0, 0.0]]), 0.0, b, c, "sc")[0] / (3 * c)
    mc = expected_objective(PredMinMax(2), np.array([[0.0, 2 * b]]), -b, b, c, "mc")[0] / c
    return sc, mc


def part_b():
    rng = np.random.default_rng(0)
    rows = []
    for b, c in ((0.5, 1.0), (1.0, 1.0), (2.0, 1.0)):
        sc_t, mc_t = tight_instances(b, c)
        for n in (3, 5):
            cs, rs = worst_case(PredMed(), n, b, c, "sc", rng, 1500)
            cm, rm = worst_case(PredMinMax(n), n, b, c, "mc", rng, 1500)
            _, r1s = worst_case(M1(), n, b, c, "sc", rng, 600)
            _, r1m = worst_case(M1(), n, b, c, "mc", rng, 600)
            rows.append({"b/c": f"{b / c:g}", "n": n,
                         "PredMed SC (cons, rob)": f"{cs:.3f}, {rs:.3f}",
                         "theory": f"1, {1 + b / c:.3f} (tight instance: {sc_t:.3f})",
                         "PredMinMax MC (cons, rob)": f"{cm:.3f}, {rm:.3f}",
                         "theory ": f"1, {max(1 + 2 * b / c, 2):.3f} (tight instance: {mc_t:.3f})",
                         "M1 SC / MC (no prediction)": f"{r1s:.3f} / {r1m:.3f}"})
            print(f"  B b/c={b / c:g} n={n}: done")
    write_markdown_table(rows, list(rows[0]), os.path.join(RES_DIR, "e7_performance.md"),
                         header="E7b - Worst consistency and robustness found by search (1,500 random profiles each, "
                                "with every optimal location as a correct prediction and peaks, valleys and random points "
                                "as wrong ones), against Theorem 14. Search values are lower bounds on the true worst case.")


def main() -> None:
    part_a()
    part_b()


if __name__ == "__main__":
    main()
