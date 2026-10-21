"""E13 - Predictions with double-peaked preferences (Theorems 20-22).

Agents have two peaks, x - b and x + b, and pay c + distance to the nearer one (Filos-Ratsikas et al. 2017).
The literal MinMaxP analogue is manipulable; the "window" mechanisms (trust the prediction within b of a generalised
median, otherwise flip a fair coin between the median's two peaks) are truthful in expectation.

* A: exhaustive truthfulness checks (every profile, agent, misreport and prediction on a grid).
* B: worst-case consistency and robustness against the theory, for several b/c and n.
* C: figure - (a) the max-cost trade-off and lower bounds; (b) ratio against prediction error.
* D: certified lower bounds for TIE mechanisms without predictions: social cost, three agents (Proposition 23);
  max cost, two agents (Proposition 24). ``--full`` re-solves the large LPs (about 15 minutes); otherwise stored
  certificates are re-verified.

Outputs: results/e13_tie.md, results/e13_performance.md, results/e13_lower_bounds.md,
results/e13_certificates/*.json, figures/e13_double_peaked.{png,pdf}
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from common import INK, INK_2, MUTED, SERIES, MARKERS, RES_DIR, apply_style, save, theory_line, write_markdown_table
from src.double_peaked import (M1, M2, Mechanism, PredMed, PredMinMax, Window, candidate_points, check_tie,
                               expected_objective, objective, optimum)
from src.double_peaked_lp import DPLowerBound, verify_dp_certificate

B, C = 1.0, 1.0
GRID = np.arange(-3, 3.001, 0.25)
PREDS = np.arange(-4, 4.001, 0.5)


class ClampAnalogue(Mechanism):
    """The literal MinMaxP analogue: clamp the prediction to [x_1 - b, x_n + b] (deterministic)."""
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
        mechs = [M1(), M2(), PredMed(), PredMinMax(n), Window(1, "Window[1]"), ClampAnalogue(), NearestMedianPeak()]
        for mech in mechs:
            t0 = time.time()
            gain, checked, ex = check_tie(mech, n, GRID, PREDS, B, C)
            rows.append({"n": n, "mechanism": mech.name, "deviations checked": f"{checked:,}",
                         "max gain": f"{gain:.4f}", "truthful": "yes" if gain < 1e-9 else "NO",
                         "example": "" if ex is None else
                         f"p={ex['prediction']:g}, peaks {tuple(round(v, 2) for v in ex['profile'])}, "
                         f"agent {ex['agent'] + 1} reports {ex['misreport']:g}"})
            print(f"  A n={n} {mech.name}: max gain {gain:.4f} ({time.time() - t0:.1f}s)")
    write_markdown_table(rows, list(rows[0]), os.path.join(RES_DIR, "e13_tie.md"),
                         header=f"E13A - Exhaustive truthfulness-in-expectation checks, double-peaked costs (b = c = 1), "
                                f"peaks and misreports on a 0.25-grid of [-3, 3], predictions on a 0.5-grid of [-4, 4].")


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
    """The instances behind the tightness claims (Theorems 21 and 22)."""
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
    write_markdown_table(rows, list(rows[0]), os.path.join(RES_DIR, "e13_performance.md"),
                         header="E13B - Worst consistency and robustness found by search (1,500 random profiles each, "
                                "with every optimal location as a correct prediction and peaks, valleys and random points "
                                "as wrong ones), against Theorems 21-22. Search values are lower bounds on the true worst case.")


def part_c():
    apply_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    # (a) max-cost trade-off: mixtures of PredMinMax (probability q) and M1 are truthful in expectation
    for col, mk, (b, c) in zip((SERIES[0], SERIES[1]), MARKERS, ((0.5, 1.0), (1.0, 1.0))):
        m1, pm = max(1 + b / c, 2), max(1 + 2 * b / c, 2)
        q = np.linspace(0, 1, 11)
        ax.plot(q + (1 - q) * m1, q * pm + (1 - q) * m1, color=col, marker=mk, markersize=3.5,
                label=f"b/c = {b / c:g}")
    ax.annotate("PredMinMax (1, 3)", (1, 3), xytext=(6, 2), textcoords="offset points", fontsize=7.5, color=INK)
    ax.annotate("PredMinMax (1, 2)", (1, 2), xytext=(6, 5), textcoords="offset points", fontsize=7.5, color=INK)
    ax.annotate("M1 (2, 2)", (2, 2), xytext=(0, -14), textcoords="offset points", fontsize=7.5, color=INK, ha="center")
    ax.plot([1], [2], marker="v", markersize=8, color=INK, linestyle="none", zorder=5)
    ax.annotate("lower bound for 1-consistent\nmechanisms: 2 (Thm 22)", (1, 2), xytext=(6, -26),
                textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_xlim(0.9, 2.15); ax.set_ylim(1.45, 3.25)
    ax.set_xlabel("consistency"); ax.set_ylabel("robustness")
    ax.set_title("(a) Max cost: mixtures of PredMinMax and M1", fontsize=9)
    ax.legend(loc="upper right", fontsize=7.5)
    # (b) worst ratio against prediction error, predictions inside the window, b = c = 1, n = 5
    rng = np.random.default_rng(3)
    edges = np.linspace(0, 1.5, 16)
    n = 5
    for col, mk, mech, kind, label in ((SERIES[0], MARKERS[0], PredMinMax(n), "mc", "PredMinMax, max cost"),
                                       (SERIES[1], MARKERS[1], PredMed(), "sc", "PredMed, social cost")):
        worst = np.ones(len(edges) - 1)
        for t in range(2500):
            X = np.sort(rng.uniform(-4, 4, n))[None, :]
            opt, o = optimum(X, B, C, kind)
            scale = opt[0] if kind == "mc" else opt[0] / n
            for eta in rng.uniform(0, 1.5, 6):
                for sgn in (-1, 1):
                    p = o[0] + sgn * eta * scale
                    locs, probs = mech.lottery(X, p, B)
                    if probs[0, 0] < 1:                       # outside the window: the fallback lottery
                        continue
                    r = expected_objective(mech, X, p, B, C, kind)[0] / opt[0]
                    k = int(np.clip(np.searchsorted(edges, eta) - 1, 0, len(worst) - 1))
                    worst[k] = max(worst[k], r)
        bx.plot(edges[1:], worst, color=col, marker=mk, markersize=4, label=label)
    e = np.linspace(0, 1.5, 50)
    theory_line(bx, e, 1 + e, "bound 1 + η (Thms 21-22)", text_xy=(0.55, 1.36), ha="left")
    bx.set_xlabel("prediction error η (bin upper edge)"); bx.set_ylabel("worst ratio found")
    bx.set_title("(b) Error dependence inside the window (b = c = 1, n = 5)", fontsize=9)
    bx.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "e13_double_peaked")


CERT_DIR = os.path.join(RES_DIR, "e13_certificates")
# name, agents, grid half-width K, grid step h, objective, a-priori ratio cap U, quick?
LB_CONFIGS = [("sc_n3_coarse", 3, 4, "1/2", "sc", "2", True), ("sc_n3_fine", 3, 8, "1/4", "sc", "2", False),
              ("mc_n2_wide", 2, 24, "1", "mc", "3", False)]


def part_d(full: bool = False):
    os.makedirs(CERT_DIR, exist_ok=True)
    for name, n, K, h, kind, U, quick in LB_CONFIGS:
        path = os.path.join(CERT_DIR, f"{name}.json")
        if quick or full or not os.path.exists(path):
            lp = DPLowerBound(n, K, h=h, b="1", c="1", objective=kind)
            lp.solve()
            cert = lp.certificate(U=U)
            cert["name"] = name
            with open(path, "w") as fh:
                json.dump(cert, fh, indent=0)
    rows = []
    for path in sorted(glob.glob(os.path.join(CERT_DIR, "*.json"))):
        cert = json.load(open(path))
        bound = verify_dp_certificate(cert)
        ok = f"{bound.numerator}/{bound.denominator}" == cert["bound"]
        rows.append({"objective": {"sc": "social cost", "mc": "max cost"}[cert["objective"]], "agents": cert["n"], "grid": f"[-{float(cert['K']) * float(eval(cert['h'])):g}, "
                                                   f"{float(cert['K']) * float(eval(cert['h'])):g}], step {cert['h']}",
                     "LP value": f"{cert['lp_value']:.6f}", "certified bound": f"{int(bound * 10000) / 10000:.4f}",
                     "exact": cert["bound"] if len(cert["bound"]) < 16 else "(long fraction)",
                     "multipliers": f"{len(cert['tie'])} + {len(cert['ratio'])}", "verified": "yes" if ok else "NO",
                     "certificate": os.path.basename(path)})
        print(f"  D {os.path.basename(path)}: bound >= {float(bound):.6f} (exact match: {ok})")
    write_markdown_table(rows, list(rows[0]), os.path.join(RES_DIR, "e13_lower_bounds.md"),
                         header="E13D - Double-peaked preferences with b = c = 1, no prediction: lower bounds on the ratio of "
                                "every truthful-in-expectation mechanism (Propositions 23-24; Filos-Ratsikas et al. prove 3/2 "
                                "for max cost and nothing for social cost), re-verified in exact arithmetic from the stored "
                                "certificates.")


def main(full: bool = False):
    part_a()
    part_b()
    print("  figure:", part_c())
    part_d(full)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="re-solve the finer lower-bound LP")
    main(ap.parse_args().full)
