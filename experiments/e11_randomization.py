"""E11 - What randomization buys (Theorems 10, 11, 13; Fang et al.'s open question on randomization).

(a) Max cost, point predictions, blocked zones: deterministic MinMaxP* with arbitrary (possibly
    infeasible) predictions [1 + 2 eta], the same with feasible predictions [1 + eta, Fang et al. 2025],
    and randomized snapping with arbitrary predictions [1 + eta].
(b) Max cost, interval predictions containing the optimum: CenterMinMaxP* [1 + omega] versus
    randomized centre snapping [1 + omega/2].
(c) Social cost, n = 7: the exact deterministic frontier of PhantomMedian[k]* (Theorem 9) versus
    report-independent mixtures with the spike mechanism of Feldman, Fiat & Golomb (universally truthful).

Outputs: figures/e11_randomization.{png,pdf}, results/e11_randomization.csv
"""

from __future__ import annotations
import csv
import os

import numpy as np

import matplotlib.pyplot as plt
from common import INK_2, MARKERS, MUTED, RES_DIR, SERIES, apply_style, save, theory_line
from src.experiment_utils import Scenario, hill_climb
from src.learning_augmented import (
    CenterMinMaxP, MinMaxP, MixtureMechanism, PhantomMedian, RandomizedPredictionSnap, SpikeMechanism,
)

ETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
OMEGAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
N_SC = 7


def worst(mech, objective, mode, eta=0.0, omega=0.0, ns=(2, 3, 5)):
    return max(hill_climb(Scenario(mech, objective, n, 2, mode, eta=eta, omega=omega), restarts=512, steps=600,
                          seed=n).best_ratio for n in ns)


def main() -> None:
    apply_style()
    rows = []
    for e in ETAS:
        rows.append({"panel": "a", "x": e,
                     "det_any": worst(MinMaxP(), "mc", "error", e),
                     "det_feasible": worst(MinMaxP(), "mc", "error_feasible", e),
                     "rand_any": worst(RandomizedPredictionSnap(MinMaxP()), "mc", "error", e)})
    for om in OMEGAS:
        rows.append({"panel": "b", "x": om,
                     "det_center": worst(CenterMinMaxP(), "mc", "interval", omega=om),
                     "rand_center": worst(RandomizedPredictionSnap(MinMaxP(), use_center=True), "mc", "interval", omega=om)})
    frontier = []
    for k in (0, 2, 4, 6):
        frontier.append({"kind": "deterministic", "k": k, "q": 0.0,
                         "consistency": (3 * N_SC - 1 - k) / (N_SC + 1 + k), "robustness": (3 * N_SC + k - 1) / (N_SC - k + 1)})
    spike_r = worst(SpikeMechanism(), "sc", "none", ns=(N_SC,))
    frontier.append({"kind": "spike", "k": None, "q": 1.0, "consistency": spike_r, "robustness": spike_r})
    for k in (4, 6):
        for q in (0.25, 0.5, 0.75):
            m = MixtureMechanism(PhantomMedian(k), SpikeMechanism(), q)
            frontier.append({"kind": "mixture", "k": k, "q": q, "consistency": worst(m, "sc", "error", ns=(N_SC,)),
                             "robustness": worst(m, "sc", "robust", ns=(N_SC,))})
    with open(os.path.join(RES_DIR, "e11_randomization.csv"), "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES_DIR, "e11_sc_frontier.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frontier[0]))
        w.writeheader()
        w.writerows(frontier)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    ax = axes[0]
    pa = [r for r in rows if r["panel"] == "a"]
    g = np.linspace(0, max(ETAS), 200)
    theory_line(ax, g, np.minimum(1 + 2 * g, 3), "min(1 + 2η, 3)", text_xy=(0.02, 2.75))
    theory_line(ax, g, np.minimum(1 + g, 3), "min(1 + η, 3)", text_xy=(1.05, 1.75))
    ax.plot([r["x"] for r in pa], [r["det_any"] for r in pa], marker=MARKERS[0], color=SERIES[0], linestyle="none",
            label="deterministic, prediction may be in a zone")
    ax.plot([r["x"] for r in pa], [r["det_feasible"] for r in pa], marker=MARKERS[1], color=SERIES[1], linestyle="none",
            label="deterministic, feasible prediction (Fang et al.)")
    ax.plot([r["x"] for r in pa], [r["rand_any"] for r in pa], marker=MARKERS[2], color=SERIES[2], linestyle="none",
            markersize=4, label="randomized snapping, any prediction")
    ax.set_title("(a) Max cost: point prediction")
    ax.set_xlabel("prediction error η")
    ax.set_ylabel("worst-case (expected) ratio")
    ax.legend(loc="lower right", fontsize=6.5)
    ax = axes[1]
    pb = [r for r in rows if r["panel"] == "b"]
    g = np.linspace(0, max(OMEGAS), 200)
    theory_line(ax, g, np.minimum(1 + g, 3), "min(1 + ω, 3)", text_xy=(0.05, 2.6))
    theory_line(ax, g, np.minimum(1 + g / 2, 3), "1 + ω/2", text_xy=(1.6, 1.62))
    ax.plot([r["x"] for r in pb], [r["det_center"] for r in pb], marker=MARKERS[0], color=SERIES[0], linestyle="none",
            label="CenterMinMaxP* (deterministic)")
    ax.plot([r["x"] for r in pb], [r["rand_center"] for r in pb], marker=MARKERS[2], color=SERIES[2], linestyle="none",
            label="randomized centre snapping")
    ax.set_title("(b) Max cost: interval prediction")
    ax.set_xlabel("interval width ω (optimum inside)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax = axes[2]
    det = [f for f in frontier if f["kind"] == "deterministic"]
    ax.plot([f["robustness"] for f in det], [f["consistency"] for f in det], marker=MARKERS[0], color=SERIES[0],
            linewidth=1.0, label="deterministic PhantomMedian[k]* (exact)")
    for f in det:
        ax.annotate(f"k={f['k']}", (f["robustness"], f["consistency"]), xytext=(4, 3), textcoords="offset points",
                    fontsize=7, color=INK_2)
    for i, k in enumerate((4, 6)):
        mix = [f for f in frontier if f["kind"] == "mixture" and f["k"] == k]
        ax.plot([f["robustness"] for f in mix], [f["consistency"] for f in mix], marker=MARKERS[1 + i],
                color=SERIES[1 + i], linewidth=1.0, label=f"mix of k={k} with spike")
    sp = [f for f in frontier if f["kind"] == "spike"][0]
    ax.plot([sp["robustness"]], [sp["consistency"]], marker="*", markersize=10, color=SERIES[3], linestyle="none",
            label="spike (no prediction)")
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 15])
    ax.set_xticklabels(["2", "3", "5", "10", "15"])
    ax.minorticks_off()
    ax.set_title(f"(c) Social cost frontier, n = {N_SC}")
    ax.set_xlabel("robustness")
    ax.set_ylabel("consistency")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    print("E11 saved", save(fig, "e11_randomization"))
    for r in rows:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for f in frontier:
        print("  ", f)


if __name__ == "__main__":
    main()
