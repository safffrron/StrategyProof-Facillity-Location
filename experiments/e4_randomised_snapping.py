"""E4 - Deterministic versus randomised snapping (report: Section 4.3, Theorems 3-5 and 9-10, Figure 2).

Adversarial search for the worst max-cost ratio, with blocked zones:
(a) point predictions with error eta, which may lie in a blocked zone: MinMaxP* against randomised
    snapping, whose proved bounds are min{1 + 2 eta, 3} and min{1 + eta, 3};
(b) interval predictions of normalised width omega that contain the optimum: the midpoint rule
    against randomised snapping of the midpoint, with bounds min{1 + omega, 3} and min{1 + omega/2, 3}.

Outputs: results/e4_randomised_snapping.md, figures/e4_randomised_snapping.{pdf,png}
"""

from __future__ import annotations
import os

import numpy as np
import matplotlib.pyplot as plt

from common import INK, MUTED, RES_DIR, SERIES, apply_style, save, write_markdown_table
from src.experiment_utils import Scenario, hill_climb
from src.learning_augmented import CenterMinMaxP, MinMaxP, RandomizedPredictionSnap

ETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
OMEGAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
BLUE, ORANGE = SERIES[0], SERIES[1]


def worst(mech, mode, eta=0.0, omega=0.0, ns=(2, 3, 5)):
    """Largest ratio found over n agents and two blocked zones (a lower bound on the worst case)."""
    return max(hill_climb(Scenario(mech, "mc", n, 2, mode, eta=eta, omega=omega), restarts=512, steps=600,
                          seed=n).best_ratio for n in ns)


def figure(point, interval):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    panels = [(axes[0], point, 2.0, 2.0, "(a) Point prediction that may be blocked", "prediction error η",
               "min{1 + 2η, 3}", "min{1 + η, 3}", (0.55, 2.35), (1.0, 1.72)),
              (axes[1], interval, 3.0, 1.0, "(b) Interval prediction, optimum inside", "interval width ω",
               "min{1 + ω, 3}", "1 + ω/2", (1.05, 2.35), (1.9, 1.7))]
    for ax, rows, xmax, slope, title, xlabel, det_label, rnd_label, det_xy, rnd_xy in panels:
        x = np.linspace(0, xmax, 400)
        det, rnd = np.minimum(1 + slope * x, 3), np.minimum(1 + slope * x / 2, 3)
        ax.fill_between(x, rnd, det, color=ORANGE, alpha=0.1, lw=0, zorder=1)
        ax.plot(x, det, color=ORANGE, lw=1.8, zorder=2, label="deterministic: best possible")
        ax.plot(x, rnd, color=BLUE, lw=1.8, zorder=2, label="randomised snapping: best possible")
        ax.plot([r["x"] for r in rows], [r["deterministic"] for r in rows], "s", color=ORANGE, ms=5, mec="white",
                zorder=3)
        ax.plot([r["x"] for r in rows], [r["randomised"] for r in rows], "o", color=BLUE, ms=5, mec="white",
                zorder=3, label="worst case found by search")
        ax.text(*det_xy, det_label, ha="right", va="center", fontsize=7.6, color=INK)
        ax.text(*rnd_xy, rnd_label, ha="left", va="center", fontsize=7.6, color=INK)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0.95, 3.15)
    axes[0].annotate("the gap randomisation removes", (1.15, 2.5), xytext=(0.05, 2.88), ha="left", va="center",
                     fontsize=7.2, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    axes[0].set_ylabel("worst-case ratio (max cost)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
    return save(fig, "e4_randomised_snapping")


def main() -> None:
    point = [{"x": e, "deterministic": worst(MinMaxP(), "error", eta=e),
              "randomised": worst(RandomizedPredictionSnap(MinMaxP()), "error", eta=e)} for e in ETAS]
    interval = [{"x": w, "deterministic": worst(CenterMinMaxP(), "interval", omega=w),
                 "randomised": worst(RandomizedPredictionSnap(MinMaxP(), use_center=True), "interval", omega=w)}
                for w in OMEGAS]
    table = []
    for kind, rows, slope in (("point, error eta", point, 2.0), ("interval, width omega", interval, 1.0)):
        for r in rows:
            table.append({"prediction": kind, "eta or omega": r["x"],
                          "deterministic (worst found)": f"{r['deterministic']:.4f}",
                          "randomised (worst found)": f"{r['randomised']:.4f}",
                          "proved bounds (det / rand)": f"{min(1 + slope * r['x'], 3):g} / "
                                                        f"{min(1 + slope * r['x'] / 2, 3):g}"})
    write_markdown_table(table, list(table[0]), os.path.join(RES_DIR, "e4_randomised_snapping.md"),
                         header="E4 - Worst max-cost ratio found by adversarial search (two, three and five agents, "
                                "two blocked zones), against the proved bounds.")
    print("  figure:", figure(point, interval))
    for r in table:
        print("  ", r)
    for r in point:
        assert abs(r["deterministic"] - min(1 + 2 * r["x"], 3)) < 2e-3 and abs(r["randomised"] - min(1 + r["x"], 3)) < 2e-3
    for r in interval:
        assert abs(r["deterministic"] - min(1 + r["x"], 3)) < 2e-3
        assert abs(r["randomised"] - min(1 + r["x"] / 2, 3)) < 2e-3


if __name__ == "__main__":
    main()
