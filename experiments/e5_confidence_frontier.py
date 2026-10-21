"""E5 - Consistency/robustness trade-off of the confidence family PhantomMedian[k]* (n = 7).

PhantomMedian[k] = median of the n reports plus k copies of the point prediction; k = 0 is the
median, k = n - 1 is MinMaxP. For each k we estimate worst-case consistency (correct prediction)
and robustness (arbitrary prediction) by adversarial search, on the open line and with two
blocked zones, for both objectives. For reference, panel (a) also draws the Pareto frontier that
Agrawal et al. prove for the 2-D utilitarian problem (CMP): sqrt(2c^2+2)/(1+c) vs sqrt(2c^2+2)/(1-c).

Outputs: figures/e5_confidence_frontier.{png,pdf}, results/e5_frontier.csv.
"""

from __future__ import annotations
import csv
import os

import numpy as np

import matplotlib.pyplot as plt
from common import INK_2, MARKERS, MUTED, RES_DIR, SERIES, apply_style, save
from src.experiment_utils import Scenario, hill_climb
from src.learning_augmented import PhantomMedian

N = 7


def search(k: int, objective: str, gaps: int, mode: str) -> float:
    sc = Scenario(PhantomMedian(k, snap=gaps > 0), objective, N, n_gaps=gaps, mode=mode, eta=0.0)
    return max(hill_climb(sc, restarts=384, steps=600, seed=s).best_ratio for s in (0, 1))


def main() -> None:
    apply_style()
    rows = []
    for objective in ("sc", "mc"):
        for gaps in (0, 2):
            for k in range(N):
                rows.append({"objective": objective, "blocked": gaps > 0, "k": k, "confidence_c": round(k / N, 3),
                             "consistency": search(k, objective, gaps, "error"),
                             "robustness": search(k, objective, gaps, "robust")})
            print(f"  {objective} blocked={gaps > 0} done")
    with open(os.path.join(RES_DIR, "e5_frontier.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, objective, title in zip(axes, ("sc", "mc"), ("(a) Social cost", "(b) Max cost")):
        if objective == "sc":
            c = np.linspace(0, 0.85, 200)
            s = np.sqrt(2 * c ** 2 + 2)
            ax.plot(s / (1 - c), s / (1 + c), color=MUTED, linewidth=1.0, zorder=1)
            ax.text(1.75, 1.27, "Agrawal et al. 2-D\nCMP frontier (proved)", fontsize=7.5, color=MUTED, va="center")
            kk = np.linspace(0, N - 1, 200)
            ax.plot(1 + 2 * (N + kk - 1) / (N - kk + 1), (3 * N - 1 - kk) / (N + 1 + kk), color=INK_2,
                    linewidth=1.0, zorder=1)
            ax.text(6.4, 1.62, "conjectured frontier,\nblocked zones (n = 7)", fontsize=7.5, color=INK_2,
                    va="center")
        for i, blocked in enumerate((False, True)):
            sub = [r for r in rows if r["objective"] == objective and r["blocked"] == blocked]
            ax.plot([r["robustness"] for r in sub], [r["consistency"] for r in sub], color=SERIES[i],
                    marker=MARKERS[i], linestyle="none",
                    label="blocked zones (snapped)" if blocked else "open line", zorder=3)
            # label distinct points; coincident ks share one label
            groups = {}
            for r in sub:
                groups.setdefault((round(r["robustness"], 2), round(r["consistency"], 2)), []).append(r["k"])
            for (x, y), ks in groups.items():
                if objective == "sc" and not blocked and ks[0] not in (0, N - 2):
                    continue
                lab = f"k={ks[0]}" if len(ks) == 1 else f"k={ks[0]}–{ks[-1]}"
                ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_xscale("log")
        ticks = [1, 2, 3, 5, 10, 15] if objective == "sc" else [1, 2, 3, 4]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        if objective == "mc":
            ax.set_xlim(0.9, 4.2)
        ax.set_title(f"{title}: PhantomMedian[k], n = {N}")
        ax.set_xlabel("robustness (worst case, any prediction)")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("consistency (worst case, correct prediction)")
    fig.tight_layout()
    print("E5 saved", save(fig, "e5_confidence_frontier"))
    for r in rows:
        print(f"  {r['objective']} blocked={r['blocked']!s:<5} k={r['k']} cons={r['consistency']:.3f} rob={r['robustness']:.3f}")


if __name__ == "__main__":
    main()
