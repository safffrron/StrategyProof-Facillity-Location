"""E9 - Asymmetric single-peaked costs: restoring strategyproofness with favourite-feasible-point rules.

Agents pay alpha per unit when the facility is on their LEFT and 1 per unit when it is on their right
(alpha = 1 is the symmetric case). Snapped rules (Median*, MinMaxP*) are manipulable for alpha != 1
(Proposition 2). Rules that work on each agent's favourite feasible point ("tops") are strategyproof for
every single-peaked cost (Proposition 14). This experiment measures what they cost:

* max cost: Tops-MinMaxP (prediction-augmented; bounds 1-consistent, (2 + alpha)-robust with blocked
  zones) and Tops-Rightmost (no prediction; bound 2 + 1/alpha), for alpha >= 1;
* social cost: Tops-Median versus the alpha-aware quantile rule (the asymmetric "median").

Outputs: figures/e9_asymmetric.{png,pdf}, results/e9_asymmetric.csv, results/e9_asymmetric.md
"""

from __future__ import annotations
import csv
import os

import numpy as np

import matplotlib.pyplot as plt
from common import INK_2, MARKERS, RES_DIR, SERIES, apply_style, save, theory_line, write_markdown_table
from src.environment import Environment1D, Interval
from src.experiment_utils import Scenario, hill_climb
from src.fast_validator import check_strategyproofness
from src.learning_augmented import MedianRule, MinMaxP, TopsGeneralizedMedian, TopsOrderStatistic

ALPHAS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def worst(mech, objective, mode="none", alpha=1.0, ns=(3, 5, 7)) -> float:
    best = 0.0
    for n in ns:
        m = TopsGeneralizedMedian(n - 1, alpha) if mech == "tops_minmaxp" else mech
        sc = Scenario(m, objective, n, n_gaps=2, mode=mode, alpha=alpha)
        best = max(best, hill_climb(sc, restarts=384, steps=500, seed=n).best_ratio)
    return best


def main() -> None:
    apply_style()
    rows = []
    for a in ALPHAS:
        rows.append({
            "alpha": a,
            "tops_minmaxp_consistency": worst("tops_minmaxp", "mc", "error", a),
            "tops_minmaxp_robustness": worst("tops_minmaxp", "mc", "robust", a),
            "tops_rightmost": worst(TopsOrderStatistic(1.0, a), "mc", "none", a),
            "tops_median_sc": worst(TopsGeneralizedMedian(0, a), "sc", "none", a),
            "tops_quantile_sc": worst(TopsOrderStatistic(a / (1 + a), a), "sc", "none", a),
        })
        print("  ", {k: round(v, 3) for k, v in rows[-1].items()})
    with open(os.path.join(RES_DIR, "e9_asymmetric.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # strategyproofness under asymmetric costs (alpha = 3), two blocked zones
    env = Environment1D(length=1.0, num_grid_points=11, blocked_intervals=[Interval(0.15, 0.35), Interval(0.55, 0.8)])
    grid = np.linspace(0, 1, 21)
    preds = list(np.round(np.linspace(-0.2, 1.2, 15), 4))
    sp_rows = []
    for mech, pr in [(MedianRule(), [None]), (MinMaxP(), preds), (TopsGeneralizedMedian(0, 3.0), [None]),
                     (TopsGeneralizedMedian(2, 3.0), preds), (TopsOrderStatistic(0.75, 3.0), [None])]:
        rep = check_strategyproofness(mech, 3, grid, env, predictions=pr, alpha=3.0)
        sp_rows.append({"Mechanism": mech.name, "Strategyproof at alpha = 3?": "yes" if rep.is_strategyproof else "**no**",
                        "Counterexample": rep.counterexample.describe() if rep.counterexample else "-"})
    table = [{"alpha": f"{r['alpha']:g}",
              "Tops-MinMaxP consistency": f"{r['tops_minmaxp_consistency']:.3f}",
              "Tops-MinMaxP robustness (bound 2+alpha)": f"{r['tops_minmaxp_robustness']:.3f} ({2 + r['alpha']:.3f})",
              "Tops-Rightmost (bound 2+1/alpha)": f"{r['tops_rightmost']:.3f} ({2 + 1 / r['alpha']:.3f})",
              "Tops-Median SC": f"{r['tops_median_sc']:.3f}",
              "Tops-Quantile SC": f"{r['tops_quantile_sc']:.3f}"} for r in rows]
    path = os.path.join(RES_DIR, "e9_asymmetric.md")
    write_markdown_table(table, list(table[0].keys()), path,
                         header="Worst case found by adversarial search, two blocked zones, n <= 7, asymmetric costs "
                                "(facility on an agent's left costs alpha per unit)")
    with open(path, "a") as fh:
        fh.write("\n")
    write_markdown_table(sp_rows, list(sp_rows[0].keys()), path + ".sp.tmp",
                         header="Exhaustive strategyproofness check under asymmetric costs (alpha = 3, n = 3, two zones)")
    with open(path, "a") as fh, open(path + ".sp.tmp") as tmp:
        fh.write(tmp.read())
    os.remove(path + ".sp.tmp")

    a = np.array(ALPHAS)
    g = np.linspace(1, a.max(), 100)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax = axes[0]
    theory_line(ax, g, 2 + g, "2 + α", text_xy=(3.9, 6.25))
    theory_line(ax, g, 2 + 1 / g, "2 + 1/α", text_xy=(3.9, 2.55))
    ax.plot(a, [r["tops_minmaxp_robustness"] for r in rows], marker=MARKERS[0], color=SERIES[0], linestyle="none",
            label="Tops-MinMaxP, robustness")
    ax.plot(a, [r["tops_rightmost"] for r in rows], marker=MARKERS[1], color=SERIES[1], linestyle="none",
            label="Tops-Rightmost (no prediction)")
    ax.plot(a, [r["tops_minmaxp_consistency"] for r in rows], marker=MARKERS[2], color=SERIES[2], linestyle="none",
            label="Tops-MinMaxP, consistency")
    ax.set_title("(a) Max cost, blocked zones")
    ax.set_xlabel("cost asymmetry α")
    ax.set_ylabel("worst-case ratio")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(a, [r["tops_median_sc"] for r in rows], marker=MARKERS[0], color=SERIES[0], label="Tops-Median")
    ax.plot(a, [r["tops_quantile_sc"] for r in rows], marker=MARKERS[1], color=SERIES[1],
            label="Tops-Quantile (q = α/(1+α))")
    ax.set_title("(b) Social cost, blocked zones")
    ax.set_xlabel("cost asymmetry α")
    ax.legend(loc="upper left")
    fig.tight_layout()
    print("E9 saved", save(fig, "e9_asymmetric"))


if __name__ == "__main__":
    main()
