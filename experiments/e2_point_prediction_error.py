"""E2 - Approximation ratio vs point-prediction error (max-cost objective).

(a) Worst case: adversarial search for MinMaxP on the open line and MinMaxP* with blocked zones,
    compared with the bounds 1 + min(eta, 1) (Agrawal et al.; Chan, Lin & Wang) and
    min(1 + 2 eta, 3) (Theorem 4 in docs/RESULTS.md).
(b) Typical case: mean ratio on random instances (uniform agents, random blocked zones generated
    with Environment1D.random_blocked) for several blocked densities.

Outputs: figures/e2_error_doubling.{png,pdf}, results/e2_worst_case.csv, results/e2_typical.csv.
"""

from __future__ import annotations
import csv
import os

import numpy as np

import matplotlib.pyplot as plt
from common import INK_2, MARKERS, RAMP, RES_DIR, SERIES, apply_style, save, theory_line
from src.experiment_utils import (
    Scenario, error_scale, hill_climb, objective_cost, optimum, point_predictions, ratio,
    sample_typical_instances,
)
from src.learning_augmented import MinMaxP

ETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
N_VALUES = (2, 3, 5)
DENSITIES = [0.0, 0.2, 0.4, 0.6]


def worst_case(eta: float, blocked: bool) -> float:
    mech = MinMaxP(snap=blocked)
    best = 0.0
    for n in N_VALUES:
        sc = Scenario(mech, "mc", n, n_gaps=2 if blocked else 0, mode="error", eta=eta)
        best = max(best, hill_climb(sc, restarts=384, steps=500, seed=n).best_ratio)
    return best


def typical(eta: float, density: float, rng) -> tuple:
    vals = []
    for env, X in sample_typical_instances(7, density, n_envs=60, profiles_per_env=50, seed=11):
        o, opt = optimum(X, env, "mc")
        pred = point_predictions(o, error_scale(opt, 7, "mc"), eta, rng)
        y = MinMaxP().locate(X, env, pred)
        vals.append(ratio(objective_cost(y, X, "mc"), opt))
    v = np.concatenate(vals)
    return float(v.mean()), float(np.quantile(v, 0.95)), float(v.max())


def main() -> None:
    apply_style()
    rng = np.random.default_rng(0)
    wc = [{"eta": e, "minmaxp_open_line": worst_case(e, False), "minmaxp_star_blocked": worst_case(e, True)}
          for e in ETAS]
    with open(os.path.join(RES_DIR, "e2_worst_case.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(wc[0]))
        w.writeheader()
        w.writerows(wc)
    ty = []
    for d in DENSITIES:
        for e in ETAS:
            mean, q95, mx = typical(e, d, rng)
            ty.append({"density": d, "eta": e, "mean": mean, "q95": q95, "max": mx})
    with open(os.path.join(RES_DIR, "e2_typical.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ty[0]))
        w.writeheader()
        w.writerows(ty)

    eta = np.array(ETAS)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
    ax = axes[0]
    grid = np.linspace(0, eta.max(), 200)
    theory_line(ax, grid, 1 + np.minimum(grid, 1), "1 + min(η, 1)", label_at=-1, dy=-12)
    theory_line(ax, grid, np.minimum(1 + 2 * grid, 3), "min(1 + 2η, 3)", label_at=-1, dy=-12)
    ax.plot(eta, [r["minmaxp_open_line"] for r in wc], marker=MARKERS[0], color=SERIES[0], linestyle="none",
            label="MinMaxP, open line", zorder=3)
    ax.plot(eta, [r["minmaxp_star_blocked"] for r in wc], marker=MARKERS[1], color=SERIES[1], linestyle="none",
            label="MinMaxP*, blocked zones", zorder=3)
    ax.set_title("(a) Worst case found by adversarial search")
    ax.set_xlabel("prediction error η (normalised by optimal cost)")
    ax.set_ylabel("approximation ratio (max cost)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.9, 3.3)

    ax = axes[1]
    for i, d in enumerate(DENSITIES):
        rows = [r for r in ty if r["density"] == d]
        ax.plot(eta, [r["mean"] for r in rows], color=RAMP[i], marker=MARKERS[i], markersize=4,
                label=f"{int(d * 100)}% blocked")
    ax.set_title("(b) Typical case: mean over 3,000 random instances")
    ax.set_xlabel("prediction error η")
    ax.legend(loc="upper left", title="blocked density", title_fontsize=8)
    fig.tight_layout()
    path = save(fig, "e2_error_doubling")
    print("E2 saved", path)
    for r in wc:
        print(f"  eta={r['eta']:<5} open={r['minmaxp_open_line']:.3f}  blocked={r['minmaxp_star_blocked']:.3f}")


if __name__ == "__main__":
    main()
