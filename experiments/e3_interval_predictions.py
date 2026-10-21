"""E3 - Interval predictions: worst-case consistency vs normalised interval width omega.

The interval always contains the optimum (eta = 0); the adversary chooses the agents, the blocked
zones and where the optimum sits inside the interval. Two ways to use an interval:
* CenterMinMaxP: run MinMaxP on the interval's midpoint;
* IntervalClamp: clamp the median into the interval, then into [min, max] (a generalised median
  with phantoms at both interval ends).

Bounds (docs/RESULTS.md): open line, max cost - Center 1 + min(omega/2, 1) (optimal among
generalised medians), Clamp 1 + min(omega, 1); blocked zones - Center* min(1 + omega, 3),
Clamp* min(1 + 2 omega, 3).

Outputs: figures/e3_interval_width.{png,pdf}, results/e3_worst_case.csv, results/e3_typical.csv.
"""

from __future__ import annotations
import csv
import os

import numpy as np

import matplotlib.pyplot as plt
from common import INK_2, MARKERS, RES_DIR, SERIES, apply_style, save, theory_line
from src.experiment_utils import (
    Scenario, error_scale, hill_climb, interval_predictions, objective_cost, optimum, ratio,
    sample_typical_instances,
)
from src.learning_augmented import CenterMinMaxP, IntervalClamp, MedianRule

OMEGAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
CASES = [("mc", 0, (2, 3, 5)), ("mc", 2, (2, 3, 5)), ("sc", 2, (3, 5, 7))]


def worst(mech, objective, gaps, ns, omega=None, mode="interval") -> float:
    best = 0.0
    for n in ns:
        sc = Scenario(mech, objective, n, n_gaps=gaps, mode=mode, omega=omega or 0.0)
        best = max(best, hill_climb(sc, restarts=384, steps=500, seed=n).best_ratio)
    return best


def main() -> None:
    apply_style()
    rows = []
    for objective, gaps, ns in CASES:
        snap = gaps > 0
        for om in OMEGAS:
            rows.append({
                "objective": objective, "blocked": snap, "omega": om,
                "center": worst(CenterMinMaxP(snap=snap), objective, gaps, ns, om),
                "clamp": worst(IntervalClamp(snap=snap), objective, gaps, ns, om),
            })
        print(f"  worst-case {objective} blocked={snap} done")
    median_sc = worst(MedianRule(), "sc", 2, (3, 5, 7), mode="none")
    with open(os.path.join(RES_DIR, "e3_worst_case.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # typical case (mean ratio, optimum uniformly placed inside the interval)
    rng = np.random.default_rng(1)
    typ = []
    for objective in ("mc", "sc"):
        for density in (0.0, 0.4):
            insts = sample_typical_instances(7, density, n_envs=60, profiles_per_env=50, seed=21)
            for om in OMEGAS:
                acc = {"center": [], "clamp": [], "median": []}
                for env, X in insts:
                    o, opt = optimum(X, env, objective)
                    I = interval_predictions(o, error_scale(opt, 7, objective), om, 0.0, rng)
                    for key, mech in (("center", CenterMinMaxP()), ("clamp", IntervalClamp()), ("median", MedianRule())):
                        y = mech.locate(X, env, I)
                        acc[key].append(ratio(objective_cost(y, X, objective), opt))
                typ.append({"objective": objective, "density": density, "omega": om,
                            **{k: float(np.concatenate(v).mean()) for k, v in acc.items()}})
    with open(os.path.join(RES_DIR, "e3_typical.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(typ[0]))
        w.writeheader()
        w.writerows(typ)

    om = np.array(OMEGAS)
    g = np.linspace(0, om.max(), 200)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3), sharey=True)
    titles = ["(a) Max cost, open line", "(b) Max cost, blocked zones", "(c) Social cost, blocked zones"]
    for ax, (objective, gaps, _), title in zip(axes, CASES, titles):
        sub = [r for r in rows if r["objective"] == objective and r["blocked"] == (gaps > 0)]
        if objective == "mc" and gaps == 0:
            theory_line(ax, g, 1 + np.minimum(g / 2, 1), "1 + min(ω/2, 1)", text_xy=(1.55, 1.55))
            theory_line(ax, g, 1 + np.minimum(g, 1), "1 + min(ω, 1)", text_xy=(0.55, 1.9), ha="right")
        elif objective == "mc":
            theory_line(ax, g, np.minimum(1 + g, 3), "min(1 + ω, 3)", text_xy=(1.3, 1.85))
            theory_line(ax, g, np.minimum(1 + 2 * g, 3), "min(1 + 2ω, 3)", text_xy=(0.08, 3.2))
        else:
            ax.axhline(median_sc, color=INK_2, linewidth=1.0, zorder=1)
            ax.annotate("Median* (ignores prediction)", (0.05, median_sc), xytext=(0, 3),
                        textcoords="offset points", fontsize=7.5, color=INK_2)
        ax.plot(om, [r["center"] for r in sub], marker=MARKERS[0], color=SERIES[0], linestyle="none",
                label="CenterMinMaxP" + ("*" if gaps else ""), zorder=3)
        ax.plot(om, [r["clamp"] for r in sub], marker=MARKERS[1], color=SERIES[1], linestyle="none",
                label="IntervalClamp" + ("*" if gaps else ""), zorder=3)
        ax.set_title(title)
        ax.set_xlabel("interval width ω (normalised by optimal cost)")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("worst-case ratio, optimum inside interval")
    axes[0].set_ylim(0.9, 3.75)
    fig.tight_layout()
    print("E3 saved", save(fig, "e3_interval_width"))
    for r in rows:
        print(f"  {r['objective']} blocked={r['blocked']!s:<5} w={r['omega']:<5} center={r['center']:.3f} clamp={r['clamp']:.3f}")
    print(f"  Median* SC worst (n<=7): {median_sc:.3f}")


if __name__ == "__main__":
    main()
