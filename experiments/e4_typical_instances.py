"""E4 - Typical instances: how blocked-zone density and the agent distribution affect performance.

Random environments come from ``Environment1D.random_blocked`` (3 zones) and agents from the
Week 1-2 generators (uniform, gaussian, bimodal, clustered, polarized); n = 7.
Mechanisms: Median* (no prediction), MinMaxP* with a point prediction of error eta = 0.25, and the
two interval rules with an interval of width omega = 0.5 containing the optimum.

Outputs: figures/e4_blocked_density.{png,pdf}, results/e4_density.csv, results/e4_distributions.md.
"""

from __future__ import annotations
import csv
import os

import numpy as np

import matplotlib.pyplot as plt
from common import MARKERS, RES_DIR, SERIES, apply_style, save, write_markdown_table
from src.experiment_utils import (
    error_scale, interval_predictions, objective_cost, optimum, point_predictions, ratio,
    sample_typical_instances,
)
from src.learning_augmented import CenterMinMaxP, IntervalClamp, MedianRule, MinMaxP

N = 7
DENSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DISTRIBUTIONS = ["uniform", "gaussian", "bimodal", "clustered", "polarized"]
MECHS = [
    ("Median* (no prediction)", MedianRule(), "none"),
    ("MinMaxP*, point η = 0.25", MinMaxP(), "point"),
    ("CenterMinMaxP*, interval ω = 0.5", CenterMinMaxP(), "interval"),
    ("IntervalClamp*, interval ω = 0.5", IntervalClamp(), "interval"),
]


def ratios_for(insts, objective, rng):
    out = {name: [] for name, _, _ in MECHS}
    for env, X in insts:
        o, opt = optimum(X, env, objective)
        scale = error_scale(opt, N, objective)
        preds = {"none": None, "point": point_predictions(o, scale, 0.25, rng),
                 "interval": interval_predictions(o, scale, 0.5, 0.0, rng)}
        for name, mech, kind in MECHS:
            y = mech.locate(X, env, preds[kind])
            out[name].append(ratio(objective_cost(y, X, objective), opt))
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    apply_style()
    rng = np.random.default_rng(4)
    rows = []
    for objective in ("mc", "sc"):
        for d in DENSITIES:
            r = ratios_for(sample_typical_instances(N, d, 60, 50, seed=31), objective, rng)
            for name, v in r.items():
                rows.append({"objective": objective, "density": d, "mechanism": name,
                             "mean": float(v.mean()), "q95": float(np.quantile(v, 0.95)), "max": float(v.max())})
    with open(os.path.join(RES_DIR, "e4_density.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    table = []
    for dist in DISTRIBUTIONS:
        for objective in ("mc", "sc"):
            r = ratios_for(sample_typical_instances(N, 0.4, 60, 50, distribution=dist, seed=41), objective, rng)
            table.append({"Distribution": dist, "Objective": "max cost" if objective == "mc" else "social cost",
                          **{name: f"{v.mean():.3f} / {np.quantile(v, 0.95):.3f}" for name, v in r.items()}})
    write_markdown_table(table, ["Distribution", "Objective"] + [m[0] for m in MECHS],
                         os.path.join(RES_DIR, "e4_distributions.md"),
                         header="Mean / 95th-percentile approximation ratio, n = 7, 40% of the line blocked "
                                "(3 random zones), 3,000 instances per cell")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
    for ax, objective, title in zip(axes, ("mc", "sc"), ("(a) Max cost", "(b) Social cost")):
        for i, (name, _, _) in enumerate(MECHS):
            sub = [r for r in rows if r["objective"] == objective and r["mechanism"] == name]
            ax.plot([100 * r["density"] for r in sub], [r["mean"] for r in sub], color=SERIES[i],
                    marker=MARKERS[i], markersize=4, label=name)
        ax.set_title(title + ": mean ratio on random instances")
        ax.set_xlabel("share of the line blocked (%)")
    axes[0].set_ylabel("mean approximation ratio")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    print("E4 saved", save(fig, "e4_blocked_density"))
    for r in rows:
        if r["density"] in (0.0, 0.4, 0.7):
            print(f"  {r['objective']} rho={r['density']} {r['mechanism']:<36} mean={r['mean']:.3f} q95={r['q95']:.3f} max={r['max']:.3f}")


if __name__ == "__main__":
    main()
