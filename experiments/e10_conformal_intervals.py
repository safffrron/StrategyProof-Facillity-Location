"""E10 - Calibrated interval predictions from data (split conformal prediction).

A planner sees, each day, public features z (e.g. weekday, event intensity) that shift where agents
cluster; agents' exact locations are private and reported strategically. From past days she learns a
model z -> optimal (feasible) location, and turns it into intervals with a coverage guarantee:

* "Conformal": absolute-residual split conformal, I = [o_hat - q, o_hat + q];
* "CQR": conformalized quantile regression (Romano, Patterson & Candes 2019), adaptive widths.

With exchangeable days, P(o in I) >= 1 - alpha. Combined with Theorem 7 / Theorem 13, with
probability >= 1 - alpha the realised ratio of CenterMinMaxP* is <= min(1 + omega, 3) (randomized
version: <= 1 + omega/2 in expectation), and it is <= 3 always. The interval depends only on past data
and public features, so the mechanisms stay strategyproof.

Outputs: figures/e10_conformal.{png,pdf}, results/e10_conformal.csv, results/e10_conformal.md
"""

from __future__ import annotations
import csv
import os

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib.pyplot as plt
from common import INK_2, MARKERS, RES_DIR, SERIES, apply_style, save, write_markdown_table
from src.environment import Environment1D, Interval
from src.experiment_utils import expected_objective, optimum
from src.learning_augmented import (
    CenterMinMaxP, IntervalClamp, MedianRule, MinMaxP, RandomizedPredictionSnap,
)

N_AGENTS = 21
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
ENV = Environment1D(length=1.0, num_grid_points=11,
                    blocked_intervals=[Interval(0.20, 0.32), Interval(0.50, 0.62), Interval(0.78, 0.84)])


def sample_days(n_days: int, rng: np.random.Generator, shift: float = 0.0):
    """Features z in [0,1]^3 and agent profiles from three neighbourhoods (hubs).

    The two outer hubs are small on many days, so whether ANY agent comes from them is random given z:
    the optimum is then bimodal given the features, and a regression model predicts a value between
    the modes, sometimes inside a blocked zone. ``shift`` moves the right hub left (concept shift).
    """
    z = rng.uniform(0, 1, (n_days, 3))
    sig = lambda t: 1.0 / (1.0 + np.exp(-t))
    w_left = 0.45 * sig(8 * (z[:, 0] - 0.6))
    w_right = 0.45 * sig(8 * (z[:, 1] - 0.6))
    w = np.stack([w_left, 1.0 - w_left - w_right, w_right], axis=1)
    centres = np.array([0.08, 0.45, 0.93 - shift])
    hub = np.array([rng.choice(3, size=N_AGENTS, p=wi) for wi in w])
    X = np.clip(rng.normal(centres[hub], 0.03 + 0.03 * z[:, 2:3]), 0.0, 1.0)
    return z, X


def conformal_intervals(model, z_cal, y_cal, z_test, alpha):
    q = np.quantile(np.abs(y_cal - model.predict(z_cal)), min(1.0, np.ceil((len(y_cal) + 1) * (1 - alpha)) / len(y_cal)))
    p = model.predict(z_test)
    return np.stack([p - q, p + q], axis=1)


def cqr_intervals(z_tr, y_tr, z_cal, y_cal, z_test, alpha, seed):
    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, n_estimators=200, max_depth=3, random_state=seed)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, n_estimators=200, max_depth=3, random_state=seed)
    lo.fit(z_tr, y_tr)
    hi.fit(z_tr, y_tr)
    scores = np.maximum(lo.predict(z_cal) - y_cal, y_cal - hi.predict(z_cal))
    q = np.quantile(scores, min(1.0, np.ceil((len(y_cal) + 1) * (1 - alpha)) / len(y_cal)))
    return np.stack([lo.predict(z_test) - q, hi.predict(z_test) + q], axis=1)


def evaluate_day_batch(X, intervals, point, objective="mc"):
    o, opt = optimum(X, ENV, objective)
    width = (intervals[:, 1] - intervals[:, 0]) / opt
    covered = (o >= intervals[:, 0] - 1e-12) & (o <= intervals[:, 1] + 1e-12)
    mechs = {
        "Median* (no prediction)": (MedianRule(), None),
        "MinMaxP* (point)": (MinMaxP(), point),
        "Rand[MinMaxP*] (point)": (RandomizedPredictionSnap(MinMaxP()), point),
        "CenterMinMaxP* (interval)": (CenterMinMaxP(), intervals),
        "Rand[centre] (interval)": (RandomizedPredictionSnap(MinMaxP(), use_center=True), intervals),
        "IntervalClamp* (interval)": (IntervalClamp(), intervals),
    }
    out = {name: expected_objective(m, X, ENV, pred, objective) / opt for name, (m, pred) in mechs.items()}
    bound_det = np.minimum(1 + width, 3)
    return out, covered, width, bound_det


def main() -> None:
    apply_style()
    rng = np.random.default_rng(2026)
    z_tr, X_tr = sample_days(3000, rng)
    z_cal, X_cal = sample_days(1500, rng)
    z_te, X_te = sample_days(3000, rng)
    z_sh, X_sh = sample_days(3000, rng, shift=0.17)       # concept shift: right hub moves left across a zone
    y_tr = optimum(X_tr, ENV, "mc")[0]
    y_cal = optimum(X_cal, ENV, "mc")[0]
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=0).fit(z_tr, y_tr)
    r2 = model.score(z_te, optimum(X_te, ENV, "mc")[0])
    print(f"  point model R^2 on test days: {r2:.3f}")

    rows = []
    for method in ("Conformal", "CQR"):
        for alpha in ALPHAS:
            for split, (z_s, X_s) in (("test", (z_te, X_te)), ("shifted", (z_sh, X_sh))):
                if method == "Conformal":
                    I = conformal_intervals(model, z_cal, y_cal, z_s, alpha)
                else:
                    I = cqr_intervals(z_tr, y_tr, z_cal, y_cal, z_s, alpha, seed=1)
                point = model.predict(z_s)
                ratios, covered, width, bound = evaluate_day_batch(X_s, I, point)
                row = {"method": method, "alpha": alpha, "split": split, "coverage": float(covered.mean()),
                       "median_width_omega": float(np.median(width)),
                       "within_bound": float((ratios["CenterMinMaxP* (interval)"] <= bound + 1e-9).mean())}
                for name, r in ratios.items():
                    row[f"mean::{name}"] = float(r.mean())
                    row[f"q95::{name}"] = float(np.quantile(r, 0.95))
                    row[f"max::{name}"] = float(r.max())
                rows.append(row)
            print(f"  {method} done")
    with open(os.path.join(RES_DIR, "e10_conformal.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    names = ["Median* (no prediction)", "MinMaxP* (point)", "Rand[MinMaxP*] (point)", "CenterMinMaxP* (interval)",
             "Rand[centre] (interval)", "IntervalClamp* (interval)"]
    table = []
    for r in rows:
        if r["alpha"] in (0.05, 0.1, 0.3):
            table.append({"intervals": r["method"], "alpha": r["alpha"], "days": r["split"],
                          "coverage": f"{r['coverage']:.3f}", "median ω": f"{r['median_width_omega']:.2f}",
                          **{n.split(" (")[0]: f"{r['mean::' + n]:.3f} / {r['max::' + n]:.2f}" for n in names}})
    write_markdown_table(table, list(table[0].keys()), os.path.join(RES_DIR, "e10_conformal.md"),
                         header=f"Max cost, n = {N_AGENTS}, three blocked zones, 3,000 days per row; cells are mean / "
                                f"worst ratio. Point-model R^2 = {r2:.3f}. 'shifted' = test days after the right hub moves left across a blocked zone.")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    interval_rules = [("CenterMinMaxP* (interval)", "CenterMinMaxP*"), ("Rand[centre] (interval)", "Rand-centre"),
                      ("IntervalClamp* (interval)", "IntervalClamp*")]
    panels = ((axes[0], "test", "mean", "(a) Average day (in distribution)", "mean ratio (max cost)"),
              (axes[1], "shifted", "max", "(b) Worst day (after distribution shift)", "worst ratio over 3,000 days"))
    for ax, split, stat, title, ylabel in panels:
        sub = [r for r in rows if r["method"] == "CQR" and r["split"] == split]
        al = [r["alpha"] for r in sub]
        for i, (n, lab) in enumerate(interval_rules):
            ax.plot(al, [r[f"{stat}::{n}"] for r in sub], marker=MARKERS[i], color=SERIES[i], label=lab + " (CQR)")
        for n, lab, dy in (("MinMaxP* (point)", "MinMaxP*, point prediction", 3), ("Median* (no prediction)", "Median*, no prediction", 3)):
            v = sub[0][f"{stat}::{n}"]
            ax.axhline(v, color=INK_2, linewidth=1.0)
            ax.annotate(lab, (0.26, v), xytext=(0, dy), textcoords="offset points", fontsize=7.5, color=INK_2)
        ax.set_title(title)
        ax.set_xlabel("miscoverage level α")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=7)
    ax = axes[2]
    for i, split in enumerate(("test", "shifted")):
        sub = [r for r in rows if r["method"] == "CQR" and r["split"] == split]
        ax.plot([1 - r["alpha"] for r in sub], [r["coverage"] for r in sub], marker=MARKERS[0], color=SERIES[i],
                label=f"coverage, {'in distribution' if split == 'test' else 'after shift'}")
        ax.plot([1 - r["alpha"] for r in sub], [r["within_bound"] for r in sub], marker=MARKERS[1], color=SERIES[i],
                linestyle="none", label=f"certificate holds, {'in distribution' if split == 'test' else 'after shift'}")
    ax.plot([0.5, 1], [0.5, 1], color=INK_2, linewidth=1.0)
    ax.annotate("1 − α", (0.63, 0.57), fontsize=7.5, color=INK_2)
    ax.set_title("(c) Coverage and per-day certificate")
    ax.set_xlabel("target coverage 1 − α")
    ax.set_ylabel("share of days")
    ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    print("E10 saved", save(fig, "e10_conformal"))
    for r in rows:
        if r["method"] == "CQR":
            print(f"  CQR a={r['alpha']:<4} {r['split']:<7} cov={r['coverage']:.3f} w={r['median_width_omega']:.2f} "
                  f"within_bound={r['within_bound']:.3f} | " + " ".join(f"{n.split(' (')[0]}={r['mean::' + n]:.3f}" for n in names))


if __name__ == "__main__":
    main()
