"""E6 - Proved / conjectured worst-case bounds vs the worst instance found by adversarial search.

Each row states a bound (with its source) and the largest ratio the cluster-aware hill-climbing
search finds. The search value is attained by an explicit instance, so it is a lower bound on the
true worst case; "match" means it reaches the stated bound to within 1%.

Outputs: results/bounds_table.md, results/bounds_table.csv.
"""

from __future__ import annotations
import csv
import os

from common import RES_DIR, write_markdown_table
from src.experiment_utils import Scenario, hill_climb
from src.learning_augmented import (
    CenterMinMaxP, IntervalClamp, MedianRule, MinMaxP, PhantomMedian, SnapTowardPrediction,
)


def found(mech, objective, n, gaps, mode="none", eta=0.0, omega=0.0) -> float:
    sc = Scenario(mech, objective, n, n_gaps=gaps, mode=mode, eta=eta, omega=omega)
    return max(hill_climb(sc, restarts=512, steps=700, seed=s).best_ratio for s in (0, 1, 2))


def rows():
    R = []

    def add(setting, mech_name, objective, quantity, n, bound, source, value):
        R.append({"Setting": setting, "Mechanism": mech_name, "Objective": objective, "Quantity": quantity,
                  "n": n, "Bound": bound if isinstance(bound, str) else f"{bound:.3f}", "Source": source,
                  "Found": f"{value:.3f}",
                  "Match": "yes" if isinstance(bound, (int, float)) and value >= 0.99 * bound else
                           ("-" if isinstance(bound, str) else "no")})

    # --- classical, for calibration
    add("open line", "Median", "max", "worst case", 3, 2.0, "Procaccia-Tennenholtz", found(MedianRule(snap=False), "mc", 3, 0))
    add("blocked", "Median*", "max", "worst case", 3, 3.0, "Walsh 2021 (tight)", found(MedianRule(), "mc", 3, 2))
    for n in (3, 5, 7):
        add("blocked", "Median*", "social", "worst case", n, 3 - 4 / (n + 1), "conjecture (→3: FFG 2016, Walsh 2021)",
            found(MedianRule(), "sc", n, 2))
    add("open line", "MinMaxP", "max", "consistency", 3, 1.0, "Agrawal et al. 2022", found(MinMaxP(snap=False), "mc", 3, 0, "error"))
    add("open line", "MinMaxP", "max", "robustness", 3, 2.0, "Agrawal et al. 2022", found(MinMaxP(snap=False), "mc", 3, 0, "robust"))
    add("open line", "MinMaxP", "max", "error η = 0.5", 3, 1.5, "Agrawal; Chan-Lin-Wang", found(MinMaxP(snap=False), "mc", 3, 0, "error", eta=0.5))
    # --- our results, point predictions with blocked zones
    add("blocked", "MinMaxP*", "max", "consistency", 5, 1.0, "Thm 4", found(MinMaxP(), "mc", 5, 2, "error"))
    add("blocked", "MinMaxP*", "max", "robustness", 3, 3.0, "Thm 4 (optimal: Walsh LB)", found(MinMaxP(), "mc", 3, 2, "robust"))
    for eta in (0.25, 0.5, 0.75):
        add("blocked", "MinMaxP*", "max", f"error η = {eta}", 3, min(1 + 2 * eta, 3), "Thm 4 (tight)",
            found(MinMaxP(), "mc", 3, 2, "error", eta=eta))
    add("blocked", "MinMaxP+SnapTowardPred", "max", "error η = 0.25", 3, "not SP", "E1",
        found(SnapTowardPrediction(), "mc", 3, 2, "error", eta=0.25))
    add("blocked", "MinMaxP*", "social", "consistency", 7, 1.0, "Prop 8(a)", found(MinMaxP(), "sc", 7, 2, "error"))
    for n in (3, 5):
        add("open line", "MinMaxP", "social", "robustness", n, n - 1, "Prop 8(b) (k = n-1)", found(MinMaxP(snap=False), "sc", n, 0, "robust"))
        add("blocked", "MinMaxP*", "social", "robustness", n, 2 * n - 1, "conjecture", found(MinMaxP(), "sc", n, 2, "robust"))
    # --- confidence family, social cost, n = 7
    n = 7
    for k in (0, 2, 4, 6):
        add("blocked", f"PhantomMedian[k={k}]*", "social", "consistency", n, (3 * n - 1 - k) / (n + 1 + k), "conjecture",
            found(PhantomMedian(k), "sc", n, 2, "error"))
        add("open line", f"PhantomMedian[k={k}]", "social", "robustness", n, max(1.0, (n + k - 1) / (n - k + 1)), "Prop 8(b)",
            found(PhantomMedian(k, snap=False), "sc", n, 0, "robust"))
        add("blocked", f"PhantomMedian[k={k}]*", "social", "robustness", n, 1 + 2 * (n + k - 1) / (n - k + 1),
            "conjecture", found(PhantomMedian(k), "sc", n, 2, "robust"))
    # --- interval predictions (optimum inside the interval)
    for om in (0.5, 1.0):
        add("open line", "CenterMinMaxP", "max", f"consistency ω = {om}", 3, 1 + min(om / 2, 1), "Thm 6 (optimal in GM class)",
            found(CenterMinMaxP(snap=False), "mc", 3, 0, "interval", omega=om))
        add("open line", "IntervalClamp", "max", f"consistency ω = {om}", 3, 1 + min(om, 1), "Thm 6",
            found(IntervalClamp(snap=False), "mc", 3, 0, "interval", omega=om))
        add("blocked", "CenterMinMaxP*", "max", f"consistency ω = {om}", 3, min(1 + om, 3), "Thm 7",
            found(CenterMinMaxP(), "mc", 3, 2, "interval", omega=om))
        add("blocked", "IntervalClamp*", "max", f"consistency ω = {om}", 3, min(1 + 2 * om, 3), "Thm 7",
            found(IntervalClamp(), "mc", 3, 2, "interval", omega=om))
    # robustness: interval far from the optimum (normalised distance 5)
    add("blocked", "CenterMinMaxP*", "max", "robustness (η = 5)", 3, 3.0, "Lemma 3",
        found(CenterMinMaxP(), "mc", 3, 2, "interval", eta=5.0, omega=0.5))
    add("blocked", "IntervalClamp*", "max", "robustness (η = 5)", 3, 3.0, "Lemma 3",
        found(IntervalClamp(), "mc", 3, 2, "interval", eta=5.0, omega=0.5))
    add("blocked", "CenterMinMaxP*", "social", "consistency ω = 3", 7, "grows with ω", "E3",
        found(CenterMinMaxP(), "sc", 7, 2, "interval", omega=3.0))
    add("blocked", "IntervalClamp*", "social", "consistency ω = 3", 7, "≈ Median* worst case", "E3",
        found(IntervalClamp(), "sc", 7, 2, "interval", omega=3.0))
    return R


def main() -> None:
    R = rows()
    cols = ["Setting", "Mechanism", "Objective", "Quantity", "n", "Bound", "Source", "Found", "Match"]
    with open(os.path.join(RES_DIR, "bounds_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(R)
    write_markdown_table(R, cols, os.path.join(RES_DIR, "bounds_table.md"),
                         header="Worst-case bounds vs adversarial search (found = ratio of an explicit instance, "
                                "a lower bound on the true worst case). Theorem numbers refer to docs/RESULTS.md.")
    for r in R:
        print(" | ".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
