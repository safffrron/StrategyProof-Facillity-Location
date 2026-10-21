"""E1 - Exhaustive strategyproofness verification.

For every mechanism, every blocked-zone layout, n in {3, 4, 5}, every profile on a grid, every
agent, every misreport (grid plus two points outside the line) and every prediction, search for a
profitable lie. Also re-checks the snapped rules under *asymmetric* single-peaked costs.

Outputs: results/sp_checks.csv (long format), results/sp_table.md (summary table).
"""

from __future__ import annotations
import csv
import os
import time

import numpy as np

from common import RES_DIR, write_markdown_table
from src.environment import Environment1D, Interval
from src.fast_validator import check_strategyproofness
from src.learning_augmented import (
    CenterMinMaxP, DictatorRule, HullNearestFeasible, IntervalClamp, LeftmostRule, MeanRule,
    MedianRule, MidpointRule, MinMaxP, PhantomMedian, PredictionOrMedian, SnapTowardPrediction,
)

SETTINGS = {
    "unconstrained": None,
    "1 zone": Environment1D(length=1.0, num_grid_points=101, blocked_intervals=[Interval(0.3, 0.7)]),
    "2 zones": Environment1D(length=1.0, num_grid_points=101,
                             blocked_intervals=[Interval(0.15, 0.35), Interval(0.55, 0.8)]),
}
GRIDS = {3: np.linspace(0, 1, 21), 4: np.linspace(0, 1, 15), 5: np.linspace(0, 1, 11)}
POINT_PREDS = list(np.round(np.linspace(-0.2, 1.2, 15), 6))
_ends = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
INTERVAL_PREDS = [(a, b) for a in _ends for b in _ends if a <= b]


def mechanisms():
    """(mechanism, predictions, expected SP for symmetric costs, family)."""
    return [
        (MedianRule(snap=False), [None], True, "baseline"),
        (MedianRule(), [None], True, "baseline"),
        (LeftmostRule(), [None], True, "baseline"),
        (DictatorRule(0), [None], True, "baseline"),
        (MeanRule(), [None], False, "baseline"),
        (MidpointRule(), [None], False, "baseline"),
        (MinMaxP(), POINT_PREDS, True, "point prediction"),
        (PhantomMedian(1), POINT_PREDS, True, "point prediction"),
        (PhantomMedian(2), POINT_PREDS, True, "point prediction"),
        (PredictionOrMedian(snap=True), POINT_PREDS, False, "point prediction"),
        (HullNearestFeasible(), POINT_PREDS, False, "point prediction"),
        (SnapTowardPrediction(), POINT_PREDS, False, "point prediction"),
        (CenterMinMaxP(), INTERVAL_PREDS, True, "interval prediction"),
        (IntervalClamp(), INTERVAL_PREDS, True, "interval prediction"),
    ]


def main() -> None:
    rows = []
    t0 = time.time()
    for mech, preds, expected, family in mechanisms():
        for n, grid in GRIDS.items():
            for setting, env in SETTINGS.items():
                rep = check_strategyproofness(mech, n, grid, env, predictions=preds, setting=setting)
                row = rep.as_row()
                row.update(family=family, expected_sp=expected)
                rows.append(row)
        print(f"  {mech.name:<28} done ({time.time() - t0:.0f}s)")
    # asymmetric costs: snapping needs symmetry
    for mech, preds in [(MedianRule(snap=False), [None]), (MedianRule(), [None]), (MinMaxP(), POINT_PREDS),
                        (IntervalClamp(), INTERVAL_PREDS)]:
        for setting in ("1 zone", "2 zones"):
            rep = check_strategyproofness(mech, 3, GRIDS[3], SETTINGS[setting], predictions=preds,
                                          alpha=3.0, setting=setting)
            row = rep.as_row()
            row.update(family="asymmetric cost (alpha=3)", expected_sp=None)
            rows.append(row)

    with open(os.path.join(RES_DIR, "sp_checks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary: one row per (family, mechanism, cost model)
    summary = {}
    for r in rows:
        key = (r["family"], r["mechanism"], r["cost_alpha"])
        s = summary.setdefault(key, {"checked": 0, "sp": True, "settings": set(), "fail_settings": set(),
                                     "example": ""})
        s["checked"] += r["deviations_checked"]
        s["settings"].add(r["setting"])
        if not r["strategyproof"]:
            s["sp"] = False
            s["fail_settings"].add(f'{r["setting"]}')
            if not s["example"]:
                s["example"] = r["counterexample"]
    table = []
    for (family, name, alpha), s in summary.items():
        table.append({
            "Family": family,
            "Mechanism": name,
            "Deviations checked": f'{s["checked"]:,}',
            "Strategyproof?": "yes" if s["sp"] else "**no**",
            "Fails in": ", ".join(sorted(s["fail_settings"])) or "-",
            "Counterexample (first found)": s["example"] or "-",
        })
    write_markdown_table(table, ["Family", "Mechanism", "Deviations checked", "Strategyproof?", "Fails in",
                                 "Counterexample (first found)"],
                         os.path.join(RES_DIR, "sp_table.md"),
                         header="Exhaustive strategyproofness checks (n = 3, 4, 5; grids of 21/15/11 points; "
                                "misreports include two points outside the line; * = snap to nearest feasible point)")
    unexpected = [r for r in rows if r["expected_sp"] is not None and r["strategyproof"] != r["expected_sp"]
                  and r["setting"] != "unconstrained"]
    print(f"E1 done in {time.time() - t0:.0f}s; {len(rows)} checks; unexpected outcomes: {len(unexpected)}")
    for r in unexpected:
        print("   UNEXPECTED:", r["mechanism"], r["setting"], r["n"], r["strategyproof"])


if __name__ == "__main__":
    main()
