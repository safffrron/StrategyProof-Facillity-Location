"""E1 - Exhaustive strategyproofness checks (report: Section 3.3, Lemma 2, Theorem 9).

For every mechanism, every allowed set (no zones, one zone, two zones), n in {3, 4, 5}, every
profile on a grid, every agent, every misreport (the grid plus two points beyond it) and every
prediction, search for a profitable lie. Randomised snapping is checked for truthfulness in
expectation: expected costs are compared exactly.

It also covers the baseline rules of the project brief (median, leftmost agent, dictator), the confidence
family PhantomMedian[k] and the brief's example rule (the prediction if it lies between the extreme reports,
else the median). Expected outcome: the mean and the brief's example rule are manipulable; every other
mechanism passes.

Output: results/e1_strategyproofness.md
"""

from __future__ import annotations
import os
import time

import numpy as np

from common import RES_DIR, write_markdown_table
from src.environment import Environment1D, Interval
from src.fast_validator import check_strategyproofness
from src.learning_augmented import (CenterMinMaxP, DictatorRule, LeftmostRule, MeanRule, MedianRule, MinMaxP,
                                    PhantomMedian, PredictionOrMedian, RandomizedPredictionSnap)

SETTINGS = {
    "no zones": None,
    "1 zone": Environment1D(length=1.0, num_grid_points=101, blocked_intervals=[Interval(0.3, 0.7)]),
    "2 zones": Environment1D(length=1.0, num_grid_points=101,
                             blocked_intervals=[Interval(0.15, 0.35), Interval(0.55, 0.8)]),
}
GRIDS = {3: np.linspace(0, 1, 21), 4: np.linspace(0, 1, 15), 5: np.linspace(0, 1, 11)}
POINT_PREDS = list(np.round(np.linspace(-0.2, 1.2, 15), 6))
_ends = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
INTERVAL_PREDS = [(a, b) for a in _ends for b in _ends if a <= b]


def mechanisms():
    """(mechanism, predictions, expected to be strategyproof)."""
    return [
        (MedianRule(), [None], True),
        (LeftmostRule(), [None], True),
        (DictatorRule(0), [None], True),
        (MeanRule(), [None], False),
        (PredictionOrMedian(), POINT_PREDS, False),
        (PhantomMedian(2), POINT_PREDS, True),
        (MinMaxP(), POINT_PREDS, True),
        (CenterMinMaxP(), INTERVAL_PREDS, True),
        (RandomizedPredictionSnap(MinMaxP()), POINT_PREDS, True),
        (RandomizedPredictionSnap(MinMaxP(), use_center=True), INTERVAL_PREDS, True),
    ]


def main() -> None:
    t0 = time.time()
    table, unexpected = [], []
    for mech, preds, expected in mechanisms():
        checked, fails, example = 0, [], ""
        for n, grid in GRIDS.items():
            for setting, env in SETTINGS.items():
                rep = check_strategyproofness(mech, n, grid, env, predictions=preds, setting=setting)
                checked += rep.n_deviations
                if not rep.is_strategyproof:
                    fails.append(setting)
                    example = example or rep.counterexample.describe()
                if rep.is_strategyproof != expected:
                    unexpected.append(f"{mech.name}, {setting}, n = {n}")
        table.append({
            "Mechanism": mech.name,
            "Deviations checked": f"{checked:,}",
            "Strategyproof?": "yes" if not fails else "**no**",
            "Fails in": ", ".join(sorted(set(fails))) or "-",
            "Counterexample (first found)": example or "-",
        })
        print(f"  {mech.name:<28} {checked:>13,} deviations, {'fails' if fails else 'passes'} "
              f"({time.time() - t0:.0f}s)")
    write_markdown_table(table, list(table[0]), os.path.join(RES_DIR, "e1_strategyproofness.md"),
                         header="E1 - Exhaustive strategyproofness checks (n = 3, 4, 5 on grids of 21, 15 and 11 "
                                "points; misreports include two points beyond the grid; * = snapped to the nearest "
                                "allowed point; Rand = randomised snapping, checked in expectation).")
    total = sum(int(r["Deviations checked"].replace(",", "")) for r in table)
    print(f"E1: {total:,} deviations in {time.time() - t0:.0f}s")
    assert not unexpected, f"unexpected outcomes: {unexpected}"


if __name__ == "__main__":
    main()
