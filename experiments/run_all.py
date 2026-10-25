"""Regenerate every result table and figure used in the report (about 10 minutes).

Usage (from the repository root):
    python3 experiments/run_all.py
"""

from __future__ import annotations
import time

import common  # noqa: F401  (sets sys.path and the plotting backend)
import e1_strategyproofness
import e2_point_prediction_error
import e3_interval_predictions
import e4_typical_instances
import e5_confidence_frontier
import e6_bounds_table
import e7_doubling_diagram
import e8_lower_bounds_milp
import e9_asymmetric_costs
import e10_conformal_intervals
import e11_randomization
import e12_candidate_lower_bound
import e13_double_peaked
import e14_unconstrained_target
import e15_social_cost_target

STEPS = [
    ("E1 strategyproofness checks", e1_strategyproofness.main),
    ("E2 point-prediction error", e2_point_prediction_error.main),
    ("E3 interval predictions", e3_interval_predictions.main),
    ("E4 typical instances", e4_typical_instances.main),
    ("E5 confidence frontier", e5_confidence_frontier.main),
    ("E6 bounds table", e6_bounds_table.main),
    ("E7 doubling diagram", e7_doubling_diagram.main),
    ("E8 lower bounds (MILP / LP)", e8_lower_bounds_milp.main),
    ("E9 asymmetric costs", e9_asymmetric_costs.main),
    ("E10 conformal intervals", e10_conformal_intervals.main),
    ("E11 randomization", e11_randomization.main),
    ("E12 randomised lower bound, candidate locations", e12_candidate_lower_bound.main),
    ("E13 double-peaked preferences", e13_double_peaked.main),
    ("E14 predictions of the unconstrained optimum", e14_unconstrained_target.main),
    ("E15 social cost, predictions of the median", e15_social_cost_target.main),
]

if __name__ == "__main__":
    t0 = time.time()
    for label, fn in STEPS:
        t = time.time()
        print(f"\n=== {label} ===")
        fn()
        print(f"--- {label}: {time.time() - t:.0f}s")
    print(f"\nAll experiments finished in {time.time() - t0:.0f}s")
