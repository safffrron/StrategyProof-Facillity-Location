"""Run every experiment of the report in order and regenerate results/ and figures/.

Usage (from the repository root):
    python3 experiments/run_all.py
"""

from __future__ import annotations
import time

import common  # noqa: F401  (sets sys.path and the plotting backend)
import e1_strategyproofness
import e2_unconstrained_target
import e3_median_predictions
import e4_randomised_snapping
import e5_optimal_mechanisms
import e6_candidate_lower_bound
import e7_double_peaked
import e8_confidence_sweep

STEPS = [
    ("E1 strategyproofness checks", e1_strategyproofness.main),
    ("E2 predictions of the unconstrained optimum", e2_unconstrained_target.main),
    ("E3 median predictions (social cost)", e3_median_predictions.main),
    ("E4 deterministic versus randomised snapping", e4_randomised_snapping.main),
    ("E5 optimisation over all mechanisms", e5_optimal_mechanisms.main),
    ("E6 certified lower bound, candidate sites", e6_candidate_lower_bound.main),
    ("E7 double-peaked preferences", e7_double_peaked.main),
    ("E8 confidence sweep (project brief)", e8_confidence_sweep.main),
]

if __name__ == "__main__":
    t0 = time.time()
    for label, fn in STEPS:
        t = time.time()
        print(f"\n=== {label} ===")
        fn()
        print(f"--- {label}: {time.time() - t:.0f}s")
    print(f"\nAll experiments finished in {time.time() - t0:.0f}s")
