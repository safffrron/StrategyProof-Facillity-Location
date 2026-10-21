Exhaustive strategyproofness checks (n = 3, 4, 5; grids of 21/15/11 points; misreports include two points outside the line; * = snap to nearest feasible point)

| Family | Mechanism | Deviations checked | Strategyproof? | Fails in | Counterexample (first found) |
|---|---|---|---|---|---|
| baseline | Median | 1,576,422 | yes | - | - |
| baseline | Median* | 1,576,422 | yes | - | - |
| baseline | Leftmost* | 1,576,422 | yes | - | - |
| baseline | Dictator[0]* | 43,649,472 | yes | - | - |
| baseline | Mean* | 1,576,422 | **no** | 1 zone, 2 zones, unconstrained | peaks=(0.5, 1, 1): agent at 0.5 reports -0.5; facility 0.833 -> 0.5, cost 0.333 -> 0 |
| baseline | Midpoint* | 1,576,422 | **no** | 1 zone, 2 zones, unconstrained | peaks=(0.25, 0.25, 1): agent at 0.25 reports -0.5; facility 0.625 -> 0.25, cost 0.375 -> 0 |
| point prediction | MinMaxP* | 23,646,330 | yes | - | - |
| point prediction | PhantomMedian[k=1]* | 23,646,330 | yes | - | - |
| point prediction | PhantomMedian[k=2]* | 23,646,330 | yes | - | - |
| point prediction | PredOrMedian* | 23,646,330 | **no** | 1 zone, 2 zones, unconstrained | peaks=(0, 0, 0.95), prediction=1: agent at 0.95 reports 1; facility 0 -> 1, cost 0.95 -> 0.05 |
| point prediction | HullNearestFeasible | 23,646,330 | **no** | 1 zone, 2 zones | peaks=(0.35, 0.35, 0.65), prediction=-0.2: agent at 0.65 reports 0.7; facility 0.3 -> 0.7, cost 0.35 -> 0.05 |
| point prediction | MinMaxP+SnapTowardPred | 23,646,330 | **no** | 1 zone, 2 zones | peaks=(0.65, 0.7, 0.7), prediction=-0.2: agent at 0.65 reports 0.7; facility 0.3 -> 0.7, cost 0.35 -> 0.05 |
| interval prediction | CenterMinMaxP* | 44,139,816 | yes | - | - |
| interval prediction | IntervalClamp* | 44,139,816 | yes | - | - |
| asymmetric cost (alpha=3) | Median | 244,398 | yes | - | - |
| asymmetric cost (alpha=3) | Median* | 244,398 | **no** | 1 zone, 2 zones | peaks=(0.5, 0.5, 0.55): agent at 0.5 reports 0.55; facility 0.3 -> 0.7, cost 0.6 -> 0.2 |
| asymmetric cost (alpha=3) | MinMaxP* | 3,665,970 | **no** | 1 zone, 2 zones | peaks=(0.5, 0.55, 0.55), prediction=-0.2: agent at 0.5 reports 0.55; facility 0.3 -> 0.7, cost 0.6 -> 0.2 |
| asymmetric cost (alpha=3) | IntervalClamp* | 6,843,144 | **no** | 1 zone, 2 zones | peaks=(0.5, 0.55, 0.55), prediction=(-0.1, -0.1): agent at 0.5 reports 0.55; facility 0.3 -> 0.7, cost 0.6 -> 0.2 |
