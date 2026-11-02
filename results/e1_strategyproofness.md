E1 - Exhaustive strategyproofness checks (n = 3, 4, 5 on grids of 21, 15 and 11 points; misreports include two points beyond the grid; * = snapped to the nearest allowed point; Rand = randomised snapping, checked in expectation).

| Mechanism | Deviations checked | Strategyproof? | Fails in | Counterexample (first found) |
|---|---|---|---|---|
| Median* | 1,576,422 | yes | - | - |
| Leftmost* | 1,576,422 | yes | - | - |
| Dictator[0]* | 43,649,472 | yes | - | - |
| Mean* | 1,576,422 | **no** | 1 zone, 2 zones, no zones | peaks=(0.5, 1, 1): agent at 0.5 reports -0.5; facility 0.833 -> 0.5, cost 0.333 -> 0 |
| PredOrMedian* | 23,646,330 | **no** | 1 zone, 2 zones, no zones | peaks=(0, 0, 0.95), prediction=1: agent at 0.95 reports 1; facility 0 -> 1, cost 0.95 -> 0.05 |
| PhantomMedian[k=2]* | 23,646,330 | yes | - | - |
| MinMaxP* | 23,646,330 | yes | - | - |
| CenterMinMaxP* | 44,139,816 | yes | - | - |
| Rand[MinMaxP*] | 23,646,330 | yes | - | - |
| Rand[MinMaxP*, centre] | 44,139,816 | yes | - | - |
