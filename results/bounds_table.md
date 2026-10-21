Worst-case bounds vs adversarial search (found = ratio of an explicit instance, a lower bound on the true worst case). Theorem numbers refer to docs/RESULTS.md.

| Setting | Mechanism | Objective | Quantity | n | Bound | Source | Found | Match |
|---|---|---|---|---|---|---|---|---|
| open line | Median | max | worst case | 3 | 2.000 | Procaccia-Tennenholtz | 2.000 | yes |
| blocked | Median* | max | worst case | 3 | 3.000 | Walsh 2021 (tight) | 3.000 | yes |
| blocked | Median* | social | worst case | 3 | 2.000 | Thm 9, k = 0 (→3: FFG 2016) | 2.000 | yes |
| blocked | Median* | social | worst case | 5 | 2.333 | Thm 9, k = 0 (→3: FFG 2016) | 2.333 | yes |
| blocked | Median* | social | worst case | 7 | 2.500 | Thm 9, k = 0 (→3: FFG 2016) | 2.500 | yes |
| open line | MinMaxP | max | consistency | 3 | 1.000 | Agrawal et al. 2022 | 1.000 | yes |
| open line | MinMaxP | max | robustness | 3 | 2.000 | Agrawal et al. 2022 | 2.000 | yes |
| open line | MinMaxP | max | error η = 0.5 | 3 | 1.500 | Agrawal; Chan-Lin-Wang | 1.500 | yes |
| blocked | MinMaxP* | max | consistency | 5 | 1.000 | Thm 4; Fang et al. Thm 1 | 1.000 | yes |
| blocked | MinMaxP* | max | robustness | 3 | 3.000 | Thm 4; Fang et al. Thm 1 (optimal) | 3.000 | yes |
| blocked | MinMaxP* | max | feasible prediction, η = 0.5 | 3 | 1.500 | Fang et al. Thm 2 | 1.500 | yes |
| blocked | MinMaxP* | max | any prediction, η = 0.25 | 3 | 1.500 | Thm 4; optimal by Thm 10 | 1.500 | yes |
| blocked | Rand[MinMaxP*] | max | any prediction, η = 0.25 | 3 | 1.250 | Thm 13 (randomized) | 1.250 | yes |
| blocked | MinMaxP* | max | any prediction, η = 0.5 | 3 | 2.000 | Thm 4; optimal by Thm 10 | 2.000 | yes |
| blocked | Rand[MinMaxP*] | max | any prediction, η = 0.5 | 3 | 1.500 | Thm 13 (randomized) | 1.500 | yes |
| blocked | MinMaxP* | max | any prediction, η = 0.75 | 3 | 2.500 | Thm 4; optimal by Thm 10 | 2.500 | yes |
| blocked | Rand[MinMaxP*] | max | any prediction, η = 0.75 | 3 | 1.750 | Thm 13 (randomized) | 1.750 | yes |
| blocked | MinMaxP+SnapTowardPred | max | error η = 0.25 | 3 | not SP | E1 | 1.500 | - |
| blocked | MinMaxP* | social | consistency | 7 | 1.000 | Prop 8(a); Fang et al. Thm 3 | 1.000 | yes |
| open line | MinMaxP | social | robustness | 3 | 2.000 | Prop 8(b) (k = n-1) | 2.000 | yes |
| blocked | MinMaxP* | social | robustness | 3 | 5.000 | Fang et al. Thm 3; Thm 9 | 5.000 | yes |
| open line | MinMaxP | social | robustness | 5 | 4.000 | Prop 8(b) (k = n-1) | 4.000 | yes |
| blocked | MinMaxP* | social | robustness | 5 | 9.000 | Fang et al. Thm 3; Thm 9 | 8.999 | yes |
| blocked | PhantomMedian[k=0]* | social | consistency | 7 | 2.500 | Thm 9 (exact) | 2.500 | yes |
| open line | PhantomMedian[k=0] | social | robustness | 7 | 1.000 | Prop 8(b) | 1.000 | yes |
| blocked | PhantomMedian[k=0]* | social | robustness | 7 | 2.500 | Thm 9 (exact) | 2.500 | yes |
| blocked | PhantomMedian[k=2]* | social | consistency | 7 | 1.800 | Thm 9 (exact) | 1.800 | yes |
| open line | PhantomMedian[k=2] | social | robustness | 7 | 1.333 | Prop 8(b) | 1.333 | yes |
| blocked | PhantomMedian[k=2]* | social | robustness | 7 | 3.667 | Thm 9 (exact) | 3.667 | yes |
| blocked | PhantomMedian[k=4]* | social | consistency | 7 | 1.333 | Thm 9 (exact) | 1.333 | yes |
| open line | PhantomMedian[k=4] | social | robustness | 7 | 2.500 | Prop 8(b) | 2.500 | yes |
| blocked | PhantomMedian[k=4]* | social | robustness | 7 | 6.000 | Thm 9 (exact) | 6.000 | yes |
| blocked | PhantomMedian[k=6]* | social | consistency | 7 | 1.000 | Thm 9 (exact) | 1.000 | yes |
| open line | PhantomMedian[k=6] | social | robustness | 7 | 6.000 | Prop 8(b) | 6.000 | yes |
| blocked | PhantomMedian[k=6]* | social | robustness | 7 | 13.000 | Thm 9 (exact) | 12.999 | yes |
| open line | CenterMinMaxP | max | consistency ω = 0.5 | 3 | 1.250 | Thm 6 (optimal in GM class) | 1.250 | yes |
| open line | IntervalClamp | max | consistency ω = 0.5 | 3 | 1.500 | Thm 6 | 1.500 | yes |
| blocked | CenterMinMaxP* | max | consistency ω = 0.5 | 3 | 1.500 | Thm 7 | 1.500 | yes |
| blocked | IntervalClamp* | max | consistency ω = 0.5 | 3 | 2.000 | Thm 7 | 2.000 | yes |
| open line | CenterMinMaxP | max | consistency ω = 1.0 | 3 | 1.500 | Thm 6 (optimal in GM class) | 1.500 | yes |
| open line | IntervalClamp | max | consistency ω = 1.0 | 3 | 2.000 | Thm 6 | 2.000 | yes |
| blocked | CenterMinMaxP* | max | consistency ω = 1.0 | 3 | 2.000 | Thm 7 | 2.000 | yes |
| blocked | IntervalClamp* | max | consistency ω = 1.0 | 3 | 3.000 | Thm 7 | 3.000 | yes |
| blocked | CenterMinMaxP* | max | robustness (η = 5) | 3 | 3.000 | Lemma 3 | 3.000 | yes |
| blocked | IntervalClamp* | max | robustness (η = 5) | 3 | 3.000 | Lemma 3 | 3.000 | yes |
| blocked | Rand[centre] | max | consistency ω = 0.5 | 3 | 1.250 | Thm 13 (randomized) | 1.250 | yes |
| blocked | Rand[centre] | max | consistency ω = 1.0 | 3 | 1.500 | Thm 13 (randomized) | 1.500 | yes |
| blocked | Tops-MinMaxP (α=2) | max (asym.) | consistency | 5 | 1.000 | Prop 15 | 1.000 | yes |
| blocked | Tops-MinMaxP (α=2) | max (asym.) | robustness | 5 | 4.000 | Prop 15 | 4.000 | yes |
| blocked | Tops-Rightmost (α=2) | max (asym.) | worst case | 3 | 2.500 | Prop 15 | 2.500 | yes |
| blocked | Tops-MinMaxP (α=3) | max (asym.) | consistency | 5 | 1.000 | Prop 15 | 1.000 | yes |
| blocked | Tops-MinMaxP (α=3) | max (asym.) | robustness | 5 | 5.000 | Prop 15 | 5.000 | yes |
| blocked | Tops-Rightmost (α=3) | max (asym.) | worst case | 3 | 2.333 | Prop 15 | 2.333 | yes |
| blocked | CenterMinMaxP* | social | consistency ω = 3 | 7 | grows with ω | E3 | 3.571 | - |
| blocked | IntervalClamp* | social | consistency ω = 3 | 7 | ≈ Median* worst case | E3 | 2.500 | - |
