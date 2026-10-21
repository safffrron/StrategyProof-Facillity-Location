Worst case found by adversarial search, two blocked zones, n <= 7, asymmetric costs (facility on an agent's left costs alpha per unit)

| alpha | Tops-MinMaxP consistency | Tops-MinMaxP robustness (bound 2+alpha) | Tops-Rightmost (bound 2+1/alpha) | Tops-Median SC | Tops-Quantile SC |
|---|---|---|---|---|---|
| 1 | 1.000 | 3.000 (3.000) | 3.000 (3.000) | 2.500 | 2.500 |
| 1.5 | 1.000 | 3.500 (3.500) | 2.667 (2.667) | 2.875 | 3.221 |
| 2 | 1.000 | 4.000 (4.000) | 2.500 (2.500) | 3.250 | 3.250 |
| 3 | 1.000 | 5.000 (5.000) | 2.333 (2.333) | 3.999 | 4.332 |
| 4 | 1.000 | 6.000 (6.000) | 2.250 (2.250) | 4.749 | 4.124 |
| 5 | 1.000 | 7.000 (7.000) | 2.200 (2.200) | 5.499 | 5.798 |

Exhaustive strategyproofness check under asymmetric costs (alpha = 3, n = 3, two zones)

| Mechanism | Strategyproof at alpha = 3? | Counterexample |
|---|---|---|
| Median* | **no** | peaks=(0.25, 0.25, 0.3): agent at 0.25 reports 0.3; facility 0.15 -> 0.35, cost 0.3 -> 0.1 |
| MinMaxP* | **no** | peaks=(0.25, 0.3, 0.3), prediction=-0.2: agent at 0.25 reports 0.3; facility 0.15 -> 0.35, cost 0.3 -> 0.1 |
| Tops[k=0, a=3] | yes | - |
| Tops[k=2, a=3] | yes | - |
| TopsQuantile[q=0.75, a=3] | yes | - |
