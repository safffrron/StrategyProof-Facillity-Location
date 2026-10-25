## (a) MinMaxP* with an exact median prediction (social cost)

| n | formula (3n-1)/(n+1) | tight instance | random worst | equals Median* |
|---|---|---|---|---|
| 3 | 2.0 | 2.0 | 1.9822 | True |
| 5 | 2.3333 | 2.3333 | 2.1295 | True |
| 7 | 2.5 | 2.5 | 2.0237 | True |
| 9 | 2.6 | 2.6 | 1.9787 | True |
| 11 | 2.6667 | 2.6667 | 1.9571 | True |
| 15 | 2.75 | 2.75 | 1.9113 | True |
| 21 | 2.8182 | 2.8182 | 1.7254 | True |

## (b) Best deterministic SP consistency, n = 3, F = {-1, 1} (MILP)

| step s | grid points | best consistency (MILP) | (4-s)/(2+s) | seconds |
|---|---|---|---|---|
| 0.5 | 7 | 1.4 | 1.4 | 0 |
| 0.25 | 13 | 1.6667 | 1.6667 | 1 |
| 0.125 | 25 | 1.8235 | 1.8235 | 22 |
