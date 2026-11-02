## E8a - Consistency and robustness of PhantomMedian[k], n = 7, no blocked zones (search values are lower bounds on the worst case)

| objective | k | consistency (search) | consistency (theory) | robustness (search) | robustness (theory) |
|---|---|---|---|---|---|
| max cost | 0 | 2.0 | 2.0 | 2.0 | 2.0 |
| max cost | 2 | 2.0 | 2.0 | 2.0 | 2.0 |
| max cost | 4 | 2.0 | 2.0 | 2.0 | 2.0 |
| max cost | 6 | 1.0 | 1.0 | 2.0 | 2.0 |
| social cost | 0 | 1.0 | 1.0 | 1.0 | 1.0 |
| social cost | 2 | 1.0 | 1.0 | 1.3333 | 1.3333 |
| social cost | 4 | 1.0 | 1.0 | 2.5 | 2.5 |
| social cost | 6 | 1.0 | 1.0 | 6.0 | 6.0 |

## E8b - Max-cost ratio against prediction error: mean over 20,000 random profiles for each k, and MinMaxP's worst case found by search

| eta | mean, k = 0 | mean, k = 2 | mean, k = 4 | mean, k = 6 | MinMaxP worst (search) | MinMaxP bound |
|---|---|---|---|---|---|---|
| 0.0 | 1.3113 | 1.0844 | 1.0103 | 1.0 | 1.0 | 1.0 |
| 0.1 | 1.3133 | 1.1445 | 1.1039 | 1.1 | 1.1 | 1.1 |
| 0.2 | 1.3115 | 1.2049 | 1.1961 | 1.2 | 1.2 | 1.2 |
| 0.3 | 1.3162 | 1.2589 | 1.286 | 1.3 | 1.3 | 1.3 |
| 0.4 | 1.3135 | 1.3084 | 1.3719 | 1.4 | 1.4 | 1.4 |
| 0.5 | 1.3117 | 1.3467 | 1.4518 | 1.5 | 1.5 | 1.5 |
| 0.6 | 1.3134 | 1.3796 | 1.5237 | 1.6 | 1.6 | 1.6 |
| 0.8 | 1.3103 | 1.4105 | 1.6352 | 1.8 | 1.8 | 1.8 |
| 1.0 | 1.3132 | 1.413 | 1.6763 | 2.0 | 2.0 | 2.0 |
| 1.25 | 1.3111 | 1.416 | 1.6769 | 2.0 | 2.0 | 2 |
| 1.5 | 1.313 | 1.4164 | 1.6775 | 2.0 | 2.0 | 2 |
