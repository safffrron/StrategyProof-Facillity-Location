E6 - Lower bounds on the max-cost ratio of truthful-in-expectation mechanisms for two agents with candidate sites (Tang et al. 2020 proved 2). Each bound is recomputed in exact rational arithmetic from the stored certificate and rounded down.

| candidates | grid | grid points | misreports imposed | LP optimum | certified lower bound | multipliers (SP + ratio) | exact match | certificate |
|---|---|---|---|---|---|---|---|---|
| {0, 2} | basic | 25 | all | 1.997006 | 1.9970 | 3 + 1 | yes | C2_basic.json |
| {0, 2, 4} | basic | 39 | all | 2.084366 | 2.0843 | 257 + 5 | yes | C3_basic.json |
| {0, 2, 4} | rich | 87 | local | 2.134070 | 2.1340 | 788 + 8 | yes | C3_rich.json |
| {0, 2, 4} | rich + mirrored | 111 | local | 2.162282 | 2.1622 | 683 + 8 | yes | C3_mirror.json |
| {0, 2, 4} | dense + mirrored | 211 | local | 2.267424 | 2.2674 | 2052 + 12 | yes | C3_dense.json |
| {0, 2, 4} | geometric + mirrored | 299 | local | 2.279516 | 2.2795 | 8169 + 19 | yes | C3_geometric.json |
