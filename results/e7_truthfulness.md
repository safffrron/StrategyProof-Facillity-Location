E7a - Exhaustive truthfulness-in-expectation checks, double-peaked costs (b = c = 1), peaks and misreports on a 0.25-grid of [-3, 3], predictions on a 0.5-grid of [-4, 4].

| n | mechanism | deviations checked | max gain | truthful | example |
|---|---|---|---|---|---|
| 2 | M1 | 265,625 | 0.0000 | yes |  |
| 2 | PredMed | 265,625 | 0.0000 | yes |  |
| 2 | PredMinMax | 265,625 | 0.0000 | yes |  |
| 2 | Window[1] | 265,625 | 0.0000 | yes |  |
| 2 | clamp(p, x_1 - b, x_n + b) | 265,625 | 1.0000 | NO | p=-3, peaks (-3.0, -1.0), agent 1 reports -1 |
| 2 | median's peak nearest the prediction | 265,625 | 1.7500 | NO | p=-2.5, peaks (-2.5, -0.75), agent 2 reports -2.75 |
| 3 | M1 | 3,453,125 | 0.0000 | yes |  |
| 3 | PredMed | 3,453,125 | 0.0000 | yes |  |
| 3 | PredMinMax | 3,453,125 | 0.0000 | yes |  |
| 3 | Window[1] | 3,453,125 | 0.0000 | yes |  |
| 3 | clamp(p, x_1 - b, x_n + b) | 3,453,125 | 1.0000 | NO | p=-3, peaks (-3.0, -1.0, -1.0), agent 1 reports -1 |
| 3 | median's peak nearest the prediction | 3,453,125 | 1.7500 | NO | p=-2.5, peaks (-2.75, -2.5, -0.75), agent 3 reports -3 |
| 4 | M1 | 31,078,125 | 0.0000 | yes |  |
| 4 | PredMed | 31,078,125 | 0.0000 | yes |  |
| 4 | PredMinMax | 31,078,125 | 0.0000 | yes |  |
| 4 | Window[1] | 31,078,125 | 0.0000 | yes |  |
| 4 | clamp(p, x_1 - b, x_n + b) | 31,078,125 | 1.0000 | NO | p=-3, peaks (-3.0, -1.0, -1.0, -1.0), agent 1 reports -1 |
| 4 | median's peak nearest the prediction | 31,078,125 | 1.7500 | NO | p=-2.5, peaks (-2.75, -2.5, -0.75, -0.75), agent 3 reports -3 |
| 5 | M1 | 217,546,875 | 0.0000 | yes |  |
| 5 | PredMed | 217,546,875 | 0.0000 | yes |  |
| 5 | PredMinMax | 217,546,875 | 0.0000 | yes |  |
| 5 | Window[1] | 217,546,875 | 0.0000 | yes |  |
| 5 | clamp(p, x_1 - b, x_n + b) | 217,546,875 | 1.0000 | NO | p=-3, peaks (-3.0, -1.0, -1.0, -1.0, -1.0), agent 1 reports -1 |
| 5 | median's peak nearest the prediction | 217,546,875 | 1.7500 | NO | p=-2.5, peaks (-3.0, -2.75, -2.5, -0.75, -0.75), agent 4 reports -3 |
