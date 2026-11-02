# Strategyproof Facility Location with Blocked Zones and Predictions

Code for our CS 6001 course project (IIT Bombay, 2026), *Project 1: Strategyproof Facility Location on a Line*.

A planner places one facility on a road. Agents report their locations and may lie, some stretches of the road
are blocked, and a machine-learned model predicts where the facility should go; the prediction may itself fall
in a blocked zone. This repository contains the mechanisms we study and every experiment in our report.
Section, theorem and figure numbers below refer to the report.

| Member | Roll No. |
|---|---|
| Prajwal Nayak | 22B4246 |
| Siva Kumar Dogga | 24M1528 |
| Adnan Khan | 25M0843 |
| Kislay Aditya Oj | 25M2111 |

## Setup

Python 3.10 or later.

```bash
git clone https://github.com/safffrron/StrategyProof-Facillity-Location.git
cd StrategyProof-Facillity-Location
pip install -r requirements.txt
```

## Run

Unit tests (about 10 seconds):

```bash
python3 -m pytest
```

All experiments, which regenerate `results/` and `figures/` (about six minutes on a laptop):

```bash
python3 experiments/run_all.py
```

Each experiment also runs on its own, for example:

```bash
python3 experiments/e4_randomised_snapping.py
```

Every experiment checks its own numbers against the proved bounds and stops with an error if one fails.

## Experiments

| Script | Report | What it shows | Output | Time |
|---|---|---|---|---|
| `e1_strategyproofness.py` | Sec. 3.3, Lemma 2, Thm 9 | Exhaustive check of every profile, agent, misreport and prediction on a grid (n = 3, 4, 5): the median, leftmost and dictator rules, PhantomMedian[k], MinMaxP\*, the midpoint rule and randomised snapping (in expectation) are strategyproof; the mean and the brief's example rule are not | `results/e1_strategyproofness.md` | 1 min |
| `e2_unconstrained_target.py` | Sec. 4.2, Thm 6 | Predictions of the unconstrained optimum m: adversarial search attains min{1 + 2η_m, 3}; 108,000 random instances never exceed it | `results/e2_worst_case.md`, `results/e2_typical.md` | 40 s |
| `e3_median_predictions.py` | Sec. 4.2, Prop 7, Thm 8 | A median prediction turns MinMaxP\* into Median\*, ratio (3n − 1)/(n + 1); an integer program over all deterministic strategyproof mechanisms returns (4 − s)/(2 + s) | `results/e3_median_predictions.md` | 20 s |
| `e4_randomised_snapping.py` | Sec. 4.3, Thms 3–5, 9–10, Figure 2 | Worst cases found by adversarial search: deterministic 1 + 2η and 1 + ω, randomised 1 + η and 1 + ω/2 | `results/e4_randomised_snapping.md`, `figures/e4_randomised_snapping.pdf` | 35 s |
| `e5_optimal_mechanisms.py` | Sec. 4.3 | Optimisation over all mechanisms (MILP for deterministic, LP for truthful in expectation): slopes 2, 1, 1 and 1/2 | `results/e5_optimal_mechanisms.md` | 1.5 min |
| `e6_candidate_lower_bound.py` | Sec. 4.4, Thm 11, Figure 3, Table 2 | With candidate sites {0, 2, 4}, every truthful-in-expectation mechanism has ratio at least 2.279; certificates re-checked in exact arithmetic | `results/e6_candidate_lower_bound.md`, `results/certificates/`, `figures/e6_candidate_lower_bound.pdf` | 5 s |
| `e7_double_peaked.py` | Sec. 4.5, Prop 12, Thms 13–14 | Double-peaked costs: window rules are truthful in expectation, the clamp rule is not; search attains the robustness bounds | `results/e7_truthfulness.md`, `results/e7_performance.md` | 2 min |
| `e8_confidence_sweep.py` | Sec. 3.3 (project brief) | The confidence family PhantomMedian[k] on the line: consistency and robustness against the theory for both objectives, and the max-cost ratio against the prediction error | `results/e8_confidence_sweep.md`, `figures/e8_confidence_sweep.pdf` | 10 s |

By default `e6_candidate_lower_bound.py` re-solves the two small linear programs and re-verifies every stored
certificate in `results/certificates/` in exact rational arithmetic. To re-solve the large grids as well (about
an hour and several GB of memory), install the HiGHS interface and pass `--full`:

```bash
pip install highspy
```

```bash
python3 experiments/e6_candidate_lower_bound.py --full
```

## The project brief, item by item

| Brief asks for | Where |
|---|---|
| Median, leftmost-agent and dictatorial mechanisms | `MedianRule`, `LeftmostRule`, `DictatorRule` in `src/learning_augmented.py` |
| Utilitarian and egalitarian optima, solved directly | `optimal_social_cost`, `optimal_max_cost` in `src/mechanisms.py` (batch versions in `src/learning_augmented.py`) |
| Exhaustive deviation checker that passes the median and flags the mean | `src/fast_validator.py`; `e1_strategyproofness.py` |
| Prediction-augmented mechanism with a confidence parameter, swept | `PhantomMedian[k]` (k copies of the prediction; k = 0 is the median, k = n - 1 is MinMaxP); `e8_confidence_sweep.py`. The brief's example rule, "the prediction if it lies between the extreme reports, else the median", is manipulable, and E1 shows a counterexample |
| Predictions of controlled error, ratio against error, consistency-robustness curve against the theory | `e8_confidence_sweep.py` (line); `e4_randomised_snapping.py` (with blocked zones) |
| Adversarial search for the worst instance, against the proved bound | `hill_climb` in `src/experiment_utils.py`; used by E2, E4, E7 and E8 |

## Repository layout

```text
src/
  environment.py            the road, its blocked zones, random layouts
  agents.py                 random agent profiles
  mechanisms.py             mechanism interface and exact optima
  learning_augmented.py     snapping, MinMaxP*, the midpoint rule, randomised snapping (vectorised)
  fast_validator.py         exhaustive strategyproofness / truthfulness-in-expectation checker
  experiment_utils.py       random instances and the adversarial search
  candidate_lower_bound.py  linear program and exact certificates for candidate sites
  double_peaked.py          double-peaked costs, M1 and the window rules
experiments/                e1-e8 (one script per claim) and run_all.py
results/                    tables written by the experiments, and the certificates
figures/                    Figures 2 and 3 of the report, and the confidence sweep
tests/                      unit tests
```
