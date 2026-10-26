# Strategyproof Facility Location with Predictions, Blocked Zones and Two-Peaked Agents

**CS 6001 Course Project — IIT Bombay**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-93%20passed-brightgreen.svg)](#run-the-tests)

| Team member | Roll No. |
|---|---|
| Prajwal Nayak | 22B4246 |
| Siva Kumar Dogga | 24M1528 |
| Adnan Khan | 25M0843 |
| Kislay Aditya Oj | 25M2111 |

📄 **Report:** [`report/report.pdf`](report/report.pdf) · **Proposal:** [`docs/proposal.pdf`](docs/proposal.pdf)

---

## The problem

A town must build one facility, such as a library, on a road. Each resident reports where they live, and may
lie to pull the facility closer to themselves. Some stretches of the road are **blocked** (a river, a railway,
private land), so the facility cannot go there. A machine-learning model gives the planner a **prediction** of
the best location, either a point or an interval.

We design rules (*mechanisms*) that are

- **strategyproof:** no resident ever gains by lying;
- **consistent:** close to optimal when the prediction is right;
- **robust:** never too bad when the prediction is wrong;

for two goals: the **maximum cost** (the longest walk anyone makes) and the **social cost** (the total walking).
We also study residents with two acceptable locations (*double-peaked* preferences).

**Example.** Residents at 0 and 2 and a blocked zone (1, 1.59). The best location is 1, where the worst-off
resident walks 1. If the model predicts 1.3, inside the zone, the natural rule moves it to the nearest open
spot, 1.59, and the worst-off resident now walks 1.59. A prediction error of 0.3 cost almost 0.6: blocked
zones *double* the error (Theorem 4), and no deterministic strategyproof rule can avoid this (Theorem 10).

## Main results

Numbering follows the report. η is the prediction error and ω the interval width, both normalised by the
optimal cost; b and c are the peak distance and minimum cost of double-peaked agents.

| # | Result | Status |
|---|---|---|
| L1 | Snapping a generalised median to the nearest feasible point stays strategyproof, which refutes the proposal's conjecture | Known (Walsh 2021) |
| P2 | The course brief's rule, "nearest feasible point to the prediction", "snap toward the prediction", snapping under asymmetric costs, and blocked-zone LRM are all manipulable | New counterexamples |
| T4 | MinMaxP\* with a prediction that may lie in a blocked zone has ratio min{1 + 2η, 3}; with an allowed prediction it is 1 + η (Fang et al. 2025) | New / Known |
| T6–T7 | Interval predictions: an interval is worth its midpoint for max cost; clamping the median into the interval is the safe social-cost design | New (the open-line upper bound follows from Chan, Lin & Wang 2025) |
| T9 | Exact finite-n consistency/robustness of the confidence family for social cost | Refines Fang et al. |
| T10–T12 | No deterministic strategyproof mechanism beats 1 + 2η (point), 1 + ω (interval, blocked) or 1 + ω/2 (interval, open line), for small η and ω | New |
| T13 | Randomised snapping of the prediction achieves 1 + η and 1 + ω/2 in expectation, which is optimal | New |
| P14–P15 | Asymmetric costs: rules on agents' favourite feasible points are strategyproof, with tight bounds 2 + α and 2 + 1/α | Strategyproofness classical (Moulin 1980; Barberà et al. 1998); bounds new |
| P16 | Conformal-prediction intervals turn coverage into per-day guarantees | New for facility location |
| T17 | No prediction, candidate locations: every randomised truthful-in-expectation mechanism has max-cost ratio ≥ **2.279**, so Tang et al.'s (2020) lower bound of 2 is not attainable | New, computer-assisted |
| P19–T22 | Double-peaked preferences: the literal MinMaxP analogue is manipulable; randomised *window* rules are truthful in expectation; PredMed is (1, 1 + b/c) for social cost and PredMinMax is (1, max{1 + 2b/c, 2}) for max cost, optimal among 1-consistent rules when b ≤ c/2 | New |
| P23–P24 | Double-peaked preferences, no prediction: first randomised social-cost lower bound above 1 (**84/79**, by hand); max-cost bound raised from 3/2 to **1.5264** | New |
| T25–C26 | Predictions of the *unconstrained* optimum: MinMaxP\* is 1-consistent for them and has ratio min{1 + 2η_m, 3}, which is optimal; intervals around it give 1 + ω | New |
| P27–T28 | Social cost: a perfect prediction of the median is worth nothing: MinMaxP\* becomes Median\* with ratio (3n−1)/(n+1), and no deterministic SP mechanism does better | New |

## How the claims are checked

- **Proofs** for every result are in the report's appendix.
- **Exhaustive strategyproofness checks**, run on grids over every profile, agent, misreport and prediction: about 2.9 × 10⁸ deviations for the single-peaked rules (E1) and 1.8 × 10⁹ for the double-peaked ones (E13).
- **Adversarial search**: a cluster-aware search attains all 52 stated bounds (E6).
- **Exact optimisation over all mechanisms** on small instances: MILP and LP (E8).
- **Exact certificates** for the computer-assisted lower bounds. The LP's dual weights are rechecked in rational arithmetic by code independent of the solver (E12, E13). The method follows Meir (SAGT 2019) and Chan, Lin & Wang (arXiv 2609.12792, 2026); the results are new.

## Repository map

```text
report/             report.tex, report.pdf               (the write-up)
docs/               RESULTS.md, LITERATURE_REVIEW.md, NOVELTY_CHECK.md, READING_LIST.md
src/                environment.py, agents.py, mechanisms.py, validator.py, visualization.py   (Week 1–2 baseline)
                    learning_augmented.py   mechanisms with predictions, intervals, snapping, randomisation
                    fast_validator.py       vectorised exhaustive strategyproofness checker
                    experiment_utils.py     predictions with controlled error, instance sampling, adversarial search
                    candidate_lower_bound.py   LP lower bounds + exact certificate checker (Theorem 17)
                    double_peaked.py, double_peaked_lp.py   two-peaked agents: mechanisms, checker, lower bounds
experiments/        e1–e15 (one script per experiment), run_all.py
reproduce.py        one-command reproduction with number-by-number checks (see REPRODUCE.md)
results/, figures/  tables and figures produced by the experiments (certificates in results/e12_*, results/e13_*)
tests/              99 unit tests
scripts/            run_week1_2_demo.py  (Week 1–2 environment and validator demo)
```

## Quick start

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` checks your environment, runs the 99 unit tests, regenerates every table and figure in the
report, and compares every regenerated number with the committed results. It ends with `ALL REPRODUCED`
or lists exactly what differed (about 15 minutes; `--quick` takes about 5). Setup for Windows, macOS and
Linux, and a table mapping each result in the report to its script and output files, are in
[`REPRODUCE.md`](REPRODUCE.md).

To run one experiment on its own: `python experiments/e14_unconstrained_target.py`. For the largest
lower-bound LPs, install the optional `highspy` and use `python reproduce.py --full`. To start from zero
background, use [`docs/READING_LIST.md`](docs/READING_LIST.md).

Full statements and proofs are in the report's appendix and in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Repository layout

```text
CS-6001-Project/
├── README.md               this file
├── requirements.txt        pinned package versions
├── report/                 report.tex and report.pdf (the write-up)
├── docs/                   RESULTS.md (statements and proofs), LITERATURE_REVIEW.md,
│                           NOVELTY_CHECK.md, READING_LIST.md, proposal.pdf
├── src/                    the library
│   ├── environment.py, agents.py, mechanisms.py, validator.py, visualization.py   Week 1–2 baseline
│   ├── learning_augmented.py    mechanisms with predictions, intervals, snapping, randomisation
│   ├── fast_validator.py        vectorised exhaustive strategyproofness checker
│   ├── experiment_utils.py      predictions with controlled error, instance sampling, adversarial search
│   ├── candidate_lower_bound.py LP lower bounds and exact certificate checker (Theorem 17)
│   └── double_peaked.py, double_peaked_lp.py   two-peaked agents: mechanisms, checker, lower bounds
├── experiments/            e1–e15, one script per experiment; run_all.py runs them all
├── results/                tables produced by the experiments (certificates in e12_*, e13_*)
├── figures/                figures produced by the experiments (used in the report)
├── tests/                  93 unit tests
└── scripts/                run_week1_2_demo.py (Week 1–2 environment and validator demo)
```

---

## Setup

Python 3.11 or newer. A virtual environment keeps the pinned versions separate from anything else on your
machine.

**Windows (PowerShell)**

```powershell
cd CS-6001-Project
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then use `.\.venv\Scripts\python.exe` wherever this page says `python`. (If `python3` on your machine belongs to
another program, for example `C:\iverilog\gtkwave\bin\python3.exe`, it fails with
`No module named 'encodings'`; the virtual environment avoids that.)

**macOS / Linux**

```bash
cd CS-6001-Project
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the tests

```bash
python -m pytest tests/
```

All 93 tests should pass in under a minute. They check the mechanisms, the strategyproofness checker, the
proved bounds and their tight instances.

## Reproduce the experiments

Every experiment writes its tables to `results/` and its figures to `figures/`. All randomness uses fixed
seeds.

```bash
python experiments/e14_unconstrained_target.py   # one experiment
python experiments/run_all.py                     # all fifteen (about 10–20 minutes)
```

The largest lower-bound LPs are stored as certificates and re-checked in exact rational arithmetic on every
run. To re-solve them from scratch, install `highspy` and run
`python experiments/e12_candidate_lower_bound.py --full` (and likewise `e13_double_peaked.py --full`); this
takes about two hours.

| Exp. | Script | Backs | Outputs |
|---|---|---|---|
| E1 | `e1_strategyproofness.py` | Lemma 1, Prop. 2: exhaustive strategyproofness checks (≈2.9 × 10⁸ deviations) | `results/sp_checks.csv`, `sp_table.md` |
| E2 | `e2_point_prediction_error.py` | Theorem 4: ratio vs point-prediction error | `results/e2_*`, `figures/e2_error_doubling` |
| E3 | `e3_interval_predictions.py` | Theorems 6–7: interval predictions vs width | `results/e3_*`, `figures/e3_interval_width` |
| E4 | `e4_typical_instances.py` | Typical instances: blocked density, agent distributions | `results/e4_*`, `figures/e4_blocked_density` |
| E5 | `e5_confidence_frontier.py` | Theorem 9, Prop. 5: the confidence frontier | `results/e5_frontier.csv`, `figures/e5_confidence_frontier` |
| E6 | `e6_bounds_table.py` | Every stated bound vs adversarial search | `results/bounds_table.*` |
| E7 | `e7_doubling_diagram.py` | Diagram of Theorem 4's tight instance | `figures/e7_doubling_diagram` |
| E8 | `e8_lower_bounds_milp.py` | Theorems 10–11: integer program over all strategyproof mechanisms | `results/e8_lower_bounds.md` |
| E9 | `e9_asymmetric_costs.py` | Props. 14–15: asymmetric costs | `results/e9_*`, `figures/e9_asymmetric` |
| E10 | `e10_conformal_intervals.py` | Prop. 16: calibrated (conformal) intervals | `results/e10_*`, `figures/e10_conformal` |
| E11 | `e11_randomization.py` | Theorem 13: what randomisation buys | `results/e11_*`, `figures/e11_randomization` |
| E12 | `e12_candidate_lower_bound.py` | Theorem 17: certified randomised lower bound 2.279 | `results/e12_*`, `figures/e12_candidate_lb` |
| E13 | `e13_double_peaked.py` | Props. 19–24: double-peaked preferences | `results/e13_*`, `figures/e13_double_peaked` |
| E14 | `e14_unconstrained_target.py` | Theorem 25, Cor. 26: predictions of the unconstrained optimum | `results/e14_*`, `figures/e14_unconstrained_target` |
| E15 | `e15_social_cost_target.py` | Prop. 27, Theorem 28: median predictions are worthless for social cost | `results/e15_median_predictions.md`, `figures/e15_social_cost_target` |

## How the claims are checked

- **Proofs** for every result are in the report's appendix.
- **Exhaustive strategyproofness checks** on grids, over every profile, agent, misreport and prediction:
  about 2.9 × 10⁸ deviations for the single-peaked rules (E1) and 1.8 × 10⁹ for the double-peaked ones (E13).
- **Adversarial search** attains every stated bound (E6).
- **Exact optimisation over all mechanisms** on small instances: integer programs and LPs (E8, E15).
- **Exact certificates** for the computer-assisted lower bounds: the LP's dual weights are rechecked in
  rational arithmetic by code independent of the solver (E12, E13).

---

## Project history

| Branch | Contents |
|---|---|
| `main` | Week 1–2 baseline: environment with blocked zones, median rules, exact optima, strategyproofness validator |
| `kislay/predictions` | Learning-augmented mechanisms, lower bounds, randomisation, asymmetric costs, conformal intervals, double-peaked preferences (E1–E13), report |
| `prajwal-unconstrained-target` | Predictions of the unconstrained optimum: Theorem 25, Corollary 26, Proposition 27, Theorem 28 (E14–E15) |

## Key references

- P. Agrawal, E. Balkanski, V. Gkatzelis, T. Ou, X. Tan. Learning-augmented mechanism design: leveraging predictions for facility location. *EC 2022; Math. Oper. Res.* 2024.
- J. Fang, Q. Fang, W. Liu, Q. Nong, A. A. Voudouris. Mechanism design with predictions for facility location games with candidate locations. *J. Comb. Optim.* 49:74, 2025.
- Z. Tang, C. Wang, M. Zhang, Y. Zhao. Mechanism design for facility location games with candidate locations. *COCOA 2020*.
- A. Filos-Ratsikas, M. Li, J. Zhang, Q. Zhang. Facility location with double-peaked preferences. *JAAMAS* 31, 2017.
- T. Walsh. Strategy proof mechanisms for facility location at limited locations. *PRICAI 2021*.
- H. Chan, A. Filos-Ratsikas, B. Li, M. Li, C. Wang. Mechanism design for facility location problems: a survey. *IJCAI 2021*.
- H. Moulin. On strategy-proofness and single peakedness. *Public Choice* 35, 1980.
- A. D. Procaccia, M. Tennenholtz. Approximate mechanism design without money. *EC 2009; ACM TEAC* 2013.

Academic coursework, IIT Bombay.
