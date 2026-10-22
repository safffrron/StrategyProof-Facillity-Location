# Strategyproof Facility Location with Predictions, Blocked Zones and Two-Peaked Agents

**CS 6001 Course Project (Project 1) — IIT Bombay**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-85%20passed-brightgreen.svg)]()

| Team member | Roll No. |
|---|---|
| Prajwal Nayak | 22B4246 |
| Siva Kumar Dogga | 24M1528 |
| Adnan Khan | 25M0843 |
| Kislay Aditya Oj | 25M2111 |

## The problem

A planner places one facility on a line. Agents report their ideal locations and may lie. Some stretches of the
line are **blocked**, and a machine-learned model gives a **prediction** of the best location: a point or an
interval. We want mechanisms that are **strategyproof** (no agent gains by lying), near-optimal when the prediction
is right (**consistency**) and never too bad when it is wrong (**robustness**), for the maximum cost (worst-off
agent) and the social cost (total distance).

The full write-up is [`report/report.pdf`](report/report.pdf). Statements and proofs are also in
[`docs/RESULTS.md`](docs/RESULTS.md), and the novelty check against the literature is in
[`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md).

## Results at a glance

Numbering follows the report. η is the prediction error and ω the interval width, both normalised by the optimum.
b and c are the peak distance and minimum cost of double-peaked agents.

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

## How the claims are checked

- **Proofs** for every result are in the report's appendix.
- **Exhaustive strategyproofness checks**, run on grids over every profile, agent, misreport and prediction: about 2.9 × 10⁸ deviations for the single-peaked rules (E1) and 1.8 × 10⁹ for the double-peaked ones (E13).
- **Adversarial search**: a cluster-aware search attains all 52 stated bounds (E6).
- **Exact optimisation over all mechanisms** on small instances: MILP and LP (E8).
- **Exact certificates** for the computer-assisted lower bounds. The LP's dual weights are rechecked in rational arithmetic by code independent of the solver (E12, E13). The method follows Meir (SAGT 2019) and Chan, Lin & Wang (arXiv 2609.12792, 2026); the results are new.

## Repository map

```text
report/             report.tex, report.pdf               (the write-up)
docs/               RESULTS.md, LITERATURE_REVIEW.md, READING_LIST.md
src/                environment.py, agents.py, mechanisms.py, validator.py, visualization.py   (Week 1–2 baseline)
                    learning_augmented.py   mechanisms with predictions, intervals, snapping, randomisation
                    fast_validator.py       vectorised exhaustive strategyproofness checker
                    experiment_utils.py     predictions with controlled error, instance sampling, adversarial search
                    candidate_lower_bound.py   LP lower bounds + exact certificate checker (Theorem 17)
                    double_peaked.py, double_peaked_lp.py   two-peaked agents: mechanisms, checker, lower bounds
experiments/        e1–e13 (one script per experiment), run_all.py
results/, figures/  tables and figures produced by the experiments (certificates in results/e12_*, results/e13_*)
tests/              85 unit tests
scripts/            run_week1_2_demo.py  (Week 1–2 environment and validator demo)
```

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python3 -m pytest tests/
```

```bash
python3 experiments/run_all.py
```

The last command regenerates every table and figure in about 10 minutes. For the largest lower-bound LPs, install the
optional `highspy` and run `experiments/e12_candidate_lower_bound.py --full`. To start from zero background, use
[`docs/READING_LIST.md`](docs/READING_LIST.md).

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
