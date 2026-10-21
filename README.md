# Strategyproof Facility Location with Interval Predictions and Feasibility Constraints

**CS 6001 Course Project — Department of Computer Science & Engineering, IIT Bombay**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-84%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👥 Authors & Team
- **Prajwal Nayak** — Roll No. `22B4246`
- **Siva Kumar Dogga** — Roll No. `24M1528` ([24m1528@iitb.ac.in](mailto:24m1528@iitb.ac.in))
- **Adnan Khan** — Roll No. `25M0843`
- **Kislay Aditya Oj** — Roll No. `________`

---

## 📌 Project Overview & Motivation

In classical theoretical mechanism design for facility location on a line (Moulin 1980; Procaccia & Tennenholtz 2009/2013), two foundational assumptions are standard:
1. **Unconstrained Physical Space**: The facility can be placed anywhere along a continuous line $[0, L]$.
2. **Prior-Free / Point Advice**: Machine learning models or external hints either do not exist or provide a perfectly certain single point $\hat{y}$.

In real-world urban and infrastructure planning (such as placing a transit hub, emergency charging station, or campus router), both assumptions break down:
1. **Blocked Zones & Feasibility Constraints**: Physical obstacles, private land, water bodies, or restricted corridors prevent placing facilities in certain sub-intervals. The feasible set $F \subset [0, L]$ is a disconnected union of intervals:
   $$F = \bigcup_{k=1}^m [a_k, b_k]$$
2. **Interval Predictions**: Machine learning models output confidence intervals $[\hat{p}_{\text{low}}, \hat{p}_{\text{high}}]$ reflecting epistemic uncertainty rather than point estimates.

Existing learning-augmented mechanisms (Agrawal et al. 2022/2024; Fang et al. 2025 for candidate locations) project a point prediction onto a bounding interval. Our proposal conjectured that snapping such outputs to the nearest feasible location would create exploitable discontinuities. **It does not**: snapping any generalised median with a fixed tie-break stays strategyproof (Walsh 2021; verified exhaustively here).

Headline findings (statements, proofs and novelty labels in [`docs/RESULTS.md`](docs/RESULTS.md); the full write-up is [`report/report.pdf`](report/report.pdf)):

- **The price of an infeasible prediction is exactly 2.**
  - When the prediction may fall in a blocked zone, which any interval or any zone-unaware model produces, MinMaxP\* has ratio $\min(1+2\eta, 3)$, and for small errors ($\eta \le 1/4$) no deterministic strategyproof mechanism does better.
  - For feasible predictions the ratio is $1+\eta$ (Fang et al.).
  - Universally truthful **randomised snapping** achieves $1+\eta$ in expectation, and that is optimal.
- **An interval prediction is worth exactly its midpoint** for max cost: $1+\omega/2$ on the open line and $1+\omega$ with blocked zones are tight for all deterministic mechanisms (for $\omega \le 1/2$), and randomisation gives $1+\omega/2$. For social cost, clamping the median into the interval is the safe design.
- **Exact finite-n social-cost frontier** for the confidence family, $(3n-1-k)/(n+1+k)$ against $(3n+k-1)/(n-k+1)$. It refines Fang et al.'s asymptotic result, and randomised mixtures with the spike mechanism beat it.
- **Asymmetric costs:** snapping fails, but rules on agents' favourite feasible points are strategyproof for every single-peaked cost, with tight bounds $2+\alpha$ and $2+1/\alpha$.
- **Calibrated intervals from data:** conformal intervals plus strategyproof interval mechanisms give per-day certificates and much better worst days. A good point prediction still wins on average.
- **Randomisation without predictions cannot reach 2.** With candidate locations, every truthful-in-expectation mechanism has max-cost ratio at least 2.279, so Tang et al.'s (2020) lower bound of 2 is not tight. The proof is computer-assisted: LP certificates checked in exact rational arithmetic (`results/e12_certificates/`).
- **Two peaks per agent.** With double-peaked preferences (Filos-Ratsikas et al. 2017) the literal MinMaxP analogue is manipulable. A randomised *window* rule is truthful in expectation: it trusts the prediction within b of a median and otherwise flips between the median's two peaks. It is 1-consistent and (1 + b/c)-robust for social cost, and for max cost it is optimal among 1-consistent mechanisms when b ≤ c/2. Without predictions, certified LPs give the first lower bound above 1 for randomised mechanisms and social cost (1.0438) and raise the max-cost bound from 3/2 to 1.5186.
- **Negative results:** the course brief's "prediction if inside, else median", "nearest feasible point to the prediction", "snap toward the prediction", and blocked-zone versions of LRM are all manipulable.

---

## 📅 Roadmap & Milestones

| Phase | Milestone Description | Status |
| :--- | :--- | :---: |
| **Week 1–2** | **Discretized Environment, Baseline Median Mechanism, and Computational Strategyproofness Validator** | **Completed** |
| **Week 3–4** | Standard Agrawal Single-Point Mechanism & Adaptation to Interval Predictions with Blocked Zones | **Completed** |
| **Week 5** | Large-Scale Simulations: Consistency-Robustness Curves Across Error & Blocked Density | **Completed** |
| **Week 6–7** | Adversarial Search for Worst-Case Bounds, Empirical Validation, and Final Report | Search, lower bounds, randomisation, asymmetric costs, conformal pipeline, randomised lower bound with candidates **completed**; report drafted |

---

## 🏗️ Repository Architecture

```text
CS 6001 Project/
├── README.md                           # Master project documentation
├── requirements.txt                    # Python package dependencies
├── proposal.pdf                        # Initial submitted course project proposal
├── src/
│   ├── __init__.py                     # Package entry point
│   ├── environment.py                  # 1D metric domain, feasible/blocked intervals, nearest projection
│   ├── agents.py                       # Synthetic agent generator (Uniform, Gaussian, Bimodal, Clustered)
│   ├── mechanisms.py                   # Baseline mechanisms (Median, Naive Projected, Mean), optimal solvers
│   ├── validator.py                    # Brute-force strategyproofness verification engine
│   └── visualization.py                # Academic plotting suite for environments, landscapes, and violations
├── tests/
│   ├── __init__.py
│   ├── test_environment.py             # 7 unit tests for intervals, complement, grid snapping, projections
│   ├── test_agents.py                  # 6 unit tests for parametric distributions, bounds, and seeding
│   ├── test_mechanisms.py              # 6 unit tests for medians, tie-breaks, SC/MC costs, exact solvers
│   └── test_validator.py               # 4 unit tests verifying truthfulness detection & manipulation flags
├── scripts/
│   └── run_week1_2_demo.py             # End-to-end execution script demonstrating all Week 1-2 components
└── figures/                            # Publication-grade diagnostic figures generated by demo
    ├── week1_2_environment_setup.png
    ├── week1_2_median_truthful_landscape.png
    ├── week1_2_mean_deviation_landscape.png
    └── week1_2_mean_manipulation_counterexample.png
```

---

## 🔬 Core Components Implemented in Week 1–2

### 1. The Discretized Environment (`src/environment.py`)
- **Metric Domain**: Continuous line $[0, L]$ discretized into $N_{\text{grid}}$ points with uniform step size $\Delta x = \frac{L}{N_{\text{grid}} - 1}$.
- **Interval Algebra**: Closed intervals $[a_k, b_k]$ with automatic merging, complementation (deriving feasible sets $F$ from blocked obstacles $B$, and vice versa), and metric distance.
- **Deterministic Nearest Projection**:
  $$y_{\text{proj}} = \arg\min_{f \in F} |x - f|$$
  with deterministic boundary tie-breaking (`left` or `right`).
- **Random Obstacle Generator**: Generates environments with target obstacle counts $m$ and blocked density $\rho \in [0, 0.9]$ using Dirichlet-distributed gap spacing.

### 2. Synthetic Agent Population (`src/agents.py`)
Provides sampling of agent peak profiles $x = (x_1, \dots, x_n)$ across multiple parametric distributions:
- **`uniform`**: Uniformly distributed over $[0, L]$.
- **`uniform_feasible`**: Uniformly distributed strictly over feasible intervals $F$.
- **`gaussian`**: Truncated normal $\mathcal{N}(\mu, \sigma^2)$ restricted to $[0, L]$.
- **`bimodal`**: Gaussian mixture model simulating two distinct demographic clusters.
- **`clustered`**: $k$-cluster mixture simulating multiple population epicenters.
- **`polarized`**: U-shaped Beta distribution concentrating agents near the boundaries.

### 3. Baseline Mechanisms & Exact Optimal Benchmarks (`src/mechanisms.py`)
- **`MedianMechanism` (Moulin 1980)**: For sorted reports $x_{(1)} \le \dots \le x_{(n)}$, selects $x_{(\lceil n/2 \rceil)}$. Deterministic left-median for even $n$ guarantees strict strategyproofness and optimal unconstrained social cost.
- **`NaiveProjectedMedianMechanism`**: Computes the unconstrained median and projects onto $F$.
- **`MeanMechanism`**: Arithmetic average included as a canonical manipulable benchmark to validate the testing engine.
- **Exact Optimal Solvers**:
  - **Optimal Social Cost (Utilitarian)**:
    $$\min_{y \in F} \sum_{i=1}^n |y - x_i|$$
    Exact global solver evaluates the finite set of extreme candidates $\{a_k, b_k\} \cup (\{x_i\} \cap F)$.
  - **Optimal Maximum Cost (Egalitarian)**:
    $$\min_{y \in F} \max_{i=1}^n |y - x_i|$$
    Exact global solver evaluates the projection of midpoint $\frac{\min(x) + \max(x)}{2}$ onto each feasible interval.
  - **Approximation Ratios**:
    $$\alpha_{\text{SC}} = \frac{SC(y, x)}{SC^*(x, F)}, \qquad \alpha_{\text{MC}} = \frac{MC(y, x)}{MC^*(x, F)}$$

### 4. Brute-Force Strategyproofness Validator (`src/validator.py`)
A computational engine that verifies incentive compatibility (dominant-strategy truthfulness):
$$\forall i \in \{1, \dots, n\}, \; \forall x'_i \in G: \quad |f(x_i, x_{-i}) - x_i| \le |f(x'_i, x_{-i}) - x_i| + \epsilon$$
- **Deviation Counterexample Extraction**: Identifies the exact agent, true peak, profitable misreport, facility shift, and personal welfare gain:
  $$\Delta_i = |f(x_i, x_{-i}) - x_i| - |f(x'_i, x_{-i}) - x_i| > \epsilon$$
- **Verification Modes**:
  1. `check_profile`: Full evaluation on an individual agent vector.
  2. `verify_distribution`: Monte Carlo verification across $M$ synthetic profiles sampled from any generator.
  3. `verify_exhaustive_grid`: Rigorous combinatorial verification over all $|G|^n$ profile combinations.

---

## 🔭 Weeks 3–5: Predictions, Blocked Zones and Worst Cases

| Component | What it does |
| :--- | :--- |
| `src/learning_augmented.py` | Vectorised mechanisms: Median, Leftmost, Dictator, MinMaxP, `PhantomMedian[k]` (confidence family), `CenterMinMaxP` and `IntervalClamp` (interval predictions). Each can optionally snap to the nearest feasible point. Also includes the manipulable baselines and exact constrained optima |
| `src/fast_validator.py` | Exhaustive checker over every profile, agent, misreport and prediction (≈180× faster than `src/validator.py`); supports asymmetric costs |
| `src/experiment_utils.py` | Predictions with controlled error and width, typical-instance sampling (reuses `Environment1D` and `generate_agents`), and a cluster-aware adversarial hill-climbing search |
| `src/candidate_lower_bound.py` | LP lower bounds for randomised mechanisms with candidate locations (grid relaxation, symmetrisation, constraint generation) and an exact-arithmetic certificate checker |
| `src/double_peaked.py` | Double-peaked preferences: M1, M2, window mechanisms (PredMed, PredMinMax), exact optima and an exhaustive truthfulness-in-expectation checker |
| `experiments/e1`–`e13` | Strategyproofness table, error-doubling curves, interval width, typical instances, confidence frontier, bounds-versus-search table (52 bounds), tight-instance diagram, MILP/LP lower bounds, asymmetric costs, conformal intervals, randomisation, certified randomised lower bound with candidates, double-peaked preferences |
| `docs/RESULTS.md` | Theorems, proofs, conjectures and how each figure supports them |

Regenerate every table and figure (≈10 minutes; `python3 experiments/e12_candidate_lower_bound.py --full` also re-solves the large lower-bound LPs):
```bash
python3 experiments/run_all.py
```

Key figures: `figures/e11_randomization.png`, `figures/e3_interval_width.png`, `figures/e10_conformal.png`, `figures/e9_asymmetric.png`, `figures/e5_confidence_frontier.png`, `figures/e2_error_doubling.png`. Key tables: `results/sp_table.md`, `results/bounds_table.md`, `results/e8_lower_bounds.md`, `results/e9_asymmetric.md`, `results/e10_conformal.md`, `results/e12_candidate_lb.md`, `results/e13_tie.md`, `results/e13_performance.md`, `results/e13_lower_bounds.md`. Reading path for newcomers: [`docs/READING_LIST.md`](docs/READING_LIST.md).

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/Sivadogga007/CS-6001-Project.git
cd CS-6001-Project
pip install -r requirements.txt
```

### 2. Running Unit Tests
Execute the full test suite (84 passing tests):
```bash
python3 -m pytest -v tests/
```

### 3. Running the Week 1–2 Demo
Run the end-to-end demonstration:
```bash
python3 scripts/run_week1_2_demo.py
```

Sample output:
```text
======================================================================
CS 6001: Week 1–2 Milestone Execution & Verification
Strategyproof Facility Location with Discretization & Constraints
======================================================================

[Step 1] Constructing Discretized 1D Environment with Blocked Zones...
  Domain:              [0, 10.0]
  Grid Resolution:     1001 points (step = 0.0100)
  Blocked Intervals:   [Interval(start=2.5, end=4.0), Interval(start=6.5, end=8.0)]
  Feasible Intervals:  [Interval(start=0.0, end=2.5), Interval(start=4.0, end=6.5), Interval(start=8.0, end=10.0)]
  Blocked Density:     30.0%
  Feasible Density:    70.0%

[Step 2] Generating Synthetic Agent Populations...
  Generated 7 bimodal agents: [9.436, 1.854, 9.213, 8.282, 1.982, 7.811, 8.601]

[Step 3] Running Baseline Mechanisms & Exact Optimal Benchmarks...
  Unconstrained Median: y = 8.2819 (Feasible? True)
  Projected Median:     y = 8.2819 (Feasible? True)
  Optimal Social Cost:  y* = 8.2819 | Min SC = 15.6031
  Optimal Max Cost:     y* = 5.6451 | Min MC = 3.7913
  Projected Median SC:  15.6031 (Approx Ratio = 1.000)
  Projected Median MC:  6.4280 (Approx Ratio = 1.695)

[Step 4] Running Computational Strategyproofness Validator...
  Median Mechanism (Unconstrained): PASSED (100 profiles, 0 violations)
  Mean Mechanism (Manipulable):     FAILED (Violation caught, Gain = 0.001490)
```

---

## 📊 Diagnostic Visualizations

The demo script automatically outputs academic figures into `figures/`:
1. `week1_2_environment_setup.png`: Visualizes 1D metric line, feasible intervals (green), blocked zones (red hatched), synthetic agent positions, and placed facility.
2. `week1_2_median_truthful_landscape.png`: Plots agent personal cost as a function of reported position under the Median rule, demonstrating the global minimum at the truthful location.
3. `week1_2_mean_deviation_landscape.png`: Shows the cost landscape under the Mean mechanism, highlighting the profitable deviation zone where lying strictly improves personal welfare.
4. `week1_2_mean_manipulation_counterexample.png`: Side-by-side comparison of truthful reporting vs strategic misreporting.

---

## 📖 References
1. **P. Agrawal, E. Balkanski, V. Gkatzelis, T. Ou, and X. Tan.** *Learning-augmented mechanism design: leveraging predictions for facility location.* Mathematics of Operations Research, 49(4):2626–2651, 2024.
2. **H. Chan, A. Filos-Ratsikas, B. Li, M. Li, and C. Wang.** *Mechanism design for facility location problems: a survey.* IJCAI, pages 4356–4365, 2021.
3. **H. Moulin.** *On strategy-proofness and single peakedness.* Public Choice, 35(4):437–455, 1980.
4. **A. D. Procaccia and M. Tennenholtz.** *Approximate mechanism design without money.* ACM Transactions on Economics and Computation, 1(4):18, 2013.

---

## 📜 License
Academic Coursework — IIT Bombay.
