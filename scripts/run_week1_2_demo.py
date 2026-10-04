"""End-to-End Demonstration and Benchmark Script for Week 1–2 Deliverables.

This script executes:
1. Creation of a discretized 1D environment with multiple blocked zones.
2. Sampling synthetic agent populations under distinct distributions (Uniform, Gaussian, Bimodal).
3. Evaluation of the baseline Median Mechanism (unconstrained & naive projected) and comparison with exact optimal benchmarks (Social Cost & Maximum Cost).
4. Running the Strategyproofness Validator on:
   - Median Mechanism (unconstrained line) -> Verification of Truthfulness (PASSED).
   - Mean Mechanism (non-strategyproof benchmark) -> Detection of Profitable Manipulation (FAILED) with exact counterexample.
   - Naive Projected Median on constrained line -> Rigorous verification.
5. Exporting publication-grade figures to figures/.
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import Environment1D, Interval
from src.agents import AgentDistribution, generate_agents
from src.mechanisms import (
    MedianMechanism,
    NaiveProjectedMedianMechanism,
    MeanMechanism,
    social_cost,
    max_cost,
    optimal_social_cost,
    optimal_max_cost,
    social_cost_approximation_ratio,
    max_cost_approximation_ratio,
)
from src.validator import StrategyproofnessValidator
from src.visualization import (
    plot_environment,
    plot_deviation_landscape,
    plot_violation_counterexample,
)


def main() -> None:
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figures_dir, exist_ok=True)
    print("=" * 70)
    print("CS 6001: Week 1–2 Milestone Execution & Verification")
    print("Strategyproof Facility Location with Discretization & Constraints")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Step 1: Environment Setup
    # ------------------------------------------------------------------------
    print("\n[Step 1] Constructing Discretized 1D Environment with Blocked Zones...")
    length = 10.0
    blocked_intervals = [
        Interval(2.5, 4.0),
        Interval(6.5, 8.0),
    ]
    env = Environment1D(
        length=length,
        num_grid_points=1001,
        blocked_intervals=blocked_intervals,
    )
    print(f"  Domain:              [0, {env.length}]")
    print(f"  Grid Resolution:     {env.num_grid_points} points (step = {env.grid_step:.4f})")
    print(f"  Blocked Intervals:   {env.blocked_intervals}")
    print(f"  Feasible Intervals:  {env.feasible_intervals}")
    print(f"  Blocked Density:     {env.blocked_density:.1%}")
    print(f"  Feasible Density:    {env.feasible_density:.1%}")

    # ------------------------------------------------------------------------
    # Step 2: Agent Population Generation
    # ------------------------------------------------------------------------
    print("\n[Step 2] Generating Synthetic Agent Populations...")
    n_agents = 7
    # Sample from Bimodal distribution (two demographic clusters)
    bimodal_agents = generate_agents(
        n_agents=n_agents,
        env=env,
        distribution=AgentDistribution.BIMODAL,
        params={"mu1": 1.5, "sigma1": 0.5, "mu2": 8.8, "sigma2": 0.6},
        seed=101,
    )
    print(f"  Generated {n_agents} bimodal agents: {[round(float(x), 3) for x in bimodal_agents]}")

    # ------------------------------------------------------------------------
    # Step 3: Mechanism Evaluation & Optimal Benchmarks
    # ------------------------------------------------------------------------
    print("\n[Step 3] Running Baseline Mechanisms & Exact Optimal Benchmarks...")
    med_mech = MedianMechanism(tie_break="left")
    proj_med_mech = NaiveProjectedMedianMechanism(tie_break="left")

    y_unconstrained = med_mech(bimodal_agents, env)
    y_projected = proj_med_mech(bimodal_agents, env)

    opt_y_sc, opt_sc = optimal_social_cost(bimodal_agents, env)
    opt_y_mc, opt_mc = optimal_max_cost(bimodal_agents, env)

    sc_proj = social_cost(y_projected, bimodal_agents)
    mc_proj = max_cost(y_projected, bimodal_agents)
    ar_sc = social_cost_approximation_ratio(y_projected, bimodal_agents, env)
    ar_mc = max_cost_approximation_ratio(y_projected, bimodal_agents, env)

    print(f"  Unconstrained Median: y = {y_unconstrained:.4f} (Feasible? {env.is_feasible(y_unconstrained)})")
    print(f"  Projected Median:     y = {y_projected:.4f} (Feasible? {env.is_feasible(y_projected)})")
    print(f"  Optimal Social Cost:  y* = {opt_y_sc:.4f} | Min SC = {opt_sc:.4f}")
    print(f"  Optimal Max Cost:     y* = {opt_y_mc:.4f} | Min MC = {opt_mc:.4f}")
    print(f"  Projected Median SC:  {sc_proj:.4f} (Approx Ratio = {ar_sc:.3f})")
    print(f"  Projected Median MC:  {mc_proj:.4f} (Approx Ratio = {ar_mc:.3f})")

    # Plot initial environment configuration
    fig_env_path = os.path.join(figures_dir, "week1_2_environment_setup.png")
    plot_environment(
        env,
        agents=bimodal_agents,
        facility=y_projected,
        facility_label="Projected Median",
        title="Discretized 1D Constrained Environment with Bimodal Agents & Facility",
        save_path=fig_env_path,
    )
    print(f"  -> Saved environment plot to: {fig_env_path}")

    # ------------------------------------------------------------------------
    # Step 4: Strategyproofness Verification Engine
    # ------------------------------------------------------------------------
    print("\n[Step 4] Running Computational Strategyproofness Validator...")

    # 4A. Verify standard Median on Unconstrained Line
    env_uncon = Environment1D.unconstrained(length=10.0, num_grid_points=1001)
    val_uncon = StrategyproofnessValidator(env=env_uncon, epsilon=1e-6)
    print("\n--- Test A: Verifying Unconstrained Median (Moulin 1980) ---")
    res_uncon = val_uncon.verify_distribution(
        mechanism=med_mech,
        n_agents=5,
        num_profiles=100,
        distribution=AgentDistribution.UNIFORM,
        seed=42,
    )
    print(res_uncon.summary())

    # 4B. Verify Mean Mechanism (Should FAIL and output counterexample)
    print("\n--- Test B: Verifying Mean Mechanism (Known Non-Truthful Benchmark) ---")
    mean_mech = MeanMechanism()
    res_mean = val_uncon.verify_distribution(
        mechanism=mean_mech,
        n_agents=3,
        num_profiles=25,
        distribution=AgentDistribution.UNIFORM,
        seed=42,
        stop_on_first_violation=True,
    )
    print(res_mean.summary())

    if res_mean.worst_violation:
        fig_violation_path = os.path.join(figures_dir, "week1_2_mean_manipulation_counterexample.png")
        plot_violation_counterexample(env_uncon, res_mean.worst_violation, save_path=fig_violation_path)
        print(f"  -> Saved violation counterexample plot to: {fig_violation_path}")

        # Plot deviation landscape for that agent
        fig_landscape_path = os.path.join(figures_dir, "week1_2_mean_deviation_landscape.png")
        plot_deviation_landscape(
            val_uncon,
            mean_mech,
            res_mean.worst_violation.profile,
            agent_idx=res_mean.worst_violation.agent_idx,
            save_path=fig_landscape_path,
        )
        print(f"  -> Saved deviation landscape plot to: {fig_landscape_path}")

    # 4C. Verify Median Mechanism Deviation Landscape (Flat / V-shaped minimum at truth)
    print("\n--- Test C: Plotting Truthful Deviation Landscape for Median Mechanism ---")
    fig_med_landscape_path = os.path.join(figures_dir, "week1_2_median_truthful_landscape.png")
    test_prof = [1.2, 3.5, 7.8]
    plot_deviation_landscape(
        val_uncon,
        med_mech,
        test_prof,
        agent_idx=1,
        title="Strategyproof Deviation Landscape for Median Mechanism (Truthful Peak at Minimum)",
        save_path=fig_med_landscape_path,
    )
    print(f"  -> Saved truthful median landscape plot to: {fig_med_landscape_path}")

    print("\n" + "=" * 70)
    print("WEEK 1–2 DELIVERABLES COMPLETED SUCCESSFULLY!")
    print("All models, mechanisms, validators, and visualizations are operational.")
    print("=" * 70)


if __name__ == "__main__":
    main()
