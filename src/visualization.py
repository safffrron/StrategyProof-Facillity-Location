"""Visualization Tools for 1D Constrained Facility Location Environments and Deviations.

This module creates publication-ready figures illustrating the 1D space (feasible vs blocked),
agent populations, facility placement, deviation landscapes, and manipulation counterexamples.
"""

from __future__ import annotations
from typing import Optional, Sequence, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.environment import Environment1D
from src.validator import Deviation, StrategyproofnessValidator
from src.mechanisms import FacilityLocationMechanism
from src.agents import AgentDistribution, generate_agents


def set_academic_style() -> None:
    """Configure matplotlib with clean publication-ready aesthetic parameters."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_environment(
    env: Environment1D,
    agents: Optional[Sequence[float]] = None,
    facility: Optional[float] = None,
    facility_label: str = "Facility y",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the 1D environment with feasible/blocked intervals, agents, and facility location.

    Args:
        env (Environment1D): The 1D metric space and constraints.
        agents (Optional[Sequence[float]]): Array of agent peak positions.
        facility (Optional[float]): Placed facility location.
        facility_label (str): Label for the facility marker in legend.
        title (Optional[str]): Figure title.
        ax (Optional[plt.Axes]): Matplotlib axes to draw on.
        save_path (Optional[str]): Path to save the figure image.

    Returns:
        plt.Figure: Generated matplotlib figure.
    """
    set_academic_style()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Draw the base 1D line domain
    y_baseline = 0.5
    ax.plot([0.0, env.length], [y_baseline, y_baseline], color="#94a3b8", lw=3, zorder=1)

    # Draw Feasible intervals
    for idx, iv in enumerate(env.feasible_intervals):
        lbl = "Feasible Zone (F)" if idx == 0 else None
        rect = patches.Rectangle(
            (iv.start, y_baseline - 0.12),
            iv.length,
            0.24,
            facecolor="#22c55e",
            edgecolor="#16a34a",
            alpha=0.35,
            zorder=2,
            label=lbl,
        )
        ax.add_patch(rect)
        # Mark endpoints
        ax.scatter([iv.start, iv.end], [y_baseline, y_baseline], color="#16a34a", s=25, zorder=3)

    # Draw Blocked intervals
    for idx, iv in enumerate(env.blocked_intervals):
        lbl = "Blocked Zone (Infeasible)" if idx == 0 else None
        rect = patches.Rectangle(
            (iv.start, y_baseline - 0.15),
            iv.length,
            0.30,
            facecolor="#ef4444",
            edgecolor="#dc2626",
            hatch="//",
            alpha=0.45,
            zorder=2,
            label=lbl,
        )
        ax.add_patch(rect)
        # Label obstacle center
        ax.text(
            (iv.start + iv.end) / 2.0,
            y_baseline,
            "BLOCKED",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#991b1b",
            zorder=4,
        )

    # Plot Agents
    if agents is not None and len(agents) > 0:
        agents_arr = np.asarray(agents, dtype=float)
        ax.scatter(
            agents_arr,
            np.full_like(agents_arr, y_baseline + 0.18),
            color="#3b82f6",
            s=80,
            marker="o",
            edgecolor="#1e3a8a",
            linewidth=1.2,
            zorder=5,
            label=f"Agents (n={len(agents_arr)})",
        )
        for i, a in enumerate(agents_arr):
            ax.vlines(a, y_baseline, y_baseline + 0.18, color="#3b82f6", linestyle=":", alpha=0.7, zorder=4)

    # Plot Facility
    if facility is not None:
        ax.scatter(
            [facility],
            [y_baseline],
            color="#f59e0b",
            s=220,
            marker="*",
            edgecolor="#78350f",
            linewidth=1.5,
            zorder=6,
            label=f"{facility_label} ({facility:.3f})",
        )

    ax.set_xlim(-0.02 * env.length, 1.02 * env.length)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("1D Line Coordinate (x)")
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontweight="bold", pad=10)

    # Organize legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_deviation_landscape(
    validator: StrategyproofnessValidator,
    mechanism: FacilityLocationMechanism,
    profile: Sequence[float],
    agent_idx: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot agent i's personal cost curve as a function of their reported location x'_i.

    Under a strategyproof mechanism, the curve's global minimum must occur at the truthful
    location x_i. If the mechanism is manipulable, points with lower cost visually appear.

    Args:
        validator (StrategyproofnessValidator): Initialized validator with environment and candidates.
        mechanism (FacilityLocationMechanism): Facility location rule to evaluate.
        profile (Sequence[float]): True agent profile.
        agent_idx (int): Agent whose deviation landscape is being analyzed.
        title (Optional[str]): Plot title.
        save_path (Optional[str]): File path to save the plot.

    Returns:
        plt.Figure: Deviation landscape plot.
    """
    set_academic_style()
    prof = np.asarray(profile, dtype=float)
    x_i_true = prof[agent_idx]
    env = validator.env

    # Baseline truthful outcome
    y_true = mechanism(prof, env)
    c_true = abs(y_true - x_i_true)

    # Sweep across candidate misreports
    cands = validator.candidates
    costs = []
    facilities = []

    for cand in cands:
        cheated_prof = prof.copy()
        cheated_prof[agent_idx] = cand
        y_cheat = mechanism(cheated_prof, env)
        c_cheat = abs(y_cheat - x_i_true)
        costs.append(c_cheat)
        facilities.append(y_cheat)

    costs = np.array(costs)
    facilities = np.array(facilities)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, dpi=150)

    # Plot 1: Facility location vs Agent Misreport
    ax1.plot(cands, facilities, color="#6366f1", lw=2, label="Facility Location y(x'_i, x_{-i})")
    ax1.scatter([x_i_true], [y_true], color="#f59e0b", s=140, marker="*", zorder=5, label=f"Truthful y = {y_true:.3f}")
    ax1.set_ylabel("Facility Location y")
    ax1.set_title("Mechanism Response Function: Facility Location vs Reported Position", fontweight="bold")
    ax1.legend(loc="best", framealpha=0.9)

    # Highlight blocked zones in ax1
    for iv in env.blocked_intervals:
        ax1.axvspan(iv.start, iv.end, color="#ef4444", alpha=0.15, hatch="//")
        ax1.axhspan(iv.start, iv.end, color="#ef4444", alpha=0.15, hatch="//")

    # Plot 2: Personal Cost vs Agent Misreport
    ax2.plot(cands, costs, color="#0284c7", lw=2, label="Agent i Cost |y(x'_i) - x_i|")
    ax2.axhline(c_true, color="#dc2626", linestyle="--", alpha=0.8, label=f"Truthful Cost = {c_true:.4f}")
    ax2.scatter([x_i_true], [c_true], color="#16a34a", s=120, zorder=5, label=f"Truthful Report x_{agent_idx} = {x_i_true:.3f}")

    # Check if there is any profitable deviation
    min_cost = float(np.min(costs))
    best_cand = float(cands[np.argmin(costs)])
    if min_cost < c_true - 1e-6:
        gain = c_true - min_cost
        ax2.scatter([best_cand], [min_cost], color="#ef4444", s=140, marker="X", zorder=6,
                    label=f"Cheating Min Cost = {min_cost:.4f} (Gain={gain:.4f})")
        ax2.fill_between(cands, costs, c_true, where=(costs < c_true), color="#ef4444", alpha=0.25,
                         label="Profitable Deviation Region")

    ax2.set_xlabel(f"Agent {agent_idx} Reported Location (x'_{agent_idx})")
    ax2.set_ylabel(f"Agent {agent_idx} Cost")
    sub_title = title or f"Deviation Landscape for Agent {agent_idx} (True Peak = {x_i_true:.3f})"
    ax2.set_title(sub_title, fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9)

    # Highlight blocked zones in ax2
    for iv in env.blocked_intervals:
        ax2.axvspan(iv.start, iv.end, color="#ef4444", alpha=0.15, hatch="//")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_violation_counterexample(
    env: Environment1D,
    deviation: Deviation,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot side-by-side comparison of truthful vs manipulated profiles for a counterexample."""
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, dpi=150)

    true_profile = np.array(deviation.profile)
    manipulated_profile = true_profile.copy()
    manipulated_profile[deviation.agent_idx] = deviation.misreport

    # Panel 1: Truthful reporting
    plot_environment(
        env,
        agents=true_profile,
        facility=deviation.true_facility,
        facility_label="Truthful Facility",
        title=f"1. Truthful Reporting: Agent {deviation.agent_idx} at x={deviation.true_location:.3f} | Cost = {deviation.true_cost:.4f}",
        ax=ax1,
    )

    # Panel 2: Manipulated reporting
    plot_environment(
        env,
        agents=manipulated_profile,
        facility=deviation.cheated_facility,
        facility_label="Manipulated Facility",
        title=f"2. Strategic Misreport: Agent {deviation.agent_idx} reports x'={deviation.misreport:.3f} | Cost drops to {deviation.cheated_cost:.4f} (Gain = {deviation.gain:.4f})",
        ax=ax2,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
