"""CS 6001 Project: Strategyproof Facility Location with Interval Predictions and Feasibility Constraints.

Modules:
    environment: Discretized 1D space, feasible/infeasible intervals, grid snapping, and random obstacle generation.
    agents: Synthetic agent population generators with controlled distributions (Uniform, Gaussian, Bimodal, Clustered).
    mechanisms: Baseline and learning-augmented facility location mechanisms and cost metrics.
    validator: Brute-force computational strategyproofness verification engine.
    visualization: Publication-quality plotting tools for 1D environments, agent profiles, and deviation landscapes.
"""

from src.environment import Environment1D, Interval
from src.agents import AgentDistribution, generate_agents
from src.mechanisms import (
    FacilityLocationMechanism,
    MedianMechanism,
    NaiveProjectedMedianMechanism,
    social_cost,
    max_cost,
    optimal_social_cost,
    optimal_max_cost,
    social_cost_approximation_ratio,
    max_cost_approximation_ratio,
)
from src.validator import StrategyproofnessValidator, ValidationResult

__all__ = [
    "Environment1D",
    "Interval",
    "AgentDistribution",
    "generate_agents",
    "FacilityLocationMechanism",
    "MedianMechanism",
    "NaiveProjectedMedianMechanism",
    "social_cost",
    "max_cost",
    "optimal_social_cost",
    "optimal_max_cost",
    "social_cost_approximation_ratio",
    "max_cost_approximation_ratio",
    "StrategyproofnessValidator",
    "ValidationResult",
]
