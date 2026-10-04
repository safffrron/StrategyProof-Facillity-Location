"""Computational Strategyproofness Validator.

This module implements a brute-force deviation checker that loops over all agents and candidate
misreports to computationally verify incentive compatibility (truthfulness) or isolate explicit
counterexamples where an agent strictly reduces their personal cost by lying.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, Dict, Any, Callable
import numpy as np

from src.environment import Environment1D
from src.agents import AgentDistribution, generate_agents
from src.mechanisms import FacilityLocationMechanism


@dataclass(frozen=True)
class Deviation:
    """Records a single profitable manipulation (strategyproofness violation)."""
    agent_idx: int
    true_location: float
    misreport: float
    true_facility: float
    cheated_facility: float
    true_cost: float
    cheated_cost: float
    gain: float
    profile: Tuple[float, ...]

    def __str__(self) -> str:
        return (
            f"Agent {self.agent_idx} at x={self.true_location:.4f} lies with x'={self.misreport:.4f} | "
            f"Facility shifts {self.true_facility:.4f} -> {self.cheated_facility:.4f} | "
            f"Cost drops {self.true_cost:.4f} -> {self.cheated_cost:.4f} (gain={self.gain:.4f})"
        )


@dataclass
class ProfileValidationResult:
    """Validation outcome for a single agent profile."""
    profile: np.ndarray
    is_strategyproof: bool
    violations: List[Deviation]
    max_gain: float


@dataclass
class ValidationResult:
    """Comprehensive validation report across multiple profiles."""
    mechanism_name: str
    is_strategyproof: bool
    num_profiles_tested: int
    num_violations_found: int
    violation_rate: float
    max_gain: float
    worst_violation: Optional[Deviation]
    all_violations: List[Deviation]

    def summary(self) -> str:
        """Return a formatted text summary of the validation report."""
        status = "PASSED (Strategyproof)" if self.is_strategyproof else "FAILED (Manipulable)"
        lines = [
            "=" * 68,
            f"Strategyproofness Report: {self.mechanism_name}",
            f"Status:                 {status}",
            f"Profiles Tested:        {self.num_profiles_tested}",
            f"Profiles with Violations: {self.num_violations_found} ({self.violation_rate:.2%})",
            f"Maximum Personal Gain:  {self.max_gain:.6f}",
        ]
        if self.worst_violation is not None:
            lines.append("-" * 68)
            lines.append("Worst-Case Violation Counterexample:")
            lines.append(f"  Agent Index:          {self.worst_violation.agent_idx}")
            lines.append(f"  True Profile:         {[round(float(x), 4) for x in self.worst_violation.profile]}")
            lines.append(f"  True Location:        {self.worst_violation.true_location:.4f}")
            lines.append(f"  Profitable Misreport: {self.worst_violation.misreport:.4f}")
            lines.append(f"  Facility (Truth):     {self.worst_violation.true_facility:.4f}")
            lines.append(f"  Facility (Misreport): {self.worst_violation.cheated_facility:.4f}")
            lines.append(f"  Cost Under Truth:     {self.worst_violation.true_cost:.6f}")
            lines.append(f"  Cost Under Lie:       {self.worst_violation.cheated_cost:.6f}")
            lines.append(f"  Personal Gain:        {self.worst_violation.gain:.6f}")
        lines.append("=" * 68)
        return "\n".join(lines)


class StrategyproofnessValidator:
    """Computational verification engine for checking incentive compatibility.

    Given a facility location mechanism f and an environment E:
    For any profile x = (x_1, ..., x_n), an agent i has no incentive to report x'_i != x_i if:
        |f(x_i, x_{-i}) - x_i| <= |f(x'_i, x_{-i}) - x_i| + epsilon
    for all candidate misreports x'_i.
    """

    def __init__(
        self,
        env: Environment1D,
        epsilon: float = 1e-6,
        candidate_misreports: Optional[np.ndarray] = None,
        feasible_misreports_only: bool = False,
    ) -> None:
        """Initialize the validator.

        Args:
            env (Environment1D): The environment with metric space and constraints.
            epsilon (float): Numerical tolerance to avoid floating-point false positives.
            candidate_misreports (Optional[np.ndarray]): Explicit grid of candidate lies.
                If None, uses env.grid or env.feasible_grid.
            feasible_misreports_only (bool): If True, candidates are restricted to feasible points.
        """
        self.env = env
        self.epsilon = float(epsilon)
        self.feasible_misreports_only = feasible_misreports_only

        if candidate_misreports is not None:
            self.candidates = np.asarray(candidate_misreports, dtype=float)
        elif feasible_misreports_only:
            self.candidates = self.env.feasible_grid.copy()
        else:
            self.candidates = self.env.grid.copy()

    def check_profile(
        self,
        mechanism: FacilityLocationMechanism,
        profile: Sequence[float],
        stop_on_first_violation: bool = False,
    ) -> ProfileValidationResult:
        """Exhaustively check all agents and all candidate misreports for a single profile.

        Args:
            mechanism (FacilityLocationMechanism): Facility location rule to verify.
            profile (Sequence[float]): Vector of true agent locations.
            stop_on_first_violation (bool): Early exit as soon as a violation is found.

        Returns:
            ProfileValidationResult: Detailed validation outcome for this profile.
        """
        prof = np.asarray(profile, dtype=float)
        n = len(prof)
        if n == 0:
            return ProfileValidationResult(prof, True, [], 0.0)

        # Baseline truthful outcome
        y_true = mechanism(prof, self.env)
        violations: List[Deviation] = []
        max_gain = 0.0

        for i in range(n):
            x_i = float(prof[i])
            c_true = abs(y_true - x_i)

            # Test all candidate misreports
            # Also include exact boundary points and other agents' reported points
            extra_cands = [iv.start for iv in self.env.feasible_intervals] + \
                          [iv.end for iv in self.env.feasible_intervals] + \
                          [float(p) for p in prof]
            all_cands = np.unique(np.concatenate([self.candidates, extra_cands]))

            for x_cand in all_cands:
                x_cand = float(x_cand)
                # Skip truthful report
                if abs(x_cand - x_i) < self.epsilon:
                    continue

                # Form perturbed profile
                cheated_prof = prof.copy()
                cheated_prof[i] = x_cand

                y_cheat = mechanism(cheated_prof, self.env)
                c_cheat = abs(y_cheat - x_i)

                gain = c_true - c_cheat
                if gain > self.epsilon:
                    dev = Deviation(
                        agent_idx=i,
                        true_location=x_i,
                        misreport=x_cand,
                        true_facility=y_true,
                        cheated_facility=y_cheat,
                        true_cost=c_true,
                        cheated_cost=c_cheat,
                        gain=gain,
                        profile=tuple(prof.tolist()),
                    )
                    violations.append(dev)
                    if gain > max_gain:
                        max_gain = gain

                    if stop_on_first_violation:
                        return ProfileValidationResult(prof, False, violations, max_gain)

        is_sp = (len(violations) == 0)
        return ProfileValidationResult(prof, is_sp, violations, max_gain)

    def verify_distribution(
        self,
        mechanism: FacilityLocationMechanism,
        n_agents: int = 5,
        num_profiles: int = 100,
        distribution: Union[AgentDistribution, str] = AgentDistribution.UNIFORM,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 42,
        stop_on_first_violation: bool = False,
    ) -> ValidationResult:
        """Run verification over multiple synthetic profiles sampled from a distribution.

        Args:
            mechanism (FacilityLocationMechanism): Mechanism to evaluate.
            n_agents (int): Number of agents in each profile.
            num_profiles (int): Number of synthetic profiles to sample and test.
            distribution (AgentDistribution | str): Distribution to draw peaks from.
            params (Optional[Dict[str, Any]]): Distribution hyperparameters.
            seed (Optional[int]): Random seed.
            stop_on_first_violation (bool): Early termination upon first violation.

        Returns:
            ValidationResult: Comprehensive diagnostic report.
        """
        all_violations: List[Deviation] = []
        worst_violation: Optional[Deviation] = None
        max_overall_gain = 0.0
        violating_profiles_count = 0

        rng = np.random.default_rng(seed)

        for p_idx in range(num_profiles):
            profile_seed = int(rng.integers(0, 1_000_000)) if seed is not None else None
            prof = generate_agents(
                n_agents=n_agents,
                env=self.env,
                distribution=distribution,
                params=params,
                seed=profile_seed,
            )

            res = self.check_profile(mechanism, prof, stop_on_first_violation=stop_on_first_violation)

            if not res.is_strategyproof:
                violating_profiles_count += 1
                all_violations.extend(res.violations)
                if res.max_gain > max_overall_gain:
                    max_overall_gain = res.max_gain
                    # Pick the violation with the highest gain
                    worst_in_res = max(res.violations, key=lambda d: d.gain)
                    worst_violation = worst_in_res

                if stop_on_first_violation:
                    break

        is_sp = (violating_profiles_count == 0)
        violation_rate = (violating_profiles_count / num_profiles) if num_profiles > 0 else 0.0

        return ValidationResult(
            mechanism_name=mechanism.name,
            is_strategyproof=is_sp,
            num_profiles_tested=p_idx + 1 if stop_on_first_violation and not is_sp else num_profiles,
            num_violations_found=violating_profiles_count,
            violation_rate=violation_rate,
            max_gain=max_overall_gain,
            worst_violation=worst_violation,
            all_violations=all_violations,
        )

    def verify_exhaustive_grid(
        self,
        mechanism: FacilityLocationMechanism,
        n_agents: int = 3,
        grid_coarseness: int = 21,
    ) -> ValidationResult:
        """Exhaustively verify ALL combinatorial profiles on a discrete sub-grid.

        For small n (e.g. n=2 or n=3), tests every tuple in G^n. This offers a rigorous
        computational proof of strategyproofness on the discretized lattice.
        """
        sub_grid = np.linspace(0.0, self.env.length, grid_coarseness)
        from itertools import product

        all_violations: List[Deviation] = []
        worst_violation: Optional[Deviation] = None
        max_overall_gain = 0.0
        violating_profiles_count = 0
        total_tested = 0

        for tuple_prof in product(sub_grid, repeat=n_agents):
            total_tested += 1
            prof = np.array(tuple_prof, dtype=float)
            res = self.check_profile(mechanism, prof, stop_on_first_violation=True)

            if not res.is_strategyproof:
                violating_profiles_count += 1
                all_violations.extend(res.violations)
                if res.max_gain > max_overall_gain:
                    max_overall_gain = res.max_gain
                    worst_violation = res.violations[0]

        is_sp = (violating_profiles_count == 0)
        return ValidationResult(
            mechanism_name=mechanism.name,
            is_strategyproof=is_sp,
            num_profiles_tested=total_tested,
            num_violations_found=violating_profiles_count,
            violation_rate=(violating_profiles_count / total_tested) if total_tested > 0 else 0.0,
            max_gain=max_overall_gain,
            worst_violation=worst_violation,
            all_violations=all_violations,
        )
