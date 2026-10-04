"""1D Environment Representation with Discretization, Feasible Intervals, and Blocked Zones.

This module models a 1D bounded metric space [0, L] with environmental feasibility constraints
(blocked zones/obstacles) and discrete grid representations for computational verification.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Union
import numpy as np


@dataclass(frozen=True)
class Interval:
    """Represents a closed 1D interval [start, end]."""
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Interval start ({self.start}) cannot exceed end ({self.end}).")

    @property
    def length(self) -> float:
        """Return the length of the interval."""
        return self.end - self.start

    def contains(self, x: Union[float, np.ndarray], tol: float = 1e-9) -> Union[bool, np.ndarray]:
        """Check whether point(s) x lie inside the interval [start, end]."""
        return (x >= self.start - tol) & (x <= self.end + tol)

    def distance_to(self, x: float) -> float:
        """Compute the shortest Euclidean distance from a point x to the interval."""
        if x < self.start:
            return self.start - x
        elif x > self.end:
            return x - self.end
        return 0.0

    def project(self, x: float) -> float:
        """Project a point x onto the interval."""
        return max(self.start, min(self.end, x))


def merge_intervals(intervals: Sequence[Interval], tol: float = 1e-9) -> List[Interval]:
    """Merge overlapping or adjacent intervals into a minimal list of disjoint intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda iv: iv.start)
    merged: List[Interval] = []

    current_start = sorted_intervals[0].start
    current_end = sorted_intervals[0].end

    for iv in sorted_intervals[1:]:
        if iv.start <= current_end + tol:
            # Overlapping or adjacent: extend current interval
            current_end = max(current_end, iv.end)
        else:
            merged.append(Interval(current_start, current_end))
            current_start = iv.start
            current_end = iv.end

    merged.append(Interval(current_start, current_end))
    return merged


def compute_complement_intervals(
    intervals: Sequence[Interval],
    domain_start: float = 0.0,
    domain_end: float = 1.0,
    tol: float = 1e-9,
) -> List[Interval]:
    """Compute the complement of a set of intervals within [domain_start, domain_end]."""
    merged = merge_intervals(intervals, tol=tol)
    complements: List[Interval] = []
    cursor = domain_start

    for iv in merged:
        # Check if interval starts after current cursor
        if iv.start > cursor + tol:
            seg_start = cursor
            seg_end = min(iv.start, domain_end)
            if seg_end > seg_start + tol:
                complements.append(Interval(seg_start, seg_end))
        cursor = max(cursor, iv.end)

    if cursor < domain_end - tol:
        complements.append(Interval(cursor, domain_end))

    return complements


class Environment1D:
    """Models a 1D bounded line [0, length] with feasible and blocked zones.

    Attributes:
        length (float): Total length of the 1D space [0, length].
        num_grid_points (int): Discretization resolution.
        feasible_intervals (List[Interval]): List of disjoint closed feasible intervals.
        blocked_intervals (List[Interval]): List of disjoint closed blocked intervals.
        grid (np.ndarray): Uniformly spaced grid points over [0, length].
        feasible_mask (np.ndarray): Boolean mask indicating feasible grid points.
        feasible_grid (np.ndarray): Grid points that fall within feasible intervals.
    """

    def __init__(
        self,
        length: float = 1.0,
        num_grid_points: int = 1001,
        feasible_intervals: Optional[Sequence[Interval]] = None,
        blocked_intervals: Optional[Sequence[Interval]] = None,
        tol: float = 1e-9,
    ) -> None:
        if length <= 0:
            raise ValueError(f"Environment length must be strictly positive, got {length}")
        if num_grid_points < 2:
            raise ValueError(f"num_grid_points must be at least 2, got {num_grid_points}")

        self.length = float(length)
        self.num_grid_points = int(num_grid_points)
        self.tol = tol

        # Resolve feasible and blocked intervals
        if feasible_intervals is not None and blocked_intervals is not None:
            raise ValueError("Specify either feasible_intervals or blocked_intervals, not both.")
        elif blocked_intervals is not None:
            # Clean and clip blocked intervals
            clipped_blocked = [
                Interval(max(0.0, b.start), min(self.length, b.end))
                for b in blocked_intervals
                if b.end > 0.0 and b.start < self.length and b.end > b.start
            ]
            self.blocked_intervals = merge_intervals(clipped_blocked, tol=self.tol)
            self.feasible_intervals = compute_complement_intervals(
                self.blocked_intervals, domain_start=0.0, domain_end=self.length, tol=self.tol
            )
        elif feasible_intervals is not None:
            clipped_feasible = [
                Interval(max(0.0, f.start), min(self.length, f.end))
                for f in feasible_intervals
                if f.end > 0.0 and f.start < self.length and f.end > f.start
            ]
            self.feasible_intervals = merge_intervals(clipped_feasible, tol=self.tol)
            self.blocked_intervals = compute_complement_intervals(
                self.feasible_intervals, domain_start=0.0, domain_end=self.length, tol=self.tol
            )
        else:
            # Default: Entire space [0, length] is feasible (unconstrained line)
            self.feasible_intervals = [Interval(0.0, self.length)]
            self.blocked_intervals = []

        if not self.feasible_intervals:
            raise ValueError("Environment must have at least one feasible interval.")

        # Discretized grid
        self.grid = np.linspace(0.0, self.length, self.num_grid_points)
        self.grid_step = self.length / (self.num_grid_points - 1)

        # Vectorized feasibility mask
        mask = np.zeros(self.num_grid_points, dtype=bool)
        for iv in self.feasible_intervals:
            mask |= (self.grid >= iv.start - self.tol) & (self.grid <= iv.end + self.tol)
        self.feasible_mask = mask
        self.feasible_grid = self.grid[self.feasible_mask]

        if len(self.feasible_grid) == 0:
            raise ValueError("Discretization produced 0 feasible grid points. Increase num_grid_points.")

    @property
    def feasible_measure(self) -> float:
        """Total measure (sum of lengths) of all feasible intervals."""
        return sum(iv.length for iv in self.feasible_intervals)

    @property
    def blocked_measure(self) -> float:
        """Total measure (sum of lengths) of all blocked intervals."""
        return sum(iv.length for iv in self.blocked_intervals)

    @property
    def blocked_density(self) -> float:
        """Fraction of [0, length] that is blocked."""
        return self.blocked_measure / self.length

    @property
    def feasible_density(self) -> float:
        """Fraction of [0, length] that is feasible."""
        return self.feasible_measure / self.length

    def is_feasible(self, x: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if point(s) x are in a feasible interval."""
        if isinstance(x, (int, float, np.floating)):
            return any(iv.contains(float(x), tol=self.tol) for iv in self.feasible_intervals)
        
        arr = np.asarray(x, dtype=float)
        res = np.zeros(arr.shape, dtype=bool)
        for iv in self.feasible_intervals:
            res |= (arr >= iv.start - self.tol) & (arr <= iv.end + self.tol)
        return res

    def project_nearest(self, x: float, tie_break: str = "left") -> float:
        """Project a point x to the nearest feasible point in the environment.

        If x is already feasible, returns x unchanged.
        If x is infeasible (inside a blocked zone or outside [0, length]),
        it finds the closest endpoint among all feasible intervals.
        In case of an exact tie (x is midway between two feasible intervals),
        tie_break='left' returns the lower endpoint, and 'right' returns the higher.

        Args:
            x (float): Query point.
            tie_break (str): Deterministic tie-breaking rule ('left' or 'right').

        Returns:
            float: Nearest feasible location.
        """
        # Clamp to [0, length]
        clamped_x = max(0.0, min(self.length, float(x)))

        # If already feasible, return clamped_x
        if self.is_feasible(clamped_x):
            return clamped_x

        # If infeasible, the nearest feasible point must be an interval boundary
        best_candidate: Optional[float] = None
        min_dist = float("inf")

        for iv in self.feasible_intervals:
            # Check projection onto this interval
            cand = iv.project(clamped_x)
            dist = abs(cand - clamped_x)

            if dist < min_dist - self.tol:
                min_dist = dist
                best_candidate = cand
            elif abs(dist - min_dist) <= self.tol:
                # Tie condition: two intervals equidistant
                if tie_break == "left" and (best_candidate is None or cand < best_candidate):
                    best_candidate = cand
                elif tie_break == "right" and (best_candidate is None or cand > best_candidate):
                    best_candidate = cand

        assert best_candidate is not None
        return best_candidate

    def snap_to_grid(self, x: float, feasible_only: bool = False) -> float:
        """Snap a continuous point x to the nearest discrete grid point.

        Args:
            x (float): Point to snap.
            feasible_only (bool): If True, snaps only to feasible grid points.

        Returns:
            float: Snapped grid point.
        """
        target_grid = self.feasible_grid if feasible_only else self.grid
        idx = int(np.abs(target_grid - x).argmin())
        return float(target_grid[idx])

    @classmethod
    def unconstrained(cls, length: float = 1.0, num_grid_points: int = 1001) -> Environment1D:
        """Create an unconstrained 1D environment where the entire [0, length] is feasible."""
        return cls(length=length, num_grid_points=num_grid_points)

    @classmethod
    def random_blocked(
        cls,
        length: float = 1.0,
        num_blocks: int = 3,
        blocked_density: float = 0.3,
        min_block_size: float = 0.05,
        num_grid_points: int = 1001,
        seed: Optional[int] = None,
    ) -> Environment1D:
        """Generate a random environment with non-overlapping blocked zones.

        Args:
            length (float): Line length.
            num_blocks (int): Target number of blocked intervals.
            blocked_density (float): Fraction of line that should be blocked in [0, 0.9].
            min_block_size (float): Minimum length of any single blocked interval.
            num_grid_points (int): Discretization grid points.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            Environment1D: Initialized environment with generated obstacles.
        """
        if not (0.0 <= blocked_density <= 0.9):
            raise ValueError(f"blocked_density must be in [0, 0.9], got {blocked_density}")
        if num_blocks <= 0 or blocked_density == 0.0:
            return cls.unconstrained(length=length, num_grid_points=num_grid_points)

        rng = np.random.default_rng(seed)
        total_blocked_len = length * blocked_density

        # Ensure min_block_size is feasible
        if num_blocks * min_block_size > total_blocked_len:
            min_block_size = total_blocked_len / (2 * num_blocks)

        # Distribute block lengths using Dirichlet distribution
        raw_proportions = rng.dirichlet(np.ones(num_blocks))
        excess = total_blocked_len - num_blocks * min_block_size
        block_lengths = min_block_size + raw_proportions * max(0.0, excess)

        # Place blocks randomly leaving gaps
        # We sample (num_blocks + 1) gap lengths
        total_gap_len = length - total_blocked_len
        raw_gaps = rng.dirichlet(np.ones(num_blocks + 1))
        gap_lengths = raw_gaps * total_gap_len

        blocked_intervals: List[Interval] = []
        cursor = gap_lengths[0]
        for b_len, g_len in zip(block_lengths, gap_lengths[1:]):
            b_start = cursor
            b_end = cursor + b_len
            blocked_intervals.append(Interval(round(b_start, 6), round(b_end, 6)))
            cursor = b_end + g_len

        return cls(
            length=length,
            num_grid_points=num_grid_points,
            blocked_intervals=blocked_intervals,
        )

    def __repr__(self) -> str:
        return (
            f"Environment1D(length={self.length}, "
            f"feasible_intervals={self.feasible_intervals}, "
            f"blocked_density={self.blocked_density:.2%}, "
            f"num_grid_points={self.num_grid_points})"
        )
