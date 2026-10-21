"""Learning-Augmented and Constrained Facility Location Mechanisms (vectorised).

Every mechanism in this module maps a *batch* of report profiles ``X`` (shape ``(P, n)``) to a
batch of facility locations (shape ``(P,)``), so that the strategyproofness checker and the
experiments can evaluate thousands of profiles per call.

Almost all mechanisms here are *generalised medians*: the median of the reports together with a
multiset of "phantom" points that does not depend on the reports (Moulin 1980). Placing phantoms
at a machine-learned prediction is how Agrawal et al. (2022) build learning-augmented rules.
For a line with blocked zones, a mechanism may additionally *snap* its output to the nearest
feasible point using a fixed (left) tie-break; the ``*`` suffix in names denotes snapping.

Feasible sets are given either as an :class:`~src.environment.Environment1D`, as ``None``
(unconstrained real line), or as a tuple ``(starts, ends)`` of arrays of shape ``(K,)`` or
``(P, K)`` holding sorted, disjoint feasible intervals (one layout per profile).
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import numpy as np

from src.environment import Environment1D
from src.mechanisms import FacilityLocationMechanism

Feasible = Union[None, Environment1D, Tuple[np.ndarray, np.ndarray]]

TIE_TOL = 1e-9


# ============================================================================
# Feasible-set operations
# ============================================================================

def feasible_bounds(feasible: Feasible) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(starts, ends)`` arrays of the feasible intervals, or ``None`` if unconstrained."""
    if feasible is None:
        return None
    if isinstance(feasible, Environment1D):
        if feasible.blocked_density < 1e-12:
            starts = np.array([0.0])
            ends = np.array([feasible.length])
        else:
            starts = np.array([iv.start for iv in feasible.feasible_intervals], dtype=float)
            ends = np.array([iv.end for iv in feasible.feasible_intervals], dtype=float)
        return starts, ends
    starts, ends = feasible
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def snap_to_feasible(y: np.ndarray, feasible: Feasible, tie_tol: float = TIE_TOL) -> np.ndarray:
    """Map each point to its nearest feasible point; exact ties go to the left.

    The map is monotone non-decreasing, which is what makes "generalised median then snap"
    strategyproof for distance costs (see ``docs/RESULTS.md``, Lemma 1).
    """
    y = np.asarray(y, dtype=float)
    bounds = feasible_bounds(feasible)
    if bounds is None:
        return y.copy()
    starts, ends = bounds
    if starts.ndim == 1:
        starts = np.broadcast_to(starts, y.shape + starts.shape)
        ends = np.broadcast_to(ends, y.shape + ends.shape)
    candidates = np.clip(y[..., None], starts, ends)
    dist = np.abs(candidates - y[..., None])
    best = dist.min(axis=-1, keepdims=True)
    first_tied = np.argmax(dist <= best + tie_tol, axis=-1)  # leftmost interval among ties
    return np.take_along_axis(candidates, first_tied[..., None], axis=-1)[..., 0]


def is_feasible_batch(y: np.ndarray, feasible: Feasible, tol: float = 1e-9) -> np.ndarray:
    """Boolean mask of which points lie in the feasible set."""
    y = np.asarray(y, dtype=float)
    bounds = feasible_bounds(feasible)
    if bounds is None:
        return np.ones(y.shape, dtype=bool)
    starts, ends = bounds
    if starts.ndim == 1:
        starts = np.broadcast_to(starts, y.shape + starts.shape)
        ends = np.broadcast_to(ends, y.shape + ends.shape)
    return np.any((y[..., None] >= starts - tol) & (y[..., None] <= ends + tol), axis=-1)


def _nearest_feasible_left(m: np.ndarray, feasible: Feasible) -> np.ndarray:
    """Largest feasible point <= m (``-inf`` if none)."""
    bounds = feasible_bounds(feasible)
    if bounds is None:
        return m.copy()
    starts, ends = bounds
    if starts.ndim == 1:
        starts = np.broadcast_to(starts, m.shape + starts.shape)
        ends = np.broadcast_to(ends, m.shape + ends.shape)
    cand = np.where(starts <= m[..., None], np.minimum(ends, m[..., None]), -np.inf)
    return cand.max(axis=-1)


def _nearest_feasible_right(m: np.ndarray, feasible: Feasible) -> np.ndarray:
    """Smallest feasible point >= m (``+inf`` if none)."""
    bounds = feasible_bounds(feasible)
    if bounds is None:
        return m.copy()
    starts, ends = bounds
    if starts.ndim == 1:
        starts = np.broadcast_to(starts, m.shape + starts.shape)
        ends = np.broadcast_to(ends, m.shape + ends.shape)
    cand = np.where(ends >= m[..., None], np.maximum(starts, m[..., None]), np.inf)
    return cand.min(axis=-1)


# ============================================================================
# Costs and exact constrained optima (batch)
# ============================================================================

def lower_median(Z: np.ndarray) -> np.ndarray:
    """Row-wise lower median (the ceil(m/2)-th smallest of m values)."""
    Z = np.sort(np.asarray(Z, dtype=float), axis=1)
    return Z[:, (Z.shape[1] - 1) // 2]


def social_cost_batch(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Sum of distances, row-wise."""
    return np.abs(X - np.asarray(Y)[:, None]).sum(axis=1)


def max_cost_batch(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Maximum distance, row-wise."""
    return np.abs(X - np.asarray(Y)[:, None]).max(axis=1)


def optimal_max_cost_batch(X: np.ndarray, feasible: Feasible) -> Tuple[np.ndarray, np.ndarray]:
    """Exact constrained egalitarian optimum.

    MC(y) = |y - mid| + h with mid the midpoint of the extreme agents and h half their spread,
    so the optimum over F is the feasible point nearest to mid.
    """
    X = np.asarray(X, dtype=float)
    mid = 0.5 * (X.min(axis=1) + X.max(axis=1))
    y = snap_to_feasible(mid, feasible)
    return y, max_cost_batch(y, X)


def optimal_social_cost_batch(X: np.ndarray, feasible: Feasible) -> Tuple[np.ndarray, np.ndarray]:
    """Exact constrained utilitarian optimum.

    SC is convex with the median as an unconstrained minimiser, so over F the optimum is one of
    the two feasible points closest to the (lower) median from either side.
    """
    X = np.asarray(X, dtype=float)
    m = lower_median(X)
    left = _nearest_feasible_left(m, feasible)
    right = _nearest_feasible_right(m, feasible)
    sc_left = np.where(np.isfinite(left), social_cost_batch(np.where(np.isfinite(left), left, 0.0), X), np.inf)
    sc_right = np.where(np.isfinite(right), social_cost_batch(np.where(np.isfinite(right), right, 0.0), X), np.inf)
    take_right = sc_right < sc_left - 1e-12
    return np.where(take_right, right, left), np.where(take_right, sc_right, sc_left)


def agent_cost(Y: np.ndarray, peaks: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Cost of an agent with peak ``p`` for facility ``y``.

    ``alpha = 1`` is the standard symmetric distance. ``alpha != 1`` gives an *asymmetric*
    single-peaked cost: ``alpha * (p - y)`` when the facility is to the left, ``(y - p)`` otherwise.
    """
    d = np.asarray(Y, dtype=float) - np.asarray(peaks, dtype=float)
    if alpha == 1.0:
        return np.abs(d)
    return np.where(d < 0, -alpha * d, d)


# ============================================================================
# Mechanism base class
# ============================================================================

class BatchMechanism(FacilityLocationMechanism):
    """A (possibly learning-augmented) mechanism that can be evaluated on a batch of profiles.

    Subclasses implement :meth:`pre_snap`. ``locate`` then optionally snaps to the feasible set.

    Attributes:
        snap: If True, the output is snapped to the nearest feasible point (left tie-break).
        prediction: Default prediction (scalar for point predictions, ``(lo, hi)`` for intervals)
            used by the scalar ``__call__`` interface.
        prediction_type: ``None``, ``"point"`` or ``"interval"``.
        anonymous: Whether the output is invariant to permuting the reports.
    """

    base_name = "Mechanism"
    prediction_type: Optional[str] = None
    anonymous = True

    def __init__(self, snap: bool = True, prediction=None) -> None:
        self.snap = snap
        self.prediction = prediction

    @property
    def name(self) -> str:
        return self.base_name + ("*" if self.snap else "")

    def pre_snap(self, X: np.ndarray, feasible: Feasible, pred) -> np.ndarray:
        raise NotImplementedError

    def locate(self, X: np.ndarray, feasible: Feasible = None, prediction=None) -> np.ndarray:
        """Facility locations for a batch of profiles ``X`` of shape ``(P, n)``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        pred = self.prediction if prediction is None else prediction
        y = self.pre_snap(X, feasible, pred)
        return snap_to_feasible(y, feasible) if self.snap else y

    def __call__(self, reports: Sequence[float], env: Optional[Environment1D] = None) -> float:
        return float(self.locate(np.asarray(reports, dtype=float)[None, :], env)[0])

    # helpers ---------------------------------------------------------------
    @staticmethod
    def _point(pred, P: int) -> np.ndarray:
        if pred is None:
            raise ValueError("This mechanism needs a point prediction.")
        return np.broadcast_to(np.asarray(pred, dtype=float), (P,))

    @staticmethod
    def _interval(pred, P: int) -> Tuple[np.ndarray, np.ndarray]:
        if pred is None:
            raise ValueError("This mechanism needs an interval prediction (lo, hi).")
        arr = np.asarray(pred, dtype=float)
        arr = np.broadcast_to(arr, (P, 2))
        lo = np.minimum(arr[:, 0], arr[:, 1])
        hi = np.maximum(arr[:, 0], arr[:, 1])
        return lo, hi


# ============================================================================
# Prediction-free mechanisms
# ============================================================================

class MedianRule(BatchMechanism):
    """Lower median of the reports (Moulin 1980); ``Median*`` snaps it (Walsh 2021)."""
    base_name = "Median"

    def pre_snap(self, X, feasible, pred):
        return lower_median(X)


class LeftmostRule(BatchMechanism):
    """Leftmost report (Procaccia & Tennenholtz): SP and 2-approximate for max cost."""
    base_name = "Leftmost"

    def pre_snap(self, X, feasible, pred):
        return X.min(axis=1)


class RightmostRule(BatchMechanism):
    base_name = "Rightmost"

    def pre_snap(self, X, feasible, pred):
        return X.max(axis=1)


class DictatorRule(BatchMechanism):
    """Facility at a fixed agent's report: SP but not anonymous."""
    anonymous = False

    def __init__(self, agent: int = 0, snap: bool = True) -> None:
        super().__init__(snap=snap)
        self.agent = agent

    @property
    def name(self) -> str:
        return f"Dictator[{self.agent}]" + ("*" if self.snap else "")

    def pre_snap(self, X, feasible, pred):
        return X[:, self.agent]


class MeanRule(BatchMechanism):
    """Arithmetic mean: the canonical manipulable rule."""
    base_name = "Mean"

    def pre_snap(self, X, feasible, pred):
        return X.mean(axis=1)


class MidpointRule(BatchMechanism):
    """Midpoint of the extreme reports: optimal for max cost but manipulable."""
    base_name = "Midpoint"

    def pre_snap(self, X, feasible, pred):
        return 0.5 * (X.min(axis=1) + X.max(axis=1))


class GeneralizedMedian(BatchMechanism):
    """Median of the reports plus a fixed multiset of phantom points (Moulin 1980)."""

    def __init__(self, phantoms: Sequence[float], snap: bool = True) -> None:
        super().__init__(snap=snap)
        self.phantoms = np.asarray(phantoms, dtype=float)

    @property
    def name(self) -> str:
        return f"GenMedian{list(np.round(self.phantoms, 3))}" + ("*" if self.snap else "")

    def pre_snap(self, X, feasible, pred):
        P = X.shape[0]
        Z = np.concatenate([X, np.broadcast_to(self.phantoms, (P, len(self.phantoms)))], axis=1)
        return lower_median(Z)


# ============================================================================
# Point-prediction mechanisms
# ============================================================================

class MinMaxP(BatchMechanism):
    """Agrawal et al. (2022), Mechanism 1: clamp the prediction into [min report, max report].

    Equivalent to a generalised median with n-1 phantoms at the prediction.
    """
    base_name = "MinMaxP"
    prediction_type = "point"

    def pre_snap(self, X, feasible, pred):
        p = self._point(pred, X.shape[0])
        return np.clip(p, X.min(axis=1), X.max(axis=1))


class PhantomMedian(BatchMechanism):
    """Confidence-parameterised family: median of the reports plus ``k`` copies of the prediction.

    ``k = 0`` is the median; ``k = n - 1`` is MinMaxP. This is the 1-D analogue of the CMP
    mechanism of Agrawal et al. with confidence ``c = k / n``.
    """
    prediction_type = "point"

    def __init__(self, k: int, snap: bool = True, prediction=None) -> None:
        super().__init__(snap=snap, prediction=prediction)
        self.k = int(k)

    @property
    def name(self) -> str:
        return f"PhantomMedian[k={self.k}]" + ("*" if self.snap else "")

    def pre_snap(self, X, feasible, pred):
        P = X.shape[0]
        p = self._point(pred, P)
        Z = np.concatenate([X, np.repeat(p[:, None], self.k, axis=1)], axis=1)
        return lower_median(Z)


class PredictionOrMedian(BatchMechanism):
    """The course brief's example rule: output the prediction if it lies in [min, max], else the median.

    Not strategyproof: an extreme agent can widen [min, max] to swallow a prediction near them.
    """
    base_name = "PredOrMedian"
    prediction_type = "point"

    def pre_snap(self, X, feasible, pred):
        p = self._point(pred, X.shape[0])
        inside = (p >= X.min(axis=1)) & (p <= X.max(axis=1))
        return np.where(inside, p, lower_median(X))


class HullNearestFeasible(BatchMechanism):
    """Naive constrained adaptation of MinMaxP: the *feasible point inside [min, max]* closest to the
    prediction; if no feasible point lies in [min, max], fall back to Median*.

    Its output is always feasible, so no snapping is applied. Not strategyproof.
    """
    base_name = "HullNearestFeasible"
    prediction_type = "point"

    def __init__(self, prediction=None) -> None:
        super().__init__(snap=False, prediction=prediction)

    @property
    def name(self) -> str:
        return self.base_name

    def pre_snap(self, X, feasible, pred):
        P = X.shape[0]
        p = self._point(pred, P)
        lo_hull, hi_hull = X.min(axis=1), X.max(axis=1)
        bounds = feasible_bounds(feasible)
        if bounds is None:
            return np.clip(p, lo_hull, hi_hull)
        starts, ends = bounds
        if starts.ndim == 1:
            starts = np.broadcast_to(starts, (P,) + starts.shape)
            ends = np.broadcast_to(ends, (P,) + ends.shape)
        lo = np.maximum(starts, lo_hull[:, None])
        hi = np.minimum(ends, hi_hull[:, None])
        valid = lo <= hi + 1e-12
        cand = np.clip(p[:, None], lo, np.maximum(lo, hi))
        dist = np.where(valid, np.abs(cand - p[:, None]), np.inf)
        best = dist.min(axis=1, keepdims=True)
        first = np.argmax(dist <= best + TIE_TOL, axis=1)
        chosen = np.take_along_axis(cand, first[:, None], axis=1)[:, 0]
        fallback = snap_to_feasible(lower_median(X), feasible)
        return np.where(np.isfinite(best[:, 0]), chosen, fallback)


class SnapTowardPrediction(BatchMechanism):
    """MinMaxP whose blocked-zone tie-break follows the prediction instead of proximity.

    A clamped point that falls inside a blocked zone is moved to the zone endpoint *nearer the
    prediction* (rather than nearer the point itself). For a fixed prediction this is still a
    monotone map, yet the rule is manipulable and does not reduce the error amplification of
    MinMaxP*, so nearest-point snapping is the right choice.
    """
    base_name = "MinMaxP+SnapTowardPred"
    prediction_type = "point"

    def __init__(self, prediction=None) -> None:
        super().__init__(snap=False, prediction=prediction)

    @property
    def name(self) -> str:
        return self.base_name

    def pre_snap(self, X, feasible, pred):
        p = self._point(pred, X.shape[0])
        y = np.clip(p, X.min(axis=1), X.max(axis=1))
        if feasible_bounds(feasible) is None:
            return y
        left = _nearest_feasible_left(y, feasible)
        right = _nearest_feasible_right(y, feasible)
        to_right = np.abs(right - p) < np.abs(left - p)
        return np.where(is_feasible_batch(y, feasible), y, np.where(to_right, right, left))


# ============================================================================
# Interval-prediction mechanisms
# ============================================================================

class CenterMinMaxP(BatchMechanism):
    """Collapse the interval prediction [lo, hi] to its centre and run MinMaxP.

    For the max-cost objective this is optimal among generalised medians: interval-consistency
    1 + min(omega/2, 1) at normalised width omega (``docs/RESULTS.md``, Theorem 3).
    """
    base_name = "CenterMinMaxP"
    prediction_type = "interval"

    def pre_snap(self, X, feasible, pred):
        lo, hi = self._interval(pred, X.shape[0])
        return np.clip(0.5 * (lo + hi), X.min(axis=1), X.max(axis=1))


class IntervalClamp(BatchMechanism):
    """Clamp the median into the predicted interval, then into [min report, max report].

    Equivalent to a generalised median with ``n - 1 - (n-1)//2`` phantoms at ``lo`` and
    ``(n-1)//2`` phantoms at ``hi``; width 0 gives MinMaxP, unbounded width gives the median.
    """
    base_name = "IntervalClamp"
    prediction_type = "interval"

    def pre_snap(self, X, feasible, pred):
        lo, hi = self._interval(pred, X.shape[0])
        return np.clip(np.clip(lower_median(X), lo, hi), X.min(axis=1), X.max(axis=1))


class IntervalPhantoms(BatchMechanism):
    """Explicit phantom form of :class:`IntervalClamp` (used to test the equivalence)."""
    base_name = "IntervalPhantoms"
    prediction_type = "interval"

    def pre_snap(self, X, feasible, pred):
        P, n = X.shape
        lo, hi = self._interval(pred, P)
        n_hi = (n - 1) // 2
        n_lo = n - 1 - n_hi
        Z = np.concatenate([X, np.repeat(lo[:, None], n_lo, axis=1), np.repeat(hi[:, None], n_hi, axis=1)], axis=1)
        return lower_median(Z)
