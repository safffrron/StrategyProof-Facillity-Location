"""Facility-location mechanisms with predictions and blocked zones (vectorised).

Every mechanism maps a *batch* of report profiles ``X`` (shape ``(P, n)``) to a batch of facility
locations (shape ``(P,)``), so the strategyproofness checker and the experiments can evaluate
thousands of profiles per call.

The deterministic mechanisms are *generalised medians*: the median of the reports together with
"phantom" points that do not depend on the reports (Moulin 1980). ``MinMaxP`` (Agrawal et al. 2022)
puts n - 1 phantoms at the prediction. On a line with blocked zones the output is *snapped* to the
nearest allowed point with a fixed (left) tie-break; a ``*`` in a name denotes snapping.
``RandomizedPredictionSnap`` is the randomised snapping of the report (Definition 3).

Allowed sets are given as an :class:`~src.environment.Environment1D`, as ``None`` (the whole
line), or as a tuple ``(starts, ends)`` of arrays of shape ``(K,)`` or ``(P, K)`` holding sorted,
disjoint allowed intervals (one layout per profile).
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


def _align(bound: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Broadcast interval bounds (K,) or (P, K) against points y of shape (P,) or (P, n): result
    has shape y.shape + (K,)."""
    if bound.ndim == 1:
        return np.broadcast_to(bound, y.shape + bound.shape)
    if y.ndim == 2:
        return np.broadcast_to(bound[:, None, :], y.shape + bound.shape[-1:])
    return bound


def snap_to_feasible(y: np.ndarray, feasible: Feasible, tie_tol: float = TIE_TOL) -> np.ndarray:
    """Map each point to its nearest feasible point; exact ties go to the left.

    The map is monotone non-decreasing, which is what makes "generalised median then snap"
    strategyproof for distance costs (Lemma 2 of the report).
    """
    y = np.asarray(y, dtype=float)
    bounds = feasible_bounds(feasible)
    if bounds is None:
        return y.copy()
    starts, ends = bounds
    starts, ends = _align(starts, y), _align(ends, y)
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
    starts, ends = _align(starts, y), _align(ends, y)
    return np.any((y[..., None] >= starts - tol) & (y[..., None] <= ends + tol), axis=-1)


def _nearest_feasible_left(m: np.ndarray, feasible: Feasible) -> np.ndarray:
    """Largest feasible point <= m (``-inf`` if none)."""
    bounds = feasible_bounds(feasible)
    if bounds is None:
        return m.copy()
    starts, ends = bounds
    starts, ends = _align(starts, m), _align(ends, m)
    cand = np.where(starts <= m[..., None], np.minimum(ends, m[..., None]), -np.inf)
    return cand.max(axis=-1)


def _nearest_feasible_right(m: np.ndarray, feasible: Feasible) -> np.ndarray:
    """Smallest feasible point >= m (``+inf`` if none)."""
    bounds = feasible_bounds(feasible)
    if bounds is None:
        return m.copy()
    starts, ends = bounds
    starts, ends = _align(starts, m), _align(ends, m)
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


def agent_cost(Y: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Cost of an agent with peak ``p`` for a facility at ``y``: the distance ``|y - p|``."""
    return np.abs(np.asarray(Y, dtype=float) - np.asarray(peaks, dtype=float))


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
    """Leftmost report (Procaccia & Tennenholtz): strategyproof and 2-approximate for max cost."""
    base_name = "Leftmost"

    def pre_snap(self, X, feasible, pred):
        return X.min(axis=1)


class DictatorRule(BatchMechanism):
    """Facility at a fixed agent's report: strategyproof but not anonymous."""
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
    """Confidence-parameterised family: the median of the reports and ``k`` copies of the prediction.

    ``k = 0`` is the median (no trust); ``k = n - 1`` is MinMaxP (full trust). A generalised median for
    every ``k``, hence strategyproof; the project brief's confidence parameter.
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
    """The example rule of the project brief: the prediction if it lies in [min, max], else the median.

    Not strategyproof: an extreme agent can widen [min, max] to swallow a prediction near her.
    """
    base_name = "PredOrMedian"
    prediction_type = "point"

    def pre_snap(self, X, feasible, pred):
        p = self._point(pred, X.shape[0])
        inside = (p >= X.min(axis=1)) & (p <= X.max(axis=1))
        return np.where(inside, p, lower_median(X))


# ============================================================================
# Interval-prediction mechanisms
# ============================================================================

class CenterMinMaxP(BatchMechanism):
    """Collapse the interval prediction [lo, hi] to its centre and run MinMaxP.

    The "midpoint rule" of the report: ratio 1 + min(omega/2, 1) on the line and min(1 + omega, 3)
    with blocked zones, at normalised width omega (Theorem 5).
    """
    base_name = "CenterMinMaxP"
    prediction_type = "interval"

    def pre_snap(self, X, feasible, pred):
        lo, hi = self._interval(pred, X.shape[0])
        return np.clip(0.5 * (lo + hi), X.min(axis=1), X.max(axis=1))


# ============================================================================
# Randomized mechanisms (returned as finite distributions over locations)
# ============================================================================

def _gap_endpoints(p: np.ndarray, feasible: Feasible) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point: (left feasible neighbour, right feasible neighbour, is_feasible)."""
    left = _nearest_feasible_left(p, feasible)
    right = _nearest_feasible_right(p, feasible)
    return left, right, is_feasible_batch(p, feasible)


def mean_preserving_rounding(p: np.ndarray, feasible: Feasible) -> Tuple[np.ndarray, np.ndarray]:
    """Round each point to a random feasible point with the same mean.

    A point inside a blocked zone (g1, g2) goes to g1 with probability (g2 - p)/(g2 - g1) and to g2
    otherwise; feasible points (and points beyond the outermost feasible interval, which have a
    single nearest feasible point) are kept deterministic. Returns ``(locs, probs)`` of shape (P, 2).
    """
    p = np.asarray(p, dtype=float)
    left, right, feas = _gap_endpoints(p, feasible)
    both = np.isfinite(left) & np.isfinite(right) & ~feas
    gap = np.where(both, right - left, 1.0)
    w_right = np.where(both, (p - np.where(both, left, 0.0)) / gap, 0.0)
    single = snap_to_feasible(p, feasible)
    loc_left = np.where(both, left, single)
    loc_right = np.where(both, right, single)
    return np.stack([loc_left, loc_right], axis=1), np.stack([1.0 - w_right, w_right], axis=1)


class RandomizedPredictionSnap(BatchMechanism):
    """Randomised snapping of the prediction (Definition 3 and Theorem 9 of the report).

    If the (point) prediction lies in a blocked zone, replace it by one of the zone's endpoints using
    :func:`mean_preserving_rounding`, then run ``base`` (a snapped generalised median such as MinMaxP*)
    with that allowed prediction. This is a report-independent mixture of two
    strategyproof mechanisms, hence universally truthful. For interval predictions, ``use_center``
    rounds the interval's midpoint instead.
    """
    randomized = True

    def __init__(self, base: BatchMechanism, use_center: bool = False) -> None:
        super().__init__(snap=True)
        if base.prediction_type != "point":
            raise ValueError("RandomizedPredictionSnap needs a point-prediction base mechanism (e.g. MinMaxP).")
        self.base = base
        self.use_center = use_center
        self.prediction_type = "interval" if use_center else "point"

    @property
    def name(self) -> str:
        return f"Rand[{self.base.name}" + (", centre]" if self.use_center else "]")

    def distribution(self, X: np.ndarray, feasible: Feasible = None, prediction=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        P = X.shape[0]
        pred = self.prediction if prediction is None else prediction
        if self.use_center:
            lo, hi = self._interval(pred, P)
            point = 0.5 * (lo + hi)
        else:
            point = self._point(pred, P)
        if feasible_bounds(feasible) is None:
            return self.base.locate(X, feasible, point)[:, None], np.ones((P, 1))
        preds, probs = mean_preserving_rounding(point, feasible)
        locs = np.stack([self.base.locate(X, feasible, preds[:, j]) for j in range(preds.shape[1])], axis=1)
        return locs, probs

    def locate(self, X, feasible=None, prediction=None):
        """Expected location (for plotting only; costs must use :meth:`distribution`)."""
        locs, probs = self.distribution(X, feasible, prediction)
        return (locs * probs).sum(axis=1)


def mechanism_distribution(mech: BatchMechanism, X: np.ndarray, feasible: Feasible = None, prediction=None):
    """(locations, probabilities) of any mechanism; deterministic ones get a single atom."""
    if getattr(mech, "randomized", False):
        return mech.distribution(X, feasible, prediction)
    y = mech.locate(X, feasible, prediction)
    return y[:, None], np.ones((y.shape[0], 1))


def expected_agent_cost(mech: BatchMechanism, X: np.ndarray, peaks: np.ndarray, feasible: Feasible = None,
                        prediction=None) -> np.ndarray:
    """Expected cost of the agents with the given peaks (row-wise) under ``mech`` on reports ``X``."""
    locs, probs = mechanism_distribution(mech, X, feasible, prediction)
    return (agent_cost(locs, peaks[:, None]) * probs).sum(axis=1)
