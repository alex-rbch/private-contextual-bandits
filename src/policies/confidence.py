"""Confidence-based policies: RegCB survivor sets, AdaCB's set, and LinUCB.

These are the only policies that read an exploration geometry. Each width is
``width_scale * sqrt(x.T V^-1 x)`` for the pooled geometry ``V``, which already
reflects every released batch, so no batch-size log factor appears here.
"""

from __future__ import annotations

import math

import numpy as np

from .base import uniform_minimizers


class CumulativeExplorationGeometry:
    """Accumulate released batch Gram matrices into ``sum_j S_j + I``.

    Predictors remain batch-local. Only this policy-side confidence geometry
    spans completed batches. ``S_j`` is a noisy Gram matrix under privacy and
    the exact Gram matrix otherwise. The identity coefficient is fixed at one,
    independent of the fitted oracle ridge and represented record count.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = int(dimension)
        self._total = np.zeros((self._dimension, self._dimension), dtype=np.float64)
        self._release_count = 0
        self._record_count = 0
        self._geometry: np.ndarray | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def release_count(self) -> int:
        return self._release_count

    @property
    def record_count(self) -> int:
        """Original records behind the pooled statistics."""

        return self._record_count

    @property
    def geometry(self) -> np.ndarray:
        if self._release_count == 0:
            raise RuntimeError("no released statistics have been accumulated")
        if self._geometry is None:
            geometry = self._total.copy()
            geometry.flat[:: self._dimension + 1] += 1.0
            geometry.setflags(write=False)
            self._geometry = geometry
        return self._geometry

    def accumulate(self, statistic: np.ndarray, *, record_count: int) -> None:
        matrix = np.asarray(statistic, dtype=np.float64)
        if matrix.shape != (self._dimension, self._dimension):
            raise ValueError("released statistic has the wrong shape")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("released statistic must be finite")
        if record_count <= 0:
            raise ValueError("record_count must be positive")
        self._total += (matrix + matrix.T) / 2.0
        self._release_count += 1
        self._record_count += int(record_count)
        self._geometry = None

    def width(self, features: np.ndarray, width_scale: float) -> np.ndarray | float:
        """Return ``width_scale * sqrt(x.T (sum_j S_j + I)^-1 x)``.

        A private Gram matrix need not be positive semidefinite, so negative
        quadratic forms are clamped to zero.
        """

        if width_scale < 0.0 or not math.isfinite(width_scale):
            raise ValueError("width_scale must be finite and nonnegative")
        matrix = np.asarray(features, dtype=np.float64)
        one = matrix.ndim == 1
        if one:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise ValueError("features have the wrong shape")
        solved = np.linalg.solve(self.geometry, matrix.T).T
        leverage = np.maximum(np.sum(matrix * solved, axis=1), 0.0)
        widths = width_scale * np.sqrt(leverage)
        return float(widths[0]) if one else widths


def confidence_bounds(
    raw_scores: np.ndarray,
    widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds in the oracle's raw score space.

    For logistic predictions, comparing these logit-space endpoints gives
    the same survivor set as applying the increasing sigmoid to both sides.
    For quadratic predictions the raw score is already the predicted loss.
    """

    return raw_scores - widths, raw_scores + widths


def confidence_survivors(
    raw_scores: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """RegCB elimination: keep actions whose lower bound beats the best upper.

    The set is recomputed for every context from the frozen predictor;
    eliminated actions are never permanently removed.
    """

    lower, upper = confidence_bounds(raw_scores, widths)
    survivors = lower <= float(np.min(upper))
    if not np.any(survivors):
        raise AssertionError("confidence set eliminated every action")
    return survivors


def regcb_allocation(survivors: np.ndarray) -> np.ndarray:
    """Play uniformly over the survivor set."""

    return survivors.astype(np.float64) / float(np.count_nonzero(survivors))


def linucb_allocation(
    predicted_losses: np.ndarray,
    radii: np.ndarray,
    ucb_scale: float,
) -> np.ndarray:
    """Maximize ``(1 - predicted_loss) + ucb_scale * radius``, ties broken evenly.

    Predictions are used raw, including values outside ``[0, 1]``: clipping
    before adding the bonus would change the ranking. The optimistic reward is
    formed explicitly rather than minimizing the equivalent
    ``predicted_loss - ucb_scale * radius``, because the tie test is relative
    and a constant shift can move which actions count as tied.
    """

    if radii.shape != predicted_losses.shape or not np.all(np.isfinite(radii)):
        raise AssertionError("LinUCB received invalid confidence geometry")
    optimistic_rewards = (1.0 - predicted_losses) + ucb_scale * radii
    return uniform_minimizers(-optimistic_rewards)
