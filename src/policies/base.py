"""Policy names, tunable parameters, and allocation helpers.

A policy turns a released predictor (and, for the confidence policies, a
released exploration geometry) into an action distribution. Nothing in this
package fits models or touches privacy mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

BANDIT_POLICIES = (
    "squarecb",
    "fastcb",
    "regcb",
    "adacb",
    "linucb",
    "vpo",
    "bpo",
)
SUPPORTED_POLICIES = (*BANDIT_POLICIES, "supervised")

# Policies that read a confidence radius when selecting an action, and so need
# a released second-moment matrix from the oracle.
EXPLORATION_POLICIES = frozenset({"linucb", "regcb", "adacb"})


def policy_needs_exploration_geometry(algorithm: str) -> bool:
    """Whether this policy reads confidence geometry when selecting actions."""

    if algorithm not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy {algorithm!r}")
    return algorithm in EXPLORATION_POLICIES


@dataclass(frozen=True)
class PolicyConfiguration:
    """Tunable policy parameters; unused fields are ignored per algorithm."""

    algorithm: str
    gamma_scale: float = 1.0
    gamma_exponent: float = 0.0
    width_scale: float = 0.1
    ucb_scale: float = 1.0
    learning_rate_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.algorithm not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported policy {self.algorithm!r}")
        for field in ("gamma_scale", "width_scale", "ucb_scale", "gamma_exponent"):
            value = getattr(self, field)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field} must be finite and nonnegative")
        if (
            self.learning_rate_scale <= 0.0
            or not math.isfinite(self.learning_rate_scale)
        ):
            raise ValueError("learning_rate_scale must be finite and positive")


def validate_probability(probability: np.ndarray) -> np.ndarray:
    """Clamp roundoff, check normalization, and renormalize."""

    result = np.asarray(probability, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)) or np.any(result < -1e-12):
        raise AssertionError("policy construction produced invalid probabilities")
    result = np.maximum(result, 0.0)
    total = float(result.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"policy probabilities sum to {total}, not one")
    return result / total


def uniform_minimizers(
    scores: np.ndarray, eligible: np.ndarray | None = None
) -> np.ndarray:
    """Spread mass evenly over the numerically tied minimizing actions."""

    if eligible is None:
        eligible = np.ones(scores.size, dtype=bool)
    candidates = np.flatnonzero(eligible)
    if candidates.size == 0:
        raise AssertionError("a policy cannot have an empty eligible action set")
    minimum = float(np.min(scores[candidates]))
    winners = eligible & np.isclose(scores, minimum, rtol=1e-12, atol=1e-15)
    probability = np.zeros(scores.size, dtype=np.float64)
    probability[winners] = 1.0 / float(np.count_nonzero(winners))
    return probability


def predicted_gaps(predictions: np.ndarray) -> np.ndarray:
    """Center finite loss predictions at the first deterministic minimizer."""

    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 1 or values.size <= 1 or not np.all(np.isfinite(values)):
        raise ValueError("predictions must be a finite action vector")
    best_action = int(np.argmin(values))
    gaps = values - values[best_action]
    gaps[best_action] = 0.0
    return np.maximum(gaps, 0.0)


def gamma_schedule(
    gamma_scale: float, gamma_exponent: float, policy_time: int
) -> float:
    """Return ``gamma_0 * tau^r``, frozen for the whole batch.

    ``policy_time`` is the number of rounds completed before the active batch
    began, using the actual geometric boundary rather than an epoch index.
    """

    if policy_time < 1:
        raise ValueError("policy_time must be at least one")
    return gamma_scale * policy_time**gamma_exponent
