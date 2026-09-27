"""Inverse-gap allocations: SquareCB, FastCB, and AdaCB's restricted variant.

SquareCB is Foster and Rakhlin (2020); FastCB is the reweighted variant of
Foster and Krishnamurthy (2021). AdaCB reuses SquareCB's allocation on the
RegCB survivor set.
"""

from __future__ import annotations

import numpy as np

from .base import uniform_minimizers, validate_probability


def inverse_gap_allocation(
    scores: np.ndarray,
    gamma: float,
    *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """SquareCB allocation ``p(a) = 1 / (K + gamma * gap_a)``.

    Restricting ``eligible`` to a survivor set gives AdaCB, with ``K`` reduced
    to the survivor count and zero mass elsewhere.
    """

    if eligible is None:
        eligible = np.ones(scores.size, dtype=bool)
    candidates = np.flatnonzero(eligible)
    if candidates.size == 0:
        raise AssertionError("inverse-gap allocation received an empty action set")
    best = int(candidates[np.argmin(scores[candidates])])
    best_score = float(scores[best])
    probability = np.zeros(scores.size, dtype=np.float64)
    mass = 0.0
    action_count = int(candidates.size)
    for action in candidates:
        if action == best:
            continue
        gap = max(0.0, float(scores[action] - best_score))
        value = 1.0 / (action_count + gamma * gap)
        probability[action] = value
        mass += value
    probability[best] = 1.0 - mass
    return validate_probability(probability)


def fastcb_supports_loss(loss: str) -> bool:
    """FastCB is defined here only for the logistic-loss oracle."""
    return loss == "logistic"


def fastcb_unsupported_run(horizon: int) -> dict[str, object]:
    """Report an unavailable FastCB run without sampling or fitting an oracle."""
    return {
        "status": "unsupported",
        "reason": "FastCB only supports logistic loss",
        "observed_losses": [-1.0] * horizon,
        "cumulative_loss": -float(horizon),
        "oracle_calls": 0,
        "actions": [],
        "action_probabilities": [],
    }


def fastcb_allocation(scores: np.ndarray, gamma: float) -> np.ndarray:
    """FastCB allocation ``p(a) = f_b / (K f_b + gamma * gap_a)``.

    Logistic predictions are used without a floor. When the best prediction
    is zero the displayed expression reaches its boundary,
    and the policy falls back to the actual minimizers without flooring them
    or forming negative weights.
    """

    best = int(np.argmin(scores))
    best_score = float(scores[best])
    if best_score <= 0.0:
        return uniform_minimizers(scores)
    probability = np.zeros(scores.size, dtype=np.float64)
    mass = 0.0
    for action in range(scores.size):
        if action == best:
            continue
        gap = max(0.0, float(scores[action] - best_score))
        value = best_score / (scores.size * best_score + gamma * gap)
        probability[action] = value
        mass += value
    probability[best] = 1.0 - mass
    return validate_probability(probability)
