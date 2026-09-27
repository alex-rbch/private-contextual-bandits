"""Policy-optimization rules: VPO and BPO.

Both follow the approach studied by Levy and Mansour (2026). The averaged
variant uses only the current predictor and averages virtual mirror-descent
iterates. The bonus variant replays every already-released predictor with an
exposure bonus; that replay reads released predictors and public batch sizes
only, never older raw rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .base import validate_probability


@dataclass(frozen=True)
class ConstantPopulationErrorEnvelope:
    """Legacy envelope retained for historical audits; active VPO ignores it."""

    value: float

    def __post_init__(self) -> None:
        if self.value < 0.0 or not math.isfinite(self.value):
            raise ValueError("population error envelope must be finite and nonnegative")

    def __call__(self, predictor_epoch: int, training_epoch_size: int) -> float:
        del predictor_epoch, training_epoch_size
        return self.value


@dataclass(frozen=True)
class PowerLawPopulationErrorEnvelope:
    """Legacy ``E_m = C h_m^(-alpha)`` schedule for historical audits only."""

    scale: float = 0.0
    exponent: float = 0.5

    def __post_init__(self) -> None:
        if self.scale < 0.0 or not math.isfinite(self.scale):
            raise ValueError("population error scale must be finite and nonnegative")
        if self.exponent < 0.0 or not math.isfinite(self.exponent):
            raise ValueError("population error exponent must be finite and nonnegative")

    def __call__(self, predictor_epoch: int, training_epoch_size: int) -> float:
        del predictor_epoch
        if training_epoch_size <= 0:
            raise ValueError("training_epoch_size must be positive")
        return self.scale * training_epoch_size ** (-self.exponent)


@dataclass(frozen=True)
class VPOParameters:
    predictor_epoch: int
    training_epoch: int
    rollout_length: int
    learning_rate: float


def vpo_parameters(
    *,
    predictor_epoch: int,
    training_epoch: int,
    training_epoch_size: int,
    num_actions: int,
    eta0: float,
) -> VPOParameters:
    """Set ``eta_m = eta0 min(1/A, 1/sqrt(A B_m))``."""

    if predictor_epoch != training_epoch + 1:
        raise ValueError("predictor_epoch must immediately follow training_epoch")
    if training_epoch_size <= 0 or num_actions <= 1:
        raise ValueError("training_epoch_size must be positive and num_actions > 1")
    if eta0 <= 0.0 or not math.isfinite(eta0):
        raise ValueError("eta0 must be finite and positive")
    return VPOParameters(
        predictor_epoch=predictor_epoch,
        training_epoch=training_epoch,
        rollout_length=training_epoch_size,
        learning_rate=eta0
        * min(1.0 / num_actions, 1.0 / math.sqrt(num_actions * training_epoch_size)),
    )


def _validate_rollout_inputs(
    predictions: np.ndarray, learning_rate: float, rollout_length: int
) -> np.ndarray:
    vector = np.asarray(predictions, dtype=np.float64)
    if vector.ndim != 1 or vector.size <= 1 or not np.all(np.isfinite(vector)):
        raise ValueError("predictions must be a finite action vector")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if rollout_length <= 0:
        raise ValueError("rollout_length must be positive")
    return vector


def virtual_vpo_update(
    probability: np.ndarray, predictions: np.ndarray, learning_rate: float
) -> np.ndarray:
    """Perform one stable negative-entropy update using predicted losses."""

    current = np.asarray(probability, dtype=np.float64)
    vector = np.asarray(predictions, dtype=np.float64)
    if (
        current.ndim != 1
        or current.shape != vector.shape
        or current.size <= 1
        or not np.all(np.isfinite(current))
        or np.any(current < 0.0)
        or not math.isclose(float(current.sum()), 1.0, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError("probability must be a valid action distribution")
    if not np.all(np.isfinite(vector)):
        raise ValueError("predictions must be finite")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    with np.errstate(divide="ignore"):
        logits = np.log(current) - learning_rate * vector
    weights = np.exp(logits - float(np.max(logits)))
    return weights / float(weights.sum())


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(centered)
    return weights / np.sum(weights, axis=1, keepdims=True)


def recursive_virtual_trajectory(
    predictions: np.ndarray, learning_rate: float, rollout_length: int
) -> np.ndarray:
    """Return ``q_0, ..., q_(B_m-1)`` using literal VPO updates."""

    vector = _validate_rollout_inputs(predictions, learning_rate, rollout_length)
    trajectory = np.empty((rollout_length, vector.size), dtype=np.float64)
    probability = np.full(vector.size, 1.0 / vector.size)
    for offset in range(rollout_length):
        trajectory[offset] = probability
        if offset + 1 < rollout_length:
            probability = virtual_vpo_update(probability, vector, learning_rate)
    return trajectory


def direct_virtual_trajectory(
    predictions: np.ndarray, learning_rate: float, rollout_length: int
) -> np.ndarray:
    """Return ``q_0, ..., q_(B_m-1)``, with ``q_k = softmax(-k eta f)``."""

    vector = _validate_rollout_inputs(predictions, learning_rate, rollout_length)
    steps = np.arange(rollout_length, dtype=np.float64)
    return _softmax_rows(-steps[:, None] * learning_rate * vector[None, :])


def averaged_virtual_policy(
    predictions: np.ndarray,
    learning_rate: float,
    rollout_length: int,
    *,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Average ``q_0, ..., q_(B_m-1)`` without retaining the trajectory."""

    vector = _validate_rollout_inputs(predictions, learning_rate, rollout_length)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    total = np.zeros(vector.size, dtype=np.float64)
    for start in range(0, rollout_length, chunk_size):
        stop = min(start + chunk_size, rollout_length)
        steps = np.arange(start, stop, dtype=np.float64)
        logits = -steps[:, None] * learning_rate * vector[None, :]
        total += np.sum(_softmax_rows(logits), axis=0)
    probability = total / rollout_length
    return probability / float(probability.sum())


def sample_virtual_mixture_component(
    predictions: np.ndarray,
    learning_rate: float,
    rollout_length: int,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """Sample K uniformly from ``0, ..., B_m-1`` and return K and ``q_K``.

    Sampling an action from the returned component has marginal distribution
    ``(1/B_m) sum_k q_k`` without constructing all virtual iterates.
    """

    vector = _validate_rollout_inputs(predictions, learning_rate, rollout_length)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    offset = int(rng.integers(rollout_length))
    logits = -offset * learning_rate * vector
    weights = np.exp(logits - float(np.max(logits)))
    return offset, weights / float(weights.sum())


def bpo_policy(
    historical_estimates: Iterable[tuple[np.ndarray, int]],
    *,
    num_actions: int,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Replay released predictors with the scaled, capped exposure bonus.

    For predictor ``j`` with training-batch size ``N_j``, the rule is

        T_j = sum_(r<=j) N_r,
        b_j(a) = min(1, gamma sqrt(T_j / K) / (1 + D_j(a))),
        adjusted_j(a) = max(0, f_j(a) - b_j(a)),
        q_(j+1)(a) proportional to q_j(a) exp(-eta N_j adjusted_j(a)),
        D_(j+1)(a) = D_j(a) + N_j q_j(a).

    Exposure uses the pre-update ``q_j``, and both T and D are reconstructed
    per context rather than accumulated across incoming contexts.
    """

    cumulative_weight = 0
    if num_actions == 2:
        # Exact scalar form of the same negative-entropy update. Avoiding tiny
        # array allocations in the historical replay is material here.
        log_odds = 0.0
        exposure_zero = 0.0
        exposure_one = 0.0
        for estimate, predictor_weight in historical_estimates:
            if predictor_weight <= 0:
                raise AssertionError("a released predictor has an empty training batch")
            if estimate.shape != (2,) or not np.all(np.isfinite(estimate)):
                raise AssertionError("BPO received invalid oracle predictions")
            if log_odds >= 0.0:
                probability_zero = 1.0 / (1.0 + math.exp(-log_odds))
            else:
                exponential = math.exp(log_odds)
                probability_zero = exponential / (1.0 + exponential)
            cumulative_weight += predictor_weight
            numerator = gamma * math.sqrt(cumulative_weight / 2.0)
            bonus_zero = min(1.0, numerator / (1.0 + exposure_zero))
            bonus_one = min(1.0, numerator / (1.0 + exposure_one))
            optimistic_difference = max(0.0, float(estimate[0]) - bonus_zero) - max(
                0.0, float(estimate[1]) - bonus_one
            )
            log_odds -= eta * predictor_weight * optimistic_difference
            exposure_zero += predictor_weight * probability_zero
            exposure_one += predictor_weight * (1.0 - probability_zero)
        if log_odds >= 0.0:
            probability_zero = 1.0 / (1.0 + math.exp(-log_odds))
        else:
            exponential = math.exp(log_odds)
            probability_zero = exponential / (1.0 + exponential)
        return np.asarray([probability_zero, 1.0 - probability_zero])

    log_weights = np.zeros(num_actions, dtype=np.float64)
    exposure = np.zeros_like(log_weights)
    for estimate, predictor_weight in historical_estimates:
        if predictor_weight <= 0:
            raise AssertionError("a released predictor has an empty training batch")
        if estimate.shape != log_weights.shape or not np.all(np.isfinite(estimate)):
            raise AssertionError("BPO received invalid oracle predictions")
        weights = np.exp(log_weights - float(np.max(log_weights)))
        probability = validate_probability(weights / float(weights.sum()))
        cumulative_weight += predictor_weight
        numerator = gamma * math.sqrt(cumulative_weight / probability.size)
        bonus = np.minimum(1.0, numerator / (1.0 + exposure))
        optimistic_loss = np.maximum(0.0, estimate - bonus)
        # Retain finite log weights so underflowed probabilities can recover.
        log_weights -= eta * predictor_weight * optimistic_loss
        # D_(j+1) adds the policy used BEFORE incorporating predictor j.
        exposure += predictor_weight * probability
    weights = np.exp(log_weights - float(np.max(log_weights)))
    return validate_probability(weights / float(weights.sum()))
