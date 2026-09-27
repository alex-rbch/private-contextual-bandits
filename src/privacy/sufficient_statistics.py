"""Sufficient-statistics perturbation for one completed regression batch.

The mechanism releases a noisy Gram matrix and a noisy response vector. Every
downstream use is post-processing of those two arrays: the quadratic-loss
oracle solves them for a predictor, and the exploration geometry sums the
released Gram matrices across batches. That is why the quadratic path needs no
separate exploration budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .accounting import (
    Adjacency,
    SufficientStatisticsBudget,
    gaussian_sigma,
    normalize_rows_to_unit_ball,
    split_sufficient_statistics_budget,
)
from .noise import symmetric_frobenius_gaussian

# The active action-block feature map has nonnegative inner products between
# any two action features (including the intercept), even across records.
# With zero-one losses, this tightens the response-vector replace-one bound.
# Rows are publicly normalized into the unit ball before statistics are formed.
BANDIT_GRAM_SENSITIVITY = math.sqrt(2.0)
BANDIT_RESPONSE_SENSITIVITY = math.sqrt(2.0)


@dataclass(frozen=True)
class SufficientStatisticsRelease:
    """One batch's released statistics plus trusted local diagnostics."""

    noisy_gram: np.ndarray
    noisy_response: np.ndarray
    budget: SufficientStatisticsBudget
    normalized_features: np.ndarray
    exact_gram: np.ndarray
    exact_response: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.noisy_gram.shape[0])


@dataclass(frozen=True)
class GramRelease:
    """One noisy, unprojected batch Gram for logistic exploration."""

    noisy_gram: np.ndarray
    exact_gram: np.ndarray
    sensitivity: float
    noise_sigma: float


def release_gram(
    features: np.ndarray,
    *,
    epsilon: float,
    delta: float,
    rng: np.random.Generator,
) -> GramRelease:
    """Release ``X.T X + E`` using the Gaussian Gram component of SSP.

    A replace-one bandit row has Frobenius sensitivity ``sqrt(2)``. The
    logistic predictor uses a separate objective perturbation release; this
    function spends only the exploration half of its per-batch budget.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a nonempty matrix")
    if (
        not np.all(np.isfinite(matrix))
        or float(np.max(np.linalg.norm(matrix, axis=1))) > 1.0 + 1e-10
    ):
        raise ValueError("private Gram release requires unit-ball features")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        exact_gram = matrix.T @ matrix
    if not np.all(np.isfinite(exact_gram)):
        raise FloatingPointError("Gram statistic is non-finite")
    sigma = gaussian_sigma(BANDIT_GRAM_SENSITIVITY, epsilon, delta)
    noisy_gram = exact_gram + symmetric_frobenius_gaussian(
        matrix.shape[1], sigma, rng
    )
    return GramRelease(noisy_gram, exact_gram, BANDIT_GRAM_SENSITIVITY, sigma)


def full_information_sensitivities(
    features: np.ndarray,
    targets: np.ndarray,
    action_count: int,
    *,
    feature_norm_bound: float,
    target_bound: float,
    adjacency: Adjacency,
    bound_tolerance: float,
) -> tuple[float, float, int]:
    """Record-level sensitivities when one record contributes K action rows.

    One original classification row contributes ``K`` mutually orthogonal
    action-block feature rows and the complete zero-one loss vector, which has
    one zero and ``K-1`` ones. The active feature map also makes cross-record
    response-vector inner products nonnegative. Under replace-one adjacency,
    this gives ``sqrt(2K) B_x^2`` for the Gram statistic and
    ``sqrt(2(K-1)) B_x B_y`` for the response statistic.
    """

    if action_count <= 1:
        raise ValueError("full-information action_count must exceed one")
    if Adjacency(adjacency) is not Adjacency.REPLACE_ONE:
        raise ValueError("full-information SSP currently requires replace-one adjacency")
    matrix = np.asarray(features, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    if matrix.shape[0] % action_count != 0:
        raise ValueError("full-information feature rows must form complete action groups")
    record_count = matrix.shape[0] // action_count
    grouped_features = matrix.reshape(record_count, action_count, matrix.shape[1])
    grouped_targets = response.reshape(record_count, action_count)
    if not np.all(
        np.isclose(grouped_targets, 0.0, atol=bound_tolerance, rtol=0.0)
        | np.isclose(grouped_targets, 1.0, atol=bound_tolerance, rtol=0.0)
    ):
        raise ValueError("full-information targets must be binary losses")
    zero_counts = np.sum(
        np.isclose(grouped_targets, 0.0, atol=bound_tolerance, rtol=0.0), axis=1
    )
    if not np.all(zero_counts == 1):
        raise ValueError("each full-information record must have exactly one zero-loss action")
    group_inner_products = grouped_features @ np.swapaxes(grouped_features, 1, 2)
    diagonal = np.zeros_like(group_inner_products)
    diagonal[:, np.arange(action_count), np.arange(action_count)] = np.diagonal(
        group_inner_products, axis1=1, axis2=2
    )
    if float(np.max(np.abs(group_inner_products - diagonal))) > max(bound_tolerance, 1e-12):
        raise ValueError("full-information features must occupy mutually orthogonal action blocks")
    return (
        math.sqrt(2.0 * action_count) * feature_norm_bound**2,
        math.sqrt(2.0 * (action_count - 1.0))
        * feature_norm_bound
        * target_bound,
        record_count,
    )


def release_sufficient_statistics(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epsilon: float,
    delta: float,
    gram_sensitivity: float = BANDIT_GRAM_SENSITIVITY,
    response_sensitivity: float = BANDIT_RESPONSE_SENSITIVITY,
    seed: int,
) -> SufficientStatisticsRelease:
    """Make one Gaussian release of ``(X.T X, X.T y)`` for this batch."""

    matrix = normalize_rows_to_unit_ball(features)
    response = np.asarray(targets, dtype=np.float64)
    if response.shape != (matrix.shape[0],):
        raise ValueError("targets must contain one value per feature row")
    if (
        not np.all(np.isfinite(response))
        or np.any(response < 0.0)
        or np.any(response > 1.0)
    ):
        raise ValueError("targets must lie in [0, 1]")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    budget = split_sufficient_statistics_budget(
        epsilon, delta, gram_sensitivity, response_sensitivity
    )
    gram_rng, response_rng = (
        np.random.default_rng(child)
        for child in np.random.SeedSequence(int(seed)).spawn(2)
    )
    # LAPACK solves of earlier indefinite noisy systems can leave floating-point
    # status flags set even when these bounded products are finite. Suppress
    # those stale warnings and validate the newly computed statistics directly.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        exact_gram = matrix.T @ matrix
        exact_response = matrix.T @ response
    if not np.all(np.isfinite(exact_gram)) or not np.all(np.isfinite(exact_response)):
        raise FloatingPointError("sufficient statistics are non-finite")
    gram_noise = symmetric_frobenius_gaussian(
        matrix.shape[1], budget.gram_noise_sigma, gram_rng
    )
    response_noise = budget.response_noise_sigma * response_rng.standard_normal(
        matrix.shape[1]
    )
    return SufficientStatisticsRelease(
        noisy_gram=exact_gram + gram_noise,
        noisy_response=exact_response + response_noise,
        budget=budget,
        normalized_features=matrix,
        exact_gram=exact_gram,
        exact_response=exact_response,
    )
