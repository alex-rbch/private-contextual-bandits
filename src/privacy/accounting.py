"""Privacy accounting: adjacency, zCDP conversion, and per-fit budget splits.

Nothing here knows about oracles or policies. These are the calibration
formulas that the mechanisms in this package draw their noise scales from.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ..data.preprocessing import normalize_rows_to_unit_ball


class Adjacency(str, Enum):
    """Supported record-level neighboring relations."""

    ADD_REMOVE = "add_remove"
    REPLACE_ONE = "replace_one"


def zcdp_rho(epsilon: float, delta: float) -> float:
    """Return the rho whose zCDP-to-approximate-DP conversion hits epsilon.

    Uses the exact inverse of the Bun and Steinke (2016) conversion
    ``epsilon = rho + 2 sqrt(rho log(1/delta))``.
    """

    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if not 0.0 < delta < 1.0 or not math.isfinite(delta):
        raise ValueError("delta must lie strictly between zero and one")
    log_inverse_delta = math.log(1.0 / delta)
    return (
        math.sqrt(log_inverse_delta + epsilon) - math.sqrt(log_inverse_delta)
    ) ** 2


def gaussian_sigma_from_rho(sensitivity: float, rho: float) -> float:
    """Gaussian scale for one rho-zCDP release of a given L2 sensitivity."""

    if sensitivity <= 0.0 or not math.isfinite(sensitivity):
        raise ValueError("sensitivity must be finite and positive")
    if rho <= 0.0 or not math.isfinite(rho):
        raise ValueError("rho must be finite and positive")
    return sensitivity / math.sqrt(2.0 * rho)


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Sufficient Gaussian calibration from the concentrated-DP bound."""

    if sensitivity <= 0.0 or not math.isfinite(sensitivity):
        raise ValueError("sensitivity must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    log_inverse_delta = -math.log(delta)
    return sensitivity / (math.sqrt(2.0) * epsilon) * (
        math.sqrt(log_inverse_delta + epsilon) + math.sqrt(log_inverse_delta)
    )


@dataclass(frozen=True)
class SufficientStatisticsBudget:
    """Per-fit zCDP split between the Gram and response releases."""

    epsilon: float
    delta: float
    rho: float
    rho_gram: float
    rho_response: float
    gram_sensitivity: float
    response_sensitivity: float
    gram_noise_sigma: float
    response_noise_sigma: float


def split_sufficient_statistics_budget(
    epsilon: float,
    delta: float,
    gram_sensitivity: float,
    response_sensitivity: float,
) -> SufficientStatisticsBudget:
    """Halve rho between the two sufficient-statistic releases of one fit.

    Both releases come from the same batch, so within a fit they compose by
    addition in zCDP. Across batches the releases are over disjoint records
    and compose in parallel, so no fit-count division appears here.
    """

    for name, value in (
        ("gram_sensitivity", gram_sensitivity),
        ("response_sensitivity", response_sensitivity),
    ):
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive")
    rho = zcdp_rho(epsilon, delta)
    rho_gram = rho_response = rho / 2.0
    converted = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    if not math.isclose(converted, epsilon, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("zCDP calibration does not convert to target epsilon")
    return SufficientStatisticsBudget(
        epsilon=epsilon,
        delta=delta,
        rho=rho,
        rho_gram=rho_gram,
        rho_response=rho_response,
        gram_sensitivity=gram_sensitivity,
        response_sensitivity=response_sensitivity,
        gram_noise_sigma=gaussian_sigma_from_rho(gram_sensitivity, rho_gram),
        response_noise_sigma=gaussian_sigma_from_rho(response_sensitivity, rho_response),
    )


@dataclass(frozen=True)
class ObjectiveGramBudget:
    """Within-batch split between a logistic fit and a Gram release."""

    objective_epsilon: float
    objective_delta: float
    gram_epsilon: float | None
    gram_delta: float | None


def split_objective_gram_budget(
    epsilon: float,
    delta: float,
    *,
    needs_gram: bool,
) -> ObjectiveGramBudget:
    """Split epsilon only; pure objective perturbation consumes no delta.

    The logistic oracle cannot reuse its fitting release as an exploration
    matrix the way the quadratic oracle reuses its noisy Gram. A confidence
    policy therefore spends half its budget on the Gram release.
    """

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    if needs_gram:
        return ObjectiveGramBudget(
            objective_epsilon=epsilon / 2.0,
            objective_delta=0.0,
            gram_epsilon=epsilon / 2.0,
            gram_delta=delta,
        )
    return ObjectiveGramBudget(epsilon, 0.0, None, None)
