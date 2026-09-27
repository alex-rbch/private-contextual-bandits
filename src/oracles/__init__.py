"""The two batch-local regression oracles and their shared types.

Exactly two loss models exist: ``quadratic`` (ridge) and ``logistic``
(log loss). Each has a non-private and a private implementation, and each
fits only the batch it is handed.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

from .base import (
    FitContext,
    Oracle,
    OracleConfiguration,
    OracleDiagnostics,
    Predictor,
    coerce_batch,
)
from .logistic_loss import (
    NonPrivateLogisticLossOracle,
    PrivateLogisticLossOracle,
    fit_logistic_objective,
    logistic_objective_and_gradient,
)
from .quadratic_loss import (
    NonPrivateQuadraticLossOracle,
    PrivateQuadraticLossOracle,
)

LOSS_MODELS = ("quadratic", "logistic")

_IMPLEMENTATIONS: dict[tuple[str, bool], type[Oracle]] = {
    ("quadratic", False): NonPrivateQuadraticLossOracle,
    ("quadratic", True): PrivateQuadraticLossOracle,
    ("logistic", False): NonPrivateLogisticLossOracle,
    ("logistic", True): PrivateLogisticLossOracle,
}

_NAMES: dict[tuple[str, bool], str] = {
    ("quadratic", False): "nonprivate_quadratic_loss",
    ("quadratic", True): "private_ssp_zcdp_quadratic_loss",
    ("logistic", False): "nonprivate_logistic_loss",
    ("logistic", True): "private_objective_perturbation_logistic_loss",
}


def oracle_class(loss: str, private: bool) -> type[Oracle]:
    """Resolve the oracle implementation for one loss model and regime."""

    try:
        return _IMPLEMENTATIONS[(loss, bool(private))]
    except KeyError as error:
        raise ValueError(f"unsupported loss model {loss!r}") from error


def oracle_name(loss: str, private: bool) -> str:
    """Stable identifier recorded in experiment metadata."""

    try:
        return _NAMES[(loss, bool(private))]
    except KeyError as error:
        raise ValueError(f"unsupported loss model {loss!r}") from error


def build_oracles(
    algorithms: Iterable[str],
    configuration: OracleConfiguration,
    *,
    private: bool,
    loss: str,
    needs_exploration_statistic: Callable[[str], bool],
) -> dict[str, Oracle]:
    """Construct one independent, identically configured oracle per algorithm.

    ``needs_exploration_statistic`` is supplied by the policy layer, because
    whether a released second-moment matrix is required is a property of the
    policy, not of the loss model.
    """

    names = tuple(algorithms)
    if not names or len(names) != len(set(names)):
        raise ValueError("algorithms must be nonempty and unique")
    implementation = oracle_class(loss, private)
    result = {
        name: implementation(
            configuration,
            needs_exploration_statistic=needs_exploration_statistic(name),
        )
        for name in names
    }
    assert {type(oracle) for oracle in result.values()} == {implementation}
    assert all(oracle.configuration == configuration for oracle in result.values())
    return result


def released_diagnostics(
    predictors: Mapping[str, Predictor]
) -> dict[str, dict[str, object]]:
    """Serialize only release-safe diagnostics for a collection of predictors."""

    return {name: p.diagnostics.release_dict() for name, p in predictors.items()}


__all__ = [
    "LOSS_MODELS",
    "FitContext",
    "NonPrivateLogisticLossOracle",
    "NonPrivateQuadraticLossOracle",
    "Oracle",
    "OracleConfiguration",
    "OracleDiagnostics",
    "Predictor",
    "PrivateLogisticLossOracle",
    "PrivateQuadraticLossOracle",
    "build_oracles",
    "coerce_batch",
    "fit_logistic_objective",
    "logistic_objective_and_gradient",
    "oracle_class",
    "oracle_name",
    "released_diagnostics",
]
