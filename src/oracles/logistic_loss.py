"""Logistic-loss (log-loss) oracles.

Both oracles minimize the *averaged* regularized logistic objective on exactly
one completed batch, so predictions ``sigmoid(x.T theta)`` are expected
zero-one losses. The configured ridge is a coefficient of the averaged
objective; the equivalent summed objective would carry ``n * lambda``.

The private oracle protects the fit with objective perturbation, which
releases only a coefficient vector. The confidence policies also release a
noisy batch Gram, using half the per-fit budget for each release. The
quadratic-loss oracle reuses the Gram from its fitting mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..privacy import (
    Adjacency,
    ObjectiveGramBudget,
    ObjectivePerturbationCalibration,
    calibrate_objective_perturbation,
    release_gram,
    sample_objective_perturbation,
    split_objective_gram_budget,
)
from .base import (
    FitContext,
    Oracle,
    OracleConfiguration,
    OracleDiagnostics,
    Predictor,
    coerce_batch,
)


@dataclass(frozen=True)
class LogisticOptimization:
    theta: np.ndarray
    objective: float
    success: bool
    iterations: int
    gradient_norm: float


def logistic_objective_and_gradient(
    theta: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    ridge_lambda: float,
    objective_perturbation: np.ndarray | None = None,
    *,
    additional_ridge: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Return averaged binary logistic loss and its analytic gradient.

    The optional perturbation is ``b`` in
    ``mean(logaddexp(0, X theta) - y X theta)
    + (lambda + Delta)/2 ||theta||^2 + b^T theta / n``. The vector b is
    already calibrated by the caller; there is no source-record multiplier.
    """

    matrix = np.asarray(features, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    coefficient = np.asarray(theta, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or response.shape != (matrix.shape[0],):
        raise ValueError("malformed logistic batch")
    if coefficient.shape != (matrix.shape[1],):
        raise ValueError("theta has the wrong dimension")
    if ridge_lambda <= 0.0 or not math.isfinite(ridge_lambda):
        raise ValueError("logistic ridge_lambda must be finite and positive")
    if additional_ridge < 0.0 or not math.isfinite(additional_ridge):
        raise ValueError("additional_ridge must be finite and nonnegative")
    perturbation = (
        np.zeros(matrix.shape[1], dtype=np.float64)
        if objective_perturbation is None
        else np.asarray(objective_perturbation, dtype=np.float64)
    )
    if perturbation.shape != coefficient.shape or not np.all(np.isfinite(perturbation)):
        raise ValueError("objective perturbation has the wrong shape or is non-finite")
    logits = matrix @ coefficient
    probabilities = expit(logits)
    count = matrix.shape[0]
    effective_ridge = ridge_lambda + additional_ridge
    value = float(
        np.mean(np.logaddexp(0.0, logits) - response * logits)
        + effective_ridge * np.dot(coefficient, coefficient) / 2.0
        + np.dot(perturbation, coefficient) / count
    )
    gradient = (
        matrix.T @ (probabilities - response) + perturbation
    ) / count + effective_ridge * coefficient
    if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("logistic objective or gradient is non-finite")
    return value, gradient


def fit_logistic_objective(
    features: np.ndarray,
    targets: np.ndarray,
    configuration: OracleConfiguration,
    objective_perturbation: np.ndarray | None = None,
    *,
    additional_ridge: float = 0.0,
) -> LogisticOptimization:
    """Run L-BFGS once and use its finite iterate, regardless of convergence."""

    matrix = np.asarray(features, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    perturbation = (
        np.zeros(matrix.shape[1], dtype=np.float64)
        if objective_perturbation is None
        else np.asarray(objective_perturbation, dtype=np.float64)
    )

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return logistic_objective_and_gradient(
            theta,
            matrix,
            response,
            configuration.ridge_lambda,
            perturbation,
            additional_ridge=additional_ridge,
        )

    result = minimize(
        objective,
        np.zeros(matrix.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "gtol": configuration.optimizer_gradient_tolerance,
            "ftol": configuration.optimizer_function_tolerance,
            "maxiter": configuration.optimizer_max_iterations,
        },
    )
    theta = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        raise FloatingPointError("L-BFGS produced non-finite logistic parameters")
    value, gradient = objective(theta)
    # The returned iterate is accepted even when L-BFGS reports nonconvergence.
    # Status and gradient norm are diagnostics only: do not impose a residual
    # cutoff, retry the solver, or redraw the noise.
    return LogisticOptimization(
        theta=theta,
        objective=value,
        success=bool(result.success),
        iterations=int(result.nit),
        gradient_norm=float(np.linalg.norm(gradient)),
    )


def _record_group_size(matrix: np.ndarray, context: FitContext) -> tuple[int, int]:
    """Return pure-DP group size K and the original source-record count."""

    action_count = context.full_information_action_count
    if action_count is None:
        return 1, matrix.shape[0]
    if action_count <= 1 or matrix.shape[0] % action_count:
        raise ValueError("full-information logistic rows must be grouped by action")
    original_count = matrix.shape[0] // action_count
    if context.sample_indices and len(context.sample_indices) != original_count:
        raise ValueError("full-information record count differs from sample indices")
    return action_count, original_count


class _LogisticLossOracle(Oracle):
    """Shared fitting and diagnostics implementation for logistic ERM."""

    private: ClassVar[bool] = False

    def __init__(
        self,
        configuration: OracleConfiguration,
        *,
        needs_exploration_statistic: bool = False,
    ) -> None:
        super().__init__(
            configuration, needs_exploration_statistic=needs_exploration_statistic
        )
        if configuration.ridge_lambda <= 0.0:
            raise ValueError("logistic objective perturbation requires ridge_lambda > 0")
        if configuration.feature_norm_bound > 1.0 + configuration.bound_tolerance:
            raise ValueError("logistic privacy analysis requires feature_norm_bound <= 1")
        if configuration.adjacency is not Adjacency.REPLACE_ONE:
            raise ValueError(
                "logistic objective perturbation currently requires replace_one adjacency"
            )

    def _objective_perturbation(
        self,
        dimension: int,
        rng: np.random.Generator,
        *,
        scalar_sample_count: int,
        group_size: int,
    ) -> tuple[
        np.ndarray,
        ObjectiveGramBudget | None,
        ObjectivePerturbationCalibration | None,
    ]:
        if not self.private:
            return np.zeros(dimension, dtype=np.float64), None, None
        budget = split_objective_gram_budget(
            self.configuration.epsilon_reg,
            self.configuration.delta_reg,
            needs_gram=self.needs_exploration_statistic,
        )
        # Protect replacement of one original supervised record with pure-DP
        # group privacy. No further K-dependent multiplier is applied to b.
        calibration = calibrate_objective_perturbation(
            scalar_sample_count,
            self.configuration.ridge_lambda,
            budget.objective_epsilon / group_size,
        )
        perturbation = sample_objective_perturbation(
            dimension, calibration.epsilon_noise, rng
        )
        return perturbation, budget, calibration

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        rng: np.random.Generator,
        *,
        context: FitContext | None = None,
    ) -> Predictor:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        context = context or FitContext()
        matrix, response, norms = coerce_batch(
            features,
            targets,
            self.configuration,
            dense_gram_required=self.needs_exploration_statistic,
        )
        if np.any(response < 0.0) or np.any(response > 1.0):
            raise ValueError("logistic loss targets must lie in [0, 1]")
        group_size, record_count = _record_group_size(matrix, context)
        if self.private and self.needs_exploration_statistic and group_size != 1:
            raise ValueError("private logistic Gram release is only calibrated for bandit rows")
        perturbation, budget, calibration = self._objective_perturbation(
            matrix.shape[1],
            rng,
            scalar_sample_count=matrix.shape[0],
            group_size=group_size,
        )
        additional_ridge = 0.0 if calibration is None else calibration.additional_ridge
        try:
            optimization = fit_logistic_objective(
                matrix,
                response,
                self.configuration,
                perturbation,
                additional_ridge=additional_ridge,
            )
        except (RuntimeError, FloatingPointError) as error:
            if not self.private:
                raise
            # A numerical failure must not disclose the raw-data gradient,
            # objective, or any other optimizer diagnostics through its text.
            raise type(error)("private logistic optimization failed; details withheld") from None
        theta = optimization.theta

        released_gram: np.ndarray | None = None
        gram_sigma: float | None = None
        gram_sensitivity: float | None = None
        min_before: float | None = None
        min_after_noise: float | None = None
        if self.needs_exploration_statistic:
            if self.private:
                assert budget is not None
                assert budget.gram_epsilon is not None
                assert budget.gram_delta is not None
                release = release_gram(
                    matrix,
                    epsilon=budget.gram_epsilon,
                    delta=budget.gram_delta,
                    rng=rng,
                )
                released_gram = release.noisy_gram
                gram_sensitivity = release.sensitivity
                gram_sigma = release.noise_sigma
                min_before = float(np.min(np.linalg.eigvalsh(release.exact_gram)))
                min_after_noise = float(np.min(np.linalg.eigvalsh(release.noisy_gram)))
            else:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    released_gram = matrix.T @ matrix
                if not np.all(np.isfinite(released_gram)):
                    raise FloatingPointError("exact Gram statistic is non-finite")
                min_before = float(np.min(np.linalg.eigvalsh(released_gram)))
                min_after_noise = min_before

        theta_norm = float(np.linalg.norm(theta))
        diagnostics = OracleDiagnostics(
            algorithm=context.algorithm,
            batch_index=context.batch_index,
            batch_size=record_count,
            sample_indices=context.sample_indices,
            oracle_class=type(self).__name__,
            private=self.private,
            epsilon_reg=(self.configuration.epsilon_reg if self.private else None),
            delta_reg=(self.configuration.delta_reg if self.private else None),
            adjacency=self.configuration.adjacency.value,
            feature_norm_bound=self.configuration.feature_norm_bound,
            target_bound=self.configuration.target_bound,
            ridge_lambda=self.configuration.ridge_lambda,
            theta_norm=theta_norm,
            feature_dimension=matrix.shape[1],
            max_observed_feature_norm=float(norms.max()),
            gram_sensitivity=gram_sensitivity,
            gram_noise_sigma=gram_sigma,
            gram_min_eigenvalue_before_noise=min_before,
            gram_min_eigenvalue_after_noise=min_after_noise,
            fraction_predictions_below_zero=0.0,
            fraction_predictions_above_one=0.0,
            nonprivate_ridge_theta=(
                None if self.private else tuple(float(value) for value in theta)
            ),
            optimizer_success=optimization.success,
            optimizer_iterations=optimization.iterations,
            optimizer_gradient_norm=optimization.gradient_norm,
            theta_private_norm=(theta_norm if self.private else None),
            ridge_parameterization=(
                "averaged_logistic_algorithm2_objective_perturbation"
                if self.private
                else "averaged_logistic_loss"
            ),
        )
        self._record_debug_diagnostics(diagnostics)
        return Predictor(
            theta,
            matrix.shape[0],
            diagnostics.redacted(),
            prediction_link="sigmoid",
            exploration_statistic=released_gram,
        )


class NonPrivateLogisticLossOracle(_LogisticLossOracle):
    """Deterministic L-BFGS logistic ERM with analytic gradient."""


class PrivateLogisticLossOracle(_LogisticLossOracle):
    """Algorithm 2 spherical objective perturbation plus optional Gram DP.

    The exact-minimizer privacy theorem does not certify finite-precision
    optimization merely because L-BFGS reports success.
    """

    private: ClassVar[bool] = True
