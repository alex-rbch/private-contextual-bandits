"""Quadratic-loss (ridge) oracles.

By default, both oracles solve the fitting-row-scaled sum-loss ridge problem

    theta = argmin_u 0.5 ||X u - y||^2 + 0.5 n_j lambda ||u||^2

on exactly one completed batch, where ``n_j`` is the number of scalar fitting
rows: ``B_j`` for bandit feedback and ``A B_j`` for full-information supervised
feedback. An isolated legacy comparison option retains the fixed ridge.

The private oracle makes a single sufficient-statistics release and then
post-processes it. Because the released noisy Gram is the same matrix the
exploration geometry needs, confidence policies cost no extra privacy budget
here, unlike the logistic path.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..privacy import (
    BANDIT_GRAM_SENSITIVITY,
    BANDIT_RESPONSE_SENSITIVITY,
    full_information_sensitivities,
    release_sufficient_statistics,
)
from .base import (
    FitContext,
    Oracle,
    OracleConfiguration,
    OracleDiagnostics,
    Predictor,
    assert_sum_loss_ridge_norm_bound,
    coerce_batch,
    out_of_range_fractions,
    solve_cholesky,
    solve_dual_sum_loss_ridge,
)


def _full_information_batch(
    matrix: np.ndarray,
    response: np.ndarray,
    context: FitContext,
    configuration: OracleConfiguration,
) -> tuple[float, float, int]:
    """Resolve sensitivities and the original-record count for one batch."""

    if context.full_information_action_count is None:
        return BANDIT_GRAM_SENSITIVITY, BANDIT_RESPONSE_SENSITIVITY, matrix.shape[0]
    gram, resp, records = full_information_sensitivities(
        matrix,
        response,
        context.full_information_action_count,
        feature_norm_bound=configuration.feature_norm_bound,
        target_bound=configuration.target_bound,
        adjacency=configuration.adjacency,
        bound_tolerance=configuration.bound_tolerance,
    )
    if context.sample_indices and len(context.sample_indices) != records:
        raise ValueError("full-information record count differs from sample indices")
    return gram, resp, records


class NonPrivateQuadraticLossOracle(Oracle):
    """Exact batch-local sum-loss ridge through the smaller exact system."""

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
        del rng
        context = context or FitContext()
        matrix, response, norms = coerce_batch(
            features, targets, self.configuration, dense_gram_required=False
        )
        _, _, record_count = _full_information_batch(
            matrix, response, context, self.configuration
        )
        ridge_lambda = self.configuration.ridge_lambda * (
            matrix.shape[0]
            if self.configuration.quadratic_ridge_scale_by_batch_records
            else 1
        )
        use_primal = (
            self.configuration.prefer_primal_ridge_when_lower_dimension
            and matrix.shape[1] < matrix.shape[0]
        )
        exploration_statistic: np.ndarray | None = None
        if use_primal:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                gram = matrix.T @ matrix
                moment = matrix.T @ response
            if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(moment)):
                raise FloatingPointError(
                    "primal ridge produced non-finite sufficient statistics"
                )
            regularized = gram.copy()
            regularized.flat[:: matrix.shape[1] + 1] += ridge_lambda
            theta = np.asarray(
                solve_cholesky(np.linalg.cholesky(regularized), moment),
                dtype=np.float64,
            ).reshape(-1)
            if not np.all(np.isfinite(theta)):
                raise FloatingPointError("primal ridge produced non-finite coefficients")
            condition_number = float(np.linalg.cond(regularized))
            if self.needs_exploration_statistic:
                exploration_statistic = gram
        else:
            theta, condition_number = solve_dual_sum_loss_ridge(
                matrix, response, ridge_lambda
            )
            if self.needs_exploration_statistic:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    exploration_statistic = matrix.T @ matrix
                if not np.all(np.isfinite(exploration_statistic)):
                    raise FloatingPointError("exact Gram statistic is non-finite")
        assert_sum_loss_ridge_norm_bound(theta, matrix.shape[0], self.configuration)
        low, high = out_of_range_fractions(theta, matrix)
        theta_norm = float(np.linalg.norm(theta))
        diagnostics = OracleDiagnostics(
            algorithm=context.algorithm,
            batch_index=context.batch_index,
            batch_size=record_count,
            sample_indices=context.sample_indices,
            oracle_class=type(self).__name__,
            private=False,
            epsilon_reg=None,
            delta_reg=None,
            adjacency=self.configuration.adjacency.value,
            feature_norm_bound=self.configuration.feature_norm_bound,
            target_bound=self.configuration.target_bound,
            ridge_lambda=ridge_lambda,
            theta_norm=theta_norm,
            feature_dimension=matrix.shape[1],
            max_observed_feature_norm=float(norms.max()),
            fraction_predictions_below_zero=low,
            fraction_predictions_above_one=high,
            theta_hat_norm=theta_norm,
            theta_private_norm=theta_norm,
            dual_matrix_dimension=None if use_primal else matrix.shape[0],
            dual_matrix_condition_number=None if use_primal else condition_number,
            prediction_out_of_range_fraction=low + high,
            mean_abs_private_minus_nonprivate_prediction=0.0,
            max_abs_private_minus_nonprivate_prediction=0.0,
            ridge_parameterization=(
                "sum_loss_fitting_row_scaled"
                if self.configuration.quadratic_ridge_scale_by_batch_records
                else "sum_loss_fixed_legacy_comparison"
            ),
        )
        self._record_debug_diagnostics(diagnostics)
        return Predictor(
            theta,
            matrix.shape[0],
            diagnostics,
            prediction_link="identity",
            exploration_statistic=exploration_statistic,
        )


class PrivateQuadraticLossOracle(Oracle):
    """One-shot zCDP sufficient-statistics perturbation for each fresh batch."""

    private: ClassVar[bool] = True

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
            features, targets, self.configuration, dense_gram_required=True
        )
        gram_sensitivity, response_sensitivity, record_count = _full_information_batch(
            matrix, response, context, self.configuration
        )
        mechanism_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        release = release_sufficient_statistics(
            matrix,
            response,
            epsilon=self.configuration.epsilon_reg,
            delta=self.configuration.delta_reg,
            gram_sensitivity=gram_sensitivity,
            response_sensitivity=response_sensitivity,
            seed=mechanism_seed,
        )
        ridge_lambda = self.configuration.ridge_lambda * (
            matrix.shape[0]
            if self.configuration.quadratic_ridge_scale_by_batch_records
            else 1
        )
        dimension = release.dimension
        private_system = release.noisy_gram.copy()
        private_system.flat[:: dimension + 1] += ridge_lambda
        theta = np.linalg.solve(private_system, release.noisy_response)
        exact_system = release.exact_gram.copy()
        exact_system.flat[:: dimension + 1] += ridge_lambda
        if ridge_lambda == 0.0:
            # Trusted evaluation only: early batches can have rank-deficient
            # X.T X. This does not alter the released private solve above.
            theta_exact = np.linalg.lstsq(
                release.normalized_features, response, rcond=None
            )[0]
        else:
            theta_exact = np.linalg.solve(exact_system, release.exact_response)
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(theta_exact)):
            raise FloatingPointError("ridge solve produced non-finite coefficients")

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            private_predictions = release.normalized_features @ theta
            exact_predictions = release.normalized_features @ theta_exact
        if not np.all(np.isfinite(private_predictions)):
            raise FloatingPointError("zCDP SSP produced non-finite predictions")
        difference = np.abs(private_predictions - exact_predictions)
        low = float(np.mean(private_predictions < 0.0))
        high = float(np.mean(private_predictions > 1.0))
        theta_norm = float(np.linalg.norm(theta))
        budget = release.budget
        diagnostics = OracleDiagnostics(
            algorithm=context.algorithm,
            batch_index=context.batch_index,
            batch_size=record_count,
            sample_indices=context.sample_indices,
            oracle_class=type(self).__name__,
            private=True,
            epsilon_reg=self.configuration.epsilon_reg,
            delta_reg=self.configuration.delta_reg,
            adjacency=self.configuration.adjacency.value,
            feature_norm_bound=self.configuration.feature_norm_bound,
            target_bound=self.configuration.target_bound,
            ridge_lambda=ridge_lambda,
            theta_norm=theta_norm,
            feature_dimension=dimension,
            max_observed_feature_norm=float(norms.max()),
            gram_sensitivity=budget.gram_sensitivity,
            response_sensitivity=budget.response_sensitivity,
            gram_noise_sigma=budget.gram_noise_sigma,
            response_noise_sigma=budget.response_noise_sigma,
            gram_min_eigenvalue_before_noise=float(
                np.min(np.linalg.eigvalsh(release.exact_gram))
            ),
            gram_min_eigenvalue_after_noise=float(
                np.min(np.linalg.eigvalsh(release.noisy_gram))
            ),
            fraction_predictions_below_zero=low,
            fraction_predictions_above_one=high,
            nonprivate_ridge_theta=tuple(float(value) for value in theta_exact),
            theta_hat_norm=float(np.linalg.norm(theta_exact)),
            theta_private_norm=theta_norm,
            prediction_out_of_range_fraction=low + high,
            mean_abs_private_minus_nonprivate_prediction=float(np.mean(difference)),
            max_abs_private_minus_nonprivate_prediction=float(np.max(difference)),
            ridge_parameterization=(
                "sum_loss_ssp_zcdp_fitting_row_scaled"
                if self.configuration.quadratic_ridge_scale_by_batch_records
                else "sum_loss_ssp_zcdp_fixed_legacy_comparison"
            ),
        )
        self._record_debug_diagnostics(diagnostics)
        return Predictor(
            theta,
            matrix.shape[0],
            diagnostics.redacted(),
            prediction_link="identity",
            exploration_statistic=(
                release.noisy_gram if self.needs_exploration_statistic else None
            ),
        )
