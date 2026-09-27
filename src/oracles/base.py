"""Shared oracle types.

An oracle turns one completed batch into a :class:`Predictor`. Every oracle in
this package is batch-local: it sees exactly the rows of the batch it is
fitting, and it never retains raw rows afterwards. Policies consume the
returned predictor and nothing else, except for the confidence policies, which
also consume the released ``exploration_statistic``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
import math
from typing import ClassVar

import numpy as np
from scipy.special import expit

from ..privacy import Adjacency

PREDICTION_LINKS = ("identity", "sigmoid")


@dataclass(frozen=True)
class FitContext:
    """Public/debug metadata associated with one oracle invocation."""

    algorithm: str = "unspecified"
    batch_index: int = 0
    sample_indices: tuple[int, ...] = ()
    full_information_action_count: int | None = None


@dataclass(frozen=True)
class OracleConfiguration:
    """Settings shared by the quadratic-loss and logistic-loss oracles."""

    ridge_lambda: float
    feature_norm_bound: float = 1.0
    target_bound: float = 1.0
    adjacency: Adjacency = Adjacency.REPLACE_ONE
    epsilon_reg: float = 1.0
    delta_reg: float = 1e-6
    validate_bounds: bool = True
    bound_tolerance: float = 1e-10
    optimizer_gradient_tolerance: float = 1e-8
    optimizer_function_tolerance: float = 1e-12
    optimizer_max_iterations: int = 2_000
    # The exact non-private sum-loss solution may use the smaller primal
    # system when d < n. The default preserves the historical dual path.
    prefer_primal_ridge_when_lower_dimension: bool = False
    # Exact SSP is dense. This guard prevents an accidental d^2 allocation.
    max_working_memory_bytes: int = 2_000_000_000
    # Legacy option name retained for comparison scripts: True scales ridge
    # by scalar fitting rows (B_j for bandit, A*B_j for supervised). False
    # preserves the historical fixed-ridge solve for isolated comparisons.
    quadratic_ridge_scale_by_batch_records: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "adjacency", Adjacency(self.adjacency))
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be finite and nonnegative")
        if not isinstance(self.prefer_primal_ridge_when_lower_dimension, bool):
            raise TypeError("prefer_primal_ridge_when_lower_dimension must be boolean")
        if not isinstance(self.quadratic_ridge_scale_by_batch_records, bool):
            raise TypeError("quadratic_ridge_scale_by_batch_records must be boolean")
        for name in ("feature_norm_bound", "target_bound", "epsilon_reg"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.delta_reg) or not 0.0 < self.delta_reg < 1.0:
            raise ValueError("delta_reg must lie strictly between zero and one")
        if self.bound_tolerance < 0.0 or not math.isfinite(self.bound_tolerance):
            raise ValueError("bound_tolerance must be finite and nonnegative")
        for name in ("optimizer_gradient_tolerance", "optimizer_function_tolerance"):
            value = getattr(self, name)
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
        if self.optimizer_max_iterations <= 0:
            raise ValueError("optimizer_max_iterations must be positive")
        if self.max_working_memory_bytes <= 0:
            raise ValueError("max_working_memory_bytes must be positive")


@dataclass(frozen=True)
class OracleDiagnostics:
    """Per-fit diagnostics; private-batch facts are redacted before release."""

    algorithm: str
    batch_index: int
    batch_size: int
    sample_indices: tuple[int, ...] | None
    oracle_class: str
    private: bool
    epsilon_reg: float | None
    delta_reg: float | None
    adjacency: str
    feature_norm_bound: float
    target_bound: float
    ridge_lambda: float
    theta_norm: float
    feature_dimension: int | None = None
    max_observed_feature_norm: float | None = None
    gram_sensitivity: float | None = None
    response_sensitivity: float | None = None
    gram_noise_sigma: float | None = None
    response_noise_sigma: float | None = None
    gram_min_eigenvalue_before_noise: float | None = None
    gram_min_eigenvalue_after_noise: float | None = None
    fraction_predictions_below_zero: float | None = None
    fraction_predictions_above_one: float | None = None
    nonprivate_ridge_theta: tuple[float, ...] | None = None
    optimizer_success: bool | None = None
    optimizer_iterations: int | None = None
    optimizer_gradient_norm: float | None = None
    theta_hat_norm: float | None = None
    theta_private_norm: float | None = None
    dual_matrix_dimension: int | None = None
    dual_matrix_condition_number: float | None = None
    prediction_out_of_range_fraction: float | None = None
    mean_abs_private_minus_nonprivate_prediction: float | None = None
    max_abs_private_minus_nonprivate_prediction: float | None = None
    ridge_parameterization: str | None = None

    # These fields inspect the unnoised private batch and must not be emitted
    # as part of a private release. They remain available for local debugging.
    _SENSITIVE_PRIVATE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "sample_indices",
            "max_observed_feature_norm",
            "gram_min_eigenvalue_before_noise",
            "fraction_predictions_below_zero",
            "fraction_predictions_above_one",
            "nonprivate_ridge_theta",
            "optimizer_success",
            "optimizer_iterations",
            "optimizer_gradient_norm",
            "theta_hat_norm",
            "dual_matrix_condition_number",
            "prediction_out_of_range_fraction",
            "mean_abs_private_minus_nonprivate_prediction",
            "max_abs_private_minus_nonprivate_prediction",
        }
    )

    def debug_dict(self) -> dict[str, object]:
        """Return complete local diagnostics, including private-batch facts."""

        payload = asdict(self)
        payload.pop("_SENSITIVE_PRIVATE_FIELDS", None)
        payload["sample_indices"] = (
            list(self.sample_indices) if self.sample_indices is not None else None
        )
        return payload

    def redacted(self) -> "OracleDiagnostics":
        """Return a model-safe copy with raw private-batch facts removed."""

        if not self.private:
            return self
        return replace(self, **{field: None for field in self._SENSITIVE_PRIVATE_FIELDS})

    def release_dict(self) -> dict[str, object]:
        """Return diagnostics safe to accompany the released private model."""

        return self.redacted().debug_dict()


@dataclass(frozen=True)
class Predictor:
    """One released predictor, plus the statistic exploration may pool.

    ``exploration_statistic`` is the unnormalized, unregularized second-moment
    matrix this batch released: ``X.T X + E`` under either private loss model,
    or ``X.T X`` without privacy. It is present only when a confidence policy
    asked for it.
    """

    theta: np.ndarray
    sample_count: int
    diagnostics: OracleDiagnostics
    prediction_link: str = "identity"
    exploration_statistic: np.ndarray | None = None

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=np.float64).copy()
        if theta.ndim != 1:
            raise ValueError("theta must be one-dimensional")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.prediction_link not in PREDICTION_LINKS:
            raise ValueError(f"prediction_link must be one of {PREDICTION_LINKS}")
        exploration = (
            None
            if self.exploration_statistic is None
            else np.asarray(self.exploration_statistic, dtype=np.float64).copy()
        )
        if exploration is not None and (
            exploration.shape != (theta.size, theta.size)
            or not np.all(np.isfinite(exploration))
        ):
            raise ValueError("exploration statistic must be a finite d-by-d matrix")
        theta.setflags(write=False)
        if exploration is not None:
            exploration.setflags(write=False)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "exploration_statistic", exploration)

    @property
    def dimension(self) -> int:
        return int(self.theta.size)

    def raw_predict(self, features: np.ndarray) -> np.ndarray | float:
        """Return the linear score, before any link function."""

        matrix = np.asarray(features, dtype=np.float64)
        if matrix.shape == (self.dimension,):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return float(matrix @ self.theta)
        if matrix.ndim < 2 or matrix.shape[-1] != self.dimension:
            raise ValueError("features have the wrong final dimension")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return matrix @ self.theta

    def predict(self, features: np.ndarray) -> np.ndarray | float:
        """Return the predicted action loss under this oracle's link."""

        raw = self.raw_predict(features)
        if self.prediction_link == "sigmoid":
            linked = expit(raw)
            return float(linked) if np.ndim(linked) == 0 else linked
        return raw


class Oracle(ABC):
    """Fit one completed batch and release a predictor."""

    private: ClassVar[bool] = False

    def __init__(
        self,
        configuration: OracleConfiguration,
        *,
        needs_exploration_statistic: bool = False,
    ) -> None:
        self.configuration = configuration
        # Releasing a dense d-by-d statistic is optional: only the confidence
        # policies read it, and under logistic loss it additionally costs
        # privacy budget, so it is requested rather than always produced.
        self.needs_exploration_statistic = bool(needs_exploration_statistic)
        self._debug_diagnostics: list[OracleDiagnostics] = []

    @property
    def debug_diagnostics(self) -> tuple[OracleDiagnostics, ...]:
        """Trusted local log; never serialize this in a private result."""

        return tuple(self._debug_diagnostics)

    def _record_debug_diagnostics(self, diagnostics: OracleDiagnostics) -> None:
        self._debug_diagnostics.append(diagnostics)

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        rng: np.random.Generator,
        *,
        context: FitContext | None = None,
    ) -> Predictor:
        raise NotImplementedError


def coerce_batch(
    features: np.ndarray,
    targets: np.ndarray,
    configuration: OracleConfiguration,
    *,
    dense_gram_required: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate one regression batch and return it with per-row norms."""

    matrix = np.asarray(features, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional matrix")
    if response.shape != (matrix.shape[0],):
        raise ValueError("targets must have one entry per feature row")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(response)):
        raise ValueError("features and targets must be finite")
    target_ceiling = min(1.0, configuration.target_bound)
    if np.any(response < 0.0) or np.any(
        response > target_ceiling + configuration.bound_tolerance
    ):
        raise ValueError("targets must lie in [0, min(1, target_bound)]")
    norms = np.linalg.norm(matrix, axis=1)
    if configuration.validate_bounds and float(norms.max()) > (
        configuration.feature_norm_bound + configuration.bound_tolerance
    ):
        raise ValueError(
            "observed feature norm exceeds feature_norm_bound: "
            f"{float(norms.max()):.12g} > {configuration.feature_norm_bound:.12g}"
        )
    if dense_gram_required:
        # NumPy's dense eigh/solve path needs several simultaneous d-by-d arrays.
        dimension = matrix.shape[1]
        estimated_bytes = 7 * dimension * dimension * np.dtype(np.float64).itemsize
        if estimated_bytes > configuration.max_working_memory_bytes:
            raise MemoryError(
                "exact dense SSP would exceed max_working_memory_bytes: "
                f"dimension={dimension}, estimated_working_bytes={estimated_bytes}, "
                f"limit={configuration.max_working_memory_bytes}. Use a public, explicitly "
                "specified lower-dimensional feature map; this oracle will not silently "
                "project data."
            )
    return matrix, response, norms


def solve_cholesky(factor: np.ndarray, response: np.ndarray) -> np.ndarray:
    return np.linalg.solve(factor.T, np.linalg.solve(factor, response))


def solve_dual_sum_loss_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, float]:
    """Solve sum-loss ridge through the ``n x n`` dual system.

    The supplied ridge is inserted directly into ``X X.T + lambda I``, with no
    multiplication by the batch size. This deliberately never forms ``X.T X``.
    """

    matrix = np.asarray(features, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional matrix")
    if response.shape != (matrix.shape[0],):
        raise ValueError("targets must have one entry per feature row")
    if ridge_lambda <= 0.0 or not math.isfinite(ridge_lambda):
        raise ValueError("ridge_lambda must be finite and positive")
    batch_size = matrix.shape[0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dual = matrix @ matrix.T
    if not np.all(np.isfinite(dual)):
        raise FloatingPointError("dual ridge produced a non-finite kernel matrix")
    dual.flat[:: batch_size + 1] += ridge_lambda
    alpha = solve_cholesky(np.linalg.cholesky(dual), response)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        theta = np.asarray(matrix.T @ alpha, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(theta)):
        raise FloatingPointError("dual ridge produced non-finite coefficients")
    return theta, float(np.linalg.cond(dual))


def assert_sum_loss_ridge_norm_bound(
    theta: np.ndarray,
    batch_size: int,
    configuration: OracleConfiguration,
) -> None:
    """Check the public norm certificate of an exact sum-loss ridge solution."""

    bound = (
        math.sqrt(float(batch_size))
        * configuration.target_bound
        / math.sqrt(configuration.ridge_lambda)
    )
    tolerance = max(configuration.bound_tolerance, 1e-10) * max(1.0, bound)
    if float(np.linalg.norm(theta)) > bound + tolerance:
        raise AssertionError(
            "exact sum-loss ridge solution violated its public norm certificate: "
            f"||theta_hat||={float(np.linalg.norm(theta)):.12g}, bound={bound:.12g}"
        )


def out_of_range_fractions(
    theta: np.ndarray, features: np.ndarray
) -> tuple[float, float]:
    """Fractions of training predictions below zero and above one."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = features @ theta
    if not np.all(np.isfinite(raw)):
        raise FloatingPointError("linear ridge produced non-finite training predictions")
    return float(np.mean(raw < 0.0)), float(np.mean(raw > 1.0))
