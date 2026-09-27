"""Chaudhuri et al. Algorithm 2 objective perturbation for logistic loss.

The mechanism adds one fixed linear term ``b.T theta / n`` to the averaged
objective before minimizing, and may raise the ridge when the budget is too
small for the unmodified objective. There is no output noise and no
per-iteration noise. The privacy theorem covers the exact minimizer; a
numerical solver reporting success is not a finite-precision privacy proof.

See https://www.jmlr.org/papers/volume12/chaudhuri11a/chaudhuri11a.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .noise import spherical_gaussian_radius_draw

# Curvature bound of the logistic loss.
LOGISTIC_CURVATURE_BOUND = 0.25


@dataclass(frozen=True)
class ObjectivePerturbationCalibration:
    """Public Algorithm 2 parameters, not a realized private noise draw."""

    epsilon_fit: float
    epsilon_noise: float
    additional_ridge: float
    scalar_sample_count: int


def calibrate_objective_perturbation(
    n: int, ridge_lambda: float, epsilon_fit: float
) -> ObjectivePerturbationCalibration:
    """Algorithm 2 for logistic loss, with curvature bound ``c = 1/4``.

    ``ridge_lambda`` and the returned extra ridge are coefficients in the
    averaged objective. ``n`` counts scalar fitting rows, including all K
    expanded rows per original supervised example. Both branches satisfy
    ``epsilon_noise + 2 log(1 + c / [n (lambda + Delta)]) = epsilon_fit``.
    """

    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda must be finite and positive")
    if not math.isfinite(epsilon_fit) or epsilon_fit <= 0.0:
        raise ValueError("epsilon_fit must be finite and positive")
    c = LOGISTIC_CURVATURE_BOUND
    u = epsilon_fit - 2.0 * math.log1p(c / (n * ridge_lambda))
    if u > 0.0:
        additional_ridge = 0.0
        epsilon_noise = u
    else:
        additional_ridge = c / (n * math.expm1(epsilon_fit / 4.0)) - ridge_lambda
        epsilon_noise = epsilon_fit / 2.0
    return ObjectivePerturbationCalibration(
        epsilon_fit=epsilon_fit,
        epsilon_noise=epsilon_noise,
        additional_ridge=additional_ridge,
        scalar_sample_count=int(n),
    )


def sample_objective_perturbation(
    dimension: int, epsilon_noise: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw the fixed linear perturbation ``b`` for one fit."""

    return spherical_gaussian_radius_draw(dimension, epsilon_noise, rng)
