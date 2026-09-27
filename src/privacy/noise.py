"""Noise draws shared by the mechanisms."""

from __future__ import annotations

import math

import numpy as np


def symmetric_frobenius_gaussian(
    dimension: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample isotropic Gaussian noise in symmetric Frobenius geometry.

    Diagonal entries have variance ``sigma^2`` and off-diagonal entries have
    variance ``sigma^2 / 2``, with symmetric entries equal.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if sigma < 0.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and nonnegative")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    raw = sigma * rng.standard_normal((dimension, dimension))
    return (raw + raw.T) / 2.0


def spherical_gaussian_radius_draw(
    dimension: int, epsilon_noise: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw b with density proportional to ``exp(-epsilon_noise ||b|| / 2)``.

    This is neither Gaussian nor coordinatewise Laplace: the direction is
    uniform on the sphere and the radius is Gamma(d, 2/epsilon_noise).
    """

    if (
        not isinstance(dimension, (int, np.integer))
        or isinstance(dimension, bool)
        or dimension <= 0
    ):
        raise ValueError("dimension must be a positive integer")
    if not math.isfinite(epsilon_noise) or epsilon_noise <= 0.0:
        raise ValueError("epsilon_noise must be finite and positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    direction = rng.normal(size=dimension)
    norm = float(np.linalg.norm(direction))
    # The zero direction has probability zero in the continuous mechanism;
    # resampling protects the finite-precision implementation.
    while norm == 0.0:
        direction = rng.normal(size=dimension)
        norm = float(np.linalg.norm(direction))
    radius = rng.gamma(shape=dimension, scale=2.0 / epsilon_noise)
    return (direction / norm) * radius
