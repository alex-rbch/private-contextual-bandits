"""Public, algorithm-independent context preprocessing."""

from __future__ import annotations

import math

import numpy as np


def normalize_rows_to_unit_ball(features: np.ndarray) -> np.ndarray:
    """Project each feature row radially onto the Euclidean unit ball."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix / np.maximum(1.0, norms)[:, None]
    if float(np.max(np.linalg.norm(normalized, axis=1))) > 1.0 + 1e-12:
        raise AssertionError("row normalization failed to enforce the unit ball")
    return normalized


def prepare_contexts(
    contexts: np.ndarray,
    *,
    use_bias: bool = True,
    project_unitball: bool = True,
) -> np.ndarray:
    """Prepare raw contexts, by default as ``(1, project(x))/sqrt(2)``.

    The fixed scale preserves the active experiment's norm-one feature bound.
    Turning projection off may invalidate that bound for large raw contexts.
    """

    matrix = np.asarray(contexts, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("contexts must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("contexts must be finite")
    if project_unitball:
        matrix = normalize_rows_to_unit_ball(matrix)
    if use_bias:
        matrix = np.concatenate((np.ones((matrix.shape[0], 1)), matrix), axis=1)
        matrix *= 1.0 / math.sqrt(2.0)
    return matrix
