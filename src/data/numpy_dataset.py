"""Validated NumPy storage for public contextual-bandit datasets."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np


def _validated_arrays(
    contexts: np.ndarray,
    labels: np.ndarray,
    num_actions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(contexts)
    targets = np.asarray(labels)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("contexts must be a nonempty two-dimensional array")
    if matrix.dtype.kind not in "fiu":
        raise ValueError("contexts must contain real numeric values")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("contexts must be finite")
    if targets.ndim != 1 or targets.shape[0] != matrix.shape[0]:
        raise ValueError("labels must have one entry per context")
    if targets.dtype.kind not in "iu":
        raise ValueError("labels must be integer action indices")
    targets = np.asarray(targets, dtype=np.int64)
    if np.any(targets < 0):
        raise ValueError("labels must be nonnegative")
    if num_actions is not None:
        if num_actions <= 1 or np.any(targets >= num_actions):
            raise ValueError(f"labels must be in 0,...,{num_actions - 1}")
    return matrix, targets


def numpy_problem_bytes(contexts: np.ndarray, labels: np.ndarray) -> bytes:
    """Encode raw contexts and zero-based labels without preprocessing them."""

    matrix, targets = _validated_arrays(contexts, labels)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, contexts=matrix, labels=targets)
    return buffer.getvalue()


def load_numpy_problem(
    path: Path, num_actions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read the two-array dataset format used by the active experiment suite."""

    metadata_path = path.with_suffix(".source.json")
    if metadata_path.is_file() and "feature_hashing" in json.loads(metadata_path.read_text()):
        raise ValueError("feature-hashed datasets are not allowed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"contexts", "labels"}:
            raise ValueError("NumPy dataset must contain only contexts and labels")
        contexts = archive["contexts"]
        labels = archive["labels"]
    return _validated_arrays(contexts, labels, num_actions)
