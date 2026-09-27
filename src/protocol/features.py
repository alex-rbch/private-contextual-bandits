"""Collision-free action blocks for already prepared context vectors.

The active data loader supplies ``z = (1, project(x))/sqrt(2)`` and the suite
uses this map without another intercept or scale. Standalone callers can still
request an intercept and feature scale explicitly. Distinct actions occupy
disjoint coordinate blocks; no hashing is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ActionBlockFeatureMap:
    """Exact collision-free ``e_a tensor z`` action features."""

    context_dimension: int
    num_actions: int
    include_intercept: bool = True
    feature_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.context_dimension <= 0 or self.num_actions <= 1:
            raise ValueError("invalid context dimension or action count")
        if not math.isfinite(self.feature_scale) or self.feature_scale <= 0.0:
            raise ValueError("feature_scale must be finite and positive")

    @property
    def block_dimension(self) -> int:
        return self.context_dimension + int(self.include_intercept)

    @property
    def dimension(self) -> int:
        return self.num_actions * self.block_dimension

    def transform(self, context: np.ndarray, action: int) -> np.ndarray:
        context = np.asarray(context, dtype=np.float64)
        if context.shape != (self.context_dimension,):
            raise ValueError("context has the wrong dimension")
        if not 0 <= action < self.num_actions:
            raise ValueError("action is out of range")
        block = np.concatenate(([1.0], context)) if self.include_intercept else context
        result = np.zeros(self.dimension, dtype=np.float64)
        start = action * self.block_dimension
        result[start : start + self.block_dimension] = self.feature_scale * block
        return result

    def transform_all(self, context: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.transform(context, action) for action in range(self.num_actions)]
        )
