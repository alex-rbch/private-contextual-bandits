"""Geometric batch boundaries.

Boundaries depend only on the public horizon and growth factor, never on the
data. Every original round falls in exactly one batch, which is what lets the
per-batch privacy releases compose in parallel rather than sequentially.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GeometricEpoch:
    """One batch, using one-based round numbering."""

    epoch: int
    start_round: int
    end_round: int
    size: int

    @property
    def sample_indices(self) -> tuple[int, ...]:
        """Return zero-based array indices for this one-based batch."""

        return tuple(range(self.start_round - 1, self.end_round))


def geometric_epochs(
    horizon: int,
    growth_factor: float = 2.0,
) -> tuple[GeometricEpoch, ...]:
    """Return batches between recursive geometric boundaries.

    The one-based boundaries start at ``s_0 = 1`` and satisfy
    ``s_m = ceil(q s_(m-1))``. A non-final batch therefore has actual length
    ``s_m - s_(m-1)``; a boundary value is never used as a batch length.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if growth_factor <= 1.0 or not math.isfinite(growth_factor):
        raise ValueError("growth_factor must be finite and greater than one")
    result: list[GeometricEpoch] = []
    boundary = 1
    epoch = 1
    while boundary <= horizon:
        next_boundary = int(math.ceil(growth_factor * boundary))
        if next_boundary <= boundary:
            raise AssertionError("geometric boundary failed to increase")
        end = min(next_boundary - 1, horizon)
        result.append(GeometricEpoch(epoch, boundary, end, end - boundary + 1))
        boundary = next_boundary
        epoch += 1
    return tuple(result)
