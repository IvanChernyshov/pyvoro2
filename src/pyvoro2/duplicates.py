"""Duplicate / near-duplicate point detection.

Voro++ contains an internal "duplicate" safeguard that can terminate the
process (via `exit(1)`) if it detects two points closer than an absolute
threshold (~1e-5 in container distance units).

This module provides a fast *Python-side* pre-check to detect such cases before
calling into the C++ library.

The check is intentionally simple:
  - spatial hashing on an integer grid with cell size == threshold
  - compare each point only to points in its own grid cell and neighboring 26
    cells

Expected complexity is O(n) for typical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import warnings

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._util import is_periodic_domain


Domain = Box | OrthorhombicCell | PeriodicCell


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    i: int
    j: int
    distance: float


class DuplicateError(ValueError):
    """Raised when near-duplicate points are detected."""

    def __init__(
        self, message: str, pairs: tuple[DuplicatePair, ...], threshold: float
    ):
        super().__init__(message)
        self.pairs = pairs
        self.threshold = float(threshold)


def duplicate_check(
    points: Any,
    *,
    threshold: float = 1e-5,
    domain: Domain | None = None,
    wrap: bool = True,
    mode: Literal['raise', 'warn', 'return'] = 'raise',
    max_pairs: int = 10,
) -> tuple[DuplicatePair, ...]:
    """Detect point pairs closer than an absolute threshold.

    Args:
        points: Array-like of shape (n, 3).
        threshold: Absolute distance threshold. The default (1e-5) matches the
            effective Voro++ duplicate check (distance < 1e-5).
        domain: Optional domain. If provided and `wrap=True`, points are first
            remapped into the primary periodic domain for periodic domains,
            matching what Voro++ will do internally.
        wrap: Whether to remap points into the primary domain when `domain` has
            periodicity.
        mode: Behavior when duplicates are found:
            - 'raise' (default): raise :class:`DuplicateError`
            - 'warn': emit a RuntimeWarning and return the pairs
            - 'return': return the pairs without warnings
        max_pairs: Maximum number of pairs to include in the report.

    Returns:
        Tuple of DuplicatePair records (possibly empty).
    """

    if mode not in ('raise', 'warn', 'return'):
        raise ValueError('mode must be one of: \'raise\', \'warn\', \'return\'')

    thr = float(threshold)
    if not np.isfinite(thr) or thr <= 0:
        raise ValueError('threshold must be a positive finite number')
    max_pairs_i = int(max_pairs)
    if max_pairs_i <= 0:
        raise ValueError('max_pairs must be > 0')

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')
    n = int(pts.shape[0])
    if n <= 1:
        return tuple()

    if domain is not None and wrap and is_periodic_domain(domain):
        # Domain remap is authoritative for how Voro++ will interpret periodic
        # coordinates. (For PeriodicCell, this matches the internal remap used
        # when inserting points.)
        pts = np.asarray(domain.remap_cart(pts), dtype=np.float64)

    h = thr
    h2 = h * h
    # grid index for each point
    g = np.floor(pts / h).astype(np.int64)

    # Precompute neighbor offsets
    neigh = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]

    buckets: dict[tuple[int, int, int], list[int]] = {}
    found: list[DuplicatePair] = []

    for i in range(n):
        key = (int(g[i, 0]), int(g[i, 1]), int(g[i, 2]))
        x = pts[i]

        # Check points in this bucket and adjacent buckets
        for dx, dy, dz in neigh:
            nk = (key[0] + dx, key[1] + dy, key[2] + dz)
            cand = buckets.get(nk)
            if not cand:
                continue
            for j in cand:
                d = x - pts[j]
                dist2 = float(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                if dist2 < h2:
                    found.append(
                        DuplicatePair(
                            i=int(j), j=int(i), distance=float(np.sqrt(dist2))
                        )
                    )
                    if len(found) >= max_pairs_i:
                        break
            if len(found) >= max_pairs_i:
                break
        if len(found) >= max_pairs_i:
            break

        buckets.setdefault(key, []).append(i)

    pairs = tuple(found)
    if not pairs:
        return pairs

    msg = (
        f'Found {len(pairs)} point pair(s) closer than threshold={thr:g}. '
        'Such near-duplicates may cause Voro++ to terminate the process.'
    )

    if mode == 'raise':
        raise DuplicateError(msg, pairs, thr)
    if mode == 'warn':
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return pairs
