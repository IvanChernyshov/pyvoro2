"""Internal shared helpers.

This module exists to avoid duplicating small pieces of domain logic across
`api`, `normalize`, and `viz3d`.

The helpers here are intentionally lightweight and have **no** dependency on
the compiled extension.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell


Domain: TypeAlias = Box | OrthorhombicCell | PeriodicCell


def is_periodic_domain(domain: Domain) -> bool:
    """Return True if *any* periodic boundary condition is active."""

    if isinstance(domain, PeriodicCell):
        return True
    if isinstance(domain, OrthorhombicCell):
        return any(domain.periodic)
    return False


def domain_length_scale(domain: Domain) -> float:
    """Return a characteristic length scale of the domain.

    The value is used for heuristic tolerances and visualization defaults.
    It is **not** guaranteed to be a rigorous bound on any geometric quantity.
    """

    if isinstance(domain, (Box, OrthorhombicCell)):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
        return float(max(xmax - xmin, ymax - ymin, zmax - zmin))

    vec = np.asarray(domain.vectors, dtype=float)
    # vectors: (3,3) where each row is a lattice vector
    return float(np.max(np.linalg.norm(vec, axis=1)))


def domain_origin(domain: Domain) -> np.ndarray:
    """Return the domain origin in Cartesian coordinates."""

    if isinstance(domain, (Box, OrthorhombicCell)):
        (xmin, _), (ymin, _), (zmin, _) = domain.bounds
        return np.array([xmin, ymin, zmin], dtype=float)
    return np.asarray(domain.origin, dtype=float)


def domain_lattice_vectors(
    domain: OrthorhombicCell | PeriodicCell,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, b, c) lattice translation vectors for the domain."""

    if isinstance(domain, OrthorhombicCell):
        return domain.lattice_vectors
    a, b, c = (np.asarray(v, dtype=float) for v in domain.vectors)
    return a, b, c
