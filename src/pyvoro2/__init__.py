"""pyvoro2 package.

This package provides Python bindings to the Voro++ cell-based Voronoi
tessellation library.

Public API:
    - Box, OrthorhombicCell, PeriodicCell
    - compute
    - locate
    - ghost_cells
"""

from __future__ import annotations

from .__about__ import __version__

from .domains import Box, OrthorhombicCell, PeriodicCell
from .api import compute, locate, ghost_cells
from .diagnostics import (
    TessellationDiagnostics,
    TessellationIssue,
    TessellationError,
    analyze_tessellation,
    validate_tessellation,
)

from .validation import (
    NormalizationDiagnostics,
    NormalizationIssue,
    NormalizationError,
    validate_normalized_topology,
)

from .duplicates import (
    DuplicatePair,
    DuplicateError,
    duplicate_check,
)
from .face_properties import annotate_face_properties
from .normalize import (
    NormalizedVertices,
    NormalizedTopology,
    normalize_vertices,
    normalize_edges_faces,
    normalize_topology,
)

from .inverse import (
    FitWeightsResult,
    radii_to_weights,
    weights_to_radii,
    fit_power_weights_from_plane_fractions,
    fit_power_weights_from_plane_positions,
)

__all__ = [
    'Box',
    'OrthorhombicCell',
    'PeriodicCell',
    'compute',
    'locate',
    'ghost_cells',
    'TessellationDiagnostics',
    'TessellationIssue',
    'TessellationError',
    'analyze_tessellation',
    'validate_tessellation',
    'NormalizationDiagnostics',
    'NormalizationIssue',
    'NormalizationError',
    'validate_normalized_topology',
    'DuplicatePair',
    'DuplicateError',
    'duplicate_check',
    'annotate_face_properties',
    'NormalizedVertices',
    'NormalizedTopology',
    'normalize_vertices',
    'normalize_edges_faces',
    'normalize_topology',
    'FitWeightsResult',
    'radii_to_weights',
    'weights_to_radii',
    'fit_power_weights_from_plane_fractions',
    'fit_power_weights_from_plane_positions',
    '__version__',
]
