"""Realized-boundary matching for resolved pairwise separator constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import warnings

import numpy as np

from .constraints import PairBisectorConstraints
from .._domain_geometry import geometry3d
from ..api import compute as compute3d
from ..diagnostics import TessellationDiagnostics as TessellationDiagnostics3D
from ..domains import Box as Box3D, OrthorhombicCell, PeriodicCell
from ..edge_properties import annotate_edge_properties
from ..face_properties import annotate_face_properties
from ..planar._domain_geometry import geometry2d
from ..planar.api import compute as compute2d
from ..planar.diagnostics import (
    TessellationDiagnostics as TessellationDiagnostics2D,
    TessellationError as TessellationError2D,
    analyze_tessellation as analyze_tessellation2d,
)
from ..planar.domains import Box as Box2D, RectangularCell

ShiftTuple = tuple[int, ...]
MeasureKey = tuple[int, int, ShiftTuple]
Domain3D = Box3D | OrthorhombicCell | PeriodicCell
Domain2D = Box2D | RectangularCell
DomainAny = Domain2D | Domain3D
TessellationDiagnosticsAny = TessellationDiagnostics2D | TessellationDiagnostics3D


def _plain_value(value: object) -> object:
    return value.item() if hasattr(value, 'item') else value


def _boundary_value(values: np.ndarray | None, index: int) -> float | None:
    if values is None or np.isnan(values[index]):
        return None
    return float(values[index])


def _supported_realization_dim(constraints: PairBisectorConstraints) -> None:
    if constraints.dim not in (2, 3):
        raise ValueError(
            'match_realized_pairs currently supports only 2D and 3D resolved '
            'constraints'
        )


@dataclass(frozen=True, slots=True)
class RealizedPairDiagnostics:
    """Diagnostics for matching candidate constraints to realized boundaries."""

    realized: np.ndarray
    unrealized: tuple[int, ...]
    realized_same_shift: np.ndarray
    realized_other_shift: np.ndarray
    realized_shifts: tuple[tuple[ShiftTuple, ...], ...]
    endpoint_i_empty: np.ndarray
    endpoint_j_empty: np.ndarray
    boundary_measure: np.ndarray | None
    cells: list[dict[str, Any]] | None
    tessellation_diagnostics: TessellationDiagnosticsAny | None

    def to_records(
        self,
        constraints: PairBisectorConstraints,
        *,
        use_ids: bool = False,
    ) -> tuple[dict[str, object], ...]:
        """Return one plain-Python record per candidate pair."""

        if constraints.n_constraints != int(self.realized.shape[0]):
            raise ValueError(
                'constraints do not match the realized diagnostics length'
            )
        left, right = constraints.pair_labels(use_ids=use_ids)
        rows: list[dict[str, object]] = []
        left_is_int = np.issubdtype(np.asarray(left).dtype, np.integer)
        right_is_int = np.issubdtype(np.asarray(right).dtype, np.integer)
        for k in range(constraints.n_constraints):
            site_i = int(left[k]) if left_is_int else _plain_value(left[k])
            site_j = int(right[k]) if right_is_int else _plain_value(right[k])
            realized_shifts = tuple(
                tuple(int(v) for v in shift)
                for shift in self.realized_shifts[k]
            )
            rows.append(
                {
                    'constraint_index': int(k),
                    'site_i': site_i,
                    'site_j': site_j,
                    'shift': tuple(int(v) for v in constraints.shifts[k]),
                    'realized': bool(self.realized[k]),
                    'realized_same_shift': bool(self.realized_same_shift[k]),
                    'realized_other_shift': bool(self.realized_other_shift[k]),
                    'realized_shifts': realized_shifts,
                    'endpoint_i_empty': bool(self.endpoint_i_empty[k]),
                    'endpoint_j_empty': bool(self.endpoint_j_empty[k]),
                    'boundary_measure': _boundary_value(self.boundary_measure, k),
                }
            )
        return tuple(rows)

    def to_report(
        self,
        constraints: PairBisectorConstraints,
        *,
        use_ids: bool = False,
    ) -> dict[str, object]:
        """Return a JSON-friendly report for realized-boundary matching."""

        from .report import build_realized_report

        return build_realized_report(self, constraints, use_ids=use_ids)


def match_realized_pairs(
    points: np.ndarray,
    *,
    domain: DomainAny,
    radii: np.ndarray,
    constraints: PairBisectorConstraints,
    return_boundary_measure: bool = False,
    return_cells: bool = False,
    return_tessellation_diagnostics: bool = False,
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'diagnose',
) -> RealizedPairDiagnostics:
    """Determine which resolved pair constraints correspond to realized boundaries.

    The matching is purely geometric: each requested ordered pair ``(i, j, shift)``
    is checked against the set of realized cell boundaries in the power
    tessellation, including explicit periodic image shifts.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] <= 0:
        raise ValueError('points must have shape (n, d) with d >= 1')
    if pts.shape[0] != constraints.n_points:
        raise ValueError('points do not match the resolved constraint set')
    if constraints.dim != pts.shape[1]:
        raise ValueError('points do not match the resolved constraint dimension')
    _supported_realization_dim(constraints)

    dim = int(pts.shape[1])
    if dim == 2:
        cells, tessellation_diagnostics, periodic = _compute_planar_cells(
            pts,
            domain=domain,
            radii=radii,
            return_boundary_measure=return_boundary_measure,
            return_tessellation_diagnostics=return_tessellation_diagnostics,
            tessellation_check=tessellation_check,
        )
        boundary_key = 'edges'
        measure_field = 'length'
        shift_dim = 2
    elif dim == 3:
        cells, tessellation_diagnostics, periodic = _compute_3d_cells(
            pts,
            domain=domain,
            radii=radii,
            return_boundary_measure=return_boundary_measure,
            return_tessellation_diagnostics=return_tessellation_diagnostics,
            tessellation_check=tessellation_check,
        )
        boundary_key = 'faces'
        measure_field = 'area'
        shift_dim = 3
    else:
        raise ValueError(
            'match_realized_pairs currently supports only 2D and 3D points'
        )

    empty_by_id, shifts_by_pair, measure_by_pair_shift = _collect_boundary_maps(
        cells,
        boundary_key=boundary_key,
        shift_dim=shift_dim,
        return_boundary_measure=return_boundary_measure,
        measure_field=measure_field,
    )

    m = constraints.n_constraints
    realized = np.zeros(m, dtype=bool)
    realized_same_shift = np.zeros(m, dtype=bool)
    realized_other_shift = np.zeros(m, dtype=bool)
    endpoint_i_empty = np.zeros(m, dtype=bool)
    endpoint_j_empty = np.zeros(m, dtype=bool)
    realized_shifts_rows: list[tuple[ShiftTuple, ...]] = []
    boundary_measure = (
        np.full(m, np.nan, dtype=np.float64) if return_boundary_measure else None
    )
    unrealized: list[int] = []

    for k in range(m):
        i = int(constraints.i[k])
        j = int(constraints.j[k])
        target_shift = tuple(int(v) for v in constraints.shifts[k])
        endpoint_i_empty[k] = bool(empty_by_id.get(i, False))
        endpoint_j_empty[k] = bool(empty_by_id.get(j, False))

        forward = shifts_by_pair.get((i, j), set())
        reverse = {
            tuple(-int(v) for v in shift)
            for shift in shifts_by_pair.get((j, i), set())
        }
        realized_set = tuple(sorted(forward | reverse))
        realized_shifts_rows.append(realized_set)
        same = target_shift in realized_set
        any_realized = len(realized_set) > 0

        realized[k] = any_realized
        realized_same_shift[k] = same
        realized_other_shift[k] = any_realized and (not same)
        if not any_realized:
            unrealized.append(k)

        if boundary_measure is not None and any_realized:
            chosen = target_shift if same else realized_set[0]
            key_f = (i, j, chosen)
            key_r = (j, i, tuple(-int(v) for v in chosen))
            if key_f in measure_by_pair_shift:
                boundary_measure[k] = measure_by_pair_shift[key_f]
            elif key_r in measure_by_pair_shift:
                boundary_measure[k] = measure_by_pair_shift[key_r]

    return RealizedPairDiagnostics(
        realized=realized,
        unrealized=tuple(unrealized),
        realized_same_shift=realized_same_shift,
        realized_other_shift=realized_other_shift,
        realized_shifts=tuple(realized_shifts_rows),
        endpoint_i_empty=endpoint_i_empty,
        endpoint_j_empty=endpoint_j_empty,
        boundary_measure=boundary_measure,
        cells=cells if return_cells else None,
        tessellation_diagnostics=tessellation_diagnostics,
    )


def _compute_3d_cells(
    points: np.ndarray,
    *,
    domain: DomainAny,
    radii: np.ndarray,
    return_boundary_measure: bool,
    return_tessellation_diagnostics: bool,
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'],
) -> tuple[list[dict[str, Any]], TessellationDiagnostics3D | None, bool]:
    if not isinstance(domain, (Box3D, OrthorhombicCell, PeriodicCell)):
        raise ValueError(
            '3D points require a 3D domain: Box, OrthorhombicCell, or '
            'PeriodicCell'
        )

    periodic = geometry3d(domain).has_any_periodic_axis
    compute_result = compute3d(
        points,
        domain=domain,
        mode='power',
        radii=np.asarray(radii, dtype=float),
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=bool(periodic),
        include_empty=True,
        return_diagnostics=return_tessellation_diagnostics,
        tessellation_check=tessellation_check,
    )
    if return_tessellation_diagnostics:
        cells, tessellation_diagnostics = compute_result
    else:
        cells = compute_result
        tessellation_diagnostics = None

    if return_boundary_measure:
        annotate_face_properties(cells, domain)
    return cells, tessellation_diagnostics, bool(periodic)


def _compute_planar_cells(
    points: np.ndarray,
    *,
    domain: DomainAny,
    radii: np.ndarray,
    return_boundary_measure: bool,
    return_tessellation_diagnostics: bool,
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'],
) -> tuple[list[dict[str, Any]], TessellationDiagnostics2D | None, bool]:
    if not isinstance(domain, (Box2D, RectangularCell)):
        raise ValueError(
            '2D points require a planar domain: pyvoro2.planar.Box or '
            'RectangularCell'
        )

    periodic = geometry2d(domain).has_any_periodic_axis
    cells = compute2d(
        points,
        domain=domain,
        mode='power',
        radii=np.asarray(radii, dtype=float),
        return_vertices=True,
        return_edges=True,
        return_adjacency=False,
        return_edge_shifts=bool(periodic),
        include_empty=True,
    )

    if return_boundary_measure:
        annotate_edge_properties(cells, domain)

    do_diag = bool(return_tessellation_diagnostics) or tessellation_check != 'none'
    tessellation_diagnostics = None
    if do_diag:
        expected = list(range(int(points.shape[0])))
        tessellation_diagnostics = analyze_tessellation2d(
            cells,
            domain,
            expected_ids=expected,
            check_reciprocity=bool(periodic),
            check_line_mismatch=bool(periodic),
            mark_edges=bool(periodic),
        )
        if tessellation_check in ('warn', 'raise'):
            ok = bool(tessellation_diagnostics.ok_area) and (
                bool(tessellation_diagnostics.ok_reciprocity)
                if bool(periodic)
                else True
            )
            if not ok:
                msg = (
                    "tessellation_check failed (mode='power'): "
                    f'area_ratio={tessellation_diagnostics.area_ratio:g}, '
                    f'orphan_edges={tessellation_diagnostics.n_edges_orphan}, '
                    'mismatched_edges='
                    f'{tessellation_diagnostics.n_edges_mismatched}'
                )
                if tessellation_check == 'raise':
                    raise TessellationError2D(msg, tessellation_diagnostics)
                warnings.warn(msg, stacklevel=2)

    if not return_tessellation_diagnostics:
        tessellation_diagnostics = None
    return cells, tessellation_diagnostics, bool(periodic)


def _collect_boundary_maps(
    cells: list[dict[str, Any]],
    *,
    boundary_key: Literal['edges', 'faces'],
    shift_dim: int,
    return_boundary_measure: bool,
    measure_field: str,
) -> tuple[
    dict[int, bool],
    dict[tuple[int, int], set[ShiftTuple]],
    dict[MeasureKey, float],
]:
    empty_by_id: dict[int, bool] = {}
    shifts_by_pair: dict[tuple[int, int], set[ShiftTuple]] = {}
    measure_by_pair_shift: dict[MeasureKey, float] = {}

    zero_shift = tuple(0 for _ in range(shift_dim))
    for cell in cells:
        ci = int(cell['id'])
        verts = np.asarray(cell.get('vertices', []), dtype=float)
        boundaries = cell.get(boundary_key, [])
        empty_by_id[ci] = bool(verts.size == 0 or len(boundaries) == 0)
        for boundary in boundaries:
            cj = int(boundary.get('adjacent_cell', -1))
            if cj < 0:
                continue
            shift = tuple(int(v) for v in boundary.get('adjacent_shift', zero_shift))
            shifts_by_pair.setdefault((ci, cj), set()).add(shift)
            if return_boundary_measure:
                measure_by_pair_shift[(ci, cj, shift)] = float(
                    boundary.get(measure_field, 0.0)
                )

    return empty_by_id, shifts_by_pair, measure_by_pair_shift
