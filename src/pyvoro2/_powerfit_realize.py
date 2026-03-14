"""Realized-face matching for resolved pairwise separator constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._powerfit_constraints import PairBisectorConstraints
from .api import compute
from .domains import Box, OrthorhombicCell, PeriodicCell
from .face_properties import annotate_face_properties


@dataclass(frozen=True, slots=True)
class RealizedPairDiagnostics:
    """Diagnostics for matching candidate constraints to realized faces."""

    realized: np.ndarray
    unrealized: tuple[int, ...]
    realized_same_shift: np.ndarray
    realized_other_shift: np.ndarray
    realized_shifts: tuple[tuple[tuple[int, int, int], ...], ...]
    endpoint_i_empty: np.ndarray
    endpoint_j_empty: np.ndarray
    boundary_measure: np.ndarray | None
    cells: list[dict[str, Any]] | None



def match_realized_pairs(
    points: np.ndarray,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    radii: np.ndarray,
    constraints: PairBisectorConstraints,
    return_boundary_measure: bool = False,
    return_cells: bool = False,
) -> RealizedPairDiagnostics:
    """Determine which resolved pair constraints correspond to realized faces."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if pts.shape[0] != constraints.n_points:
        raise ValueError('points do not match the resolved constraint set')

    periodic = isinstance(domain, PeriodicCell) or (
        isinstance(domain, OrthorhombicCell) and any(domain.periodic)
    )

    cells = compute(
        pts,
        domain=domain,
        mode='power',
        radii=np.asarray(radii, dtype=float),
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=bool(periodic),
        include_empty=True,
    )

    if return_boundary_measure:
        annotate_face_properties(cells, domain)

    empty_by_id: dict[int, bool] = {}
    shifts_by_pair: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
    measure_by_pair_shift: dict[tuple[int, int, int, int, int], float] = {}

    for cell in cells:
        ci = int(cell['id'])
        verts = np.asarray(cell.get('vertices', []), dtype=float)
        faces = cell.get('faces', [])
        empty_by_id[ci] = bool(verts.size == 0 or len(faces) == 0)
        for face in faces:
            cj = int(face.get('adjacent_cell', -1))
            if cj < 0:
                continue
            sh = face.get('adjacent_shift', (0, 0, 0))
            shift = (int(sh[0]), int(sh[1]), int(sh[2]))
            shifts_by_pair.setdefault((ci, cj), set()).add(shift)
            if return_boundary_measure:
                measure = float(face.get('area', 0.0))
                measure_by_pair_shift[(ci, cj, shift[0], shift[1], shift[2])] = measure

    m = constraints.n_constraints
    realized = np.zeros(m, dtype=bool)
    realized_same_shift = np.zeros(m, dtype=bool)
    realized_other_shift = np.zeros(m, dtype=bool)
    endpoint_i_empty = np.zeros(m, dtype=bool)
    endpoint_j_empty = np.zeros(m, dtype=bool)
    realized_shifts_rows: list[tuple[tuple[int, int, int], ...]] = []
    boundary_measure = (
        np.full(m, np.nan, dtype=np.float64) if return_boundary_measure else None
    )
    unrealized: list[int] = []

    for k in range(m):
        i = int(constraints.i[k])
        j = int(constraints.j[k])
        target_shift = (
            int(constraints.shifts[k, 0]),
            int(constraints.shifts[k, 1]),
            int(constraints.shifts[k, 2]),
        )
        endpoint_i_empty[k] = bool(empty_by_id.get(i, False))
        endpoint_j_empty[k] = bool(empty_by_id.get(j, False))

        forward = shifts_by_pair.get((i, j), set())
        reverse = {(-sx, -sy, -sz) for (sx, sy, sz) in shifts_by_pair.get((j, i), set())}
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
            if same:
                key_f = (i, j, target_shift[0], target_shift[1], target_shift[2])
                key_r = (j, i, -target_shift[0], -target_shift[1], -target_shift[2])
                if key_f in measure_by_pair_shift:
                    boundary_measure[k] = measure_by_pair_shift[key_f]
                elif key_r in measure_by_pair_shift:
                    boundary_measure[k] = measure_by_pair_shift[key_r]
            else:
                chosen = realized_set[0]
                key_f = (i, j, chosen[0], chosen[1], chosen[2])
                key_r = (j, i, -chosen[0], -chosen[1], -chosen[2])
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
    )
