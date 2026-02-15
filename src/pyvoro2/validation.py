"""Strict validation utilities.

This module provides *post-hoc* validators that can be used to sanity check:

1) tessellation outputs (via :func:`pyvoro2.validate_tessellation`, implemented
   as a thin wrapper around :func:`pyvoro2.analyze_tessellation`), and
2) topology/normalization outputs (via :func:`validate_normalized_topology`).

The core goal is to turn subtle periodic bookkeeping mistakes into explicit,
actionable errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._util import is_periodic_domain
from .normalize import NormalizedVertices, NormalizedTopology


Domain = Box | OrthorhombicCell | PeriodicCell


@dataclass(frozen=True, slots=True)
class NormalizationIssue:
    code: str
    severity: Literal['info', 'warning', 'error']
    message: str
    examples: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class NormalizationDiagnostics:
    n_cells: int
    n_global_vertices: int
    n_global_edges: int | None
    n_global_faces: int | None
    is_periodic_domain: bool
    fully_periodic_domain: bool
    has_wall_faces: bool

    n_vertex_face_shift_mismatch: int
    n_face_vertex_set_mismatch: int
    n_vertices_low_incidence: int
    n_edges_low_incidence: int
    n_cells_bad_euler: int

    issues: tuple[NormalizationIssue, ...]

    ok_vertex_face_shift: bool
    ok_face_vertex_sets: bool
    ok_incidence: bool
    ok_euler: bool
    ok: bool


class NormalizationError(ValueError):
    """Raised when strict normalization validation fails."""

    def __init__(self, message: str, diagnostics: NormalizationDiagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _as_shift(s: Any) -> tuple[int, int, int]:
    return int(s[0]), int(s[1]), int(s[2])


def _fully_periodic(domain: Domain) -> bool:
    if isinstance(domain, PeriodicCell):
        return True
    if isinstance(domain, OrthorhombicCell):
        return bool(all(domain.periodic))
    return False


def _iter_face_vertex_indices(face: dict[str, Any]) -> list[int]:
    idx = face.get('vertices')
    if idx is None:
        return []
    # Keep as Python ints for safe dict/set use.
    return [int(x) for x in idx]


def validate_normalized_topology(
    normalized: NormalizedVertices | NormalizedTopology,
    domain: Domain,
    *,
    level: Literal['basic', 'strict'] = 'basic',
    check_vertex_face_shift: bool = True,
    check_face_vertex_sets: bool = True,
    check_incidence: bool = True,
    check_euler: bool = True,
    max_examples: int = 10,
) -> NormalizationDiagnostics:
    """Validate periodic shift/topology consistency after normalization.

    The most important invariant (and the one most likely to detect subtle
    periodic bookkeeping bugs) is:

        For every periodic face i -> j with adjacent_shift = s,
        every shared vertex gid must satisfy:

            vertex_shift_i(gid) == vertex_shift_j(gid) + s

    where vertex_shift_* are the per-cell lattice-image shifts returned by
    :func:`pyvoro2.normalize_vertices`.

    Args:
        normalized: Output of :func:`pyvoro2.normalize_vertices` or
            :func:`pyvoro2.normalize_topology`.
        domain: Domain used to compute the tessellation.
        level: 'basic' returns diagnostics; 'strict' raises
            :class:`NormalizationError` on any error-level issue.
        check_vertex_face_shift: Check the key vertex/face shift invariant.
        check_face_vertex_sets: Check that reciprocal faces reference the same
            set of global vertex IDs.
        check_incidence: In fully periodic domains, check minimal incidence
            counts for vertices (>=4) and edges (>=3). Only runs when
            `normalized` includes edges (i.e., is a NormalizedTopology).
        check_euler: Check Euler characteristic per cell (V - E + F == 2)
            as a warning-level sanity check.
        max_examples: Max number of example tuples to attach per issue.

    Returns:
        NormalizationDiagnostics
    """

    if level not in ('basic', 'strict'):
        raise ValueError('level must be \'basic\' or \'strict\'')

    cells = list(normalized.cells)
    n_cells = len(cells)
    n_global_vertices = int(getattr(normalized, 'global_vertices').shape[0])

    n_global_edges: int | None = None
    n_global_faces: int | None = None
    if isinstance(normalized, NormalizedTopology):
        n_global_edges = len(normalized.global_edges)
        n_global_faces = len(normalized.global_faces)

    periodic = bool(is_periodic_domain(domain))
    fully_periodic = bool(_fully_periodic(domain))

    # Detect wall faces (adjacent_cell < 0). This matters for incidence checks.
    has_wall_faces = False
    for c in cells:
        faces = c.get('faces') or []
        for f in faces:
            if int(f.get('adjacent_cell', -1)) < 0:
                has_wall_faces = True
                break
        if has_wall_faces:
            break

    issues: list[NormalizationIssue] = []

    # ------------------------------------------------------------------
    # Build id->cell mapping and per-cell gid->shift mapping
    # ------------------------------------------------------------------
    cell_by_id: dict[int, dict[str, Any]] = {}
    gid_shift_by_cell: dict[int, dict[int, tuple[int, int, int]]] = {}

    for c in cells:
        cid = int(c.get('id', -1))
        if cid < 0:
            continue
        cell_by_id[cid] = c

        gids = c.get('vertex_global_id')
        vsh = c.get('vertex_shift')
        if gids is None or vsh is None:
            continue
        m: dict[int, tuple[int, int, int]] = {}
        dup_examples: list[
            tuple[int, int, tuple[int, int, int], tuple[int, int, int]]
        ] = []
        for k, gid in enumerate(gids):
            g = int(gid)
            s = _as_shift(vsh[k])
            if g in m and m[g] != s:
                if len(dup_examples) < max_examples:
                    dup_examples.append((cid, g, m[g], s))
            else:
                m[g] = s
        if dup_examples:
            issues.append(
                NormalizationIssue(
                    code='DUPLICATE_GID_DIFFERENT_SHIFT',
                    severity='error',
                    message=(
                        'A cell contains the same global vertex id with different '
                        'vertex_shift values. This indicates a broken normalization.'
                    ),
                    examples=tuple(dup_examples),
                )
            )
        gid_shift_by_cell[cid] = m

    # ------------------------------------------------------------------
    # Check 1: vertex_shift <-> face adjacent_shift consistency
    # ------------------------------------------------------------------
    n_vfs_mismatch = 0
    if periodic and check_vertex_face_shift:
        examples: list[
            tuple[
                int,
                int,
                tuple[int, int, int],
                int,
                tuple[int, int, int],
                tuple[int, int, int],
            ]
        ] = []
        missing_neighbor_cells: list[tuple[int, int, tuple[int, int, int]]] = []
        missing_shared_vertex: list[tuple[int, int, tuple[int, int, int], int]] = []

        for c in cells:
            cid = int(c.get('id', -1))
            if cid < 0 or bool(c.get('empty', False)):
                continue
            faces = c.get('faces') or []
            gids = c.get('vertex_global_id')
            vsh = c.get('vertex_shift')
            if gids is None or vsh is None:
                continue

            gids_list = [int(x) for x in gids]
            vsh_list = [_as_shift(x) for x in vsh]

            for f in faces:
                j = int(f.get('adjacent_cell', -1))
                if j < 0:
                    continue
                if 'adjacent_shift' not in f:
                    issues.append(
                        NormalizationIssue(
                            code='FACE_MISSING_ADJACENT_SHIFT',
                            severity='error',
                            message=(
                                'A periodic neighbor face is missing adjacent_shift. '
                                'Ensure compute(..., return_face_shifts=True) was used.'
                            ),
                            examples=((cid, j),),
                        )
                    )
                    continue

                s = _as_shift(f.get('adjacent_shift', (0, 0, 0)))
                cj = cell_by_id.get(j)
                if cj is None:
                    if len(missing_neighbor_cells) < max_examples:
                        missing_neighbor_cells.append((cid, j, s))
                    continue
                map_j = gid_shift_by_cell.get(j)
                if map_j is None:
                    continue

                idx = _iter_face_vertex_indices(f)
                for vk in idx:
                    if vk < 0 or vk >= len(gids_list):
                        continue
                    gid = gids_list[vk]
                    ti = vsh_list[vk]
                    tj = map_j.get(gid)
                    if tj is None:
                        if len(missing_shared_vertex) < max_examples:
                            missing_shared_vertex.append((cid, j, s, gid))
                        continue
                    exp = (tj[0] + s[0], tj[1] + s[1], tj[2] + s[2])
                    if ti != exp:
                        n_vfs_mismatch += 1
                        if len(examples) < max_examples:
                            examples.append((cid, j, s, gid, ti, tj))

        if missing_neighbor_cells:
            issues.append(
                NormalizationIssue(
                    code='NEIGHBOR_CELL_MISSING',
                    severity='error',
                    message=(
                        'A face references a neighbor cell id that is not present '
                        'in the normalized output.'
                    ),
                    examples=tuple(missing_neighbor_cells),
                )
            )
        if missing_shared_vertex:
            issues.append(
                NormalizationIssue(
                    code='SHARED_VERTEX_MISSING_IN_NEIGHBOR',
                    severity='error',
                    message=(
                        'A periodic neighbor face references a global vertex id '
                        'that is not present in the neighbor cell. '
                        'This suggests inconsistent vertex normalization.'
                    ),
                    examples=tuple(missing_shared_vertex),
                )
            )
        if examples:
            issues.append(
                NormalizationIssue(
                    code='VERTEX_FACE_SHIFT_MISMATCH',
                    severity='error',
                    message=(
                        'vertex_shift and adjacent_shift are inconsistent across '
                        'a periodic face: expected vertex_shift_i == '
                        'vertex_shift_j + adjacent_shift.'
                    ),
                    examples=tuple(examples),
                )
            )

    # ------------------------------------------------------------------
    # Check 2: reciprocal faces have matching vertex-id sets
    # ------------------------------------------------------------------
    n_face_set_mismatch = 0
    if periodic and check_face_vertex_sets:
        face_map: dict[tuple[int, int, tuple[int, int, int]], dict[str, Any]] = {}
        for c in cells:
            i = int(c.get('id', -1))
            if i < 0 or bool(c.get('empty', False)):
                continue
            faces = c.get('faces') or []
            for f in faces:
                j = int(f.get('adjacent_cell', -1))
                if j < 0:
                    continue
                if 'adjacent_shift' not in f:
                    continue
                s = _as_shift(f.get('adjacent_shift', (0, 0, 0)))
                face_map[(i, j, s)] = f

        checked: set[tuple[int, int, tuple[int, int, int]]] = set()
        examples: list[
            tuple[int, int, tuple[int, int, int], tuple[int, ...], tuple[int, ...]]
        ] = []

        def _face_gid_set(cid: int, face: dict[str, Any]) -> set[int]:
            c = cell_by_id.get(cid)
            if c is None:
                return set()
            gids = c.get('vertex_global_id')
            if gids is None:
                return set()
            gids_list = [int(x) for x in gids]
            out: set[int] = set()
            for vk in _iter_face_vertex_indices(face):
                if 0 <= vk < len(gids_list):
                    out.add(gids_list[vk])
            return out

        for (i, j, s), f in list(face_map.items()):
            if (i, j, s) in checked:
                continue
            r = (j, i, (-s[0], -s[1], -s[2]))
            checked.add((i, j, s))
            checked.add(r)
            fr = face_map.get(r)
            if fr is None:
                continue
            si = _face_gid_set(i, f)
            sj = _face_gid_set(j, fr)
            if si != sj:
                n_face_set_mismatch += 1
                if len(examples) < max_examples:
                    examples.append((i, j, s, tuple(sorted(si)), tuple(sorted(sj))))

        if examples:
            issues.append(
                NormalizationIssue(
                    code='RECIPROCAL_FACE_VERTEX_SET_MISMATCH',
                    severity='error',
                    message=(
                        'Reciprocal periodic faces do not reference the same set of '
                        'global vertex ids.'
                    ),
                    examples=tuple(examples),
                )
            )

    # ------------------------------------------------------------------
    # Check 3: incidence sanity (fully periodic domains)
    # ------------------------------------------------------------------
    n_vertices_low = 0
    n_edges_low = 0
    if (
        check_incidence
        and fully_periodic
        and periodic
        and (not has_wall_faces)
        and isinstance(normalized, NormalizedTopology)
    ):
        vertex_to_cells: dict[int, set[int]] = {}
        edge_to_cells: dict[int, set[int]] = {}

        for c in cells:
            cid = int(c.get('id', -1))
            if cid < 0 or bool(c.get('empty', False)):
                continue
            gids = c.get('vertex_global_id')
            if gids is not None:
                for gid in set(int(x) for x in gids):
                    vertex_to_cells.setdefault(gid, set()).add(cid)
            eids = c.get('edge_global_id')
            if eids is not None:
                for eid in set(int(x) for x in eids):
                    edge_to_cells.setdefault(eid, set()).add(cid)

        v_warn, v_err = 4, 3
        e_warn, e_err = 3, 2

        bad_v_warn: list[tuple[int, int]] = []
        bad_v_err: list[tuple[int, int]] = []
        for gid, ss in vertex_to_cells.items():
            deg = len(ss)
            if deg < v_err:
                n_vertices_low += 1
                if len(bad_v_err) < max_examples:
                    bad_v_err.append((gid, deg))
            elif deg < v_warn:
                n_vertices_low += 1
                if len(bad_v_warn) < max_examples:
                    bad_v_warn.append((gid, deg))

        bad_e_warn: list[tuple[int, int]] = []
        bad_e_err: list[tuple[int, int]] = []
        for eid, ss in edge_to_cells.items():
            deg = len(ss)
            if deg < e_err:
                n_edges_low += 1
                if len(bad_e_err) < max_examples:
                    bad_e_err.append((eid, deg))
            elif deg < e_warn:
                n_edges_low += 1
                if len(bad_e_warn) < max_examples:
                    bad_e_warn.append((eid, deg))

        if bad_v_err:
            issues.append(
                NormalizationIssue(
                    code='VERTEX_INCIDENCE_TOO_LOW',
                    severity='error',
                    message=(
                        'In a fully periodic tessellation, some global vertices are '
                        'incident to fewer than 3 cells.'
                    ),
                    examples=tuple(bad_v_err),
                )
            )
        if bad_v_warn:
            issues.append(
                NormalizationIssue(
                    code='VERTEX_INCIDENCE_LOW',
                    severity='warning',
                    message=(
                        'Some global vertices are incident to fewer than 4 cells '
                        '(may indicate degeneracy or issues).'
                    ),
                    examples=tuple(bad_v_warn),
                )
            )
        if bad_e_err:
            issues.append(
                NormalizationIssue(
                    code='EDGE_INCIDENCE_TOO_LOW',
                    severity='error',
                    message=(
                        'In a fully periodic tessellation, some global edges are '
                        'incident to fewer than 2 cells.'
                    ),
                    examples=tuple(bad_e_err),
                )
            )
        if bad_e_warn:
            issues.append(
                NormalizationIssue(
                    code='EDGE_INCIDENCE_LOW',
                    severity='warning',
                    message=(
                        'Some global edges are incident to fewer than 3 cells '
                        '(may indicate degeneracy or issues).'
                    ),
                    examples=tuple(bad_e_warn),
                )
            )

    # ------------------------------------------------------------------
    # Check 4: Euler characteristic per cell (warning-level)
    # ------------------------------------------------------------------
    n_bad_euler = 0
    if check_euler:
        examples: list[tuple[int, int, int, int, int]] = []
        for c in cells:
            cid = int(c.get('id', -1))
            if cid < 0 or bool(c.get('empty', False)):
                continue
            faces = c.get('faces') or []
            face_cycles: list[list[int]] = []
            for f in faces:
                idx = _iter_face_vertex_indices(f)
                if len(idx) >= 3:
                    face_cycles.append(idx)
            if not face_cycles:
                continue

            vset: set[int] = set()
            eset: set[tuple[int, int]] = set()
            for cyc in face_cycles:
                for v in cyc:
                    vset.add(int(v))
                m = len(cyc)
                for k in range(m):
                    a = int(cyc[k])
                    b = int(cyc[(k + 1) % m])
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    eset.add((a, b))
            V = len(vset)
            E = len(eset)
            F = len(face_cycles)
            chi = V - E + F
            if chi != 2:
                n_bad_euler += 1
                if len(examples) < max_examples:
                    examples.append((cid, chi, V, E, F))

        if examples:
            issues.append(
                NormalizationIssue(
                    code='EULER_CHARACTERISTIC_MISMATCH',
                    severity='warning',
                    message=(
                        'Some cells do not satisfy Euler characteristic V - E + F == 2 '
                        '(may indicate degeneracy).'
                    ),
                    examples=tuple(examples),
                )
            )

    ok_vertex_face = not any(
        i.severity == 'error'
        and i.code
        in (
            'DUPLICATE_GID_DIFFERENT_SHIFT',
            'FACE_MISSING_ADJACENT_SHIFT',
            'NEIGHBOR_CELL_MISSING',
            'SHARED_VERTEX_MISSING_IN_NEIGHBOR',
            'VERTEX_FACE_SHIFT_MISMATCH',
        )
        for i in issues
    )
    ok_face_sets = n_face_set_mismatch == 0
    ok_inc = not any(
        i.severity == 'error'
        and i.code
        in (
            'VERTEX_INCIDENCE_TOO_LOW',
            'EDGE_INCIDENCE_TOO_LOW',
        )
        for i in issues
    )
    ok_euler = n_bad_euler == 0
    ok = not any(i.severity == 'error' for i in issues)

    diag = NormalizationDiagnostics(
        n_cells=int(n_cells),
        n_global_vertices=int(n_global_vertices),
        n_global_edges=(int(n_global_edges) if n_global_edges is not None else None),
        n_global_faces=(int(n_global_faces) if n_global_faces is not None else None),
        is_periodic_domain=bool(periodic),
        fully_periodic_domain=bool(fully_periodic),
        has_wall_faces=bool(has_wall_faces),
        n_vertex_face_shift_mismatch=int(n_vfs_mismatch),
        n_face_vertex_set_mismatch=int(n_face_set_mismatch),
        n_vertices_low_incidence=int(n_vertices_low),
        n_edges_low_incidence=int(n_edges_low),
        n_cells_bad_euler=int(n_bad_euler),
        issues=tuple(issues),
        ok_vertex_face_shift=bool(ok_vertex_face),
        ok_face_vertex_sets=bool(ok_face_sets),
        ok_incidence=bool(ok_inc),
        ok_euler=bool(ok_euler),
        ok=bool(ok),
    )

    if level == 'strict' and not diag.ok:
        err = next((x for x in diag.issues if x.severity == 'error'), None)
        msg = err.message if err is not None else 'Normalization validation failed'
        raise NormalizationError(msg, diag)
    return diag
