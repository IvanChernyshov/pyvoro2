"""Topology-level post-processing utilities.

Voro++ returns each Voronoi cell with its own *local* vertex list. In periodic
systems, many of those local vertices represent the same geometric vertex but in
different periodic images.

This module provides a correctness-first normalisation routine that builds a
global vertex pool in the 000 cell (primary periodic domain) and, for each cell,
maps local vertices to:
    - a global vertex index, and
    - an integer lattice shift (na, nb, nc) such that:

        v_local ~= v_global + na*a + nb*b + nc*c

All coordinates exposed by the public API are Cartesian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import warnings

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._util import domain_length_scale, is_periodic_domain


@dataclass(frozen=True)
class NormalizedVertices:
    """Result of :func:`normalize_vertices`.

    Attributes:
        global_vertices: Array of unique vertices in Cartesian coordinates,
            remapped into the primary cell for periodic domains.
        cells: A list of per-cell dictionaries. Each dictionary contains the
            original fields returned by :func:`pyvoro2.compute` plus:
                - vertex_global_id: list[int] of length n_local_vertices
                - vertex_shift: list[tuple[int,int,int]] aligned with vertices
    """

    global_vertices: np.ndarray
    cells: List[Dict[str, Any]]


@dataclass(frozen=True)
class NormalizedTopology:
    """Result of :func:`normalize_topology`.

    This extends :class:`NormalizedVertices` with globally deduplicated edges
    and faces. These are useful for building periodic Voronoi graphs.

    All coordinates exposed here are Cartesian. Periodicity is represented via
    integer lattice shifts (na, nb, nc) relative to the domain lattice vectors.

    Attributes:
        global_vertices: Array of unique vertices in Cartesian coordinates,
            remapped into the primary cell for periodic domains.
        global_edges: List of unique edges. Each edge dictionary contains:
            - vertices: (gid0, gid1) global vertex indices
            - vertex_shifts: ((0,0,0), (na,nb,nc)) such that the second endpoint
              is V[gid1] + na*a + nb*b + nc*c, while the first is V[gid0].
        global_faces: List of unique faces. Each face dictionary contains:
            - cells: (cid0, cid1) particle ids (first is the canonical anchor)
            - cell_shifts: ((0,0,0), (na,nb,nc)) shift of cid1 relative to cid0
            - vertices: list[int] global vertex ids in canonical cyclic order
            - vertex_shifts: list[(na,nb,nc)] shifts aligned with vertices,
              with the first vertex shift always (0,0,0).
        cells: Per-cell dictionaries (copies by default) including the
            vertex mapping fields plus:
            - edges: list[(u,v)] local vertex index pairs (u<v)
            - edge_global_id: list[int] aligned with edges
            - face_global_id: list[int] aligned with faces
    """

    global_vertices: np.ndarray
    global_edges: List[Dict[str, Any]]
    global_faces: List[Dict[str, Any]]
    cells: List[Dict[str, Any]]


def _quant_key(coord: np.ndarray, tol: float) -> Tuple[int, int, int]:
    q = np.rint(coord / tol).astype(np.int64)
    return int(q[0]), int(q[1]), int(q[2])


def _canonical_incident_key(
    incident: Sequence[Tuple[int, Tuple[int, int, int]]]
) -> Tuple[Any, ...]:
    """Canonicalize an incident cell-image set up to global lattice translation.

    `incident` is a collection of (cell_id, shift) pairs expressed relative to
    some reference cell (i.e., shifts are only defined up to adding a constant
    vector to *all* shifts).

    We canonicalize by considering all anchors in the set: for each anchor
    shift s_a, subtract s_a from all shifts and take the lexicographically
    minimal resulting representation.

    This makes the key invariant to adding a constant shift to all elements.
    """
    # Deduplicate exactly identical tuples first
    uniq = sorted(
        set((int(cid), (int(s[0]), int(s[1]), int(s[2]))) for cid, s in incident)
    )
    if not uniq:
        return tuple()

    best: Tuple[Any, ...] | None = None
    for _cid_a, s_a in uniq:
        sa = np.array(s_a, dtype=np.int64)
        rep = []
        for cid, s in uniq:
            ss = np.array(s, dtype=np.int64) - sa
            rep.append((cid, int(ss[0]), int(ss[1]), int(ss[2])))
        rep_sorted = tuple(sorted(rep))
        if best is None or rep_sorted < best:
            best = rep_sorted
    assert best is not None
    return best


def normalize_vertices(
    cells: List[Dict[str, Any]],
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    tol: float | None = None,
    require_face_shifts: bool = True,
    copy_cells: bool = True,
) -> NormalizedVertices:
    """Build a global vertex pool and per-cell vertex mappings.

    Args:
        cells: Output list from :func:`pyvoro2.compute`. Must include local
            vertices (`return_vertices=True`). For periodic domains, faces and
            face shifts are required unless `require_face_shifts=False`.
        domain: The domain used for the computation.
        tol: Quantization tolerance used for coordinate keys and residual
            verification. If None, defaults to 1e-8 * L where L is a domain
            length scale.
        require_face_shifts: If True and domain is PeriodicCell, require
            face-level `adjacent_shift` entries to build robust topology keys.
        copy_cells: If True, return shallow copies of the cell dicts with
            added mapping fields. If False, mutate the input dictionaries.

    Returns:
        NormalizedVertices with global_vertices and augmented cell dicts.

    Raises:
        ValueError: if required fields are missing.
    """
    L = domain_length_scale(domain)
    is_periodic = is_periodic_domain(domain)
    if tol is None:
        if not np.isfinite(L) or float(L) <= 0.0:
            raise ValueError('domain has an invalid length scale; pass tol explicitly')
        tol = 1e-8 * float(L)
        # If the user relies on defaults under a suspicious unit system,
        # highlight that pyvoro2 expects explicit rescaling.
        if float(L) < 1e-3 or float(L) > 1e9:
            warnings.warn(
                'normalize_vertices is using a default tolerance proportional to the '
                f'domain length scale (L≈{float(L):.3g}). For very small/large units '
                'this may be too strict/too loose. Consider rescaling your coordinates '
                'or passing an explicit tol=... .',
                RuntimeWarning,
                stacklevel=2,
            )
    if tol <= 0:
        raise ValueError('tol must be positive')

    if not isinstance(cells, list):
        raise ValueError('cells must be a list of dicts')

    # Prepare output cell containers
    out_cells: List[Dict[str, Any]]
    if copy_cells:
        out_cells = [dict(c) for c in cells]
    else:
        out_cells = cells

    # Global storage
    global_vertices: List[np.ndarray] = []
    key_to_gid: Dict[Tuple[Any, ...], int] = {}

    if not is_periodic:
        # For non-periodic boxes, coordinate-based deduplication is sufficient.
        for c in out_cells:
            verts = np.asarray(c.get('vertices', []), dtype=float)
            if verts.size == 0:
                verts = verts.reshape((0, 3))
            if verts.ndim != 2 or verts.shape[1] != 3:
                raise ValueError('cells must include vertices with shape (m,3)')

            gids: List[int] = []
            shifts: List[Tuple[int, int, int]] = []
            for v in verts:
                key = ('box',) + _quant_key(v, tol)
                gid = key_to_gid.get(key)
                if gid is None:
                    gid = len(global_vertices)
                    key_to_gid[key] = gid
                    global_vertices.append(v.astype(np.float64))
                gids.append(gid)
                shifts.append((0, 0, 0))

            c['vertex_global_id'] = gids
            c['vertex_shift'] = shifts

        return NormalizedVertices(
            global_vertices=(
                np.stack(global_vertices, axis=0)
                if global_vertices
                else np.zeros((0, 3))
            ),
            cells=out_cells,
        )

    # Periodic domain (PeriodicCell or partially periodic OrthorhombicCell)
    if is_periodic and require_face_shifts:
        # Validate presence of shifts
        for c in out_cells:
            faces = c.get('faces')
            if faces is None:
                raise ValueError('cells must include faces for periodic normalization')
            for f in faces:
                if 'adjacent_shift' not in f:
                    raise ValueError(
                        'cells must include face adjacent_shift '
                        '(compute with return_face_shifts=True)'
                    )

    # Build per-cell local->global mapping
    # Process deterministically by sorted cell id then vertex index.
    out_cells_sorted = sorted(out_cells, key=lambda cc: int(cc.get('id', 0)))

    for c in out_cells_sorted:
        verts = np.asarray(c.get('vertices', []), dtype=float)
        if verts.size == 0:
            verts = verts.reshape((0, 3))
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError('cells must include vertices with shape (m,3)')
        faces = c.get('faces')
        if faces is None:
            raise ValueError('cells must include faces for periodic normalization')

        # Build vertex -> incident faces list
        v_faces: List[List[Dict[str, Any]]] = [[] for _ in range(int(verts.shape[0]))]
        for f in faces:
            vids = f.get('vertices')
            if vids is None:
                continue
            for vid in vids:
                iv = int(vid)
                if 0 <= iv < len(v_faces):
                    v_faces[iv].append(f)

        gids: List[int] = []
        shifts: List[Tuple[int, int, int]] = []

        # Precompute remapped vertices and remap shifts
        # type: ignore[arg-type]
        remapped, rem_shifts = domain.remap_cart(verts, return_shifts=True)
        # NOTE: Due to floating-point round-off in the internal<->Cartesian
        # rotations, a point that was remapped into the primary cell can land
        # extremely close to a boundary. A subsequent remap may then report a
        # nonzero lattice shift (e.g. [-1, 1, 0]) even though the point is
        # already geometrically in the primary parallelepiped.
        #
        # To make global vertices strictly '000-normalized' under our own
        # remap_cart() definition (and keep tests stable), we apply remapping
        # once more and accumulate the extra shift into the local->global
        # shift mapping.
        for _ in range(2):
            # type: ignore[arg-type]
            remapped2, extra = domain.remap_cart(remapped, return_shifts=True)
            remapped = remapped2
            rem_shifts = rem_shifts + extra
            if not np.any(extra):
                break

        for k in range(int(verts.shape[0])):
            v0 = remapped[k]
            s0 = tuple(int(x) for x in rem_shifts[k])

            # Build incident set: include this cell (id, shift=(0,0,0)) plus
            # each adjacent cell image meeting at this vertex.
            incident: List[Tuple[int, Tuple[int, int, int]]] = []
            cid_here = int(c.get('id', 0))
            incident.append((cid_here, (0, 0, 0)))
            for f in v_faces[k]:
                adj = int(f.get('adjacent_cell', -999999))
                sh = f.get('adjacent_shift', (0, 0, 0))
                sh_t = (int(sh[0]), int(sh[1]), int(sh[2]))
                incident.append((adj, sh_t))

            topo_key = _canonical_incident_key(incident)
            coord_key = _quant_key(v0, tol)
            key: Tuple[Any, ...] = ('pbc',) + topo_key + ('@',) + coord_key

            gid = key_to_gid.get(key)
            if gid is None:
                gid = len(global_vertices)
                key_to_gid[key] = gid
                global_vertices.append(v0.astype(np.float64))
            else:
                # Sanity: ensure coordinate consistency (after remap) within tolerance.
                dv = np.linalg.norm(global_vertices[gid] - v0)
                if dv > 10 * tol:
                    raise ValueError(
                        'vertex key collision: same topology key '
                        'but significantly different coordinates; '
                        f'gid={gid}, dv={dv}'
                    )

            gids.append(gid)
            shifts.append(s0)

        c['vertex_global_id'] = gids
        c['vertex_shift'] = shifts

    gv = np.stack(global_vertices, axis=0) if global_vertices else np.zeros((0, 3))
    return NormalizedVertices(global_vertices=gv, cells=out_cells)


def _as_shift(s: Any) -> Tuple[int, int, int]:
    return int(s[0]), int(s[1]), int(s[2])


def _canon_edge(
    a: Tuple[int, Tuple[int, int, int]],
    b: Tuple[int, Tuple[int, int, int]],
) -> Tuple[
    Tuple[Any, ...], Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]
]:
    """Canonicalize an edge defined by two (gid, shift) endpoints.

    Returns:
        key: a hashable canonical key.
        rep: two endpoint records (gid, sx, sy, sz) where the first endpoint
            is guaranteed to have (0,0,0) shift.
    """
    gid0, s0 = a
    gid1, s1 = b
    s0a = np.array(s0, dtype=np.int64)
    s1a = np.array(s1, dtype=np.int64)

    # Two translation anchors: subtract s0 or subtract s1.
    candidates = []
    for ga, sa, gb, sb in ((gid0, s0a, gid1, s1a), (gid1, s1a, gid0, s0a)):
        d = sb - sa
        recs = (
            (int(ga), 0, 0, 0),
            (int(gb), int(d[0]), int(d[1]), int(d[2])),
        )
        candidates.append(tuple(sorted(recs)))
    best = min(candidates)

    # Normalize so the first record has zero shift.
    g0, x0, y0, z0 = best[0]
    g1, x1, y1, z1 = best[1]
    rep = (
        (int(g0), 0, 0, 0),
        (int(g1), int(x1 - x0), int(y1 - y0), int(z1 - z0)),
    )
    key = (
        'e',
        int(rep[0][0]),
        int(rep[1][0]),
        int(rep[1][1]),
        int(rep[1][2]),
        int(rep[1][3]),
    )
    return key, rep


def _rotate(seq: Sequence[Any], k: int) -> List[Any]:
    k = int(k) % len(seq)
    return list(seq[k:]) + list(seq[:k])


def _canon_polygon(
    verts: Sequence[Tuple[int, Tuple[int, int, int]]]
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Canonicalize a face vertex cycle up to translation, rotation, and reversal.

    Each input vertex is (gid, shift).
    Output is a tuple of (gid, sx, sy, sz) records with the first shift (0,0,0).
    """
    if not verts:
        return tuple()
    vv = [(int(g), (int(s[0]), int(s[1]), int(s[2]))) for g, s in verts]
    n = len(vv)

    # Consider both orientations.
    candidates: List[Tuple[Tuple[int, int, int, int], ...]] = []
    for base in (vv, list(reversed(vv))):
        for r in range(n):
            seq = _rotate(base, r)
            sa = np.array(seq[0][1], dtype=np.int64)
            rep = []
            for g, s in seq:
                ss = np.array(s, dtype=np.int64) - sa
                rep.append((int(g), int(ss[0]), int(ss[1]), int(ss[2])))
            candidates.append(tuple(rep))
    return min(candidates)


def _canon_face_pair(
    cid_here: int,
    adj: int,
    adj_shift: Tuple[int, int, int],
) -> Tuple[int, int, int, int, int, int, int, int]:
    sx, sy, sz = int(adj_shift[0]), int(adj_shift[1]), int(adj_shift[2])
    rep1 = (int(cid_here), 0, 0, 0, int(adj), sx, sy, sz)
    rep2 = (int(adj), 0, 0, 0, int(cid_here), -sx, -sy, -sz)
    return rep2 if rep2 < rep1 else rep1


def normalize_edges_faces(
    nv: NormalizedVertices,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    tol: float | None = None,
    copy_cells: bool = True,
) -> NormalizedTopology:
    """Build global edge and face pools based on an existing vertex normalization."""
    L = domain_length_scale(domain)
    if tol is None:
        if not np.isfinite(L) or float(L) <= 0.0:
            raise ValueError('domain has an invalid length scale; pass tol explicitly')
        tol = 1e-8 * float(L)
        if float(L) < 1e-3 or float(L) > 1e9:
            msg = (
                'normalize_edges_faces is using a default tolerance proportional '
                'to the '
                'domain length scale '
                f'(L≈{float(L):.3g}). '
                'For very small/large units this may be too strict/too loose. '
                'Consider rescaling your coordinates or passing an explicit tol=... .'
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    if tol <= 0:
        raise ValueError('tol must be positive')

    # Copy or reuse cells
    if copy_cells:
        cells = [dict(c) for c in nv.cells]
    else:
        cells = nv.cells

    global_edges: List[Dict[str, Any]] = []
    edge_key_to_id: Dict[Tuple[Any, ...], int] = {}

    global_faces: List[Dict[str, Any]] = []
    face_key_to_id: Dict[Tuple[Any, ...], int] = {}

    domain_periodic = is_periodic_domain(domain)

    # Deterministic processing order
    cells_sorted = sorted(cells, key=lambda cc: int(cc.get('id', 0)))

    # Build edges and faces
    for c in cells_sorted:
        # Validate required fields
        verts = c.get('vertices')
        if verts is None:
            raise ValueError('cells must include vertices')
        faces = c.get('faces')
        if faces is None:
            raise ValueError('cells must include faces')
        gids = c.get('vertex_global_id')
        vsh = c.get('vertex_shift')
        if gids is None or vsh is None:
            raise ValueError(
                'cells must include vertex_global_id and vertex_shift '
                '(call normalize_vertices first)'
            )

        # Extract local edges from face vertex cycles.
        edge_set: set[Tuple[int, int]] = set()
        for f in faces:
            vids = f.get('vertices')
            if vids is None:
                continue
            vv = [int(x) for x in vids]
            if len(vv) < 2:
                continue
            for u, v in zip(vv, vv[1:] + vv[:1]):
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edge_set.add((a, b))

        edges_local = sorted(edge_set)
        c['edges'] = [(int(u), int(v)) for u, v in edges_local]

        # Map edges to global ids
        edge_ids: List[int] = []
        for u, v in edges_local:
            ea = (int(gids[u]), _as_shift(vsh[u]))
            eb = (int(gids[v]), _as_shift(vsh[v]))
            ekey, erep = _canon_edge(ea, eb)
            eid = edge_key_to_id.get(ekey)
            if eid is None:
                eid = len(global_edges)
                edge_key_to_id[ekey] = eid
                global_edges.append(
                    {
                        'vertices': (int(erep[0][0]), int(erep[1][0])),
                        'vertex_shifts': (
                            (0, 0, 0),
                            (int(erep[1][1]), int(erep[1][2]), int(erep[1][3])),
                        ),
                    }
                )
            edge_ids.append(eid)
        c['edge_global_id'] = edge_ids

        # Faces -> global ids
        face_ids: List[int] = []
        cid_here = int(c.get('id', 0))
        for f in faces:
            adj = int(f.get('adjacent_cell', -999999))
            if domain_periodic and adj >= 0:
                if 'adjacent_shift' not in f:
                    raise ValueError(
                        'Periodic domain face missing adjacent_shift; '
                        'compute with return_face_shifts=True'
                    )
                adj_shift = _as_shift(f.get('adjacent_shift'))
            else:
                adj_shift = (0, 0, 0)

            pair = _canon_face_pair(cid_here, adj, adj_shift)
            vids = f.get('vertices')
            if vids is None:
                poly = tuple()
            else:
                desc = [(int(gids[int(v)]), _as_shift(vsh[int(v)])) for v in vids]
                poly = _canon_polygon(desc)

            if domain_periodic and adj >= 0:
                # In periodic tessellations, a given (cell_id, neighbor_id,
                # neighbor_shift) pair uniquely identifies a face (two convex
                # polyhedra share at most one face). Use only the canonical
                # cell-pair key for deduplication to avoid sensitivity to
                # boundary-vertex remapping noise in polygon keys.
                fkey: Tuple[Any, ...] = ('f',) + pair
            else:
                # For non-periodic domains *or* wall faces in partially periodic
                # orthorhombic domains, adjacent_cell may be a shared wall id.
                # Include the canonical polygon to distinguish distinct boundary faces.
                fkey: Tuple[Any, ...] = ('f',) + pair + ('|',) + poly
            fid = face_key_to_id.get(fkey)
            if fid is None:
                fid = len(global_faces)
                face_key_to_id[fkey] = fid
                # Convert canonical poly back to lists
                gv_ids = [int(t[0]) for t in poly]
                gv_sh = [(int(t[1]), int(t[2]), int(t[3])) for t in poly]
                global_faces.append(
                    {
                        'cells': (int(pair[0]), int(pair[4])),
                        'cell_shifts': (
                            (0, 0, 0),
                            (int(pair[5]), int(pair[6]), int(pair[7])),
                        ),
                        'vertices': gv_ids,
                        'vertex_shifts': gv_sh,
                    }
                )
            else:
                # Keep the stored polygon deterministic across both (cid->adj)
                # and (adj->cid) occurrences by choosing the lexicographically
                # smallest canonical polygon.
                if poly:
                    old = global_faces[fid]
                    old_poly = tuple(
                        (int(gid), int(sh[0]), int(sh[1]), int(sh[2]))
                        for gid, sh in zip(
                            old.get('vertices', []), old.get('vertex_shifts', [])
                        )
                    )
                    if (not old_poly) or poly < old_poly:
                        old['vertices'] = [int(t[0]) for t in poly]
                        old['vertex_shifts'] = [
                            (int(t[1]), int(t[2]), int(t[3])) for t in poly
                        ]
            face_ids.append(fid)
        c['face_global_id'] = face_ids

    return NormalizedTopology(
        global_vertices=nv.global_vertices,
        global_edges=global_edges,
        global_faces=global_faces,
        cells=cells,
    )


def normalize_topology(
    cells: List[Dict[str, Any]],
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    tol: float | None = None,
    require_face_shifts: bool = True,
    copy_cells: bool = True,
) -> NormalizedTopology:
    """Convenience wrapper: normalize vertices, then edges/faces."""
    nv = normalize_vertices(
        cells,
        domain=domain,
        tol=tol,
        require_face_shifts=require_face_shifts,
        copy_cells=copy_cells,
    )
    return normalize_edges_faces(nv, domain=domain, tol=tol, copy_cells=False)
