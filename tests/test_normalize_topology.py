import numpy as np

from pyvoro2 import OrthorhombicCell, PeriodicCell, compute, normalize_topology


def _edge_vector_from_global(
    global_vertices: np.ndarray, vectors: np.ndarray, edge: dict
) -> np.ndarray:
    """Return the edge direction vector in Cartesian coordinates.

    global edge format:
        {
            'vertices': (gid0, gid1),
            'vertex_shifts': ((0,0,0), (na,nb,nc)),
        }
    """
    a, b, c = vectors
    gid0, gid1 = edge['vertices']
    (_s0, s1) = edge['vertex_shifts']
    s1 = np.asarray(s1, dtype=int)
    v0 = global_vertices[int(gid0)]
    v1 = global_vertices[int(gid1)] + s1[0] * a + s1[1] * b + s1[2] * c
    return v1 - v0


def _min_signed_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between vectors up to sign."""
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def test_normalize_topology_periodic_cubic_edges_and_faces():
    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float)

    cells = compute(
        pts,
        domain=cell,
        mode='standard',
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=True,
        face_shift_search=1,
    )

    nt = normalize_topology(cells, domain=cell)
    gv = np.asarray(nt.global_vertices, dtype=float)
    vec = np.asarray(cell.vectors, dtype=float)

    assert len(nt.global_faces) > 0
    assert len(nt.global_edges) > 0

    c0 = next(c for c in nt.cells if int(c['id']) == 0)
    c1 = next(c for c in nt.cells if int(c['id']) == 1)

    # Faces: the boundary face between 0 and 1 across x-wrap should deduplicate.
    idx0 = [
        i
        for i, f in enumerate(c0['faces'])
        if int(f['adjacent_cell']) == 1
        and tuple(int(x) for x in f['adjacent_shift']) == (-1, 0, 0)
    ]
    idx1 = [
        i
        for i, f in enumerate(c1['faces'])
        if int(f['adjacent_cell']) == 0
        and tuple(int(x) for x in f['adjacent_shift']) == (1, 0, 0)
    ]
    assert len(idx0) == 1
    assert len(idx1) == 1
    gid0 = int(c0['face_global_id'][idx0[0]])
    gid1 = int(c1['face_global_id'][idx1[0]])
    assert gid0 == gid1

    # Edges: each local edge vector should match its global canonical representative
    # up to sign.
    tol = 1e-6
    for c in (c0, c1):
        verts = np.asarray(c['vertices'], dtype=float)
        for (u, v), eid in zip(c['edges'], c['edge_global_id']):
            dv_local = verts[int(v)] - verts[int(u)]
            dv_global = _edge_vector_from_global(gv, vec, nt.global_edges[int(eid)])
            assert _min_signed_diff(dv_local, dv_global) < tol


def test_normalize_topology_copy_cells_false_mutates_input():
    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float)

    cells = compute(
        pts,
        domain=cell,
        mode='standard',
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=True,
        face_shift_search=1,
    )

    c0 = next(c for c in cells if int(c['id']) == 0)
    assert 'vertex_global_id' not in c0
    assert 'face_global_id' not in c0

    nt = normalize_topology(cells, domain=cell, copy_cells=False)

    # In-place mode must mutate the original dictionaries.
    assert nt.cells is cells
    assert 'vertex_global_id' in c0
    assert 'vertex_shift' in c0
    assert 'edge_global_id' in c0
    assert 'face_global_id' in c0


def test_normalize_topology_periodic_sheared_edges_and_faces():
    # A triclinic but simple shear cell
    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.2, 1.0, 0.0), (0.1, 0.1, 1.0)))
    pts = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float)

    cells = compute(
        pts,
        domain=cell,
        mode='standard',
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=True,
        face_shift_search=1,
    )

    nt = normalize_topology(cells, domain=cell)
    gv = np.asarray(nt.global_vertices, dtype=float)
    vec = np.asarray(cell.vectors, dtype=float)

    # Global vertices must be strictly remapped into the primary cell.
    gv2, shifts = cell.remap_cart(gv, return_shifts=True)
    assert np.all(shifts == 0)
    assert np.allclose(gv2, gv, atol=1e-10)

    c0 = next(c for c in nt.cells if int(c['id']) == 0)
    c1 = next(c for c in nt.cells if int(c['id']) == 1)

    idx0 = [
        i
        for i, f in enumerate(c0['faces'])
        if int(f['adjacent_cell']) == 1
        and tuple(int(x) for x in f['adjacent_shift']) == (-1, 0, 0)
    ]
    idx1 = [
        i
        for i, f in enumerate(c1['faces'])
        if int(f['adjacent_cell']) == 0
        and tuple(int(x) for x in f['adjacent_shift']) == (1, 0, 0)
    ]
    assert len(idx0) == 1
    assert len(idx1) == 1
    assert int(c0['face_global_id'][idx0[0]]) == int(c1['face_global_id'][idx1[0]])

    tol = 1e-6
    for c in (c0, c1):
        verts = np.asarray(c['vertices'], dtype=float)
        for (u, v), eid in zip(c['edges'], c['edge_global_id']):
            dv_local = verts[int(v)] - verts[int(u)]
            dv_global = _edge_vector_from_global(gv, vec, nt.global_edges[int(eid)])
            assert _min_signed_diff(dv_local, dv_global) < tol


def test_normalize_topology_orthorhombic_partial_periodic_edges_and_faces():
    dom = OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        periodic=(True, True, False),
    )
    pts = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float)

    cells = compute(
        pts,
        domain=dom,
        mode='standard',
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=True,
        face_shift_search=1,
    )

    nt = normalize_topology(cells, domain=dom)
    gv = np.asarray(nt.global_vertices, dtype=float)
    vec = np.stack(dom.lattice_vectors, axis=0)

    c0 = next(c for c in nt.cells if int(c['id']) == 0)
    c1 = next(c for c in nt.cells if int(c['id']) == 1)

    # Periodic faces across x-wrap should deduplicate.
    idx0 = [
        i
        for i, f in enumerate(c0['faces'])
        if int(f['adjacent_cell']) == 1
        and tuple(int(x) for x in f.get('adjacent_shift', (0, 0, 0))) == (-1, 0, 0)
    ]
    idx1 = [
        i
        for i, f in enumerate(c1['faces'])
        if int(f['adjacent_cell']) == 0
        and tuple(int(x) for x in f.get('adjacent_shift', (0, 0, 0))) == (1, 0, 0)
    ]
    assert len(idx0) == 1
    assert len(idx1) == 1
    assert int(c0['face_global_id'][idx0[0]]) == int(c1['face_global_id'][idx1[0]])

    # Local edges must match the global canonical representative up to sign.
    tol = 1e-6
    for c in (c0, c1):
        verts = np.asarray(c['vertices'], dtype=float)
        for (u, v), eid in zip(c['edges'], c['edge_global_id']):
            dv_local = verts[int(v)] - verts[int(u)]
            dv_global = _edge_vector_from_global(gv, vec, nt.global_edges[int(eid)])
            assert _min_signed_diff(dv_local, dv_global) < tol
