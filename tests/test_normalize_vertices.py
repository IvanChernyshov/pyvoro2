import numpy as np

from pyvoro2 import OrthorhombicCell, PeriodicCell, compute, normalize_vertices


def _reconstruct_vertices(
    global_vertices: np.ndarray,
    gids: np.ndarray,
    shifts: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    """Reconstruct local vertex coordinates from (gid, shift) mapping."""
    a, b, c = vectors
    v0 = global_vertices[gids]
    return v0 + shifts[:, 0:1] * a + shifts[:, 1:2] * b + shifts[:, 2:3] * c


def test_normalize_vertices_periodic_reconstructs_local_vertices_cubic():
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

    nv = normalize_vertices(cells, domain=cell)
    gv = np.asarray(nv.global_vertices, dtype=float)
    vec = np.asarray(cell.vectors, dtype=float)

    assert gv.ndim == 2 and gv.shape[1] == 3
    assert len(nv.cells) == len(cells)

    # Global vertices must be in the primary cell
    gv2, shifts = cell.remap_cart(gv, return_shifts=True)
    assert np.all(shifts == 0)
    assert np.allclose(gv2, gv, atol=1e-10)

    # Each cell must be reconstructable from (gid, shift)
    for c in nv.cells:
        verts = np.asarray(c['vertices'], dtype=float)
        gids = np.asarray(c['vertex_global_id'], dtype=int)
        sh = np.asarray(c['vertex_shift'], dtype=int)
        rec = _reconstruct_vertices(gv, gids, sh, vec)
        assert np.allclose(rec, verts, atol=1e-7)


def test_normalize_vertices_periodic_reconstructs_local_vertices_sheared():
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

    nv = normalize_vertices(cells, domain=cell)
    gv = np.asarray(nv.global_vertices, dtype=float)
    vec = np.asarray(cell.vectors, dtype=float)

    gv2, shifts = cell.remap_cart(gv, return_shifts=True)
    assert np.all(shifts == 0)
    assert np.allclose(gv2, gv, atol=1e-10)

    for c in nv.cells:
        verts = np.asarray(c['vertices'], dtype=float)
        gids = np.asarray(c['vertex_global_id'], dtype=int)
        sh = np.asarray(c['vertex_shift'], dtype=int)
        rec = _reconstruct_vertices(gv, gids, sh, vec)
        assert np.allclose(rec, verts, atol=1e-7)


def test_normalize_vertices_orthorhombic_partial_periodic_reconstructs_local_vertices():
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

    nv = normalize_vertices(cells, domain=dom)
    gv = np.asarray(nv.global_vertices, dtype=float)
    vec = np.stack(dom.lattice_vectors, axis=0)

    # Global vertices must be in the primary cell for the periodic axes.
    gv2, shifts = dom.remap_cart(gv, return_shifts=True)
    assert np.all(shifts == 0)
    assert np.allclose(gv2, gv, atol=1e-10)

    # Each cell must be reconstructable from (gid, shift)
    for c in nv.cells:
        verts = np.asarray(c['vertices'], dtype=float)
        gids = np.asarray(c['vertex_global_id'], dtype=int)
        sh = np.asarray(c['vertex_shift'], dtype=int)
        rec = _reconstruct_vertices(gv, gids, sh, vec)
        assert np.allclose(rec, verts, atol=1e-7)
