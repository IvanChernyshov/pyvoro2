import numpy as np

from pyvoro2 import PeriodicCell, compute


def _sheared_cell() -> PeriodicCell:
    # Choose lattice vectors already in Voro++'s lower-triangular form so that
    # the internal basis equals Cartesian (rotation = identity).
    #
    # a = (bx, 0, 0)
    # b = (bxy, by, 0)
    # c = (bxz, byz, bz)
    return PeriodicCell(vectors=((10.0, 0.0, 0.0), (2.0, 10.0, 0.0), (1.0, 3.0, 10.0)))


def test_remap_internal_couples_x_when_wrapping_y() -> None:
    cell = _sheared_cell()
    pts_i = np.array([[1.0, 12.0, 0.0]], dtype=float)

    rem, shifts = cell.remap_internal(pts_i, return_shifts=True)
    assert rem.shape == (1, 3)
    assert shifts.shape == (1, 3)

    # Wrapping y by -by also shifts x by -bxy in the sheared basis.
    assert np.allclose(rem[0], [9.0, 2.0, 0.0])

    # Reconstruction: p = rem + na*a + nb*b + nc*c (in internal coords).
    bx, bxy, by, bxz, byz, bz = cell.to_internal_params()
    a = np.array([bx, 0.0, 0.0])
    b = np.array([bxy, by, 0.0])
    c = np.array([bxz, byz, bz])
    na, nb, nc = shifts[0]
    rec = rem[0] + na * a + nb * b + nc * c
    assert np.allclose(rec, pts_i[0])


def test_remap_internal_couples_xy_when_wrapping_z() -> None:
    cell = _sheared_cell()
    pts_i = np.array([[1.0, 1.0, 12.0]], dtype=float)

    rem, shifts = cell.remap_internal(pts_i, return_shifts=True)
    assert np.allclose(rem[0], [2.0, 8.0, 2.0])

    bx, bxy, by, bxz, byz, bz = cell.to_internal_params()
    a = np.array([bx, 0.0, 0.0])
    b = np.array([bxy, by, 0.0])
    c = np.array([bxz, byz, bz])
    na, nb, nc = shifts[0]
    rec = rem[0] + na * a + nb * b + nc * c
    assert np.allclose(rec, pts_i[0])


def test_remap_cart_respects_origin() -> None:
    cell = PeriodicCell(
        vectors=((10.0, 0.0, 0.0), (2.0, 10.0, 0.0), (1.0, 3.0, 10.0)),
        origin=(10.0, 0.0, 0.0),
    )
    pt = np.array([[11.0, 12.0, 0.0]], dtype=float)  # origin + [1,12,0]
    rem = cell.remap_cart(pt)
    assert np.allclose(rem[0], [19.0, 2.0, 0.0])


def test_compute_does_not_pre_wrap_periodic_points(monkeypatch) -> None:
    """Ensure compute() passes raw internal coordinates to the C++ layer.

    Voro++ applies periodic remapping internally. Pre-wrapping in Python is
    both redundant and can be incorrect for sheared cells.
    """
    cell = _sheared_cell()
    pts = np.array([[1.0, 12.0, 0.0], [5.0, 5.0, 5.0]], dtype=float)

    captured = {}

    def fake_compute_periodic_standard(
        pts_i, ids_internal, cell_params, blocks, init_mem, opts
    ):
        captured['pts_i'] = np.asarray(pts_i, dtype=float)
        # Minimal stub cells.
        return [{'id': int(i), 'volume': 0.0} for i in range(len(pts_i))]

    monkeypatch.setattr(
        'pyvoro2.api._core.compute_periodic_standard',
        fake_compute_periodic_standard,
    )

    compute(
        pts,
        domain=cell,
        mode='standard',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
    )

    assert 'pts_i' in captured
    # Internal basis equals Cartesian for this cell, so the captured point
    # should remain unwrapped (y==12, not y==2).
    assert np.isclose(captured['pts_i'][0, 1], 12.0)
