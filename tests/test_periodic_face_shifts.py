import numpy as np


from pyvoro2 import PeriodicCell, compute


def _as_tuple3(x):
    # adjacent_shift may be stored as tuple or list
    return tuple(int(v) for v in x)


def test_return_face_shifts_requires_faces_and_vertices():
    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

    # Missing faces
    try:
        compute(
            pts,
            domain=cell,
            mode='standard',
            return_vertices=True,
            return_faces=False,
            return_adjacency=False,
            return_face_shifts=True,
        )
        assert False, 'expected ValueError'
    except ValueError:
        pass

    # Missing vertices
    try:
        compute(
            pts,
            domain=cell,
            mode='standard',
            return_vertices=False,
            return_faces=True,
            return_adjacency=False,
            return_face_shifts=True,
        )
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_periodic_face_shifts_detect_wraparound_standard():
    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

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

    c0 = next(c for c in cells if c['id'] == 0)
    c1 = next(c for c in cells if c['id'] == 1)

    # Site coordinates are returned in Cartesian and should match inputs here.
    assert np.allclose(np.asarray(c0['site'], dtype=float), pts[0])
    assert np.allclose(np.asarray(c1['site'], dtype=float), pts[1])

    s01 = {
        _as_tuple3(f['adjacent_shift']) for f in c0['faces'] if f['adjacent_cell'] == 1
    }
    s10 = {
        _as_tuple3(f['adjacent_shift']) for f in c1['faces'] if f['adjacent_cell'] == 0
    }

    # At least one face should correspond to the nearest periodic image across x.
    assert (-1, 0, 0) in s01
    assert (1, 0, 0) in s10

    # Reciprocity: if i sees j with shift s, j should see i with shift -s.
    assert tuple(-v for v in (-1, 0, 0)) in s10


def test_periodic_face_shifts_power_mode():
    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    # NOTE: in power (Laguerre) diagrams, sufficiently different weights can
    # produce *empty* cells (Voro++ compute_cell returns false). To robustly
    # test shift recovery in power mode, use a configuration where both cells
    # are guaranteed non-empty.
    pts = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float)

    # Small weight difference; both cells remain non-empty.
    radii = np.array([1.0, 1.1], dtype=float)
    cells = compute(
        pts,
        domain=cell,
        mode='power',
        radii=radii,
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=True,
        face_shift_search=1,
    )

    c0 = next(c for c in cells if c['id'] == 0)
    c1 = next(c for c in cells if c['id'] == 1)
    s01 = {
        _as_tuple3(f['adjacent_shift']) for f in c0['faces'] if f['adjacent_cell'] == 1
    }
    s10 = {
        _as_tuple3(f['adjacent_shift']) for f in c1['faces'] if f['adjacent_cell'] == 0
    }

    assert (-1, 0, 0) in s01
    assert (1, 0, 0) in s10
