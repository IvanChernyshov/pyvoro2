import numpy as np

from pyvoro2 import Box, PeriodicCell, compute


def test_power_box_equal_radii_matches_standard():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    box = Box(bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    radii = np.ones(len(pts), dtype=float)

    cells_std = compute(
        pts,
        domain=box,
        mode='standard',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
    )
    cells_pow = compute(
        pts,
        domain=box,
        mode='power',
        radii=radii,
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
    )
    v0 = sorted([c['volume'] for c in cells_std])
    v1 = sorted([c['volume'] for c in cells_pow])
    assert np.allclose(v0, v1)


def test_power_periodic_equal_radii_matches_standard():
    cell = PeriodicCell(vectors=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)))
    pts = np.array(
        [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [8.0, 2.0, 7.0],
            [3.0, 9.0, 4.0],
        ],
        dtype=float,
    )
    radii = np.ones(len(pts), dtype=float)

    cells_std = compute(
        pts,
        domain=cell,
        mode='standard',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
    )
    cells_pow = compute(
        pts,
        domain=cell,
        mode='power',
        radii=radii,
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
    )
    v0 = sorted([c['volume'] for c in cells_std])
    v1 = sorted([c['volume'] for c in cells_pow])
    assert np.allclose(v0, v1)
