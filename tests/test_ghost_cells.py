import numpy as np

import pyvoro2


def test_ghost_cells_box_standard_inside_outside() -> None:
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)
    queries = np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    box = pyvoro2.Box(bounds=((-1, 1), (-1, 1), (-1, 1)))

    cells = pyvoro2.ghost_cells(
        pts,
        queries,
        domain=box,
        mode='standard',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
        include_empty=True,
    )

    assert isinstance(cells, list)
    assert len(cells) == 2

    c0 = cells[0]
    c1 = cells[1]

    assert c0['query_index'] == 0
    assert np.allclose(np.asarray(c0['query'], dtype=float), queries[0])
    assert c0['empty'] is False
    # Expected half-space volume: x in [0.25, 1] -> 0.75 * 2 * 2 = 3.0
    assert abs(float(c0['volume']) - 3.0) < 1e-6

    assert c1['query_index'] == 1
    assert np.allclose(np.asarray(c1['query'], dtype=float), queries[1])
    assert c1['empty'] is True
    assert float(c1['volume']) == 0.0


def test_ghost_cells_box_power_volume_shift() -> None:
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)
    radii = np.array([0.3], dtype=float)
    queries = np.array([[0.5, 0.0, 0.0]], dtype=float)
    box = pyvoro2.Box(bounds=((-1, 1), (-1, 1), (-1, 1)))

    cells = pyvoro2.ghost_cells(
        pts,
        queries,
        domain=box,
        mode='power',
        radii=radii,
        ghost_radius=0.0,
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
        include_empty=True,
    )

    assert len(cells) == 1
    c = cells[0]
    assert c['empty'] is False

    # Power bisector between (0, r0) and (0.5, rg=0):
    # x = 0.25 + r0^2 - rg^2 = 0.25 + 0.09 = 0.34
    # Ghost region x in [0.34, 1] => (1-0.34)*4 = 2.64
    assert abs(float(c['volume']) - 2.64) < 1e-4


def test_ghost_cells_ids_remap_faces() -> None:
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)
    ids = [123]
    queries = np.array([[0.5, 0.0, 0.0]], dtype=float)
    box = pyvoro2.Box(bounds=((-1, 1), (-1, 1), (-1, 1)))

    cells = pyvoro2.ghost_cells(
        pts,
        queries,
        domain=box,
        ids=ids,
        mode='standard',
        return_vertices=True,
        return_adjacency=False,
        return_faces=True,
        include_empty=True,
    )

    assert len(cells) == 1
    c = cells[0]
    assert c['empty'] is False
    neigh = [int(f['adjacent_cell']) for f in c.get('faces', [])]
    assert 123 in neigh


def test_ghost_cells_orthorhombic_periodic_query_wrapping() -> None:
    domain = pyvoro2.OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        periodic=(True, False, False),
    )
    pts = np.array([[0.1, 0.1, 0.1]], dtype=float)
    queries = np.array([[1.2, 0.1, 0.1]], dtype=float)  # wraps to x=0.2

    cells = pyvoro2.ghost_cells(
        pts,
        queries,
        domain=domain,
        mode='standard',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
        include_empty=True,
    )

    assert len(cells) == 1
    c = cells[0]
    assert c['empty'] is False

    site = np.asarray(c['site'], dtype=float)
    assert abs(site[0] - 0.2) < 1e-12
    assert abs(site[1] - 0.1) < 1e-12
    assert abs(site[2] - 0.1) < 1e-12
