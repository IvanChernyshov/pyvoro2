import numpy as np

from pyvoro2 import Box, OrthorhombicCell, locate


def test_locate_box_standard_two_points():
    pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    box = Box(bounds=((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)))

    q = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)

    out = locate(pts, q, domain=box, mode='standard')
    assert out['found'].shape == (2,)
    assert out['owner_id'].shape == (2,)
    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [0, 1]

    out2 = locate(pts, q, domain=box, mode='standard', return_owner_position=True)
    assert 'owner_pos' in out2
    pos = np.asarray(out2['owner_pos'], dtype=float)
    assert pos.shape == (2, 3)
    assert np.allclose(pos[0], pts[0])
    assert np.allclose(pos[1], pts[1])


def test_locate_box_standard_remaps_ids():
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    ids = [10, 20]
    box = Box(bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    q = np.array([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=float)

    out = locate(pts, q, domain=box, ids=ids, mode='standard')
    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [10, 20]


def test_locate_orthorhombic_periodic_x_returns_image_positions():
    # 1D periodic wire along x: [0,1) with periodic x, nonperiodic y/z.
    dom = OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), periodic=(True, False, False)
    )
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

    q = np.array(
        [
            [1.05, 0.5, 0.5],  # wraps to 0.05 -> closer to 0.1 (id 0)
            [-0.05, 0.5, 0.5],
        ],  # wraps to 0.95 -> closer to 0.9 (id 1)
        dtype=float,
    )

    out = locate(pts, q, domain=dom, mode='standard', return_owner_position=True)
    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [0, 1]

    # Voro++ returns owner positions in the periodic image consistent with the query.
    pos = np.asarray(out['owner_pos'], dtype=float)
    Lx = dom.bounds[0][1] - dom.bounds[0][0]
    assert np.isclose(pos[0, 0], pts[0, 0] + 1 * Lx)
    assert np.isclose(pos[1, 0], pts[1, 0] - 1 * Lx)
    assert np.allclose(pos[:, 1:], pts[:, 1:])
