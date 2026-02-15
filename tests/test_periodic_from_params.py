import numpy as np


def test_periodiccell_from_params_roundtrip():
    """PeriodicCell.from_params should preserve Voro++ parameters."""
    from pyvoro2.domains import PeriodicCell

    bx, bxy, by, bxz, byz, bz = 5.0, 1.25, 4.0, -0.75, 0.5, 6.0
    origin = (0.1, -0.2, 0.3)

    cell = PeriodicCell.from_params(bx, bxy, by, bxz, byz, bz, origin=origin)
    bx2, bxy2, by2, bxz2, byz2, bz2 = cell.to_internal_params()

    assert np.allclose([bx2, bxy2, by2, bxz2, byz2, bz2], [bx, bxy, by, bxz, byz, bz])


def test_periodiccell_from_params_internal_identity():
    """For from_params cells, cart<->internal is identity up to origin."""
    from pyvoro2.domains import PeriodicCell

    cell = PeriodicCell.from_params(
        3.0, 0.4, 2.5, -0.1, 0.2, 4.0, origin=(1.0, 2.0, 3.0)
    )

    pts = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 2.2, 3.3],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    # cart_to_internal subtracts origin, internal_to_cart adds it back.
    internal = cell.cart_to_internal(pts)
    back = cell.internal_to_cart(internal)
    assert np.allclose(back, pts)
