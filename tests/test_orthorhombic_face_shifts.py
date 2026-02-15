import numpy as np

from pyvoro2 import OrthorhombicCell, compute


def test_orthorhombic_slab_face_shifts_have_zero_z_shift():
    # Periodic in x,y but open in z.
    domain = OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), periodic=(True, True, False)
    )

    pts = np.array(
        [
            [0.10, 0.50, 0.50],
            [0.90, 0.50, 0.50],  # close across x-boundary
            [0.50, 0.10, 0.50],
            [0.50, 0.90, 0.50],  # close across y-boundary
        ],
        dtype=float,
    )

    cells = compute(
        pts,
        domain=domain,
        mode='standard',
        return_vertices=True,
        return_faces=True,
        return_face_shifts=True,
    )

    # All shifts must have sz=0 in a slab.
    # At least one interior face should use a non-zero (sx,sy) shift.
    seen_nonzero = False

    for c in cells:
        for f in c.get('faces', []):
            s = f.get('adjacent_shift')
            assert s is not None
            sx, sy, sz = int(s[0]), int(s[1]), int(s[2])
            assert sz == 0
            if int(f.get('adjacent_cell', -999999)) >= 0:
                if sx != 0 or sy != 0:
                    seen_nonzero = True

    assert seen_nonzero
