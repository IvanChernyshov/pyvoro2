import numpy as np

from pyvoro2 import Box, compute


def test_box_standard_two_points_volume_partition():
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    box = Box(bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    cells = compute(
        pts, domain=box, mode='standard', return_adjacency=False, return_faces=False
    )
    vols = sorted([c['volume'] for c in cells])
    assert len(vols) == 2
    assert abs(sum(vols) - 1000.0) < 1e-6  # 10*10*10
