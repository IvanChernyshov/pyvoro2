import numpy as np

from pyvoro2 import PeriodicCell, compute


def test_include_empty_power_periodic_returns_all_ids():
    # In power/Laguerre mode, a heavily weighted site may dominate
    # and some cells can be empty.
    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)
    radii = np.array([1.0, 2.0], dtype=float)

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
        include_empty=True,
    )

    assert len(cells) == 2
    c0 = next(c for c in cells if int(c['id']) == 0)
    assert c0.get('empty') is True
    assert float(c0.get('volume', 0.0)) == 0.0
    assert c0.get('vertices', []) == []
    assert c0.get('faces', []) == []
