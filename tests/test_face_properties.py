import numpy as np

from pyvoro2 import (
    PeriodicCell,
    compute,
    analyze_tessellation,
    annotate_face_properties,
)


def _find_face(cell_dict, *, adjacent_cell, adjacent_shift):
    for f in cell_dict.get('faces', []):
        if int(f.get('adjacent_cell', -999999)) != int(adjacent_cell):
            continue
        s = f.get('adjacent_shift', (0, 0, 0))
        s = (int(s[0]), int(s[1]), int(s[2]))
        if s == tuple(int(x) for x in adjacent_shift):
            return f
    raise AssertionError('face not found')


def test_face_properties_standard_periodic_midface():
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
    # Mark faces (orphan/mismatch flags)
    analyze_tessellation(
        cells, cell, expected_ids=[0, 1], mode='standard', mark_faces=True
    )

    annotate_face_properties(cells, cell)

    c0 = next(c for c in cells if int(c['id']) == 0)

    # Face between 0 and 1 without wrapping: should be the x=0.5 plane.
    f = _find_face(c0, adjacent_cell=1, adjacent_shift=(0, 0, 0))
    cent = np.asarray(f['centroid'], dtype=float)
    assert np.allclose(cent, [0.5, 0.5, 0.5], atol=1e-10)

    other = np.asarray(f['other_site'], dtype=float)
    assert np.allclose(other, [0.75, 0.5, 0.5], atol=1e-10)

    n = np.asarray(f['normal'], dtype=float)
    # Normal oriented from site0 -> face should point in +x.
    assert n[0] > 0.0
    assert abs(n[1]) < 1e-12
    assert abs(n[2]) < 1e-12

    x = np.asarray(f['intersection'], dtype=float)
    assert np.allclose(x, cent, atol=1e-10)
    assert bool(f['intersection_inside']) is True
    assert abs(float(f['intersection_centroid_dist'])) < 1e-10
    assert abs(float(f['intersection_edge_min_dist']) - 0.5) < 1e-10
