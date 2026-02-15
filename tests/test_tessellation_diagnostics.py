import numpy as np

from pyvoro2 import (
    PeriodicCell,
    compute,
    analyze_tessellation,
)


def test_diagnostics_periodic_standard_ok():
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

    diag = analyze_tessellation(cells, cell, expected_ids=[0, 1], mode='standard')
    assert diag.ok_volume
    assert diag.reciprocity_checked
    assert diag.ok_reciprocity
    assert diag.ok
    assert abs(diag.volume_ratio - 1.0) < 1e-8
    assert diag.n_faces_orphan == 0
    assert diag.n_faces_mismatched == 0


def test_compute_return_diagnostics_periodic():
    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float)

    out = compute(
        pts,
        domain=cell,
        mode='standard',
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=True,
        face_shift_search=1,
        return_diagnostics=True,
        tessellation_check='diagnose',
    )
    cells, diag = out
    assert isinstance(cells, list)
    assert diag.ok
