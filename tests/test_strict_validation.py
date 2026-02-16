from __future__ import annotations

import numpy as np
import pytest

import pyvoro2


def _make_periodic_domain(L: float = 10.0) -> pyvoro2.OrthorhombicCell:
    return pyvoro2.OrthorhombicCell(
        bounds=((0.0, L), (0.0, L), (0.0, L)),
        periodic=(True, True, True),
    )


def _make_partial_periodic_domain(L: float = 10.0) -> pyvoro2.OrthorhombicCell:
    # Slab-like: periodic in x/y, walls in z.
    return pyvoro2.OrthorhombicCell(
        bounds=((0.0, L), (0.0, L), (0.0, L)),
        periodic=(True, True, False),
    )


def test_validate_tessellation_strict_passes_for_simple_periodic_case() -> None:
    dom = _make_periodic_domain(8.0)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 8.0, size=(25, 3))

    cells = pyvoro2.compute(
        pts,
        domain=dom,
        return_vertices=True,
        return_faces=True,
        return_face_shifts=True,
    )

    diag = pyvoro2.validate_tessellation(cells, dom, level='strict')
    assert diag.ok_volume


def test_validate_tessellation_strict_detects_orphan_face_after_corruption() -> None:
    dom = _make_periodic_domain(8.0)
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.0, 8.0, size=(20, 3))

    cells = pyvoro2.compute(
        pts,
        domain=dom,
        return_vertices=True,
        return_faces=True,
        return_face_shifts=True,
    )
    # Sanity: should pass initially.
    pyvoro2.validate_tessellation(cells, dom, level='strict')

    # Corrupt: remove one interior face from the first non-empty cell.
    removed = False
    for c in cells:
        if c.get('empty', False):
            continue
        faces = c.get('faces') or []
        for k, f in enumerate(list(faces)):
            if int(f.get('adjacent_cell', -1)) >= 0:
                faces.pop(k)
                removed = True
                break
        if removed:
            break
    assert removed

    with pytest.raises(pyvoro2.TessellationError):
        pyvoro2.validate_tessellation(cells, dom, level='strict')


def test_validate_tessellation_strict_passes_for_power_mode_periodic_case() -> None:
    dom = _make_periodic_domain(8.0)
    rng = np.random.default_rng(3)
    pts = rng.uniform(0.0, 8.0, size=(25, 3))
    radii = rng.uniform(0.1, 0.5, size=(25,))

    cells = pyvoro2.compute(
        pts,
        domain=dom,
        mode='power',
        radii=radii,
        return_vertices=True,
        return_faces=True,
        return_face_shifts=True,
        repair_face_shifts=True,
    )
    diag = pyvoro2.validate_tessellation(cells, dom, level='strict')
    assert diag.ok_volume


def test_validate_tessellation_strict_passes_for_partial_periodic_case() -> None:
    dom = _make_partial_periodic_domain(8.0)
    rng = np.random.default_rng(4)
    pts = rng.uniform(0.0, 8.0, size=(30, 3))

    cells = pyvoro2.compute(
        pts,
        domain=dom,
        return_vertices=True,
        return_faces=True,
        return_face_shifts=True,
    )

    diag = pyvoro2.validate_tessellation(cells, dom, level='strict')
    assert diag.ok_volume


def test_validate_normalized_topology_strict_detects_vertex_shift_corruption() -> None:
    dom = _make_periodic_domain(9.0)
    rng = np.random.default_rng(2)
    pts = rng.uniform(0.0, 9.0, size=(30, 3))

    cells = pyvoro2.compute(
        pts,
        domain=dom,
        return_vertices=True,
        return_faces=True,
        return_face_shifts=True,
    )

    nt = pyvoro2.normalize_topology(cells, domain=dom)
    diag = pyvoro2.validate_normalized_topology(nt, dom, level='strict')
    assert diag.ok

    # Corrupt a vertex_shift entry: this should violate the vertex-face shift invariant.
    # (All faces are interior in a fully periodic orthorhombic cell.)
    c0 = nt.cells[0]
    assert 'vertex_shift' in c0 and len(c0['vertex_shift']) > 0
    s0 = c0['vertex_shift'][0]
    c0['vertex_shift'][0] = (int(s0[0]) + 1, int(s0[1]), int(s0[2]))

    with pytest.raises(pyvoro2.NormalizationError):
        pyvoro2.validate_normalized_topology(nt, dom, level='strict')


def test_validate_normalized_topology_strict_detects_adjacent_shift_corruption() -> (
    None
):
    dom = _make_periodic_domain(9.0)
    rng = np.random.default_rng(5)
    pts = rng.uniform(0.0, 9.0, size=(30, 3))

    cells = pyvoro2.compute(
        pts,
        domain=dom,
        return_vertices=True,
        return_faces=True,
        return_face_shifts=True,
    )

    nt = pyvoro2.normalize_topology(cells, domain=dom)
    pyvoro2.validate_normalized_topology(nt, dom, level='strict')

    # Corrupt one interior face's adjacent_shift.
    corrupted = False
    for c in nt.cells:
        faces = c.get('faces') or []
        for f in faces:
            if int(f.get('adjacent_cell', -1)) >= 0:
                sx, sy, sz = f.get('adjacent_shift', (0, 0, 0))
                f['adjacent_shift'] = (int(sx) + 1, int(sy), int(sz))
                corrupted = True
                break
        if corrupted:
            break
    assert corrupted

    with pytest.raises(pyvoro2.NormalizationError):
        pyvoro2.validate_normalized_topology(nt, dom, level='strict')
