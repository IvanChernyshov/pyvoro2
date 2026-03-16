from __future__ import annotations

import importlib.util

import numpy as np
import pytest


if importlib.util.find_spec('pyvoro2._core2d') is None:
    pytest.skip('pyvoro2._core2d is not available', allow_module_level=True)

import pyvoro2.planar as pv2


def _periodic_sample() -> tuple[np.ndarray, pv2.RectangularCell]:
    pts = np.array([[0.2, 0.2], [0.8, 0.25], [0.4, 0.8]], dtype=float)
    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    return pts, domain


def test_planar_analyze_tessellation_box_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    box = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    cells = pv2.compute(pts, domain=box, return_vertices=True, return_edges=True)

    diag = pv2.analyze_tessellation(cells, box)

    assert diag.ok is True
    assert diag.ok_area is True
    assert diag.reciprocity_checked is False
    assert diag.area_ratio == pytest.approx(1.0)


def test_planar_analyze_tessellation_reports_missing_edge_shifts() -> None:
    pts, domain = _periodic_sample()
    cells = pv2.compute(pts, domain=domain, return_vertices=True, return_edges=True)

    diag = pv2.analyze_tessellation(cells, domain)

    assert diag.edge_shift_available is False
    assert diag.reciprocity_checked is False
    assert any(issue.code == 'NO_EDGE_SHIFTS' for issue in diag.issues)


def test_planar_periodic_hidden_adjacency_is_resolved() -> None:
    pts, domain = _periodic_sample()
    cells = pv2.compute(
        pts,
        domain=domain,
        return_vertices=True,
        return_edges=True,
        return_edge_shifts=True,
    )

    assert all(
        int(edge['adjacent_cell']) >= 0
        for cell in cells
        for edge in cell['edges']
    )

    diag = pv2.validate_tessellation(cells, domain, level='strict')
    assert diag.ok is True
    assert diag.n_edges_orphan == 0
    assert diag.n_edges_mismatched == 0


def test_planar_partially_periodic_walls_remain_negative() -> None:
    pts = np.array([[0.2, 0.2], [0.8, 0.25], [0.4, 0.8]], dtype=float)
    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, False))
    cells = pv2.compute(
        pts,
        domain=domain,
        return_vertices=True,
        return_edges=True,
        return_edge_shifts=True,
    )

    assert any(
        int(edge['adjacent_cell']) < 0
        for cell in cells
        for edge in cell['edges']
    )


def test_planar_validate_tessellation_strict_raises_on_area_gap() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    box = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    cells = pv2.compute(pts, domain=box, return_vertices=True, return_edges=True)
    broken = cells[:1]

    with pytest.raises(pv2.TessellationError):
        pv2.validate_tessellation(broken, box, level='strict')


def test_planar_compute_periodic_diagnostics_auto_enable_edge_shifts() -> None:
    pts, domain = _periodic_sample()
    cells, diag = pv2.compute(
        pts,
        domain=domain,
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        return_diagnostics=True,
    )

    assert all(set(cell.keys()) == {'id', 'area', 'site'} for cell in cells)
    assert diag.reciprocity_checked is True
    assert diag.ok_reciprocity is True
    assert diag.ok is True


def test_planar_compute_tessellation_check_raise_uses_internal_shifts() -> None:
    pts, domain = _periodic_sample()
    cells = pv2.compute(
        pts,
        domain=domain,
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        tessellation_check='raise',
    )

    assert len(cells) == len(pts)
    assert all(set(cell.keys()) == {'id', 'area', 'site'} for cell in cells)
