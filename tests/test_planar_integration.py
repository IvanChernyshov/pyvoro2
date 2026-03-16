from __future__ import annotations

import importlib.util

import numpy as np
import pytest


if importlib.util.find_spec('pyvoro2._core2d') is None:
    pytest.skip('pyvoro2._core2d is not available', allow_module_level=True)

import pyvoro2.planar as pv2


def test_planar_compute_standard_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    cells = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_vertices=True,
        return_edges=True,
    )

    assert len(cells) == 2
    assert {int(cell['id']) for cell in cells} == {0, 1}
    assert all('area' in cell for cell in cells)


def test_planar_locate_standard_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    queries = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    out = pv2.locate(pts, queries, domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))))

    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [0, 1]


def test_planar_ghost_cells_standard_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    queries = np.array([[0.5, 0.5]], dtype=float)
    cells = pv2.ghost_cells(
        pts,
        queries,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_vertices=True,
        return_edges=True,
    )

    assert len(cells) == 1
    assert cells[0]['query_index'] == 0
    assert cells[0]['empty'] is False


def test_planar_compute_return_result_only_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    result = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_result=True,
    )

    assert isinstance(result, pv2.PlanarComputeResult)
    assert result.tessellation_diagnostics is None
    assert result.normalized_vertices is None
    assert result.normalized_topology is None
    assert len(result.cells) == 2


def test_planar_locate_power_asymmetric_weights_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    radii = np.array([0.05, 0.15], dtype=float)
    queries = np.array([[0.15, 0.5], [0.9, 0.5]], dtype=float)
    out = pv2.locate(
        pts,
        queries,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        mode='power',
        radii=radii,
    )

    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [0, 1]


def test_planar_ghost_cells_power_asymmetric_weights_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    radii = np.array([0.05, 0.15], dtype=float)
    queries = np.array([[0.5, 0.5]], dtype=float)
    cells = pv2.ghost_cells(
        pts,
        queries,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        mode='power',
        radii=radii,
        ghost_radius=0.1,
        return_vertices=True,
        return_edges=True,
    )

    assert len(cells) == 1
    assert cells[0]['query_index'] == 0
    assert cells[0]['empty'] is False
    assert float(cells[0]['area']) > 0.0


def test_planar_compute_result_vertices_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    result = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        normalize='vertices',
    )

    assert isinstance(result, pv2.PlanarComputeResult)
    assert result.global_vertices is not None
    assert result.global_vertices.shape == (6, 2)
    assert set(result.cells[0].keys()) == {'id', 'area', 'site'}


def test_planar_compute_result_topology_periodic_smoke() -> None:
    pts = np.array([[0.2, 0.2], [0.8, 0.25], [0.4, 0.8]], dtype=float)
    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    result = pv2.compute(
        pts,
        domain=domain,
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        return_diagnostics=True,
        normalize='topology',
    )

    assert isinstance(result, pv2.PlanarComputeResult)
    assert result.require_tessellation_diagnostics().ok is True
    assert set(result.cells[0].keys()) == {'id', 'area', 'site'}

    topo = result.require_normalized_topology()
    diag = pv2.validate_normalized_topology(topo, domain, level='strict')

    assert topo.global_vertices.shape == (6, 2)
    assert len(topo.global_edges) == 9
    assert diag.ok is True


def test_planar_compute_power_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    radii = np.array([0.1, 0.12], dtype=float)
    cells = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        mode='power',
        radii=radii,
        include_empty=True,
        return_vertices=True,
        return_edges=True,
    )

    assert len(cells) == 2
    assert all('area' in cell for cell in cells)


def test_planar_locate_power_return_owner_position_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    radii = np.array([0.1, 0.1], dtype=float)
    queries = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    out = pv2.locate(
        pts,
        queries,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        mode='power',
        radii=radii,
        return_owner_position=True,
    )

    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [0, 1]
    assert out['owner_pos'].shape == (2, 2)


def test_planar_ghost_cells_power_with_ghost_radius_smoke() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    radii = np.array([0.1, 0.1], dtype=float)
    queries = np.array([[0.5, 0.5]], dtype=float)
    cells = pv2.ghost_cells(
        pts,
        queries,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        mode='power',
        radii=radii,
        ghost_radius=0.08,
        return_vertices=True,
        return_edges=True,
        include_empty=True,
    )

    assert len(cells) == 1
    assert cells[0]['query_index'] == 0
    assert 'area' in cells[0]


def test_planar_ghost_cells_periodic_edge_shifts_smoke() -> None:
    pts = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]], dtype=float)
    queries = np.array([[1.15, 0.2]], dtype=float)
    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    cells = pv2.ghost_cells(
        pts,
        queries,
        domain=domain,
        return_vertices=True,
        return_edges=True,
        return_edge_shifts=True,
    )

    assert len(cells) == 1
    assert cells[0]['query_index'] == 0
    assert any(
        'adjacent_shift' in edge
        for edge in (cells[0].get('edges') or [])
        if int(edge.get('adjacent_cell', -1)) >= 0
    )
