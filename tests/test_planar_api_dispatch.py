from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import pyvoro2.planar as pv2
import pyvoro2.planar.api as api2d


@dataclass
class FakeCore2D:
    last_call: tuple[str, tuple] | None = None

    def compute_box_standard(
        self,
        points,
        ids,
        bounds,
        blocks,
        periodic,
        init_mem,
        opts,
    ):
        self.last_call = (
            'compute_box_standard',
            (bounds, blocks, periodic, init_mem, opts),
        )
        return [
            {
                'id': 0,
                'area': 0.5,
                'site': [0.1, 0.5],
                'vertices': [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
                'adjacency': [[1, 3], [2, 0], [3, 1], [0, 2]],
                'edges': [
                    {'adjacent_cell': -1, 'vertices': [0, 1]},
                    {'adjacent_cell': 1, 'vertices': [1, 2]},
                    {'adjacent_cell': -2, 'vertices': [2, 3]},
                    {'adjacent_cell': 1, 'vertices': [3, 0]},
                ],
            },
            {
                'id': 1,
                'area': 0.5,
                'site': [0.9, 0.5],
                'vertices': [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                'adjacency': [[1, 3], [2, 0], [3, 1], [0, 2]],
                'edges': [
                    {'adjacent_cell': -1, 'vertices': [0, 1]},
                    {'adjacent_cell': 0, 'vertices': [1, 2]},
                    {'adjacent_cell': -2, 'vertices': [2, 3]},
                    {'adjacent_cell': 0, 'vertices': [3, 0]},
                ],
            },
        ]

    def compute_box_power(
        self,
        points,
        ids,
        radii,
        bounds,
        blocks,
        periodic,
        init_mem,
        opts,
    ):
        self.last_call = ('compute_box_power', (radii.copy(), bounds, blocks, periodic))
        return [
            {
                'id': 1,
                'area': 1.0,
                'site': [0.7, 0.7],
                'vertices': [],
                'adjacency': [],
                'edges': [],
            }
        ]

    def locate_box_standard(
        self,
        points,
        ids,
        bounds,
        blocks,
        periodic,
        init_mem,
        queries,
    ):
        self.last_call = ('locate_box_standard', (bounds, blocks, periodic, init_mem))
        return (
            np.array([True, False]),
            np.array([1, -1]),
            np.array([[1.0, 0.0], [np.nan, np.nan]]),
        )

    def locate_box_power(
        self,
        points,
        ids,
        radii,
        bounds,
        blocks,
        periodic,
        init_mem,
        queries,
    ):
        self.last_call = ('locate_box_power', (radii.copy(), bounds, blocks, periodic))
        return np.array([True]), np.array([0]), np.array([[0.0, 0.0]])

    def ghost_box_standard(
        self,
        points,
        ids,
        bounds,
        blocks,
        periodic,
        init_mem,
        opts,
        queries,
    ):
        self.last_call = (
            'ghost_box_standard',
            (bounds, blocks, periodic, init_mem, opts),
        )
        return [
            {
                'id': -1,
                'empty': False,
                'area': 0.25,
                'site': [0.25, 0.25],
                'vertices': [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                'adjacency': [[1, 3], [2, 0], [3, 1], [0, 2]],
                'edges': [
                    {'adjacent_cell': 0, 'vertices': [0, 1]},
                    {'adjacent_cell': 1, 'vertices': [1, 2]},
                    {'adjacent_cell': -1, 'vertices': [2, 3]},
                    {'adjacent_cell': -2, 'vertices': [3, 0]},
                ],
                'query_index': 0,
            },
            {
                'id': -1,
                'empty': True,
                'area': 0.0,
                'site': [0.0, 0.0],
                'vertices': [],
                'adjacency': [],
                'edges': [],
                'query_index': 1,
            },
        ]

    def ghost_box_power(
        self,
        points,
        ids,
        radii,
        bounds,
        blocks,
        periodic,
        init_mem,
        opts,
        queries,
        ghost_radii,
    ):
        self.last_call = (
            'ghost_box_power',
            (
                radii.copy(),
                ghost_radii.copy(),
                bounds,
                blocks,
                periodic,
                init_mem,
                opts,
            ),
        )
        return []


@pytest.fixture()
def fake_core(monkeypatch) -> FakeCore2D:
    fake = FakeCore2D()
    monkeypatch.setattr(api2d, '_core2d', fake, raising=False)
    monkeypatch.setattr(api2d, '_CORE2D_IMPORT_ERROR', None, raising=False)
    return fake


def test_planar_compute_remaps_ids_and_adds_edge_shifts(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    out = pv2.compute(
        pts,
        domain=pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, False)),
        ids=[10, 20],
        return_edge_shifts=True,
        edge_shift_search=1,
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == 'compute_box_standard'
    assert [cell['id'] for cell in out] == [10, 20]

    c0 = out[0]
    c1 = out[1]
    shifts01 = {
        tuple(int(v) for v in edge['adjacent_shift'])
        for edge in c0['edges']
        if edge['adjacent_cell'] == 20
    }
    shifts10 = {
        tuple(int(v) for v in edge['adjacent_shift'])
        for edge in c1['edges']
        if edge['adjacent_cell'] == 10
    }
    assert shifts01 == {(-1, 0), (0, 0)}
    assert shifts10 == {(0, 0), (1, 0)}


def test_planar_compute_power_inserts_empty_cells(fake_core) -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    out = pv2.compute(
        pts,
        domain=pv2.RectangularCell(((0.0, 2.0), (0.0, 1.0)), periodic=(True, False)),
        mode='power',
        radii=np.array([1.0, 2.0]),
        include_empty=True,
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == 'compute_box_power'
    assert len(out) == 2
    assert out[0]['id'] == 0
    assert out[0]['empty'] is True
    assert out[0]['area'] == 0.0
    assert out[1]['id'] == 1


def test_planar_locate_remaps_owner_ids(fake_core) -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    queries = np.array([[0.9, 0.0], [5.0, 5.0]], dtype=float)
    out = pv2.locate(
        pts,
        queries,
        domain=pv2.Box(((0.0, 2.0), (-1.0, 1.0))),
        ids=[100, 200],
        return_owner_position=True,
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == 'locate_box_standard'
    assert out['found'].tolist() == [True, False]
    assert out['owner_id'].tolist() == [200, -1]
    assert out['owner_pos'].shape == (2, 2)


def test_planar_ghost_cells_remap_neighbor_ids(fake_core) -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    queries = np.array([[0.5, 0.5], [9.0, 9.0]], dtype=float)
    out = pv2.ghost_cells(
        pts,
        queries,
        domain=pv2.Box(((0.0, 2.0), (0.0, 1.0))),
        ids=[10, 20],
        include_empty=False,
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == 'ghost_box_standard'
    assert len(out) == 1
    assert out[0]['edges'][0]['adjacent_cell'] == 10
    assert out[0]['edges'][1]['adjacent_cell'] == 20


def test_planar_return_edge_shifts_requires_periodicity(fake_core) -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match='periodic domains'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 2.0), (0.0, 1.0))),
            return_edge_shifts=True,
        )


def test_planar_compute_return_diagnostics(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    cells, diag = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_diagnostics=True,
        tessellation_check='diagnose',
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == 'compute_box_standard'
    assert isinstance(cells, list)
    assert diag.ok is True
    assert diag.ok_area is True
    assert diag.area_ratio == pytest.approx(1.0)


def test_planar_compute_periodic_diagnostics_strip_internal_geometry(
    fake_core,
) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    cells, diag = pv2.compute(
        pts,
        domain=pv2.RectangularCell(
            ((0.0, 1.0), (0.0, 1.0)),
            periodic=(True, False),
        ),
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        return_diagnostics=True,
        tessellation_check='diagnose',
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == 'compute_box_standard'
    assert fake_core.last_call[1][-1] == (True, False, True)

    assert 'vertices' not in cells[0]
    assert 'adjacency' not in cells[0]
    assert 'edges' not in cells[0]
    assert diag.reciprocity_checked is True
    assert diag.ok_reciprocity is True


def test_planar_compute_tessellation_check_raise(fake_core) -> None:
    def broken_compute_box_standard(*args, **kwargs):
        fake_core.last_call = ('compute_box_standard', tuple())
        return [
            {
                'id': 0,
                'area': 0.25,
                'site': [0.1, 0.5],
                'vertices': [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
                'adjacency': [[1, 3], [2, 0], [3, 1], [0, 2]],
                'edges': [
                    {'adjacent_cell': -1, 'vertices': [0, 1]},
                    {'adjacent_cell': -1, 'vertices': [1, 2]},
                    {'adjacent_cell': -1, 'vertices': [2, 3]},
                    {'adjacent_cell': -1, 'vertices': [3, 0]},
                ],
            }
        ]

    fake_core.compute_box_standard = broken_compute_box_standard

    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    with pytest.raises(pv2.TessellationError, match='tessellation_check failed'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            tessellation_check='raise',
        )


def test_planar_compute_invalid_tessellation_check(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    with pytest.raises(ValueError, match='tessellation_check'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            tessellation_check='nope',  # type: ignore[arg-type]
        )
