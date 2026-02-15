from __future__ import annotations

import numpy as np
import pytest

import pyvoro2


def _box() -> pyvoro2.Box:
    return pyvoro2.Box(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))


def test_compute_rejects_nonfinite_points() -> None:
    dom = _box()
    pts = np.array([[0.1, 0.2, 0.3], [np.nan, 0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match='finite'):
        pyvoro2.compute(
            pts,
            domain=dom,
            return_vertices=False,
            return_adjacency=False,
            return_faces=False,
        )


def test_locate_rejects_nonfinite_queries() -> None:
    dom = _box()
    pts = np.array([[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]], dtype=float)
    q = np.array([[0.1, 0.2, np.inf]], dtype=float)
    with pytest.raises(ValueError, match='finite'):
        pyvoro2.locate(pts, q, domain=dom)


def test_power_mode_rejects_negative_radii() -> None:
    dom = _box()
    pts = np.array([[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]], dtype=float)
    rr = np.array([0.1, -0.2], dtype=float)
    with pytest.raises(ValueError, match='non-negative'):
        pyvoro2.compute(
            pts,
            domain=dom,
            mode='power',
            radii=rr,
            return_vertices=False,
            return_adjacency=False,
            return_faces=False,
        )


def test_ghost_cells_rejects_negative_ghost_radius() -> None:
    dom = _box()
    pts = np.array([[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]], dtype=float)
    q = np.array([[0.2, 0.2, 0.2]], dtype=float)
    rr = np.array([0.1, 0.2], dtype=float)
    with pytest.raises(ValueError, match='ghost_radius'):
        pyvoro2.ghost_cells(
            pts,
            q,
            domain=dom,
            mode='power',
            radii=rr,
            ghost_radius=-1.0,
            return_vertices=False,
            return_adjacency=False,
            return_faces=False,
        )


def test_ids_must_be_unique_and_nonnegative() -> None:
    dom = _box()
    pts = np.array([[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]], dtype=float)

    with pytest.raises(ValueError, match='unique'):
        pyvoro2.compute(
            pts,
            domain=dom,
            ids=[0, 0],
            return_vertices=False,
            return_adjacency=False,
            return_faces=False,
        )

    with pytest.raises(ValueError, match='non-negative'):
        pyvoro2.compute(
            pts,
            domain=dom,
            ids=[-1, 2],
            return_vertices=False,
            return_adjacency=False,
            return_faces=False,
        )


def test_duplicate_check_argument_raises_before_cpp() -> None:
    dom = _box()
    pts = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=float)

    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.compute(
            pts,
            domain=dom,
            duplicate_check='raise',
            return_vertices=False,
            return_adjacency=False,
            return_faces=False,
        )

    # locate/ghost_cells also support the same pre-check.
    q = np.array([[0.3, 0.3, 0.3]], dtype=float)
    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.locate(pts, q, domain=dom, duplicate_check='raise')
    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.ghost_cells(
            pts,
            q,
            domain=dom,
            duplicate_check='raise',
            return_vertices=False,
            return_adjacency=False,
            return_faces=False,
        )


def test_duplicate_check_argument_warns(monkeypatch) -> None:
    dom = _box()
    pts = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=float)

    # Prevent calling into Voro++ with duplicates (it can terminate the process).
    def fake_compute_box_standard(
        pts, ids_internal, bounds, blocks, periodic_flags, init_mem, opts
    ):
        # Minimal stub cells.
        return [{'id': int(i), 'volume': 0.0} for i in range(len(pts))]

    monkeypatch.setattr(
        'pyvoro2.api._core.compute_box_standard',
        fake_compute_box_standard,
    )

    with pytest.warns(RuntimeWarning):
        pyvoro2.compute(
            pts,
            domain=dom,
            duplicate_check='warn',
            return_vertices=False,
            return_adjacency=False,
            return_faces=False,
        )
