from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from conftest import rng_for_run


if importlib.util.find_spec('pyvoro2._core2d') is None:
    pytest.skip('pyvoro2._core2d is not available', allow_module_level=True)

import pyvoro2.planar as pv2


def _sample_points_in_bounds(
    rng: np.random.Generator,
    n: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    pad_frac: float = 0.05,
) -> np.ndarray:
    (xmin, xmax), (ymin, ymax) = bounds
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = pad_frac * dx
    pad_y = pad_frac * dy
    low = np.array([xmin + pad_x, ymin + pad_y], dtype=float)
    high = np.array([xmax - pad_x, ymax - pad_y], dtype=float)
    if np.any(high <= low):
        low = np.array([xmin, ymin], dtype=float)
        high = np.array([xmax, ymax], dtype=float)
    return rng.uniform(low, high, size=(n, 2))


@pytest.mark.fuzz
def test_fuzz_planar_compute_box_standard_with_diagnostics(fuzz_settings) -> None:
    n_runs = int(fuzz_settings['n'])
    seed = int(fuzz_settings['seed'])

    bounds = ((-5.0, 5.0), (-4.0, 6.0))
    domain = pv2.Box(bounds)

    for run in range(n_runs):
        rng = rng_for_run(seed, 3000 + run)
        pts = _sample_points_in_bounds(rng, 40, bounds)

        cells, diag = pv2.compute(
            pts,
            domain=domain,
            mode='standard',
            return_diagnostics=True,
        )

        assert diag.ok_area
        assert len(cells) == len(pts)
        assert all('area' in cell for cell in cells)


@pytest.mark.fuzz
def test_fuzz_planar_compute_periodic_with_convenience_diagnostics(
    fuzz_settings,
) -> None:
    n_runs = int(fuzz_settings['n'])
    seed = int(fuzz_settings['seed'])

    bounds = ((0.0, 1.0), (0.0, 1.0))
    domain = pv2.RectangularCell(bounds, periodic=(True, True))

    for run in range(n_runs):
        rng = rng_for_run(seed, 4000 + run)
        pts = _sample_points_in_bounds(rng, 35, bounds)

        cells, diag = pv2.compute(
            pts,
            domain=domain,
            mode='standard',
            return_vertices=False,
            return_adjacency=False,
            return_edges=False,
            return_diagnostics=True,
        )

        assert diag.ok_area
        assert diag.ok_reciprocity
        assert all(set(cell.keys()) == {'id', 'area', 'site'} for cell in cells)


@pytest.mark.fuzz
def test_fuzz_planar_compute_periodic_power_with_diagnostics(
    fuzz_settings,
) -> None:
    n_runs = max(1, int(fuzz_settings['n']) // 2)
    seed = int(fuzz_settings['seed'])

    bounds = ((0.0, 1.0), (0.0, 1.0))
    domain = pv2.RectangularCell(bounds, periodic=(True, True))

    for run in range(n_runs):
        rng = rng_for_run(seed, 5000 + run)
        pts = _sample_points_in_bounds(rng, 24, bounds)
        radii = rng.uniform(0.0, 0.08, size=(24,))

        cells, diag = pv2.compute(
            pts,
            domain=domain,
            mode='power',
            radii=radii,
            include_empty=True,
            return_vertices=False,
            return_adjacency=False,
            return_edges=False,
            return_diagnostics=True,
        )

        assert len(cells) == len(pts)
        assert diag.ok_area
        assert diag.ok_reciprocity
        assert all(set(cell.keys()) >= {'id', 'area', 'site'} for cell in cells)
