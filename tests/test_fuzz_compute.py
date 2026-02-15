from __future__ import annotations

import pytest
import numpy as np

import pyvoro2 as pv


def rng_for_run(seed: int, run: int) -> np.random.Generator:
    """Deterministic per-run RNG for fuzz tests."""
    mixed = (seed + 0x9E3779B97F4A7C15 + 104729 * int(run)) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(mixed)


def _sample_points_in_bounds(
    rng: np.random.Generator,
    n: int,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    pad_frac: float = 0.05,
) -> np.ndarray:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    pad_x = pad_frac * dx
    pad_y = pad_frac * dy
    pad_z = pad_frac * dz
    low = np.array([xmin + pad_x, ymin + pad_y, zmin + pad_z], dtype=float)
    high = np.array([xmax - pad_x, ymax - pad_y, zmax - pad_z], dtype=float)
    if np.any(high <= low):
        low = np.array([xmin, ymin, zmin], dtype=float)
        high = np.array([xmax, ymax, zmax], dtype=float)
    return rng.uniform(low, high, size=(n, 3))


@pytest.mark.fuzz
def test_fuzz_compute_box_standard(fuzz_settings):
    n_runs = int(fuzz_settings['n'])
    seed = int(fuzz_settings['seed'])

    bounds = ((-5.0, 5.0), (-4.0, 6.0), (-3.0, 7.0))
    domain = pv.Box(bounds)

    for run in range(n_runs):
        rng = rng_for_run(seed, run)
        pts = _sample_points_in_bounds(rng, 30, bounds)

        cells, diag = pv.compute(
            pts,
            domain=domain,
            mode='standard',
            return_faces=True,
            return_vertices=True,
            return_diagnostics=True,
        )

        assert diag.ok_volume
        assert len(cells) == len(pts)
        for c in cells:
            assert 'volume' in c
            assert c.get('empty', False) in (False, True)


@pytest.mark.fuzz
def test_fuzz_compute_orthorhombic_periodic_with_face_shifts(fuzz_settings):
    n_runs = int(fuzz_settings['n'])
    seed = int(fuzz_settings['seed'])

    bounds = ((0.0, 10.0), (0.0, 8.0), (0.0, 12.0))
    # slab-like partial periodicity (x,y periodic)
    domain = pv.OrthorhombicCell(bounds, periodic=(True, True, False))

    for run in range(n_runs):
        rng = rng_for_run(seed, 1000 + run)
        pts = _sample_points_in_bounds(rng, 40, bounds)

        cells, diag = pv.compute(
            pts,
            domain=domain,
            mode='standard',
            return_faces=True,
            return_vertices=True,
            return_face_shifts=True,
            return_diagnostics=True,
        )

        assert diag.ok_volume
        assert diag.ok_reciprocity

        # For slab periodicity, z-shifts must always be zero.
        for c in cells:
            for f in c.get('faces', []):
                if f.get('adjacent_cell', -1) >= 0:
                    shift = tuple(f.get('adjacent_shift', (0, 0, 0)))
                    assert len(shift) == 3
                    assert int(shift[2]) == 0


@pytest.mark.fuzz
def test_fuzz_compute_periodic_triclinic_with_face_shifts(fuzz_settings):
    n_runs = max(1, int(fuzz_settings['n']) // 2)  # a bit heavier
    seed = int(fuzz_settings['seed'])

    for run in range(n_runs):
        rng = rng_for_run(seed, 2000 + run)

        # Random-but-reasonable triclinic cell parameters.
        bx = float(rng.uniform(8.0, 15.0))
        by = float(rng.uniform(7.0, 14.0))
        bz = float(rng.uniform(6.0, 13.0))
        bxy = float(rng.uniform(-0.3, 0.3) * bx)
        bxz = float(rng.uniform(-0.3, 0.3) * bx)
        byz = float(rng.uniform(-0.3, 0.3) * by)

        cell = pv.PeriodicCell.from_params(bx, bxy, by, bxz, byz, bz)

        # Sample fractional coordinates in [0,1)
        frac = rng.uniform(0.0, 1.0, size=(35, 3))
        a, b, c = cell.vectors
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        c = np.asarray(c, dtype=float)
        pts = frac[:, 0:1] * a + frac[:, 1:2] * b + frac[:, 2:3] * c

        cells, diag = pv.compute(
            pts,
            domain=cell,
            mode='standard',
            return_faces=True,
            return_vertices=True,
            return_face_shifts=True,
            return_diagnostics=True,
        )

        assert diag.ok_volume
        assert diag.ok_reciprocity

        # Basic sanity: each non-wall face should have a shift.
        for ccell in cells:
            for f in ccell.get('faces', []):
                if f.get('adjacent_cell', -1) >= 0:
                    assert 'adjacent_shift' in f
