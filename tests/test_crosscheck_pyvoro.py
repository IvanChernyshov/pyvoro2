from __future__ import annotations

import inspect

import numpy as np
import pytest

import pyvoro2 as pv


def _sample_points(rng: np.random.Generator, n: int, bounds, pad_frac: float = 0.05):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    dx, dy, dz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    pad_x, pad_y, pad_z = pad_frac * dx, pad_frac * dy, pad_frac * dz
    low = np.array([xmin + pad_x, ymin + pad_y, zmin + pad_z], dtype=float)
    high = np.array([xmax - pad_x, ymin + dy - pad_y, zmin + dz - pad_z], dtype=float)
    return rng.uniform(low, high, size=(n, 3))


def _rng_for_run(seed: int, run: int) -> np.random.Generator:
    mixed = (seed + 0x9E3779B97F4A7C15 + 104729 * int(run)) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(mixed)


@pytest.mark.fuzz
@pytest.mark.pyvoro
def test_crosscheck_pyvoro_box_standard_volumes(fuzz_settings):
    pyvoro = pytest.importorskip('pyvoro')

    compute_fn = getattr(pyvoro, 'compute_voronoi', None)
    if compute_fn is None:
        pytest.skip('pyvoro.compute_voronoi not found')

    sig = inspect.signature(compute_fn)

    n_runs = int(fuzz_settings['n'])
    seed = int(fuzz_settings['seed'])

    bounds = ((-5.0, 5.0), (-4.0, 6.0), (-3.0, 7.0))
    limits = [
        [bounds[0][0], bounds[0][1]],
        [bounds[1][0], bounds[1][1]],
        [bounds[2][0], bounds[2][1]],
    ]
    domain = pv.Box(bounds)

    block_size = 2.5

    for run in range(n_runs):
        rng = _rng_for_run(seed, 3000 + run)
        pts = _sample_points(rng, 40, bounds)

        cells2 = pv.compute(
            pts,
            domain=domain,
            mode='standard',
            block_size=block_size,
            return_faces=True,
            return_vertices=True,
        )

        # pyvoro expects plain python lists
        cells1 = compute_fn(pts.tolist(), limits, block_size)

        vols2 = np.sort(np.asarray([c['volume'] for c in cells2], dtype=float))
        vols1 = np.sort(np.asarray([c['volume'] for c in cells1], dtype=float))

        assert vols1.shape == vols2.shape

        # Be tolerant to small numeric and version differences.
        np.testing.assert_allclose(vols1, vols2, rtol=1e-6, atol=1e-8)

        # Total volume should be the domain volume in both.
        expected_vol = (
            (bounds[0][1] - bounds[0][0])
            * (bounds[1][1] - bounds[1][0])
            * (bounds[2][1] - bounds[2][0])
        )
        assert abs(vols1.sum() - expected_vol) / expected_vol < 1e-7
        assert abs(vols2.sum() - expected_vol) / expected_vol < 1e-7

    # Silence unused variable warning; we intentionally inspect the signature
    assert sig is not None


@pytest.mark.fuzz
@pytest.mark.pyvoro
def test_crosscheck_pyvoro_orthorhombic_periodic_if_supported(fuzz_settings):
    pyvoro = pytest.importorskip('pyvoro')

    compute_fn = getattr(pyvoro, 'compute_voronoi', None)
    if compute_fn is None:
        pytest.skip('pyvoro.compute_voronoi not found')

    sig = inspect.signature(compute_fn)
    if 'periodic' not in sig.parameters:
        pytest.skip('this pyvoro build does not expose periodic boundaries')

    n_runs = max(1, int(fuzz_settings['n']) // 2)
    seed = int(fuzz_settings['seed'])

    bounds = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
    limits = [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]
    domain = pv.OrthorhombicCell(bounds, periodic=(True, True, True))

    block_size = 2.5

    for run in range(n_runs):
        rng = _rng_for_run(seed, 4000 + run)
        pts = _sample_points(rng, 50, bounds)

        cells2 = pv.compute(pts, domain=domain, mode='standard', block_size=block_size)
        cells1 = compute_fn(
            pts.tolist(), limits, block_size, periodic=[True, True, True]
        )

        vols2 = np.sort(np.asarray([c['volume'] for c in cells2], dtype=float))
        vols1 = np.sort(np.asarray([c['volume'] for c in cells1], dtype=float))
        np.testing.assert_allclose(vols1, vols2, rtol=1e-6, atol=1e-8)

    assert sig is not None
