from __future__ import annotations

import numpy as np
import pytest

import pyvoro2


def test_duplicate_check_returns_empty_for_small_inputs() -> None:
    assert pyvoro2.duplicate_check(np.zeros((0, 3))) == tuple()
    assert pyvoro2.duplicate_check(np.zeros((1, 3))) == tuple()


def test_duplicate_check_detects_exact_duplicate() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.duplicate_check(pts)


def test_duplicate_check_detects_near_duplicate() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [0.5e-5, 0.0, 0.0]], dtype=float)
    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.duplicate_check(pts, threshold=1e-5)


def test_duplicate_check_periodic_wrap_catches_modulo_duplicates() -> None:
    L = 10.0
    dom = pyvoro2.OrthorhombicCell(
        bounds=((0.0, L), (0.0, L), (0.0, L)), periodic=(True, True, True)
    )
    pts = np.array(
        [
            [0.1, 0.2, 0.3],
            [L + 0.1, 0.2, 0.3],  # same point modulo x-periodicity
        ],
        dtype=float,
    )
    # With wrapping, they coincide -> duplicate.
    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.duplicate_check(pts, domain=dom, wrap=True)

    # Without wrapping, distance is ~L -> no duplicate.
    pairs = pyvoro2.duplicate_check(pts, domain=dom, wrap=False, mode='return')
    assert pairs == tuple()
