from __future__ import annotations

import pytest

import pyvoro2


def test_periodic_cell_near_coplanar_raises() -> None:
    # Nearly coplanar vectors: extremely small normalized volume.
    with pytest.raises(ValueError, match='degenerate'):
        pyvoro2.PeriodicCell(
            vectors=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1e-12, 1e-12, 1e-25),
            )
        )


def test_periodic_cell_ill_conditioned_warns() -> None:
    # A very large aspect ratio should warn but still be allowed.
    with pytest.warns(RuntimeWarning):
        pyvoro2.PeriodicCell(
            vectors=(
                (1e12, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
