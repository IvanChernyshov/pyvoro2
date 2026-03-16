import numpy as np
import pytest


def test_resolve_pair_bisector_constraints_preserves_explicit_periodic_shift():
    from pyvoro2 import PeriodicCell, resolve_pair_bisector_constraints

    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.5, (-1, 0, 0))],
        measurement='fraction',
        domain=cell,
        image='given_only',
    )

    assert bool(constraints.explicit_shift[0]) is True
    assert tuple(int(v) for v in constraints.shifts[0]) == (-1, 0, 0)
    assert np.isclose(constraints.distance[0], 0.2)
    assert np.isclose(constraints.target_fraction[0], 0.5)
    assert np.isclose(constraints.target_position[0], 0.1)


def test_resolve_pair_bisector_constraints_rejects_shifts_on_nonperiodic_axes():
    from pyvoro2 import OrthorhombicCell, resolve_pair_bisector_constraints

    domain = OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), periodic=(True, False, True)
    )
    pts = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=float)

    with pytest.raises(ValueError, match='non-periodic axes|non-periodic'):
        resolve_pair_bisector_constraints(
            pts,
            [(0, 1, 0.5, (0, 1, 0))],
            measurement='fraction',
            domain=domain,
            image='given_only',
        )


def test_resolved_constraints_export_records_and_ids():
    from pyvoro2 import Box, resolve_pair_bisector_constraints

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    resolved = resolve_pair_bisector_constraints(
        pts,
        [(10, 20, 0.25)],
        ids=[10, 20],
        index_mode='id',
        measurement='fraction',
        domain=domain,
    )

    rows_idx = resolved.to_records()
    rows_id = resolved.to_records(use_ids=True)
    assert rows_idx[0]['site_i'] == 0
    assert rows_idx[0]['site_j'] == 1
    assert rows_id[0]['site_i'] == 10
    assert rows_id[0]['site_j'] == 20
    assert rows_id[0]['measurement'] == 'fraction'


def test_resolve_pair_bisector_constraints_warns_on_triclinic_search_boundary():
    from pyvoro2 import PeriodicCell, resolve_pair_bisector_constraints

    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.2, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=cell,
        image='nearest',
        image_search=1,
    )

    assert tuple(int(v) for v in constraints.shifts[0]) == (-1, 0, 0)
    assert any('image_search boundary' in msg for msg in constraints.warnings)


def test_resolve_pair_bisector_constraints_supports_planar_box() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2 import resolve_pair_bisector_constraints

    pts = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    domain = pv2.Box(((-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
        domain=domain,
    )

    assert constraints.dim == 2
    assert tuple(int(v) for v in constraints.shifts[0]) == (0, 0)
    assert np.isclose(constraints.distance[0], 2.0)
    assert np.isclose(constraints.target_position[0], 0.5)


def test_resolve_pair_bisector_constraints_supports_planar_periodic_shift() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2 import resolve_pair_bisector_constraints

    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)

    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.5, (-1, 0))],
        measurement='fraction',
        domain=domain,
        image='given_only',
    )

    assert bool(constraints.explicit_shift[0]) is True
    assert tuple(int(v) for v in constraints.shifts[0]) == (-1, 0)
    assert np.isclose(constraints.distance[0], 0.2)
