import numpy as np


def test_match_realized_pairs_flags_unrealized_constraints():
    from pyvoro2 import (
        Box,
        FitModel,
        Interval,
        fit_power_weights,
        match_realized_pairs,
        resolve_pair_bisector_constraints,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 2, 0.5)],
        measurement='fraction',
        domain=domain,
    )
    fit = fit_power_weights(
        pts,
        constraints,
        model=FitModel(feasible=Interval(0.0, 1.0)),
        solver='admm',
        max_iter=5000,
    )
    diag = match_realized_pairs(pts, domain=domain, radii=fit.radii, constraints=constraints)

    assert diag.realized.shape == (1,)
    assert bool(diag.realized[0]) is False
    assert diag.unrealized == (0,)


def test_match_realized_pairs_reports_boundary_measure_when_requested():
    from pyvoro2 import (
        Box,
        fit_power_weights,
        match_realized_pairs,
        resolve_pair_bisector_constraints,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=domain,
    )
    fit = fit_power_weights(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=domain,
        radii=fit.radii,
        constraints=constraints,
        return_boundary_measure=True,
    )

    assert bool(diag.realized[0]) is True
    assert diag.boundary_measure is not None
    assert np.isfinite(diag.boundary_measure[0])
    assert diag.boundary_measure[0] > 0.0
