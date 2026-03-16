import numpy as np
import pytest


def test_powerfit_rejects_nonfinite_points_values_and_confidence():
    from pyvoro2 import fit_power_weights, resolve_pair_bisector_constraints

    pts_bad = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match='finite'):
        resolve_pair_bisector_constraints(pts_bad, [(0, 1, 0.5)])
    with pytest.raises(ValueError, match='finite'):
        fit_power_weights(pts_bad, [(0, 1, 0.5)])

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match='finite'):
        resolve_pair_bisector_constraints(pts, [(0, 1, np.nan)])
    with pytest.raises(ValueError, match='finite'):
        resolve_pair_bisector_constraints(pts, [(0, 1, 0.5)], confidence=[np.inf])


def test_powerfit_constraint_ids_must_match_points_and_be_unique():
    from pyvoro2 import resolve_pair_bisector_constraints

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )

    with pytest.raises(ValueError, match='unique'):
        resolve_pair_bisector_constraints(
            pts,
            [(10, 20, 0.5)],
            ids=[10, 10, 20],
            index_mode='id',
        )

    with pytest.raises(ValueError, match='length n_points'):
        resolve_pair_bisector_constraints(
            pts,
            [(10, 20, 0.5)],
            ids=[10, 20],
            index_mode='id',
        )


def test_zero_confidence_constraints_do_not_crash_quadratic_fit():
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
        confidence=[0.0],
    )

    assert res.status == 'optimal'
    assert np.allclose(res.weights, np.array([0.0, 0.0]))
    assert np.allclose(res.predicted_fraction, np.array([0.5]))
    assert any('zero-confidence' in msg for msg in res.warnings)


def test_zero_confidence_rows_do_not_join_effective_components():
    from pyvoro2 import fit_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    res = fit_power_weights(
        pts,
        [(0, 1, 0.25), (1, 2, 0.9)],
        measurement='fraction',
        confidence=[1.0, 0.0],
    )

    assert res.status == 'optimal'
    assert np.allclose(res.weights[0] - res.weights[1], -2.0, atol=1e-10)
    assert np.allclose(res.weights[2], 0.0, atol=1e-12)


def test_empty_resolved_constraints_use_regularization_only_solution():
    from pyvoro2 import FitModel, L2Regularization, fit_power_weights
    from pyvoro2.powerfit.constraints import resolve_pair_bisector_constraints

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    constraints = resolve_pair_bisector_constraints(
        pts,
        [],
        measurement='fraction',
        allow_empty=True,
    )
    model = FitModel(
        regularization=L2Regularization(
            strength=1.0,
            reference=np.array([3.0, 5.0], dtype=float),
        )
    )

    res = fit_power_weights(pts, constraints, model=model)

    assert res.status == 'optimal'
    assert np.allclose(res.weights, np.array([3.0, 5.0]))
    assert any('regularization-only' in msg for msg in res.warnings)


def test_weight_radius_conversions_reject_nonfinite_values():
    from pyvoro2 import radii_to_weights, weights_to_radii

    with pytest.raises(ValueError, match='finite'):
        radii_to_weights(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match='finite'):
        weights_to_radii(np.array([0.0, np.inf]))


def test_fit_power_weights_returns_numerical_failure_on_internal_solver_error(
    monkeypatch,
):
    import pyvoro2.powerfit.solver as solver_mod
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError('synthetic failure')

    monkeypatch.setattr(solver_mod, '_solve_component_analytic', boom)

    res = fit_power_weights(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
        solver='analytic',
    )

    assert res.status == 'numerical_failure'
    assert res.converged is False
    assert res.weights is None
    assert res.radii is None
    assert res.predicted is None
    assert any('numerical solver failure' in msg for msg in res.warnings)


def test_active_set_propagates_numerical_failure(monkeypatch):
    import pyvoro2.powerfit.active as active_mod
    from pyvoro2 import Box
    from pyvoro2.powerfit.solver import PowerWeightFitResult

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))

    def fake_fit_power_weights(points, constraints, **kwargs):
        return PowerWeightFitResult(
            status='numerical_failure',
            hard_feasible=True,
            weights=None,
            radii=None,
            weight_shift=None,
            measurement=constraints.measurement,
            target=constraints.target.copy(),
            predicted=None,
            predicted_fraction=None,
            predicted_position=None,
            residuals=None,
            rms_residual=None,
            max_residual=None,
            used_shifts=constraints.shifts.copy(),
            solver='analytic',
            n_iter=0,
            converged=False,
            conflict=None,
            warnings=('synthetic fit failure',),
        )

    monkeypatch.setattr(active_mod, 'fit_power_weights', fake_fit_power_weights)

    res = active_mod.solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=domain,
    )

    assert res.termination == 'numerical_failure'
    assert res.converged is False
    assert res.fit.status == 'numerical_failure'
    assert res.diagnostics.status == ('numerical_failure',)
    assert any('synthetic fit failure' in msg for msg in res.warnings)


def test_fit_power_weights_accepts_pre_resolved_lower_dim_constraints():
    from pyvoro2 import PairBisectorConstraints, fit_power_weights

    pts = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    constraints = PairBisectorConstraints(
        n_points=2,
        i=np.array([0], dtype=np.int64),
        j=np.array([1], dtype=np.int64),
        shifts=np.zeros((1, 2), dtype=np.int64),
        target=np.array([0.25], dtype=np.float64),
        confidence=np.array([1.0], dtype=np.float64),
        measurement='fraction',
        distance=np.array([2.0], dtype=np.float64),
        distance2=np.array([4.0], dtype=np.float64),
        delta=np.array([[2.0, 0.0]], dtype=np.float64),
        target_fraction=np.array([0.25], dtype=np.float64),
        target_position=np.array([0.5], dtype=np.float64),
        input_index=np.array([0], dtype=np.int64),
        explicit_shift=np.array([False], dtype=bool),
        ids=None,
        warnings=tuple(),
    )

    res = fit_power_weights(pts, constraints, measurement='fraction')

    assert res.status == 'optimal'
    assert np.allclose(res.weights[1] - res.weights[0], 2.0)
    assert np.allclose(res.predicted_fraction, np.array([0.25]))


def test_pre_resolved_constraints_expose_dimension_property():
    from pyvoro2 import PairBisectorConstraints

    constraints = PairBisectorConstraints(
        n_points=2,
        i=np.array([0], dtype=np.int64),
        j=np.array([1], dtype=np.int64),
        shifts=np.zeros((1, 2), dtype=np.int64),
        target=np.array([0.25], dtype=np.float64),
        confidence=np.array([1.0], dtype=np.float64),
        measurement='fraction',
        distance=np.array([2.0], dtype=np.float64),
        distance2=np.array([4.0], dtype=np.float64),
        delta=np.array([[2.0, 0.0]], dtype=np.float64),
        target_fraction=np.array([0.25], dtype=np.float64),
        target_position=np.array([0.5], dtype=np.float64),
        input_index=np.array([0], dtype=np.int64),
        explicit_shift=np.array([False], dtype=bool),
        ids=None,
        warnings=tuple(),
    )

    assert constraints.dim == 2


def test_match_realized_pairs_supports_pre_resolved_planar_constraints():
    import pyvoro2.planar as pv2
    from pyvoro2 import PairBisectorConstraints, match_realized_pairs

    pts = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    constraints = PairBisectorConstraints(
        n_points=2,
        i=np.array([0], dtype=np.int64),
        j=np.array([1], dtype=np.int64),
        shifts=np.zeros((1, 2), dtype=np.int64),
        target=np.array([0.25], dtype=np.float64),
        confidence=np.array([1.0], dtype=np.float64),
        measurement='fraction',
        distance=np.array([2.0], dtype=np.float64),
        distance2=np.array([4.0], dtype=np.float64),
        delta=np.array([[2.0, 0.0]], dtype=np.float64),
        target_fraction=np.array([0.25], dtype=np.float64),
        target_position=np.array([0.5], dtype=np.float64),
        input_index=np.array([0], dtype=np.int64),
        explicit_shift=np.array([False], dtype=bool),
        ids=None,
        warnings=tuple(),
    )

    diag = match_realized_pairs(
        pts,
        domain=pv2.Box(((-5.0, 5.0), (-5.0, 5.0))),
        radii=np.array([0.0, 0.0]),
        constraints=constraints,
    )

    assert bool(diag.realized[0]) is True
    assert diag.unrealized == tuple()


def test_active_set_supports_pre_resolved_planar_constraints():
    import pyvoro2.planar as pv2
    from pyvoro2 import (
        PairBisectorConstraints,
        solve_self_consistent_power_weights,
    )

    pts = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    constraints = PairBisectorConstraints(
        n_points=2,
        i=np.array([0], dtype=np.int64),
        j=np.array([1], dtype=np.int64),
        shifts=np.zeros((1, 2), dtype=np.int64),
        target=np.array([0.25], dtype=np.float64),
        confidence=np.array([1.0], dtype=np.float64),
        measurement='fraction',
        distance=np.array([2.0], dtype=np.float64),
        distance2=np.array([4.0], dtype=np.float64),
        delta=np.array([[2.0, 0.0]], dtype=np.float64),
        target_fraction=np.array([0.25], dtype=np.float64),
        target_position=np.array([0.5], dtype=np.float64),
        input_index=np.array([0], dtype=np.int64),
        explicit_shift=np.array([False], dtype=bool),
        ids=None,
        warnings=tuple(),
    )

    res = solve_self_consistent_power_weights(
        pts,
        constraints,
        measurement='fraction',
        domain=pv2.Box(((-5.0, 5.0), (-5.0, 5.0))),
    )

    assert res.termination == 'self_consistent'
    assert bool(res.realized.realized_same_shift[0]) is True
