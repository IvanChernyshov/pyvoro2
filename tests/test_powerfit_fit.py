import numpy as np
import pytest


def test_fit_power_weights_fraction_two_points_analytic():
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(pts, [(0, 1, 0.25)], measurement='fraction')

    assert np.allclose(res.weights[0] - res.weights[1], -2.0, atol=1e-10)
    assert np.allclose(res.predicted[0], 0.25, atol=1e-10)
    assert res.solver == 'analytic'
    assert res.status == 'optimal'


def test_fit_result_exposes_algebraic_edge_diagnostics():
    from pyvoro2 import fit_power_weights, resolve_pair_bisector_constraints

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(pts, [(0, 1, 0.25)], measurement='fraction')

    assert res.edge_diagnostics is not None
    diag = res.edge_diagnostics
    assert np.allclose(diag.alpha, np.array([0.125]))
    assert np.allclose(diag.beta, np.array([0.5]))
    assert np.allclose(diag.z_obs, np.array([-2.0]))
    assert np.allclose(diag.z_fit, np.array([-2.0]))
    assert np.allclose(diag.residual, np.array([0.0]))
    assert np.allclose(diag.edge_weight, np.array([0.015625]))
    assert np.isclose(diag.weighted_l2, 0.0)
    assert np.isclose(diag.weighted_rmse, 0.0)
    assert np.isclose(diag.rmse, 0.0)
    assert np.isclose(diag.mae, 0.0)

    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
    )
    rows = res.to_records(constraints)
    assert rows[0]['z_obs'] == pytest.approx(-2.0)
    assert rows[0]['z_fit'] == pytest.approx(-2.0)
    assert rows[0]['algebraic_residual'] == pytest.approx(0.0)
    assert rows[0]['edge_weight'] == pytest.approx(0.015625)


def test_fit_power_weights_fraction_allows_values_outside_segment():
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(pts, [(0, 1, 1.2)], measurement='fraction')

    assert np.allclose(res.predicted[0], 1.2, atol=1e-10)
    assert np.all(res.radii >= 0)


def test_fit_power_weights_fraction_hard_interval_clips_prediction():
    from pyvoro2 import FitModel, Interval, fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(
        pts,
        [(0, 1, -0.2)],
        measurement='fraction',
        model=FitModel(feasible=Interval(0.0, 1.0)),
        solver='admm',
        max_iter=5000,
    )

    assert 0.0 <= res.predicted[0] <= 1.0
    assert np.allclose(res.predicted[0], 0.0, atol=1e-5)


def test_r_min_sets_minimum_radius_via_weight_shift():
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(pts, [(0, 1, 0.25)], measurement='fraction', r_min=1.0)

    assert np.min(res.radii) == np.min(res.radii)
    assert np.allclose(np.min(res.radii), 1.0, atol=1e-12)
    assert np.allclose(res.radii * res.radii, res.weights + res.weight_shift)


def test_soft_interval_penalty_prefers_inside_interval():
    from pyvoro2 import FitModel, SoftIntervalPenalty, fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    res0 = fit_power_weights(pts, [(0, 1, -0.2)], measurement='fraction')
    assert np.allclose(res0.predicted[0], -0.2, atol=1e-10)

    res = fit_power_weights(
        pts,
        [(0, 1, -0.2)],
        measurement='fraction',
        model=FitModel(penalties=(SoftIntervalPenalty(0.0, 1.0, 100.0),)),
        solver='admm',
        max_iter=5000,
    )

    assert res.predicted[0] > res0.predicted[0]


def test_exponential_boundary_penalty_pushes_away_from_boundary():
    from pyvoro2 import (
        ExponentialBoundaryPenalty,
        FitModel,
        Interval,
        fit_power_weights,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    res_hard = fit_power_weights(
        pts,
        [(0, 1, 1e-3)],
        measurement='fraction',
        model=FitModel(feasible=Interval(0.0, 1.0)),
        solver='admm',
        max_iter=5000,
    )

    res_repulse = fit_power_weights(
        pts,
        [(0, 1, 1e-3)],
        measurement='fraction',
        model=FitModel(
            feasible=Interval(0.0, 1.0),
            penalties=(
                ExponentialBoundaryPenalty(
                    lower=0.0,
                    upper=1.0,
                    margin=0.05,
                    strength=1.0,
                    tau=0.01,
                ),
            ),
        ),
        solver='admm',
        max_iter=8000,
    )

    assert res_repulse.predicted[0] >= res_hard.predicted[0] - 1e-6
    assert res_repulse.predicted[0] > 0.01


def test_position_measurement_uses_absolute_position_space():
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(pts, [(0, 1, 1.0)], measurement='position')

    assert np.allclose(res.predicted[0], 1.0, atol=1e-10)
    assert np.allclose(res.predicted_position[0], 1.0, atol=1e-10)
    assert np.allclose(res.predicted_fraction[0], 0.25, atol=1e-10)


def test_infeasible_hard_constraints_are_reported():
    from pyvoro2 import FixedValue, FitModel, fit_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    # Impossible equalities on a 3-cycle: z01=0, z12=0, z02=2.
    res = fit_power_weights(
        pts,
        [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 3.0)],
        measurement='position',
        model=FitModel(feasible=FixedValue(0.0)),
        solver='admm',
    )

    assert res.status == 'infeasible_hard_constraints'
    assert res.hard_feasible is False
    assert res.weights is None
    assert res.is_infeasible is True
    assert res.conflicting_constraint_indices == (0, 1, 2)


def test_huber_loss_is_available_as_an_alternative_mismatch():
    from pyvoro2 import FitModel, HuberLoss, fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(
        pts,
        [(0, 1, 1.2)],
        measurement='fraction',
        model=FitModel(mismatch=HuberLoss(delta=0.1)),
        solver='admm',
        max_iter=5000,
    )

    assert res.status == 'optimal'
    assert res.converged is True
    assert res.predicted is not None


def test_fit_power_weights_accepts_explicit_weight_shift_for_radii():
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
        weight_shift=2.0,
    )

    assert np.allclose(res.weights[0] - res.weights[1], -2.0, atol=1e-10)
    assert np.allclose(res.weight_shift, 2.0, atol=1e-12)
    assert np.allclose(res.radii * res.radii, res.weights + 2.0, atol=1e-12)


def test_disconnected_components_use_mean_zero_gauge_and_connectivity_diagnostics():
    from pyvoro2 import fit_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=float,
    )
    res = fit_power_weights(
        pts,
        [(0, 1, 0.25), (2, 3, 0.75)],
        measurement='fraction',
        connectivity_check='diagnose',
    )

    assert np.allclose(res.weights[[0, 1]], np.array([-1.0, 1.0]), atol=1e-10)
    assert np.allclose(res.weights[[2, 3]], np.array([1.0, -1.0]), atol=1e-10)
    assert res.connectivity is not None
    assert res.connectivity.candidate_graph.n_components == 2
    assert res.connectivity.effective_graph.n_components == 2
    assert res.connectivity.offsets_identified_in_objective is False
    assert 'mean zero' in res.connectivity.gauge_policy


def test_disconnected_components_can_align_to_zero_strength_reference_means():
    from pyvoro2 import FitModel, L2Regularization, fit_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=float,
    )
    model = FitModel(
        regularization=L2Regularization(
            strength=0.0,
            reference=np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
        )
    )
    res = fit_power_weights(
        pts,
        [(0, 1, 0.25), (2, 3, 0.75)],
        measurement='fraction',
        model=model,
        connectivity_check='diagnose',
    )

    assert np.allclose(res.weights[0] - res.weights[1], -2.0, atol=1e-10)
    assert np.allclose(res.weights[2] - res.weights[3], 2.0, atol=1e-10)
    assert np.allclose(np.mean(res.weights[:2]), 15.0, atol=1e-10)
    assert np.allclose(np.mean(res.weights[2:]), 35.0, atol=1e-10)
    assert res.connectivity is not None
    assert 'reference mean' in res.connectivity.gauge_policy


def test_fit_power_weights_can_raise_connectivity_diagnostics():
    from pyvoro2 import ConnectivityDiagnosticsError, fit_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=float,
    )

    with pytest.raises(ConnectivityDiagnosticsError):
        fit_power_weights(
            pts,
            [(0, 1, 0.25), (2, 3, 0.75)],
            measurement='fraction',
            connectivity_check='raise',
        )
