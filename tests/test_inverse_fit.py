import numpy as np


def test_fit_power_weights_fraction_two_points_analytic():
    from pyvoro2 import fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(pts, [(0, 1, 0.25)], measurement='fraction')

    assert np.allclose(res.weights[0] - res.weights[1], -2.0, atol=1e-10)
    assert np.allclose(res.predicted[0], 0.25, atol=1e-10)
    assert res.solver == 'analytic'
    assert res.status == 'optimal'


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
    assert np.allclose(res.weights[0], 0.0, atol=1e-12)


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
    from pyvoro2 import ExponentialBoundaryPenalty, FitModel, Interval, fit_power_weights

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
    assert res.infeasible_constraints is not None


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

    assert res.status in ('optimal', 'max_iter')
    assert res.predicted is not None
