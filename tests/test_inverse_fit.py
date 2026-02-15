import numpy as np


def test_fit_power_weights_fraction_two_points_analytic():
    from pyvoro2 import fit_power_weights_from_plane_fractions

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights_from_plane_fractions(pts, [(0, 1, 0.25)])

    # Expected: w0 - w1 = d^2 (2t-1) = 4 * (-0.5) = -2
    assert np.allclose(res.weights[0] - res.weights[1], -2.0, atol=1e-10)
    assert np.allclose(res.t_pred[0], 0.25, atol=1e-10)
    assert res.solver == 'analytic'


def test_fit_power_weights_fraction_allows_t_outside_segment():
    from pyvoro2 import fit_power_weights_from_plane_fractions

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights_from_plane_fractions(
        pts, [(0, 1, 1.2)], t_bounds_mode='none'
    )

    assert np.allclose(res.t_pred[0], 1.2, atol=1e-10)
    # Radii are defined via a gauge shift; should be non-negative.
    assert np.all(res.radii >= 0)


def test_fit_power_weights_fraction_hard_bounds_clips_prediction():
    from pyvoro2 import fit_power_weights_from_plane_fractions

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights_from_plane_fractions(
        pts,
        [(0, 1, -0.2)],
        t_bounds=(0.0, 1.0),
        t_bounds_mode='hard',
        solver='admm',
        max_iter=5000,
    )

    assert 0.0 <= res.t_pred[0] <= 1.0
    assert np.allclose(res.t_pred[0], 0.0, atol=1e-5)


def test_r_min_sets_minimum_radius_via_weight_shift():
    from pyvoro2 import fit_power_weights_from_plane_fractions

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights_from_plane_fractions(pts, [(0, 1, 0.25)], r_min=1.0)

    assert np.min(res.radii) == np.min(res.radii)  # not NaN
    assert np.allclose(np.min(res.radii), 1.0, atol=1e-12)
    # The underlying weights are not shifted; shift is reported separately.
    assert np.allclose(res.weights[0], 0.0, atol=1e-12)


def test_fit_power_weights_fraction_soft_quadratic_penalty_prefers_inside_interval():
    from pyvoro2 import fit_power_weights_from_plane_fractions

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    # Without restriction, the solver can match t=-0.2 exactly.
    res0 = fit_power_weights_from_plane_fractions(
        pts, [(0, 1, -0.2)], t_bounds_mode='none'
    )
    assert np.allclose(res0.t_pred[0], -0.2, atol=1e-10)

    # With a soft penalty for leaving [0,1], prediction should move toward the interval.
    res = fit_power_weights_from_plane_fractions(
        pts,
        [(0, 1, -0.2)],
        t_bounds=(0.0, 1.0),
        t_bounds_mode='soft_quadratic',
        alpha_out=100.0,
        solver='admm',
        max_iter=5000,
    )

    assert res.t_pred[0] > res0.t_pred[0]


def test_fit_power_weights_fraction_near_boundary_penalty_pushes_away():
    from pyvoro2 import fit_power_weights_from_plane_fractions

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    # Target is very close to 0. With hard bounds only, it should fit near 0.
    res_hard = fit_power_weights_from_plane_fractions(
        pts,
        [(0, 1, 1e-3)],
        t_bounds=(0.0, 1.0),
        t_bounds_mode='hard',
        solver='admm',
        max_iter=5000,
    )

    # With near-boundary repulsion, the optimum should move away from the boundary.
    res_repulse = fit_power_weights_from_plane_fractions(
        pts,
        [(0, 1, 1e-3)],
        t_bounds=(0.0, 1.0),
        t_bounds_mode='hard',
        t_near_penalty='exp',
        beta_near=1.0,
        t_margin=0.05,
        t_tau=0.01,
        solver='admm',
        max_iter=8000,
    )

    assert res_repulse.t_pred[0] >= res_hard.t_pred[0] - 1e-6
    assert res_repulse.t_pred[0] > 0.01  # should not hug the boundary


def test_fit_power_weights_check_contacts_flags_inactive_constraints():
    from pyvoro2 import Box, fit_power_weights_from_plane_fractions

    # Three collinear points: neighbors are (0-1) and (1-2), not (0-2).
    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))

    res = fit_power_weights_from_plane_fractions(
        pts,
        constraints=[(0, 2, 0.5)],
        domain=domain,
        check_contacts=True,
    )

    assert res.is_contact is not None
    assert res.inactive_constraints is not None
    assert res.is_contact.shape == (1,)
    assert bool(res.is_contact[0]) is False
    assert tuple(res.inactive_constraints) == (0,)
