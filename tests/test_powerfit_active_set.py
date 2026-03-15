import numpy as np


def test_self_consistent_solver_drops_unrealized_pair():
    from pyvoro2 import Box, solve_self_consistent_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    res = solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
        measurement='fraction',
        domain=domain,
        return_history=True,
        return_boundary_measure=True,
        return_tessellation_diagnostics=True,
    )

    assert res.termination == 'self_consistent'
    assert bool(res.active_mask[0]) is True
    assert bool(res.active_mask[1]) is True
    assert bool(res.active_mask[2]) is False
    assert bool(res.realized.realized_same_shift[2]) is False
    assert res.history is not None
    assert len(res.history) >= 1
    assert res.constraints.n_constraints == 3
    assert res.tessellation_diagnostics is not None
    assert res.tessellation_diagnostics.ok is True
    assert np.isfinite(res.rms_residual_all)
    assert np.isfinite(res.max_residual_all)
    assert np.array_equal(res.diagnostics.site_i, np.array([0, 1, 0]))
    assert np.array_equal(res.diagnostics.site_j, np.array([1, 2, 2]))
    assert res.diagnostics.boundary_measure is not None
    assert res.diagnostics.status[0] == 'stable_active'
    assert res.diagnostics.status[1] == 'stable_active'
    assert res.diagnostics.status[2] in {'toggled_inactive', 'stable_inactive'}



def test_self_consistent_solver_can_start_from_empty_active_set():
    from pyvoro2 import ActiveSetOptions, Box, solve_self_consistent_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    res = solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=domain,
        active0=np.array([False]),
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=5),
    )

    assert res.termination == 'self_consistent'
    assert bool(res.active_mask[0]) is True
    assert bool(res.realized.realized_same_shift[0]) is True
    assert res.diagnostics.status == ('toggled_active',)
