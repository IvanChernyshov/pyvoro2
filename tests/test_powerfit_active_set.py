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


def test_self_consistent_solver_respects_add_hysteresis_from_empty_start():
    from pyvoro2 import ActiveSetOptions, Box, solve_self_consistent_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    res = solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=domain,
        active0=np.array([False]),
        options=ActiveSetOptions(add_after=2, drop_after=1, max_iter=6),
        return_history=True,
    )

    assert res.termination == 'self_consistent'
    assert res.history is not None
    assert len(res.history) >= 2
    assert [row.n_active for row in res.history[:2]] == [0, 1]
    assert bool(res.active_mask[0]) is True
    assert int(res.diagnostics.first_realized_iter[0]) == 1
    assert int(res.diagnostics.toggle_count[0]) == 1


def test_self_consistent_solver_under_relaxation_records_nonzero_weight_step():
    from pyvoro2 import ActiveSetOptions, Box, solve_self_consistent_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    res = solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
        domain=domain,
        active0=np.array([False]),
        options=ActiveSetOptions(
            add_after=2,
            drop_after=1,
            relax=0.5,
            max_iter=30,
            weight_step_tol=1e-6,
        ),
        return_history=True,
    )

    assert res.history is not None
    assert any(row.weight_step_norm > 0.0 for row in res.history)
    assert res.fit.weights is not None
    assert bool(res.active_mask[0]) is True
    assert res.termination == 'self_consistent'


def test_self_consistent_solver_reports_realized_other_shift_for_periodic_pair():
    from pyvoro2 import (
        ActiveSetOptions,
        PeriodicCell,
        solve_self_consistent_power_weights,
    )

    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)
    res = solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5, (1, 0, 0))],
        measurement='fraction',
        domain=cell,
        image='given_only',
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=5),
    )

    assert res.termination == 'self_consistent'
    assert bool(res.active_mask[0]) is False
    assert bool(res.realized.realized[0]) is True
    assert bool(res.realized.realized_same_shift[0]) is False
    assert bool(res.realized.realized_other_shift[0]) is True
    assert res.diagnostics.status == ('realized_other_shift',)
    assert (-1, 0, 0) in res.diagnostics.realized_shifts[0]


def test_self_consistent_solver_detects_active_mask_cycle(monkeypatch):
    import pyvoro2.powerfit.active as active_mod
    from pyvoro2 import ActiveSetOptions, Box
    from pyvoro2.powerfit.realize import RealizedPairDiagnostics

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    realized_masks = [
        np.array([True, False], dtype=bool),
        np.array([False, True], dtype=bool),
        np.array([True, False], dtype=bool),
    ]
    state = {'calls': 0}

    def fake_match_realized_pairs(*args, **kwargs):
        idx = min(state['calls'], len(realized_masks) - 1)
        same = realized_masks[idx]
        state['calls'] += 1
        return RealizedPairDiagnostics(
            realized=same.copy(),
            unrealized=tuple(np.flatnonzero(~same).tolist()),
            realized_same_shift=same.copy(),
            realized_other_shift=np.zeros(2, dtype=bool),
            realized_shifts=tuple(((0, 0, 0),) if bool(v) else tuple() for v in same),
            endpoint_i_empty=np.zeros(2, dtype=bool),
            endpoint_j_empty=np.zeros(2, dtype=bool),
            boundary_measure=None,
            cells=None,
            tessellation_diagnostics=None,
        )

    monkeypatch.setattr(active_mod, 'match_realized_pairs', fake_match_realized_pairs)

    res = active_mod.solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5), (1, 2, 0.5)],
        measurement='fraction',
        domain=domain,
        options=ActiveSetOptions(add_after=1, drop_after=1, cycle_window=4, max_iter=8),
        return_history=True,
    )

    assert res.termination == 'cycle_detected'
    assert res.converged is False
    assert res.cycle_length == 2
    assert set(res.marginal_constraints) == {0, 1}
    assert res.diagnostics.status == ('cycle_member', 'cycle_member')


def test_self_consistent_result_exports_records_with_ids():
    from pyvoro2 import (
        ActiveSetOptions,
        Box,
        FitModel,
        Interval,
        solve_self_consistent_power_weights,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    box = Box(((-5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0)))
    res = solve_self_consistent_power_weights(
        pts,
        [(101, 202, 0.5)],
        ids=[101, 202],
        index_mode='id',
        measurement='fraction',
        domain=box,
        model=FitModel(feasible=Interval(0.0, 1.0)),
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=3),
    )

    rows = res.to_records(use_ids=True)
    assert len(rows) == 1
    assert rows[0]['site_i'] == 101
    assert rows[0]['site_j'] == 202
    assert rows[0]['status'] in {
        'stable_active',
        'stable_inactive',
        'active_unrealized',
    }


def test_self_consistent_solver_supports_planar_box() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2 import solve_self_consistent_power_weights

    pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=float,
    )
    domain = pv2.Box(((-5.0, 5.0), (-5.0, 5.0)))
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
    assert res.tessellation_diagnostics is not None
    assert res.tessellation_diagnostics.ok is True
    assert res.diagnostics.boundary_measure is not None
    assert np.isfinite(res.rms_residual_all)


def test_self_consistent_solver_supports_planar_periodic_wrong_shift() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2 import ActiveSetOptions, solve_self_consistent_power_weights

    cell = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    res = solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5, (1, 0))],
        measurement='fraction',
        domain=cell,
        image='given_only',
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=5),
    )

    assert res.termination == 'self_consistent'
    assert bool(res.active_mask[0]) is False
    assert bool(res.realized.realized[0]) is True
    assert bool(res.realized.realized_same_shift[0]) is False
    assert bool(res.realized.realized_other_shift[0]) is True
    assert res.diagnostics.status == ('realized_other_shift',)
    assert (-1, 0) in res.diagnostics.realized_shifts[0]


def test_self_consistent_solver_reports_active_connectivity_and_unaccounted_pairs():
    from pyvoro2 import (
        ActiveSetOptions,
        Box,
        solve_self_consistent_power_weights,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0)))
    res = solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5), (2, 3, 0.5), (0, 2, 0.5)],
        measurement='fraction',
        domain=box,
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=5),
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
    )

    assert res.termination == 'self_consistent'
    assert np.array_equal(res.active_mask, np.array([True, True, False]))
    assert {(pair.site_i, pair.site_j) for pair in res.realized.unaccounted_pairs} == {
        (1, 2),
    }
    assert res.connectivity is not None
    assert res.connectivity.candidate_graph.n_components == 1
    assert res.connectivity.active_graph is not None
    assert res.connectivity.active_graph.n_components == 2
    assert res.connectivity.active_offsets_identified_by_data is False


def test_self_consistent_solver_preserves_active_component_offsets_on_final_refit(
    monkeypatch,
):
    import pyvoro2.powerfit.active as active_mod
    from pyvoro2 import ActiveSetOptions, Box
    from pyvoro2.powerfit.realize import RealizedPairDiagnostics
    from pyvoro2.powerfit.solver import PowerWeightFitResult, weights_to_radii

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=float,
    )
    domain = Box(((-5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0)))

    def fake_fit_power_weights(points, constraints, **kwargs):
        if constraints.n_constraints == 3:
            weights = np.array([10.0, 12.0, 30.0, 28.0], dtype=float)
        else:
            weights = np.array([-1.0, 1.0, 1.0, -1.0], dtype=float)
        radii, shift = weights_to_radii(weights)
        return PowerWeightFitResult(
            status='optimal',
            hard_feasible=True,
            weights=weights,
            radii=radii,
            weight_shift=shift,
            measurement=constraints.measurement,
            target=constraints.target.copy(),
            predicted=np.zeros(constraints.n_constraints, dtype=float),
            predicted_fraction=np.zeros(constraints.n_constraints, dtype=float),
            predicted_position=np.zeros(constraints.n_constraints, dtype=float),
            residuals=np.zeros(constraints.n_constraints, dtype=float),
            rms_residual=0.0,
            max_residual=0.0,
            used_shifts=constraints.shifts.copy(),
            solver='analytic',
            n_iter=0,
            converged=True,
            conflict=None,
            warnings=tuple(),
        )

    def fake_match_realized_pairs(*args, **kwargs):
        same = np.array([True, True, False], dtype=bool)
        return RealizedPairDiagnostics(
            realized=same.copy(),
            unrealized=(2,),
            realized_same_shift=same.copy(),
            realized_other_shift=np.zeros(3, dtype=bool),
            realized_shifts=(((0, 0, 0),), ((0, 0, 0),), tuple()),
            endpoint_i_empty=np.zeros(3, dtype=bool),
            endpoint_j_empty=np.zeros(3, dtype=bool),
            boundary_measure=None,
            cells=None,
            tessellation_diagnostics=None,
            unaccounted_pairs=tuple(),
            warnings=tuple(),
        )

    monkeypatch.setattr(active_mod, 'fit_power_weights', fake_fit_power_weights)
    monkeypatch.setattr(active_mod, 'match_realized_pairs', fake_match_realized_pairs)

    res = active_mod.solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5), (2, 3, 0.5), (0, 2, 0.5)],
        measurement='fraction',
        domain=domain,
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=5),
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
    )

    assert res.termination == 'self_consistent'
    assert np.allclose(res.fit.weights, np.array([10.0, 12.0, 30.0, 28.0]))
    assert np.allclose(res.fit.weights[:2], np.array([10.0, 12.0]))
    assert np.allclose(res.fit.weights[2:], np.array([30.0, 28.0]))


def test_self_consistent_solver_reports_transient_path_disconnectivity():
    from pyvoro2 import ActiveSetOptions, Box, solve_self_consistent_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    res = solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
        measurement='fraction',
        domain=box,
        active0=np.array([False, False, False]),
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=6),
        return_history=True,
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
    )

    assert res.path_summary is not None
    assert res.path_summary.n_iterations == len(res.history)
    assert res.path_summary.ever_fit_active_graph_disconnected is True
    assert res.path_summary.ever_fit_active_effective_graph_disconnected is True
    assert res.path_summary.ever_fit_active_offsets_unidentified_by_data is True
    assert res.path_summary.first_fit_active_graph_disconnected_iter == 1
    assert res.path_summary.first_fit_active_effective_graph_disconnected_iter == 1
    assert res.path_summary.max_fit_active_graph_components == 3
    assert res.path_summary.max_fit_active_effective_graph_components == 3
    assert res.path_summary.ever_unaccounted_pairs is False
    assert res.connectivity is not None
    assert res.connectivity.active_graph is not None
    assert res.connectivity.active_graph.n_components == 1
    assert res.history is not None
    assert res.history[0].n_active_fit == 0
    assert res.history[0].n_active == 2
    assert res.history[0].fit_active_graph_n_components == 3
    assert res.history[0].fit_active_effective_graph_n_components == 3
    assert res.history[0].fit_active_offsets_identified_by_data is False
    assert res.history[0].n_unaccounted_pairs == 0
    assert res.history[-1].fit_active_graph_n_components == 1


def test_self_consistent_solver_tracks_transient_unaccounted_pairs(
    monkeypatch,
):
    import pyvoro2.powerfit.active as active_mod
    from pyvoro2 import ActiveSetOptions, Box
    from pyvoro2.powerfit.realize import (
        RealizedPairDiagnostics,
        UnaccountedRealizedPair,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    state = {'calls': 0}

    def fake_match_realized_pairs(*args, **kwargs):
        state['calls'] += 1
        same = np.array([True, True], dtype=bool)
        unaccounted = (
            (
                UnaccountedRealizedPair(
                    site_i=0,
                    site_j=2,
                    realized_shifts=((0, 0, 0),),
                    boundary_measure=None,
                ),
            )
            if state['calls'] == 1
            else tuple()
        )
        return RealizedPairDiagnostics(
            realized=same.copy(),
            unrealized=tuple(),
            realized_same_shift=same.copy(),
            realized_other_shift=np.zeros(2, dtype=bool),
            realized_shifts=(((0, 0, 0),), ((0, 0, 0),)),
            endpoint_i_empty=np.zeros(2, dtype=bool),
            endpoint_j_empty=np.zeros(2, dtype=bool),
            boundary_measure=None,
            cells=None,
            tessellation_diagnostics=None,
            unaccounted_pairs=unaccounted,
            warnings=tuple(),
        )

    monkeypatch.setattr(active_mod, 'match_realized_pairs', fake_match_realized_pairs)

    res = active_mod.solve_self_consistent_power_weights(
        pts,
        [(0, 1, 0.5), (1, 2, 0.5)],
        measurement='fraction',
        domain=box,
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=4),
        return_history=True,
        connectivity_check='diagnose',
        unaccounted_pair_check='warn',
    )

    assert res.termination == 'self_consistent'
    assert res.path_summary is not None
    assert res.path_summary.ever_unaccounted_pairs is True
    assert res.path_summary.max_n_unaccounted_pairs == 1
    assert res.path_summary.first_unaccounted_pairs_iter == 1
    assert res.history is not None
    assert res.history[0].n_unaccounted_pairs == 1
    assert res.realized.unaccounted_pairs == tuple()
    assert all('candidate-absent point pairs' not in msg for msg in res.warnings)
