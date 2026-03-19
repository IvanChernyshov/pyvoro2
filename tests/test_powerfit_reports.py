import json
import numpy as np


def test_fit_report_exports_nested_plain_python_payload():
    from pyvoro2 import (
        Box,
        FixedValue,
        FitModel,
        build_fit_report,
        fit_power_weights,
        resolve_pair_bisector_constraints,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(10, 20, 0.5), (20, 30, 0.5), (10, 30, 3.0)],
        ids=[10, 20, 30],
        index_mode='id',
        measurement='position',
        domain=box,
    )
    fit = fit_power_weights(
        pts,
        constraints,
        model=FitModel(feasible=FixedValue(0.0)),
        solver='admm',
    )

    report = build_fit_report(fit, constraints, use_ids=True)
    report_via_method = fit.to_report(constraints, use_ids=True)

    assert report['summary']['status'] == 'infeasible_hard_constraints'
    assert report['summary']['is_infeasible'] is True
    assert report['conflict'] is not None
    assert report['conflict']['constraint_indices'] == [0, 1, 2]
    assert report['constraints'][0]['site_i'] == 10
    assert report['constraints'][0]['site_j'] == 20
    assert len(report['fit_records']) == 3
    assert report_via_method == report


def test_active_set_report_collects_nested_diagnostics_and_history():
    from pyvoro2 import (
        ActiveSetOptions,
        Box,
        FitModel,
        Interval,
        build_active_set_report,
        resolve_pair_bisector_constraints,
        solve_self_consistent_power_weights,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    box = Box(((-5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(100, 200, 0.5)],
        ids=[100, 200],
        index_mode='id',
        measurement='fraction',
        domain=box,
    )
    result = solve_self_consistent_power_weights(
        pts,
        constraints,
        domain=box,
        model=FitModel(feasible=Interval(0.0, 1.0)),
        options=ActiveSetOptions(max_iter=5),
        return_history=True,
        return_tessellation_diagnostics=True,
    )

    report = build_active_set_report(result, use_ids=True)
    report_via_method = result.to_report(use_ids=True)

    assert report['summary']['n_constraints'] == 1
    assert report['summary']['n_active_final'] in {0, 1}
    assert report['constraints'][0]['site_i'] == 100
    assert report['fit']['summary']['measurement'] == 'fraction'
    assert report['realized']['summary']['n_constraints'] == 1
    assert report['diagnostics'][0]['site_j'] == 200
    assert report['history'] is not None
    assert len(report['history']) >= 1
    assert report['path_summary'] is not None
    assert report['history'][0]['n_active_fit'] is not None
    assert report['tessellation_diagnostics'] is not None
    assert report_via_method == report


def test_report_json_helpers_roundtrip_plain_report(tmp_path):
    from pyvoro2 import (
        Box,
        FixedValue,
        FitModel,
        build_fit_report,
        dumps_report_json,
        fit_power_weights,
        resolve_pair_bisector_constraints,
        write_report_json,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(10, 20, 0.5), (20, 30, 0.5), (10, 30, 3.0)],
        ids=[10, 20, 30],
        index_mode='id',
        measurement='position',
        domain=box,
    )
    fit = fit_power_weights(
        pts,
        constraints,
        model=FitModel(feasible=FixedValue(0.0)),
        solver='admm',
    )

    report = build_fit_report(fit, constraints, use_ids=True)
    payload = dumps_report_json(report, sort_keys=True)
    loaded = json.loads(payload)

    assert loaded['kind'] == 'power_weight_fit'
    assert loaded['summary']['status'] == 'infeasible_hard_constraints'
    assert loaded['conflict'] is not None

    out_path = tmp_path / 'fit_report.json'
    write_report_json(report, out_path, sort_keys=True)
    assert json.loads(out_path.read_text(encoding='utf-8')) == loaded


def test_active_set_report_supports_planar_tessellation_diagnostics() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2 import (
        ActiveSetOptions,
        FitModel,
        Interval,
        build_active_set_report,
        resolve_pair_bisector_constraints,
        solve_self_consistent_power_weights,
    )

    pts = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    box = pv2.Box(((-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(100, 200, 0.5)],
        ids=[100, 200],
        index_mode='id',
        measurement='fraction',
        domain=box,
    )
    result = solve_self_consistent_power_weights(
        pts,
        constraints,
        domain=box,
        model=FitModel(feasible=Interval(0.0, 1.0)),
        options=ActiveSetOptions(max_iter=5),
        return_tessellation_diagnostics=True,
    )

    report = build_active_set_report(result, use_ids=True)

    assert report['constraints'][0]['site_i'] == 100
    assert report['tessellation_diagnostics'] is not None
    assert report['tessellation_diagnostics']['dimension'] == 2
    assert report['tessellation_diagnostics']['domain_area'] > 0.0
    assert report['tessellation_diagnostics']['ok_area'] is True


def test_fit_report_includes_edge_diagnostics_and_algebraic_rows():
    from pyvoro2 import (
        build_fit_report,
        fit_power_weights,
        resolve_pair_bisector_constraints,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(10, 20, 0.25)],
        ids=[10, 20],
        index_mode='id',
        measurement='fraction',
    )
    fit = fit_power_weights(pts, constraints)

    report = build_fit_report(fit, constraints, use_ids=True)

    assert report['edge_diagnostics']['z_obs'] == [-2.0]
    assert report['edge_diagnostics']['z_fit'] == [-2.0]
    assert report['edge_diagnostics']['residual'] == [0.0]
    assert report['edge_diagnostics']['edge_weight'] == [0.015625]
    assert report['fit_records'][0]['site_i'] == 10
    assert report['fit_records'][0]['z_obs'] == -2.0
    assert report['fit_records'][0]['z_fit'] == -2.0
    assert report['fit_records'][0]['algebraic_residual'] == 0.0
    assert report['fit_records'][0]['edge_weight'] == 0.015625


def test_fit_report_includes_connectivity_diagnostics():
    from pyvoro2 import (
        build_fit_report,
        fit_power_weights,
        resolve_pair_bisector_constraints,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=float,
    )
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(10, 20, 0.25), (30, 40, 0.75)],
        ids=[10, 20, 30, 40],
        index_mode='id',
        measurement='fraction',
    )
    fit = fit_power_weights(
        pts,
        constraints,
        connectivity_check='diagnose',
    )

    report = build_fit_report(fit, constraints, use_ids=True)

    assert report['connectivity'] is not None
    assert report['connectivity']['candidate_graph']['n_components'] == 2
    assert report['connectivity']['candidate_graph']['connected_components'] == [
        [10, 20],
        [30, 40],
    ]


def test_realized_report_includes_unaccounted_pairs_and_warnings():
    from pyvoro2 import (
        Box,
        build_realized_report,
        fit_power_weights,
        match_realized_pairs,
        resolve_pair_bisector_constraints,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(10, 30, 0.5)],
        ids=[10, 20, 30],
        index_mode='id',
        measurement='fraction',
        domain=box,
    )
    fit = fit_power_weights(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=box,
        radii=fit.radii,
        constraints=constraints,
        return_boundary_measure=True,
        unaccounted_pair_check='warn',
    )

    report = build_realized_report(diag, constraints, use_ids=True)

    assert report['summary']['n_unaccounted_pairs'] == 2
    assert {(row['site_i'], row['site_j']) for row in report['unaccounted_pairs']} == {
        (10, 20),
        (20, 30),
    }
    assert report['warnings']


def test_active_set_report_uses_final_active_subset_and_top_level_connectivity():
    from pyvoro2 import (
        ActiveSetOptions,
        Box,
        build_active_set_report,
        solve_self_consistent_power_weights,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 15.0), (-5.0, 5.0), (-5.0, 5.0)))
    result = solve_self_consistent_power_weights(
        pts,
        [(10, 20, 0.5), (30, 40, 0.5), (10, 30, 0.5)],
        ids=[10, 20, 30, 40],
        index_mode='id',
        measurement='fraction',
        domain=box,
        options=ActiveSetOptions(add_after=1, drop_after=1, max_iter=5),
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
    )

    report = build_active_set_report(result, use_ids=True)

    assert report['summary']['n_constraints'] == 3
    assert report['fit']['summary']['n_constraints'] == 2
    assert len(report['fit']['fit_records']) == 2
    assert report['realized']['summary']['n_unaccounted_pairs'] == 1
    assert report['connectivity'] is not None
    assert report['connectivity']['candidate_graph']['n_components'] == 1
    assert report['connectivity']['active_graph']['n_components'] == 2
    assert {
        (row['site_i'], row['site_j'])
        for row in report['realized']['unaccounted_pairs']
    } == {(20, 30)}


def test_active_set_report_includes_transient_path_summary_fields():
    from pyvoro2 import (
        ActiveSetOptions,
        Box,
        build_active_set_report,
        solve_self_consistent_power_weights,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    result = solve_self_consistent_power_weights(
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

    report = build_active_set_report(result)

    assert report['path_summary'] is not None
    assert report['path_summary']['ever_fit_active_graph_disconnected'] is True
    assert report['path_summary']['max_fit_active_graph_components'] == 3
    assert report['path_summary']['first_fit_active_graph_disconnected_iter'] == 1
    assert report['path_summary']['ever_unaccounted_pairs'] is False
    assert report['history'] is not None
    assert report['history'][0]['n_active_fit'] == 0
    assert report['history'][0]['n_active'] == 2
    assert report['history'][0]['fit_active_graph_n_components'] == 3
    assert report['history'][0]['fit_active_effective_graph_n_components'] == 3
    assert report['history'][0]['fit_active_offsets_identified_by_data'] is False
    assert report['history'][0]['n_unaccounted_pairs'] == 0
