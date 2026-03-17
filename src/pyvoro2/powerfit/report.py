"""Plain-Python report helpers for power-fitting results.

These helpers sit one layer above the numerical result objects. They keep the
solver API array-oriented while making it easy to export nested diagnostics into
JSON-friendly dictionaries and row lists for downstream packages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constraints import PairBisectorConstraints
from .realize import RealizedPairDiagnostics
from .solver import (
    ConnectivityDiagnostics,
    ConstraintGraphDiagnostics,
    HardConstraintConflict,
    PowerWeightFitResult,
)


def _label_nodes(nodes: tuple[int, ...], ids: np.ndarray | None) -> list[object]:
    labeled: list[object] = []
    for node in nodes:
        if ids is None:
            labeled.append(int(node))
            continue
        value = ids[int(node)]
        labeled.append(value.item() if hasattr(value, 'item') else value)
    return labeled


def _graph_record(
    graph: ConstraintGraphDiagnostics | None,
    *,
    ids: np.ndarray | None,
) -> dict[str, object] | None:
    if graph is None:
        return None
    return {
        'n_points': int(graph.n_points),
        'n_constraints': int(graph.n_constraints),
        'n_edges': int(graph.n_edges),
        'isolated_points': _label_nodes(graph.isolated_points, ids),
        'connected_components': [
            _label_nodes(component, ids)
            for component in graph.connected_components
        ],
        'n_components': int(graph.n_components),
        'fully_connected': bool(graph.fully_connected),
    }


def _connectivity_record(
    diagnostics: ConnectivityDiagnostics | None,
    *,
    ids: np.ndarray | None,
) -> dict[str, object] | None:
    if diagnostics is None:
        return None
    return {
        'unconstrained_points': _label_nodes(diagnostics.unconstrained_points, ids),
        'candidate_graph': _graph_record(diagnostics.candidate_graph, ids=ids),
        'effective_graph': _graph_record(diagnostics.effective_graph, ids=ids),
        'active_graph': _graph_record(diagnostics.active_graph, ids=ids),
        'active_effective_graph': _graph_record(
            diagnostics.active_effective_graph,
            ids=ids,
        ),
        'candidate_offsets_identified_by_data': bool(
            diagnostics.candidate_offsets_identified_by_data
        ),
        'active_offsets_identified_by_data': (
            None
            if diagnostics.active_offsets_identified_by_data is None
            else bool(diagnostics.active_offsets_identified_by_data)
        ),
        'offsets_identified_in_objective': bool(
            diagnostics.offsets_identified_in_objective
        ),
        'gauge_policy': diagnostics.gauge_policy,
        'messages': list(diagnostics.messages),
    }


def _path_summary_record(summary: Any | None) -> dict[str, object] | None:
    if summary is None:
        return None
    return {
        'n_iterations': int(summary.n_iterations),
        'ever_fit_active_graph_disconnected': bool(
            summary.ever_fit_active_graph_disconnected
        ),
        'ever_fit_active_effective_graph_disconnected': bool(
            summary.ever_fit_active_effective_graph_disconnected
        ),
        'ever_fit_active_offsets_unidentified_by_data': bool(
            summary.ever_fit_active_offsets_unidentified_by_data
        ),
        'ever_unaccounted_pairs': bool(summary.ever_unaccounted_pairs),
        'max_fit_active_graph_components': int(
            summary.max_fit_active_graph_components
        ),
        'max_fit_active_effective_graph_components': int(
            summary.max_fit_active_effective_graph_components
        ),
        'max_n_unaccounted_pairs': int(summary.max_n_unaccounted_pairs),
        'first_fit_active_graph_disconnected_iter': (
            None
            if summary.first_fit_active_graph_disconnected_iter is None
            else int(summary.first_fit_active_graph_disconnected_iter)
        ),
        'first_fit_active_effective_graph_disconnected_iter': (
            None
            if summary.first_fit_active_effective_graph_disconnected_iter is None
            else int(summary.first_fit_active_effective_graph_disconnected_iter)
        ),
        'first_unaccounted_pairs_iter': (
            None
            if summary.first_unaccounted_pairs_iter is None
            else int(summary.first_unaccounted_pairs_iter)
        ),
    }


def _tessellation_record(diagnostics: Any | None) -> dict[str, object] | None:
    if diagnostics is None:
        return None
    issue_rows = []
    for issue in diagnostics.issues:
        issue_rows.append(
            {
                'code': issue.code,
                'message': issue.message,
                'severity': issue.severity,
                'examples': list(issue.examples),
            }
        )

    if hasattr(diagnostics, 'domain_volume'):
        return {
            'dimension': 3,
            'n_sites_expected': int(diagnostics.n_sites_expected),
            'n_cells_returned': int(diagnostics.n_cells_returned),
            'sum_cell_measure': float(diagnostics.sum_cell_volume),
            'domain_measure': float(diagnostics.domain_volume),
            'measure_ratio': float(diagnostics.volume_ratio),
            'measure_gap': float(diagnostics.volume_gap),
            'measure_overlap': float(diagnostics.volume_overlap),
            'sum_cell_volume': float(diagnostics.sum_cell_volume),
            'domain_volume': float(diagnostics.domain_volume),
            'volume_ratio': float(diagnostics.volume_ratio),
            'volume_gap': float(diagnostics.volume_gap),
            'volume_overlap': float(diagnostics.volume_overlap),
            'missing_ids': [int(value) for value in diagnostics.missing_ids],
            'empty_ids': [int(value) for value in diagnostics.empty_ids],
            'boundary_shift_available': bool(diagnostics.face_shift_available),
            'face_shift_available': bool(diagnostics.face_shift_available),
            'reciprocity_checked': bool(diagnostics.reciprocity_checked),
            'n_boundaries_total': int(diagnostics.n_faces_total),
            'n_boundaries_orphan': int(diagnostics.n_faces_orphan),
            'n_boundaries_mismatched': int(diagnostics.n_faces_mismatched),
            'n_faces_total': int(diagnostics.n_faces_total),
            'n_faces_orphan': int(diagnostics.n_faces_orphan),
            'n_faces_mismatched': int(diagnostics.n_faces_mismatched),
            'ok_measure': bool(diagnostics.ok_volume),
            'ok_volume': bool(diagnostics.ok_volume),
            'ok_reciprocity': bool(diagnostics.ok_reciprocity),
            'ok': bool(diagnostics.ok),
            'issues': issue_rows,
        }

    if hasattr(diagnostics, 'domain_area'):
        return {
            'dimension': 2,
            'n_sites_expected': int(diagnostics.n_sites_expected),
            'n_cells_returned': int(diagnostics.n_cells_returned),
            'sum_cell_measure': float(diagnostics.sum_cell_area),
            'domain_measure': float(diagnostics.domain_area),
            'measure_ratio': float(diagnostics.area_ratio),
            'measure_gap': float(diagnostics.area_gap),
            'measure_overlap': float(diagnostics.area_overlap),
            'sum_cell_area': float(diagnostics.sum_cell_area),
            'domain_area': float(diagnostics.domain_area),
            'area_ratio': float(diagnostics.area_ratio),
            'area_gap': float(diagnostics.area_gap),
            'area_overlap': float(diagnostics.area_overlap),
            'missing_ids': [int(value) for value in diagnostics.missing_ids],
            'empty_ids': [int(value) for value in diagnostics.empty_ids],
            'boundary_shift_available': bool(diagnostics.edge_shift_available),
            'edge_shift_available': bool(diagnostics.edge_shift_available),
            'reciprocity_checked': bool(diagnostics.reciprocity_checked),
            'n_boundaries_total': int(diagnostics.n_edges_total),
            'n_boundaries_orphan': int(diagnostics.n_edges_orphan),
            'n_boundaries_mismatched': int(diagnostics.n_edges_mismatched),
            'n_edges_total': int(diagnostics.n_edges_total),
            'n_edges_orphan': int(diagnostics.n_edges_orphan),
            'n_edges_mismatched': int(diagnostics.n_edges_mismatched),
            'ok_measure': bool(diagnostics.ok_area),
            'ok_area': bool(diagnostics.ok_area),
            'ok_reciprocity': bool(diagnostics.ok_reciprocity),
            'ok': bool(diagnostics.ok),
            'issues': issue_rows,
        }

    raise TypeError('unsupported tessellation diagnostics object')


def _conflict_record(
    conflict: HardConstraintConflict | None,
    *,
    ids: np.ndarray | None,
) -> dict[str, object] | None:
    if conflict is None:
        return None
    return {
        'message': conflict.message,
        'component_nodes': _label_nodes(conflict.component_nodes, ids),
        'cycle_nodes': _label_nodes(conflict.cycle_nodes, ids),
        'constraint_indices': list(conflict.constraint_indices),
        'terms': list(conflict.to_records(ids=ids)),
    }


def build_fit_report(
    result: PowerWeightFitResult,
    constraints: PairBisectorConstraints,
    *,
    use_ids: bool = False,
) -> dict[str, object]:
    """Return a JSON-friendly report for a low-level fit result."""

    ids = constraints.ids if use_ids else None
    return {
        'kind': 'power_weight_fit',
        'summary': {
            'status': result.status,
            'is_optimal': bool(result.is_optimal),
            'is_infeasible': bool(result.is_infeasible),
            'hard_feasible': bool(result.hard_feasible),
            'solver': result.solver,
            'measurement': result.measurement,
            'n_constraints': int(constraints.n_constraints),
            'n_points': int(constraints.n_points),
            'converged': bool(result.converged),
            'n_iter': int(result.n_iter),
            'rms_residual': (
                None if result.rms_residual is None else float(result.rms_residual)
            ),
            'max_residual': (
                None if result.max_residual is None else float(result.max_residual)
            ),
            'conflicting_constraint_indices': list(
                result.conflicting_constraint_indices
            ),
        },
        'constraints': list(constraints.to_records(use_ids=use_ids)),
        'fit_records': list(result.to_records(constraints, use_ids=use_ids)),
        'weights': None if result.weights is None else result.weights.tolist(),
        'radii': None if result.radii is None else result.radii.tolist(),
        'weight_shift': (
            None if result.weight_shift is None else float(result.weight_shift)
        ),
        'used_shifts': [
            tuple(int(v) for v in shift_row) for shift_row in result.used_shifts
        ],
        'warnings': list(result.warnings),
        'conflict': _conflict_record(result.conflict, ids=ids),
        'connectivity': _connectivity_record(result.connectivity, ids=ids),
    }


def build_realized_report(
    diagnostics: RealizedPairDiagnostics,
    constraints: PairBisectorConstraints,
    *,
    use_ids: bool = False,
) -> dict[str, object]:
    """Return a JSON-friendly report for realized-face matching."""

    ids = constraints.ids if use_ids else None
    return {
        'kind': 'realized_pair_diagnostics',
        'summary': {
            'n_constraints': int(constraints.n_constraints),
            'n_realized': int(np.count_nonzero(diagnostics.realized)),
            'n_same_shift': int(np.count_nonzero(diagnostics.realized_same_shift)),
            'n_other_shift': int(np.count_nonzero(diagnostics.realized_other_shift)),
            'n_unrealized': int(len(diagnostics.unrealized)),
            'n_unaccounted_pairs': int(len(diagnostics.unaccounted_pairs)),
        },
        'records': list(diagnostics.to_records(constraints, use_ids=use_ids)),
        'unrealized': [int(idx) for idx in diagnostics.unrealized],
        'unaccounted_pairs': list(diagnostics.unaccounted_records(ids=ids)),
        'warnings': list(diagnostics.warnings),
        'tessellation_diagnostics': _tessellation_record(
            diagnostics.tessellation_diagnostics
        ),
    }


def build_active_set_report(
    result: Any,
    *,
    use_ids: bool = False,
) -> dict[str, object]:
    """Return a JSON-friendly report for a self-consistent active-set result."""

    # Import lazily to avoid a module cycle during package initialization.
    from .active import SelfConsistentPowerFitResult

    if not isinstance(result, SelfConsistentPowerFitResult):
        raise TypeError(
            'build_active_set_report expects a SelfConsistentPowerFitResult'
        )

    history_rows: list[dict[str, object]] | None = None
    if result.history is not None:
        history_rows = []
        for row in result.history:
            history_rows.append(
                {
                    'iteration': int(row.iteration),
                    'n_active': int(row.n_active),
                    'n_realized': int(row.n_realized),
                    'n_added': int(row.n_added),
                    'n_removed': int(row.n_removed),
                    'rms_residual_all': float(row.rms_residual_all),
                    'max_residual_all': float(row.max_residual_all),
                    'weight_step_norm': float(row.weight_step_norm),
                    'n_active_fit': (
                        None
                        if row.n_active_fit is None
                        else int(row.n_active_fit)
                    ),
                    'fit_active_graph_n_components': (
                        None
                        if row.fit_active_graph_n_components is None
                        else int(row.fit_active_graph_n_components)
                    ),
                    'fit_active_effective_graph_n_components': (
                        None
                        if row.fit_active_effective_graph_n_components is None
                        else int(row.fit_active_effective_graph_n_components)
                    ),
                    'fit_active_offsets_identified_by_data': (
                        None
                        if row.fit_active_offsets_identified_by_data is None
                        else bool(row.fit_active_offsets_identified_by_data)
                    ),
                    'n_unaccounted_pairs': (
                        None
                        if row.n_unaccounted_pairs is None
                        else int(row.n_unaccounted_pairs)
                    ),
                }
            )

    diagnostic_rows = list(result.to_records(use_ids=use_ids))
    marginal_rows = [diagnostic_rows[int(idx)] for idx in result.marginal_constraints]

    return {
        'kind': 'self_consistent_power_fit',
        'summary': {
            'termination': result.termination,
            'converged': bool(result.converged),
            'n_outer_iter': int(result.n_outer_iter),
            'cycle_length': (
                None if result.cycle_length is None else int(result.cycle_length)
            ),
            'n_constraints': int(result.constraints.n_constraints),
            'n_active_final': int(np.count_nonzero(result.active_mask)),
            'n_realized_final': int(np.count_nonzero(result.realized.realized)),
            'rms_residual_all': float(result.rms_residual_all),
            'max_residual_all': float(result.max_residual_all),
            'marginal_constraint_indices': [
                int(idx) for idx in result.marginal_constraints
            ],
        },
        'constraints': list(result.constraints.to_records(use_ids=use_ids)),
        'fit': build_fit_report(
            result.fit,
            result.constraints.subset(result.active_mask),
            use_ids=use_ids,
        ),
        'realized': build_realized_report(
            result.realized,
            result.constraints,
            use_ids=use_ids,
        ),
        'diagnostics': diagnostic_rows,
        'marginal_records': marginal_rows,
        'history': history_rows,
        'path_summary': _path_summary_record(result.path_summary),
        'tessellation_diagnostics': _tessellation_record(
            result.tessellation_diagnostics
        ),
        'warnings': list(result.warnings),
        'connectivity': _connectivity_record(
            result.connectivity,
            ids=(result.constraints.ids if use_ids else None),
        ),
    }


def _jsonable_report_value(value: Any) -> Any:
    """Convert nested report payloads into plain JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(key): _jsonable_report_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_report_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable_report_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dumps_report_json(
    report: dict[str, object],
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> str:
    """Serialize a powerfit report into a JSON string."""

    return json.dumps(
        _jsonable_report_value(report),
        indent=indent,
        sort_keys=sort_keys,
    )


def write_report_json(
    report: dict[str, object],
    path: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Write a powerfit report to a JSON file."""

    output_path = Path(path)
    text = dumps_report_json(report, indent=indent, sort_keys=sort_keys)
    if indent > 0 and not text.endswith('\n'):
        text += '\n'
    output_path.write_text(text, encoding='utf-8')


__all__ = [
    'build_fit_report',
    'build_realized_report',
    'build_active_set_report',
    'dumps_report_json',
    'write_report_json',
]
