"""Self-consistent active-set refinement for pairwise separator constraints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .constraints import PairBisectorConstraints, resolve_pair_bisector_constraints
from .model import FitModel
from .realize import RealizedPairDiagnostics, match_realized_pairs
from .solver import (
    ConnectivityDiagnostics,
    PowerWeightFitResult,
    _apply_connectivity_policy,
    _build_active_set_connectivity_diagnostics,
    _connected_components,
    _difference_identifying_mask,
    _predict_measurements,
    fit_power_weights,
    weights_to_radii,
)
from ..diagnostics import TessellationDiagnostics as TessellationDiagnostics3D
from ..domains import Box as Box3D, OrthorhombicCell, PeriodicCell
from ..planar.diagnostics import TessellationDiagnostics as TessellationDiagnostics2D
from ..planar.domains import Box as Box2D, RectangularCell

ShiftTuple = tuple[int, ...]


def _label_value(
    values: np.ndarray,
    index: int,
    ids: np.ndarray | None,
) -> object:
    if ids is None:
        return int(values[index])
    item = ids[int(values[index])]
    return item.item() if hasattr(item, 'item') else item


def _boundary_value(values: np.ndarray | None, index: int) -> float | None:
    if values is None or np.isnan(values[index]):
        return None
    return float(values[index])


def _require_self_consistent_supported_dim(
    constraints: PairBisectorConstraints,
) -> None:
    if constraints.dim not in (2, 3):
        raise ValueError(
            'solve_self_consistent_power_weights currently supports only 2D '
            'and 3D resolved constraints'
        )


@dataclass(frozen=True, slots=True)
class ActiveSetOptions:
    add_after: int = 1
    drop_after: int = 2
    relax: float = 1.0
    max_iter: int = 25
    cycle_window: int = 8
    weight_step_tol: float = 1e-8

    def __post_init__(self) -> None:
        if int(self.add_after) <= 0:
            raise ValueError('ActiveSetOptions.add_after must be > 0')
        if int(self.drop_after) <= 0:
            raise ValueError('ActiveSetOptions.drop_after must be > 0')
        if not (0.0 < float(self.relax) <= 1.0):
            raise ValueError('ActiveSetOptions.relax must lie in (0, 1]')
        if int(self.max_iter) <= 0:
            raise ValueError('ActiveSetOptions.max_iter must be > 0')
        if int(self.cycle_window) <= 0:
            raise ValueError('ActiveSetOptions.cycle_window must be > 0')
        if float(self.weight_step_tol) < 0.0:
            raise ValueError('ActiveSetOptions.weight_step_tol must be >= 0')


@dataclass(frozen=True, slots=True)
class ActiveSetIteration:
    iteration: int
    n_active: int
    n_realized: int
    n_added: int
    n_removed: int
    rms_residual_all: float
    max_residual_all: float
    weight_step_norm: float


@dataclass(frozen=True, slots=True)
class PairConstraintDiagnostics:
    site_i: np.ndarray
    site_j: np.ndarray
    shift: np.ndarray
    target: np.ndarray
    confidence: np.ndarray
    predicted: np.ndarray
    predicted_fraction: np.ndarray
    predicted_position: np.ndarray
    residuals: np.ndarray
    active: np.ndarray
    realized: np.ndarray
    realized_same_shift: np.ndarray
    realized_other_shift: np.ndarray
    realized_shifts: tuple[tuple[ShiftTuple, ...], ...]
    endpoint_i_empty: np.ndarray
    endpoint_j_empty: np.ndarray
    boundary_measure: np.ndarray | None
    toggle_count: np.ndarray
    realized_toggle_count: np.ndarray
    first_realized_iter: np.ndarray
    last_realized_iter: np.ndarray
    marginal: np.ndarray
    status: tuple[str, ...]

    def to_records(
        self, *, ids: np.ndarray | None = None
    ) -> tuple[dict[str, object], ...]:
        """Return one plain-Python record per candidate pair."""

        rows: list[dict[str, object]] = []
        for k in range(int(self.site_i.shape[0])):
            realized_shifts = tuple(
                tuple(int(v) for v in shift)
                for shift in self.realized_shifts[k]
            )
            rows.append(
                {
                    'constraint_index': int(k),
                    'site_i': _label_value(self.site_i, k, ids),
                    'site_j': _label_value(self.site_j, k, ids),
                    'shift': tuple(int(v) for v in self.shift[k]),
                    'target': float(self.target[k]),
                    'confidence': float(self.confidence[k]),
                    'predicted': float(self.predicted[k]),
                    'predicted_fraction': float(self.predicted_fraction[k]),
                    'predicted_position': float(self.predicted_position[k]),
                    'residual': float(self.residuals[k]),
                    'active': bool(self.active[k]),
                    'realized': bool(self.realized[k]),
                    'realized_same_shift': bool(self.realized_same_shift[k]),
                    'realized_other_shift': bool(self.realized_other_shift[k]),
                    'realized_shifts': realized_shifts,
                    'endpoint_i_empty': bool(self.endpoint_i_empty[k]),
                    'endpoint_j_empty': bool(self.endpoint_j_empty[k]),
                    'boundary_measure': _boundary_value(self.boundary_measure, k),
                    'toggle_count': int(self.toggle_count[k]),
                    'realized_toggle_count': int(self.realized_toggle_count[k]),
                    'first_realized_iter': int(self.first_realized_iter[k]),
                    'last_realized_iter': int(self.last_realized_iter[k]),
                    'marginal': bool(self.marginal[k]),
                    'status': self.status[k],
                }
            )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class SelfConsistentPowerFitResult:
    constraints: PairBisectorConstraints
    fit: PowerWeightFitResult
    realized: RealizedPairDiagnostics
    diagnostics: PairConstraintDiagnostics
    active_mask: np.ndarray
    n_outer_iter: int
    converged: bool
    termination: Literal[
        'self_consistent',
        'cycle_detected',
        'max_outer_iter',
        'infeasible_active_set',
        'numerical_failure',
    ]
    cycle_length: int | None
    marginal_constraints: tuple[int, ...]
    rms_residual_all: float
    max_residual_all: float
    tessellation_diagnostics: (
        TessellationDiagnostics2D | TessellationDiagnostics3D | None
    )
    history: tuple[ActiveSetIteration, ...] | None
    warnings: tuple[str, ...]
    connectivity: ConnectivityDiagnostics | None = None

    def to_records(self, *, use_ids: bool = False) -> tuple[dict[str, object], ...]:
        """Return one plain-Python record per candidate pair."""

        ids = self.constraints.ids if use_ids else None
        return self.diagnostics.to_records(ids=ids)

    def to_report(self, *, use_ids: bool = False) -> dict[str, object]:
        """Return a JSON-friendly report for this active-set solve."""

        from .report import build_active_set_report

        return build_active_set_report(self, use_ids=use_ids)


def solve_self_consistent_power_weights(
    points: np.ndarray,
    constraints: PairBisectorConstraints | list[tuple] | tuple[tuple, ...],
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: Box2D | RectangularCell | Box3D | OrthorhombicCell | PeriodicCell,
    ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: list[float] | tuple[float, ...] | np.ndarray | None = None,
    model: FitModel | None = None,
    active0: np.ndarray | None = None,
    options: ActiveSetOptions | None = None,
    r_min: float = 0.0,
    weight_shift: float | None = None,
    fit_solver: Literal['auto', 'analytic', 'admm'] = 'auto',
    fit_max_iter: int = 2000,
    fit_rho: float = 1.0,
    fit_tol_abs: float = 1e-6,
    fit_tol_rel: float = 1e-5,
    return_history: bool = False,
    return_cells: bool = False,
    return_boundary_measure: bool = False,
    return_tessellation_diagnostics: bool = False,
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'diagnose',
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
    unaccounted_pair_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
) -> SelfConsistentPowerFitResult:
    """Iteratively refine an active pair set against realized power-diagram
    boundaries."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] <= 0:
        raise ValueError('points must have shape (n, d) with d >= 1')
    if connectivity_check not in ('none', 'diagnose', 'warn', 'raise'):
        raise ValueError(
            'connectivity_check must be none, diagnose, warn, or raise'
        )
    if unaccounted_pair_check not in ('none', 'diagnose', 'warn', 'raise'):
        raise ValueError(
            'unaccounted_pair_check must be none, diagnose, warn, or raise'
        )

    if model is None:
        model = FitModel()
    if options is None:
        options = ActiveSetOptions()

    if isinstance(constraints, PairBisectorConstraints):
        resolved = constraints
        if resolved.n_points != pts.shape[0]:
            raise ValueError('resolved constraints do not match the number of points')
        if resolved.dim != pts.shape[1]:
            raise ValueError('resolved constraints do not match the point dimension')
        _require_self_consistent_supported_dim(resolved)
    else:
        if pts.shape[1] not in (2, 3):
            raise ValueError(
                'solve_self_consistent_power_weights currently supports only '
                '2D and 3D points'
            )
        resolved = resolve_pair_bisector_constraints(
            pts,
            constraints,
            measurement=measurement,
            domain=domain,
            ids=ids,
            index_mode=index_mode,
            image=image,
            image_search=image_search,
            confidence=confidence,
            allow_empty=True,
        )

    m = resolved.n_constraints
    if active0 is None:
        active = np.ones(m, dtype=bool)
    else:
        active = np.asarray(active0, dtype=bool).copy()
        if active.shape != (m,):
            raise ValueError('active0 must have shape (m,)')

    warnings_list = list(resolved.warnings)
    add_streak = np.zeros(m, dtype=np.int64)
    drop_streak = np.zeros(m, dtype=np.int64)
    toggle_count = np.zeros(m, dtype=np.int64)
    realized_toggle_count = np.zeros(m, dtype=np.int64)
    first_realized_iter = np.full(m, -1, dtype=np.int64)
    last_realized_iter = np.full(m, -1, dtype=np.int64)
    history_rows: list[ActiveSetIteration] = []
    prev_weights_eval: np.ndarray | None = None
    prev_realized_same: np.ndarray | None = None
    seen_masks: dict[bytes, int] = {active.tobytes(): 0}

    termination: Literal[
        'self_consistent',
        'cycle_detected',
        'max_outer_iter',
        'infeasible_active_set',
        'numerical_failure',
    ] = 'max_outer_iter'
    cycle_length: int | None = None
    converged = False
    last_diag: RealizedPairDiagnostics | None = None

    for outer_iter in range(1, options.max_iter + 1):
        active_constraints = resolved.subset(active)
        fit = fit_power_weights(
            pts,
            active_constraints,
            model=model,
            r_min=r_min,
            weight_shift=weight_shift,
            solver=fit_solver,
            max_iter=fit_max_iter,
            rho=fit_rho,
            tol_abs=fit_tol_abs,
            tol_rel=fit_tol_rel,
            connectivity_check='diagnose',
        )
        if fit.weights is None:
            warnings_list.extend(fit.warnings)
            termination = (
                'numerical_failure'
                if fit.status == 'numerical_failure'
                else 'infeasible_active_set'
            )
            final_realized = _empty_realized_pair_diagnostics(
                m,
                return_boundary_measure=return_boundary_measure,
            )
            diag_all = PairConstraintDiagnostics(
                site_i=resolved.i.copy(),
                site_j=resolved.j.copy(),
                shift=resolved.shifts.copy(),
                target=resolved.target.copy(),
                confidence=resolved.confidence.copy(),
                predicted=np.full(m, np.nan, dtype=np.float64),
                predicted_fraction=np.full(m, np.nan, dtype=np.float64),
                predicted_position=np.full(m, np.nan, dtype=np.float64),
                residuals=np.full(m, np.nan, dtype=np.float64),
                active=active.copy(),
                realized=final_realized.realized.copy(),
                realized_same_shift=final_realized.realized_same_shift.copy(),
                realized_other_shift=final_realized.realized_other_shift.copy(),
                realized_shifts=final_realized.realized_shifts,
                endpoint_i_empty=final_realized.endpoint_i_empty.copy(),
                endpoint_j_empty=final_realized.endpoint_j_empty.copy(),
                boundary_measure=(
                    None
                    if final_realized.boundary_measure is None
                    else final_realized.boundary_measure.copy()
                ),
                toggle_count=toggle_count.copy(),
                realized_toggle_count=realized_toggle_count.copy(),
                first_realized_iter=first_realized_iter.copy(),
                last_realized_iter=last_realized_iter.copy(),
                marginal=np.zeros(m, dtype=bool),
                status=tuple(termination for _ in range(m)),
            )
            connectivity = None
            if connectivity_check != 'none':
                connectivity = _build_active_set_connectivity_diagnostics(
                    resolved,
                    active,
                    model=model,
                    gauge_policy=_self_consistent_gauge_policy_description(),
                )
                _apply_connectivity_policy(
                    connectivity_check,
                    connectivity,
                    warnings_list,
                )
            return SelfConsistentPowerFitResult(
                constraints=resolved,
                fit=fit,
                realized=final_realized,
                diagnostics=diag_all,
                active_mask=active.copy(),
                n_outer_iter=outer_iter,
                converged=False,
                termination=termination,
                cycle_length=None,
                marginal_constraints=tuple(),
                rms_residual_all=float('nan'),
                max_residual_all=float('nan'),
                tessellation_diagnostics=None,
                history=tuple(history_rows) if return_history else None,
                warnings=tuple(warnings_list),
                connectivity=connectivity,
            )

        weights_exact = fit.weights.copy()
        if prev_weights_eval is not None:
            weights_exact = _align_weights_to_reference(
                weights_exact,
                prev_weights_eval,
                _active_alignment_components(active_constraints, model),
            )
            weights_eval = (
                (1.0 - float(options.relax)) * prev_weights_eval
                + float(options.relax) * weights_exact
            )
            step_norm = float(np.linalg.norm(weights_eval - prev_weights_eval))
        else:
            weights_eval = weights_exact
            step_norm = 0.0

        radii_eval, _ = weights_to_radii(
            weights_eval,
            r_min=r_min,
            weight_shift=weight_shift,
        )
        diag = match_realized_pairs(
            pts,
            domain=domain,
            radii=radii_eval,
            constraints=resolved,
            return_boundary_measure=False,
            return_cells=False,
            return_tessellation_diagnostics=False,
            tessellation_check='none',
            unaccounted_pair_check='none',
        )
        last_diag = diag
        realized_same = diag.realized_same_shift
        if prev_realized_same is not None:
            realized_toggle_count += prev_realized_same != realized_same
        newly_realized = realized_same & (first_realized_iter < 0)
        first_realized_iter[newly_realized] = outer_iter
        last_realized_iter[realized_same] = outer_iter

        new_active = active.copy()
        for k in range(m):
            if realized_same[k]:
                add_streak[k] += 1
                drop_streak[k] = 0
            else:
                drop_streak[k] += 1
                add_streak[k] = 0

            if active[k]:
                if drop_streak[k] >= options.drop_after:
                    new_active[k] = False
            else:
                if add_streak[k] >= options.add_after:
                    new_active[k] = True

        toggled = new_active != active
        toggle_count += toggled
        n_added = int(np.count_nonzero((~active) & new_active))
        n_removed = int(np.count_nonzero(active & (~new_active)))

        pred_fraction, pred_position, pred = _predict_measurements(
            weights_eval,
            resolved,
        )
        target = (
            resolved.target_fraction
            if resolved.measurement == 'fraction'
            else resolved.target_position
        )
        residuals = pred - target
        history_rows.append(
            ActiveSetIteration(
                iteration=outer_iter,
                n_active=int(np.count_nonzero(new_active)),
                n_realized=int(np.count_nonzero(realized_same)),
                n_added=n_added,
                n_removed=n_removed,
                rms_residual_all=float(np.sqrt(np.mean(residuals * residuals)))
                if residuals.size
                else 0.0,
                max_residual_all=float(np.max(np.abs(residuals)))
                if residuals.size
                else 0.0,
                weight_step_norm=step_norm,
            )
        )

        if (
            np.array_equal(new_active, active)
            and np.array_equal(realized_same, active)
            and step_norm <= float(options.weight_step_tol)
        ):
            active = new_active
            prev_weights_eval = weights_eval
            prev_realized_same = realized_same.copy()
            termination = 'self_consistent'
            converged = True
            break

        active_key = new_active.tobytes()
        if np.any(toggled):
            if (
                active_key in seen_masks
                and outer_iter - seen_masks[active_key] <= options.cycle_window
            ):
                cycle_length = outer_iter - seen_masks[active_key]
                active = new_active
                prev_weights_eval = weights_eval
                prev_realized_same = realized_same.copy()
                termination = 'cycle_detected'
                converged = False
                break
            seen_masks[active_key] = outer_iter

        active = new_active
        prev_weights_eval = weights_eval
        prev_realized_same = realized_same.copy()
    else:
        termination = 'max_outer_iter'

    active_constraints = resolved.subset(active)
    final_fit = fit_power_weights(
        pts,
        active_constraints,
        model=model,
        r_min=r_min,
        weight_shift=weight_shift,
        solver=fit_solver,
        max_iter=fit_max_iter,
        rho=fit_rho,
        tol_abs=fit_tol_abs,
        tol_rel=fit_tol_rel,
        connectivity_check='diagnose',
    )
    warnings_list.extend(final_fit.warnings)

    if final_fit.status == 'numerical_failure':
        termination = 'numerical_failure'
        converged = False

    if final_fit.weights is not None:
        final_weights = final_fit.weights.copy()
        if prev_weights_eval is not None:
            final_weights = _align_weights_to_reference(
                final_weights,
                prev_weights_eval,
                _active_alignment_components(active_constraints, model),
            )
        final_fit = _rebuild_fit_with_weights(
            final_fit,
            active_constraints,
            final_weights,
            r_min=r_min,
            weight_shift=weight_shift,
        )
        final_realized = match_realized_pairs(
            pts,
            domain=domain,
            radii=final_fit.radii,
            constraints=resolved,
            return_boundary_measure=return_boundary_measure,
            return_cells=return_cells,
            return_tessellation_diagnostics=return_tessellation_diagnostics,
            tessellation_check=tessellation_check,
            unaccounted_pair_check=unaccounted_pair_check,
        )
        warnings_list.extend(final_realized.warnings)
        pred_fraction, pred_position, pred = _predict_measurements(
            final_fit.weights,
            resolved,
        )
    else:
        final_realized = last_diag
        pred_fraction = np.full(m, np.nan, dtype=np.float64)
        pred_position = np.full(m, np.nan, dtype=np.float64)
        pred = np.full(m, np.nan, dtype=np.float64)
        if final_realized is None:
            final_realized = _empty_realized_pair_diagnostics(
                m,
                return_boundary_measure=return_boundary_measure,
            )

    target = (
        resolved.target_fraction
        if resolved.measurement == 'fraction'
        else resolved.target_position
    )
    residuals = pred - target
    rms_residual_all = (
        float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0
    )
    max_residual_all = float(np.max(np.abs(residuals))) if residuals.size else 0.0

    marginal = (toggle_count > 0) | final_realized.realized_other_shift
    if termination == 'cycle_detected':
        marginal = marginal | (realized_toggle_count > 0)
    marginal_constraints = tuple(np.flatnonzero(marginal).tolist())
    status = _build_constraint_statuses(
        active=active,
        realized=final_realized,
        toggle_count=toggle_count,
        realized_toggle_count=realized_toggle_count,
        termination=termination,
    )

    diag_all = PairConstraintDiagnostics(
        site_i=resolved.i.copy(),
        site_j=resolved.j.copy(),
        shift=resolved.shifts.copy(),
        target=resolved.target.copy(),
        confidence=resolved.confidence.copy(),
        predicted=pred,
        predicted_fraction=pred_fraction,
        predicted_position=pred_position,
        residuals=residuals,
        active=active.copy(),
        realized=final_realized.realized.copy(),
        realized_same_shift=final_realized.realized_same_shift.copy(),
        realized_other_shift=final_realized.realized_other_shift.copy(),
        realized_shifts=final_realized.realized_shifts,
        endpoint_i_empty=final_realized.endpoint_i_empty.copy(),
        endpoint_j_empty=final_realized.endpoint_j_empty.copy(),
        boundary_measure=(
            None
            if final_realized.boundary_measure is None
            else final_realized.boundary_measure.copy()
        ),
        toggle_count=toggle_count.copy(),
        realized_toggle_count=realized_toggle_count.copy(),
        first_realized_iter=first_realized_iter.copy(),
        last_realized_iter=last_realized_iter.copy(),
        marginal=marginal.copy(),
        status=status,
    )

    connectivity = None
    if connectivity_check != 'none':
        connectivity = _build_active_set_connectivity_diagnostics(
            resolved,
            active,
            model=model,
            gauge_policy=_self_consistent_gauge_policy_description(),
        )
        _apply_connectivity_policy(
            connectivity_check,
            connectivity,
            warnings_list,
        )

    return SelfConsistentPowerFitResult(
        constraints=resolved,
        fit=final_fit,
        realized=final_realized,
        diagnostics=diag_all,
        active_mask=active.copy(),
        n_outer_iter=len(history_rows),
        converged=converged,
        termination=termination,
        cycle_length=cycle_length,
        marginal_constraints=marginal_constraints,
        rms_residual_all=rms_residual_all,
        max_residual_all=max_residual_all,
        tessellation_diagnostics=final_realized.tessellation_diagnostics,
        history=tuple(history_rows) if return_history else None,
        warnings=tuple(warnings_list),
        connectivity=connectivity,
    )


def _align_weights_to_reference(
    weights: np.ndarray, reference: np.ndarray, comps: list[list[int]]
) -> np.ndarray:
    aligned = np.asarray(weights, dtype=np.float64).copy()
    ref = np.asarray(reference, dtype=np.float64)
    if aligned.shape != ref.shape:
        raise ValueError('weights and reference must have the same shape')
    for comp in comps:
        idx = np.asarray(comp, dtype=np.int64)
        if idx.size == 0:
            continue
        shift = float(np.mean(aligned[idx] - ref[idx]))
        aligned[idx] -= shift
    return aligned


def _active_alignment_components(
    constraints: PairBisectorConstraints,
    model: FitModel,
) -> list[list[int]]:
    effective_mask = _difference_identifying_mask(constraints, model)
    return _connected_components(
        constraints.n_points,
        constraints.i[effective_mask],
        constraints.j[effective_mask],
    )


def _self_consistent_gauge_policy_description() -> str:
    return (
        'each connected active effective component is aligned to the previous '
        'iterate; the first iterate falls back to the standalone component-mean '
        'gauge'
    )


def _rebuild_fit_with_weights(
    fit: PowerWeightFitResult,
    constraints: PairBisectorConstraints,
    weights: np.ndarray,
    *,
    r_min: float,
    weight_shift: float | None,
) -> PowerWeightFitResult:
    radii, shift = weights_to_radii(
        weights,
        r_min=r_min,
        weight_shift=weight_shift,
    )
    pred_fraction, pred_position, pred = _predict_measurements(weights, constraints)
    target = (
        constraints.target_fraction
        if constraints.measurement == 'fraction'
        else constraints.target_position
    )
    residuals = pred - target
    rms = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0
    mx = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    return replace(
        fit,
        weights=np.asarray(weights, dtype=np.float64).copy(),
        radii=radii,
        weight_shift=shift,
        predicted=pred,
        predicted_fraction=pred_fraction,
        predicted_position=pred_position,
        residuals=residuals,
        rms_residual=rms,
        max_residual=mx,
    )


def _empty_realized_pair_diagnostics(
    m: int, *, return_boundary_measure: bool
) -> RealizedPairDiagnostics:
    return RealizedPairDiagnostics(
        realized=np.zeros(m, dtype=bool),
        unrealized=tuple(range(m)),
        realized_same_shift=np.zeros(m, dtype=bool),
        realized_other_shift=np.zeros(m, dtype=bool),
        realized_shifts=tuple(() for _ in range(m)),
        endpoint_i_empty=np.zeros(m, dtype=bool),
        endpoint_j_empty=np.zeros(m, dtype=bool),
        boundary_measure=(
            np.full(m, np.nan, dtype=np.float64) if return_boundary_measure else None
        ),
        cells=None,
        tessellation_diagnostics=None,
        unaccounted_pairs=tuple(),
        warnings=tuple(),
    )


def _build_constraint_statuses(
    *,
    active: np.ndarray,
    realized: RealizedPairDiagnostics,
    toggle_count: np.ndarray,
    realized_toggle_count: np.ndarray,
    termination: str,
) -> tuple[str, ...]:
    rows: list[str] = []
    for k in range(active.shape[0]):
        if termination == 'numerical_failure':
            rows.append('numerical_failure')
            continue
        if termination == 'cycle_detected' and (
            bool(toggle_count[k] > 0) or bool(realized_toggle_count[k] > 0)
        ):
            rows.append('cycle_member')
            continue
        if bool(realized.realized_other_shift[k]):
            rows.append('realized_other_shift')
            continue
        if bool(realized.endpoint_i_empty[k] or realized.endpoint_j_empty[k]):
            rows.append('endpoint_empty')
            continue
        if bool(active[k]) and bool(realized.realized_same_shift[k]):
            rows.append(
                'toggled_active'
                if bool(toggle_count[k] > 0)
                else 'stable_active'
            )
            continue
        if (not bool(active[k])) and (not bool(realized.realized[k])):
            rows.append(
                'toggled_inactive'
                if bool(toggle_count[k] > 0)
                else 'stable_inactive'
            )
            continue
        if bool(active[k]) and (not bool(realized.realized_same_shift[k])):
            rows.append('active_unrealized')
            continue
        if (not bool(active[k])) and bool(realized.realized_same_shift[k]):
            rows.append('inactive_realized')
            continue
        rows.append('unresolved')
    return tuple(rows)
