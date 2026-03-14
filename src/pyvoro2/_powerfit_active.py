"""Self-consistent active-set refinement for pairwise separator constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._powerfit_constraints import PairBisectorConstraints, resolve_pair_bisector_constraints
from ._powerfit_model import FitModel
from ._powerfit_realize import RealizedPairDiagnostics, match_realized_pairs
from ._powerfit_solver import (
    PowerWeightFitResult,
    _connected_components,
    _predict_measurements,
    fit_power_weights,
    weights_to_radii,
)
from .domains import Box, OrthorhombicCell, PeriodicCell


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
    predicted: np.ndarray
    predicted_fraction: np.ndarray
    predicted_position: np.ndarray
    residuals: np.ndarray
    active: np.ndarray
    realized: np.ndarray
    realized_same_shift: np.ndarray
    realized_other_shift: np.ndarray
    toggle_count: np.ndarray
    realized_toggle_count: np.ndarray
    first_realized_iter: np.ndarray
    last_realized_iter: np.ndarray
    marginal: np.ndarray


@dataclass(frozen=True, slots=True)
class SelfConsistentPowerFitResult:
    fit: PowerWeightFitResult
    realized: RealizedPairDiagnostics
    diagnostics: PairConstraintDiagnostics
    active_mask: np.ndarray
    n_outer_iter: int
    converged: bool
    termination: Literal[
        'self_consistent', 'cycle_detected', 'max_outer_iter', 'infeasible_active_set'
    ]
    cycle_length: int | None
    marginal_constraints: tuple[int, ...]
    history: tuple[ActiveSetIteration, ...] | None
    warnings: tuple[str, ...]



def solve_self_consistent_power_weights(
    points: np.ndarray,
    constraints: PairBisectorConstraints | list[tuple] | tuple[tuple, ...],
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: Box | OrthorhombicCell | PeriodicCell,
    ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: list[float] | tuple[float, ...] | np.ndarray | None = None,
    model: FitModel | None = None,
    active0: np.ndarray | None = None,
    options: ActiveSetOptions = ActiveSetOptions(),
    r_min: float = 0.0,
    fit_solver: Literal['auto', 'analytic', 'admm'] = 'auto',
    fit_max_iter: int = 2000,
    fit_rho: float = 1.0,
    fit_tol_abs: float = 1e-6,
    fit_tol_rel: float = 1e-5,
    return_history: bool = False,
    return_cells: bool = False,
    return_boundary_measure: bool = False,
) -> SelfConsistentPowerFitResult:
    """Iteratively refine an active pair set against realized power-diagram faces."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')

    if model is None:
        model = FitModel()

    if isinstance(constraints, PairBisectorConstraints):
        resolved = constraints
        if resolved.n_points != pts.shape[0]:
            raise ValueError('resolved constraints do not match the number of points')
    else:
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
    comps = _connected_components(resolved.n_points, resolved.i, resolved.j)

    termination: Literal[
        'self_consistent', 'cycle_detected', 'max_outer_iter', 'infeasible_active_set'
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
            solver=fit_solver,
            max_iter=fit_max_iter,
            rho=fit_rho,
            tol_abs=fit_tol_abs,
            tol_rel=fit_tol_rel,
        )
        if fit.weights is None:
            warnings_list.extend(fit.warnings)
            termination = 'infeasible_active_set'
            final_realized = RealizedPairDiagnostics(
                realized=np.zeros(m, dtype=bool),
                unrealized=tuple(range(m)),
                realized_same_shift=np.zeros(m, dtype=bool),
                realized_other_shift=np.zeros(m, dtype=bool),
                realized_shifts=tuple(() for _ in range(m)),
                endpoint_i_empty=np.zeros(m, dtype=bool),
                endpoint_j_empty=np.zeros(m, dtype=bool),
                boundary_measure=(
                    np.full(m, np.nan, dtype=np.float64)
                    if return_boundary_measure
                    else None
                ),
                cells=None,
            )
            diag_all = PairConstraintDiagnostics(
                predicted=np.full(m, np.nan, dtype=np.float64),
                predicted_fraction=np.full(m, np.nan, dtype=np.float64),
                predicted_position=np.full(m, np.nan, dtype=np.float64),
                residuals=np.full(m, np.nan, dtype=np.float64),
                active=active.copy(),
                realized=final_realized.realized.copy(),
                realized_same_shift=final_realized.realized_same_shift.copy(),
                realized_other_shift=final_realized.realized_other_shift.copy(),
                toggle_count=toggle_count.copy(),
                realized_toggle_count=realized_toggle_count.copy(),
                first_realized_iter=first_realized_iter.copy(),
                last_realized_iter=last_realized_iter.copy(),
                marginal=np.zeros(m, dtype=bool),
            )
            return SelfConsistentPowerFitResult(
                fit=fit,
                realized=final_realized,
                diagnostics=diag_all,
                active_mask=active.copy(),
                n_outer_iter=outer_iter,
                converged=False,
                termination=termination,
                cycle_length=None,
                marginal_constraints=tuple(),
                history=tuple(history_rows) if return_history else None,
                warnings=tuple(warnings_list),
            )

        weights_exact = fit.weights.copy()
        if prev_weights_eval is not None:
            weights_exact = _align_weights_to_reference(weights_exact, prev_weights_eval, comps)
            weights_eval = (
                (1.0 - float(options.relax)) * prev_weights_eval
                + float(options.relax) * weights_exact
            )
            step_norm = float(np.linalg.norm(weights_eval - prev_weights_eval))
        else:
            weights_eval = weights_exact
            step_norm = 0.0

        radii_eval, _ = weights_to_radii(weights_eval, r_min=r_min)
        diag = match_realized_pairs(
            pts,
            domain=domain,
            radii=radii_eval,
            constraints=resolved,
            return_boundary_measure=False,
            return_cells=False,
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

        pred_fraction, pred_position, pred = _predict_measurements(weights_eval, resolved)
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
            if active_key in seen_masks and outer_iter - seen_masks[active_key] <= options.cycle_window:
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
        solver=fit_solver,
        max_iter=fit_max_iter,
        rho=fit_rho,
        tol_abs=fit_tol_abs,
        tol_rel=fit_tol_rel,
    )
    warnings_list.extend(final_fit.warnings)

    if final_fit.weights is not None and final_fit.radii is not None:
        final_realized = match_realized_pairs(
            pts,
            domain=domain,
            radii=final_fit.radii,
            constraints=resolved,
            return_boundary_measure=return_boundary_measure,
            return_cells=return_cells,
        )
        pred_fraction, pred_position, pred = _predict_measurements(final_fit.weights, resolved)
    else:
        final_realized = last_diag
        pred_fraction = np.full(m, np.nan, dtype=np.float64)
        pred_position = np.full(m, np.nan, dtype=np.float64)
        pred = np.full(m, np.nan, dtype=np.float64)
        if final_realized is None:
            final_realized = RealizedPairDiagnostics(
                realized=np.zeros(m, dtype=bool),
                unrealized=tuple(range(m)),
                realized_same_shift=np.zeros(m, dtype=bool),
                realized_other_shift=np.zeros(m, dtype=bool),
                realized_shifts=tuple(() for _ in range(m)),
                endpoint_i_empty=np.zeros(m, dtype=bool),
                endpoint_j_empty=np.zeros(m, dtype=bool),
                boundary_measure=(
                    np.full(m, np.nan, dtype=np.float64)
                    if return_boundary_measure
                    else None
                ),
                cells=None,
            )

    target = (
        resolved.target_fraction
        if resolved.measurement == 'fraction'
        else resolved.target_position
    )
    residuals = pred - target
    marginal = (toggle_count > 0) | final_realized.realized_other_shift
    if termination == 'cycle_detected':
        marginal = marginal | (realized_toggle_count > 0)
    marginal_constraints = tuple(np.flatnonzero(marginal).tolist())

    diag_all = PairConstraintDiagnostics(
        predicted=pred,
        predicted_fraction=pred_fraction,
        predicted_position=pred_position,
        residuals=residuals,
        active=active.copy(),
        realized=final_realized.realized.copy(),
        realized_same_shift=final_realized.realized_same_shift.copy(),
        realized_other_shift=final_realized.realized_other_shift.copy(),
        toggle_count=toggle_count.copy(),
        realized_toggle_count=realized_toggle_count.copy(),
        first_realized_iter=first_realized_iter.copy(),
        last_realized_iter=last_realized_iter.copy(),
        marginal=marginal.copy(),
    )

    return SelfConsistentPowerFitResult(
        fit=final_fit,
        realized=final_realized,
        diagnostics=diag_all,
        active_mask=active.copy(),
        n_outer_iter=len(history_rows),
        converged=converged,
        termination=termination,
        cycle_length=cycle_length,
        marginal_constraints=marginal_constraints,
        history=tuple(history_rows) if return_history else None,
        warnings=tuple(warnings_list),
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
