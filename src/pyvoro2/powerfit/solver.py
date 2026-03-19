"""Low-level inverse solver for fitting power weights from pairwise constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .constraints import PairBisectorConstraints, resolve_pair_bisector_constraints
from .model import (
    ExponentialBoundaryPenalty,
    FitModel,
    FixedValue,
    HardConstraint,
    HuberLoss,
    Interval,
    L2Regularization,
    ReciprocalBoundaryPenalty,
    SoftIntervalPenalty,
    SquaredLoss,
)
from ..domains import Box, OrthorhombicCell, PeriodicCell


def _plain_value(value: object) -> object:
    return value.item() if hasattr(value, 'item') else value


@dataclass(frozen=True, slots=True)
class ConstraintGraphDiagnostics:
    """Connectivity summary for a graph induced by constraint rows."""

    n_points: int
    n_constraints: int
    n_edges: int
    isolated_points: tuple[int, ...]
    connected_components: tuple[tuple[int, ...], ...]
    fully_connected: bool

    @property
    def n_components(self) -> int:
        """Return the number of connected components."""

        return int(len(self.connected_components))


@dataclass(frozen=True, slots=True)
class ConnectivityDiagnostics:
    """Structured connectivity diagnostics for the inverse-fit graph."""

    unconstrained_points: tuple[int, ...]
    candidate_graph: ConstraintGraphDiagnostics
    effective_graph: ConstraintGraphDiagnostics
    active_graph: ConstraintGraphDiagnostics | None = None
    active_effective_graph: ConstraintGraphDiagnostics | None = None
    candidate_offsets_identified_by_data: bool = False
    active_offsets_identified_by_data: bool | None = None
    offsets_identified_in_objective: bool = False
    gauge_policy: str = ''
    messages: tuple[str, ...] = ()


class ConnectivityDiagnosticsError(ValueError):
    """Raised when connectivity_check='raise' detects a graph issue."""

    def __init__(
        self,
        message: str,
        diagnostics: ConnectivityDiagnostics,
    ) -> None:
        super().__init__(message, diagnostics)
        self.diagnostics = diagnostics

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True, slots=True)
class AlgebraicEdgeDiagnostics:
    """Edge-space diagnostics matching the paper-side algebraic formulas."""

    alpha: np.ndarray
    beta: np.ndarray
    z_obs: np.ndarray
    z_fit: np.ndarray | None
    residual: np.ndarray | None
    edge_weight: np.ndarray
    weighted_l2: float | None
    weighted_rmse: float | None
    rmse: float | None
    mae: float | None


@dataclass(frozen=True, slots=True)
class PowerWeightFitResult:
    """Result of inverse fitting of power weights."""

    status: Literal[
        'optimal', 'infeasible_hard_constraints', 'max_iter', 'numerical_failure'
    ]
    hard_feasible: bool
    weights: np.ndarray | None
    radii: np.ndarray | None
    weight_shift: float | None
    measurement: Literal['fraction', 'position']
    target: np.ndarray
    predicted: np.ndarray | None
    predicted_fraction: np.ndarray | None
    predicted_position: np.ndarray | None
    residuals: np.ndarray | None
    rms_residual: float | None
    max_residual: float | None
    used_shifts: np.ndarray
    solver: str
    n_iter: int
    converged: bool
    conflict: 'HardConstraintConflict | None'
    warnings: tuple[str, ...]
    connectivity: ConnectivityDiagnostics | None = None
    edge_diagnostics: AlgebraicEdgeDiagnostics | None = None

    @property
    def is_optimal(self) -> bool:
        """Whether the fit terminated with a final solution."""

        return self.status == 'optimal'

    @property
    def is_infeasible(self) -> bool:
        """Whether hard feasibility failed before optimization."""

        return self.status == 'infeasible_hard_constraints'

    @property
    def conflicting_constraint_indices(self) -> tuple[int, ...]:
        """Constraint rows participating in the infeasibility witness."""

        if self.conflict is None:
            return tuple()
        return self.conflict.constraint_indices

    def to_records(
        self,
        constraints: PairBisectorConstraints,
        *,
        use_ids: bool = False,
    ) -> tuple[dict[str, object], ...]:
        """Return one plain-Python record per fitted constraint row."""

        if constraints.n_constraints != int(self.target.shape[0]):
            raise ValueError('constraints do not match the fit result length')
        left, right = constraints.pair_labels(use_ids=use_ids)
        edge_diag = _edge_diagnostics_for_result(self, constraints)
        rows: list[dict[str, object]] = []
        left_is_int = np.issubdtype(np.asarray(left).dtype, np.integer)
        right_is_int = np.issubdtype(np.asarray(right).dtype, np.integer)
        for k in range(constraints.n_constraints):
            site_i = int(left[k]) if left_is_int else _plain_value(left[k])
            site_j = int(right[k]) if right_is_int else _plain_value(right[k])
            rows.append(
                {
                    'constraint_index': int(k),
                    'site_i': site_i,
                    'site_j': site_j,
                    'shift': tuple(int(v) for v in constraints.shifts[k]),
                    'measurement': self.measurement,
                    'target': float(self.target[k]),
                    'predicted': (
                        None
                        if self.predicted is None
                        else float(self.predicted[k])
                    ),
                    'predicted_fraction': (
                        None
                        if self.predicted_fraction is None
                        else float(self.predicted_fraction[k])
                    ),
                    'predicted_position': (
                        None
                        if self.predicted_position is None
                        else float(self.predicted_position[k])
                    ),
                    'residual': (
                        None
                        if self.residuals is None
                        else float(self.residuals[k])
                    ),
                    'alpha': float(edge_diag.alpha[k]),
                    'beta': float(edge_diag.beta[k]),
                    'z_obs': float(edge_diag.z_obs[k]),
                    'z_fit': (
                        None
                        if edge_diag.z_fit is None
                        else float(edge_diag.z_fit[k])
                    ),
                    'algebraic_residual': (
                        None
                        if edge_diag.residual is None
                        else float(edge_diag.residual[k])
                    ),
                    'edge_weight': float(edge_diag.edge_weight[k]),
                }
            )
        return tuple(rows)

    def to_report(
        self,
        constraints: PairBisectorConstraints,
        *,
        use_ids: bool = False,
    ) -> dict[str, object]:
        """Return a JSON-friendly report for this fit result."""

        from .report import build_fit_report

        return build_fit_report(self, constraints, use_ids=use_ids)


@dataclass(frozen=True, slots=True)
class HardConstraintConflictTerm:
    """One bound relation participating in an infeasibility witness.

    Each term refers back to one input constraint row and states which bound on
    ``w_i - w_j`` participates in the contradiction cycle.
    """

    constraint_index: int
    site_i: int
    site_j: int
    relation: Literal['<=', '>=']
    bound_value: float

    def to_record(self, *, ids: np.ndarray | None = None) -> dict[str, object]:
        """Return a plain-Python record for this conflict term."""

        site_i = int(self.site_i) if ids is None else ids[self.site_i].item()
        site_j = int(self.site_j) if ids is None else ids[self.site_j].item()
        return {
            'constraint_index': int(self.constraint_index),
            'site_i': site_i,
            'site_j': site_j,
            'relation': self.relation,
            'bound_value': float(self.bound_value),
        }


@dataclass(frozen=True, slots=True)
class HardConstraintConflict:
    """Compact witness for inconsistent hard separator restrictions."""

    component_nodes: tuple[int, ...]
    cycle_nodes: tuple[int, ...]
    terms: tuple[HardConstraintConflictTerm, ...]
    message: str

    @property
    def constraint_indices(self) -> tuple[int, ...]:
        """Sorted unique input rows participating in the conflict."""

        return tuple(sorted({int(term.constraint_index) for term in self.terms}))

    def to_records(
        self, *, ids: np.ndarray | None = None
    ) -> tuple[dict[str, object], ...]:
        """Return plain-Python records for the witness terms."""

        return tuple(term.to_record(ids=ids) for term in self.terms)


@dataclass(frozen=True, slots=True)
class _DifferenceEdge:
    source: int
    target: int
    weight: float
    constraint_index: int
    site_i: int
    site_j: int
    relation: Literal['<=', '>=']
    bound_value: float


@dataclass(frozen=True, slots=True)
class _MeasurementGeometry:
    alpha: np.ndarray
    beta: np.ndarray
    target: np.ndarray
    target_fraction: np.ndarray
    target_position: np.ndarray


class _NumericalFailure(RuntimeError):
    """Raised when the numerical backend fails before producing a result."""


def radii_to_weights(radii: np.ndarray) -> np.ndarray:
    """Convert radii to power weights (``w = r^2``)."""

    r = np.asarray(radii, dtype=float)
    if r.ndim != 1:
        raise ValueError('radii must be 1D')
    if not np.all(np.isfinite(r)):
        raise ValueError('radii must contain only finite values')
    if np.any(r < 0):
        raise ValueError('radii must be non-negative')
    return r * r


def weights_to_radii(
    weights: np.ndarray,
    *,
    r_min: float = 0.0,
    weight_shift: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convert power weights to radii using an explicit global shift.

    By default, the returned radii use the minimal additive shift that makes the
    smallest radius equal to ``r_min``. Pass ``weight_shift`` to request an
    explicit gauge instead.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError('weights must be 1D')
    if not np.all(np.isfinite(w)):
        raise ValueError('weights must contain only finite values')

    if weight_shift is not None:
        if r_min != 0.0:
            raise ValueError('specify at most one of r_min and weight_shift')
        C = float(weight_shift)
        if not np.isfinite(C):
            raise ValueError('weight_shift must be finite')
    else:
        r_min = float(r_min)
        if r_min < 0:
            raise ValueError('r_min must be >= 0')
        w_min = float(np.min(w)) if w.size else 0.0
        C = (r_min * r_min) - w_min

    w_shifted = w + C
    if np.any(w_shifted < -1e-14):
        raise ValueError('weight shift produced negative values (numerical issue)')
    w_shifted = np.maximum(w_shifted, 0.0)
    return np.sqrt(w_shifted), float(C)


def fit_power_weights(
    points: np.ndarray,
    constraints: PairBisectorConstraints | list[tuple] | tuple[tuple, ...],
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: Box | OrthorhombicCell | PeriodicCell | None = None,
    ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: list[float] | tuple[float, ...] | np.ndarray | None = None,
    model: FitModel | None = None,
    r_min: float = 0.0,
    weight_shift: float | None = None,
    solver: Literal['auto', 'analytic', 'admm'] = 'auto',
    max_iter: int = 2000,
    rho: float = 1.0,
    tol_abs: float = 1e-6,
    tol_rel: float = 1e-5,
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
) -> PowerWeightFitResult:
    """Fit power weights from resolved pairwise separator constraints.

    The raw constraint tuples are ``(i, j, value[, shift])`` where ``shift`` is
    the integer lattice image applied to site ``j``. The returned radii use the
    minimal non-negative global gauge by default; pass ``weight_shift`` for an
    explicit output gauge or ``r_min`` for the legacy minimum-radius helper.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] <= 0:
        raise ValueError('points must have shape (n, d) with d >= 1')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')

    if model is None:
        model = FitModel()

    if isinstance(constraints, PairBisectorConstraints):
        resolved = constraints
        if resolved.n_points != pts.shape[0]:
            raise ValueError('resolved constraints do not match the number of points')
        if resolved.dim != pts.shape[1]:
            raise ValueError(
                'resolved constraints do not match the point dimension'
            )
        if resolved.measurement != measurement:
            measurement = resolved.measurement
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
        measurement = resolved.measurement

    return _fit_power_weights_resolved(
        resolved,
        model=model,
        r_min=r_min,
        weight_shift=weight_shift,
        solver=solver,
        max_iter=max_iter,
        rho=rho,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        connectivity_check=connectivity_check,
    )


def _fit_power_weights_resolved(
    constraints: PairBisectorConstraints,
    *,
    model: FitModel,
    r_min: float,
    weight_shift: float | None,
    solver: Literal['auto', 'analytic', 'admm'],
    max_iter: int,
    rho: float,
    tol_abs: float,
    tol_rel: float,
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'],
) -> PowerWeightFitResult:
    n = int(constraints.n_points)
    m = int(constraints.n_constraints)
    warnings_list = list(constraints.warnings)

    if max_iter <= 0:
        raise ValueError('max_iter must be > 0')
    if rho <= 0:
        raise ValueError('rho must be > 0')
    if tol_abs <= 0 or tol_rel <= 0:
        raise ValueError('tol_abs and tol_rel must be > 0')
    if connectivity_check not in ('none', 'diagnose', 'warn', 'raise'):
        raise ValueError(
            'connectivity_check must be none, diagnose, warn, or raise'
        )

    reg = model.regularization
    lam = float(reg.strength)
    w0 = _regularization_reference(reg, n)
    reference = None if reg.reference is None else w0

    geom = _measurement_geometry(constraints)
    z_target = (geom.target - geom.beta) / geom.alpha
    a = constraints.confidence * (geom.alpha**2)

    hard = _hard_constraint_bounds(model.feasible, geom.alpha, geom.beta)
    z_lo = hard[0] if hard is not None else None
    z_hi = hard[1] if hard is not None else None
    hard_measurement = _hard_constraint_measurement_bounds(
        model.feasible,
        constraints.n_constraints,
    )
    y_lo = hard_measurement[0] if hard_measurement is not None else None
    y_hi = hard_measurement[1] if hard_measurement is not None else None

    nonquadratic = _requires_admm(model)
    if solver == 'auto':
        solver_eff = 'analytic' if not nonquadratic else 'admm'
    else:
        solver_eff = solver
    if solver_eff not in ('analytic', 'admm'):
        raise ValueError('solver must be auto, analytic, or admm')
    if solver_eff == 'analytic' and nonquadratic:
        raise ValueError(
            'analytic solver cannot be used with hard constraints '
            'or non-quadratic penalties'
        )

    effective_mask = _difference_identifying_mask(constraints, model)
    comps = _connected_components(
        n,
        constraints.i[effective_mask],
        constraints.j[effective_mask],
    )
    connectivity = None
    if connectivity_check != 'none':
        connectivity = _build_fit_connectivity_diagnostics(
            constraints,
            model=model,
            gauge_policy=_standalone_gauge_policy_description(reg),
        )
        _apply_connectivity_policy(
            connectivity_check,
            connectivity,
            warnings_list,
        )

    if hard is not None:
        feasible, conflict = _check_hard_feasibility(
            n,
            constraints.i,
            constraints.j,
            z_lo,
            z_hi,
        )
        if not feasible:
            warnings_list.append('hard feasibility check failed before optimization')
            if conflict is not None:
                warnings_list.append(conflict.message)
            return PowerWeightFitResult(
                status='infeasible_hard_constraints',
                hard_feasible=False,
                weights=None,
                radii=None,
                weight_shift=None,
                measurement=constraints.measurement,
                target=geom.target.copy(),
                predicted=None,
                predicted_fraction=None,
                predicted_position=None,
                residuals=None,
                rms_residual=None,
                max_residual=None,
                used_shifts=constraints.shifts.copy(),
                solver='none',
                n_iter=0,
                converged=False,
                conflict=conflict,
                warnings=tuple(warnings_list),
                connectivity=connectivity,
                edge_diagnostics=_compute_edge_diagnostics(
                    constraints,
                    weights=None,
                    geom=geom,
                ),
            )
    else:
        conflict = None

    if m == 0:
        if lam > 0.0:
            weights = w0.copy()
            warnings_list.append(
                'empty constraint set; using the regularization-only solution'
            )
        elif reference is not None:
            weights = reference.copy()
            warnings_list.append(
                'empty constraint set; no pair data are present, so weights '
                'follow the zero-strength reference gauge convention'
            )
        else:
            weights = np.zeros(n, dtype=np.float64)
            warnings_list.append(
                'empty constraint set; returning the mean-zero gauge solution'
            )
        radii, shift = weights_to_radii(
            weights,
            r_min=r_min,
            weight_shift=weight_shift,
        )
        pred_fraction = np.zeros(0, dtype=np.float64)
        pred_position = np.zeros(0, dtype=np.float64)
        pred = pred_fraction if constraints.measurement == 'fraction' else pred_position
        return PowerWeightFitResult(
            status='optimal',
            hard_feasible=True,
            weights=weights,
            radii=radii,
            weight_shift=shift,
            measurement=constraints.measurement,
            target=geom.target.copy(),
            predicted=pred,
            predicted_fraction=pred_fraction,
            predicted_position=pred_position,
            residuals=np.zeros(0, dtype=np.float64),
            rms_residual=0.0,
            max_residual=0.0,
            used_shifts=constraints.shifts.copy(),
            solver='analytic',
            n_iter=0,
            converged=True,
            conflict=conflict,
            warnings=tuple(warnings_list),
            connectivity=connectivity,
            edge_diagnostics=_compute_edge_diagnostics(
                constraints,
                weights=weights,
                geom=geom,
            ),
        )

    weights = np.zeros(n, dtype=np.float64)
    converged_all = True
    n_iter_max = 0

    try:
        for nodes in comps:
            idx_nodes = np.asarray(nodes, dtype=np.int64)
            if idx_nodes.size <= 1:
                if lam > 0.0 and idx_nodes.size == 1:
                    weights[idx_nodes[0]] = w0[idx_nodes[0]]
                continue

            node_set = set(nodes)
            mask = effective_mask & np.fromiter(
                (
                    (int(i) in node_set) and (int(j) in node_set)
                    for i, j in zip(constraints.i, constraints.j)
                ),
                dtype=bool,
                count=m,
            )
            local_index = {int(node): k for k, node in enumerate(nodes)}
            ii = np.array(
                [local_index[int(i)] for i in constraints.i[mask]],
                dtype=np.int64,
            )
            jj = np.array(
                [local_index[int(j)] for j in constraints.j[mask]],
                dtype=np.int64,
            )
            a_c = a[mask]
            b_c = z_target[mask]
            alpha_c = geom.alpha[mask]
            beta_c = geom.beta[mask]
            target_c = geom.target[mask]
            conf_c = constraints.confidence[mask]
            w0_c = w0[idx_nodes]
            if solver_eff == 'analytic':
                w_c = _solve_component_analytic(ii, jj, a_c, b_c, w0_c, lam)
                iters = 1
                conv = True
            else:
                w_c, iters, conv = _solve_component_admm(
                    ii,
                    jj,
                    alpha_c,
                    beta_c,
                    target_c,
                    conf_c,
                    w0_c,
                    model=model,
                    lambda_regularize=lam,
                    rho=rho,
                    max_iter=max_iter,
                    tol_abs=tol_abs,
                    tol_rel=tol_rel,
                    y_lo=None if y_lo is None else y_lo[mask],
                    y_hi=None if y_hi is None else y_hi[mask],
                )
            if not np.all(np.isfinite(w_c)):
                raise _NumericalFailure('component solver returned non-finite weights')
            weights[idx_nodes] = w_c
            converged_all = converged_all and conv
            n_iter_max = max(n_iter_max, iters)

        if lam == 0.0:
            weights = _apply_component_mean_gauge(
                weights,
                comps,
                reference=reference,
            )
        if not np.all(np.isfinite(weights)):
            raise _NumericalFailure('assembled weight vector is non-finite')
        try:
            radii, shift = weights_to_radii(
                weights,
                r_min=r_min,
                weight_shift=weight_shift,
            )
        except ValueError as exc:
            raise _NumericalFailure(str(exc)) from exc
        pred_fraction, pred_position, pred = _predict_measurements(weights, constraints)
        residuals = pred - geom.target
        if not np.all(np.isfinite(residuals)):
            raise _NumericalFailure(
                'predicted measurements or residuals are non-finite'
            )
        rms = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0
        mx = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    except (np.linalg.LinAlgError, FloatingPointError, _NumericalFailure) as exc:
        warnings_list.append(f'numerical solver failure: {exc}')
        return PowerWeightFitResult(
            status='numerical_failure',
            hard_feasible=True,
            weights=None,
            radii=None,
            weight_shift=None,
            measurement=constraints.measurement,
            target=geom.target.copy(),
            predicted=None,
            predicted_fraction=None,
            predicted_position=None,
            residuals=None,
            rms_residual=None,
            max_residual=None,
            used_shifts=constraints.shifts.copy(),
            solver=solver_eff,
            n_iter=int(n_iter_max),
            converged=False,
            conflict=conflict,
            warnings=tuple(warnings_list),
            connectivity=connectivity,
            edge_diagnostics=_compute_edge_diagnostics(
                constraints,
                weights=None,
                geom=geom,
            ),
        )

    if converged_all:
        status: Literal['optimal', 'max_iter', 'numerical_failure'] = 'optimal'
    else:
        status = 'max_iter'
        warnings_list.append('iterative solver reached max_iter before convergence')

    return PowerWeightFitResult(
        status=status,
        hard_feasible=True,
        weights=weights,
        radii=radii,
        weight_shift=shift,
        measurement=constraints.measurement,
        target=geom.target.copy(),
        predicted=pred,
        predicted_fraction=pred_fraction,
        predicted_position=pred_position,
        residuals=residuals,
        rms_residual=rms,
        max_residual=mx,
        used_shifts=constraints.shifts.copy(),
        solver=solver_eff,
        n_iter=int(n_iter_max),
        converged=bool(converged_all),
        conflict=conflict,
        warnings=tuple(warnings_list),
        connectivity=connectivity,
        edge_diagnostics=_compute_edge_diagnostics(
            constraints,
            weights=weights,
            geom=geom,
        ),
    )


def _measurement_geometry(constraints: PairBisectorConstraints) -> _MeasurementGeometry:
    d = constraints.distance
    d2 = constraints.distance2
    if constraints.measurement == 'fraction':
        alpha = 1.0 / (2.0 * d2)
        beta = np.full_like(alpha, 0.5)
        target = constraints.target_fraction
    else:
        alpha = 1.0 / (2.0 * d)
        beta = 0.5 * d
        target = constraints.target_position
    return _MeasurementGeometry(
        alpha=np.asarray(alpha, dtype=np.float64),
        beta=np.asarray(beta, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        target_fraction=constraints.target_fraction.copy(),
        target_position=constraints.target_position.copy(),
    )


def _predict_measurements(
    weights: np.ndarray, constraints: PairBisectorConstraints
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_pred = weights[constraints.i] - weights[constraints.j]
    t_pred = 0.5 + z_pred / (2.0 * constraints.distance2)
    x_pred = constraints.distance * t_pred
    pred = t_pred if constraints.measurement == 'fraction' else x_pred
    return (
        np.asarray(t_pred, dtype=np.float64),
        np.asarray(x_pred, dtype=np.float64),
        np.asarray(pred, dtype=np.float64),
    )


def _compute_edge_diagnostics(
    constraints: PairBisectorConstraints,
    *,
    weights: np.ndarray | None,
    geom: _MeasurementGeometry | None = None,
) -> AlgebraicEdgeDiagnostics:
    if geom is None:
        geom = _measurement_geometry(constraints)

    alpha = np.asarray(geom.alpha, dtype=np.float64).copy()
    beta = np.asarray(geom.beta, dtype=np.float64).copy()
    z_obs = (geom.target - beta) / alpha
    edge_weight = (
        np.asarray(constraints.confidence, dtype=np.float64) * (alpha * alpha)
    )

    if weights is None:
        return AlgebraicEdgeDiagnostics(
            alpha=alpha,
            beta=beta,
            z_obs=np.asarray(z_obs, dtype=np.float64),
            z_fit=None,
            residual=None,
            edge_weight=np.asarray(edge_weight, dtype=np.float64),
            weighted_l2=None,
            weighted_rmse=None,
            rmse=None,
            mae=None,
        )

    w = np.asarray(weights, dtype=np.float64)
    z_fit = w[constraints.i] - w[constraints.j]
    residual = z_obs - z_fit
    weighted_sq = edge_weight * residual * residual
    if residual.size:
        weighted_l2 = float(np.linalg.norm(np.sqrt(edge_weight) * residual))
        weighted_rmse = float(np.sqrt(np.mean(weighted_sq)))
        rmse = float(np.sqrt(np.mean(residual * residual)))
        mae = float(np.mean(np.abs(residual)))
    else:
        weighted_l2 = 0.0
        weighted_rmse = 0.0
        rmse = 0.0
        mae = 0.0

    return AlgebraicEdgeDiagnostics(
        alpha=alpha,
        beta=beta,
        z_obs=np.asarray(z_obs, dtype=np.float64),
        z_fit=np.asarray(z_fit, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        edge_weight=np.asarray(edge_weight, dtype=np.float64),
        weighted_l2=weighted_l2,
        weighted_rmse=weighted_rmse,
        rmse=rmse,
        mae=mae,
    )


def _edge_diagnostics_for_result(
    result: PowerWeightFitResult,
    constraints: PairBisectorConstraints,
) -> AlgebraicEdgeDiagnostics:
    if result.edge_diagnostics is not None:
        return result.edge_diagnostics
    return _compute_edge_diagnostics(
        constraints,
        weights=result.weights,
    )


def _regularization_reference(reg: L2Regularization, n: int) -> np.ndarray:
    if reg.reference is None:
        return np.zeros(n, dtype=np.float64)
    w0 = np.asarray(reg.reference, dtype=float)
    if w0.shape != (n,):
        raise ValueError('regularization.reference must have shape (n,)')
    return w0.astype(np.float64)


def _difference_identifying_mask(
    constraints: PairBisectorConstraints,
    model: FitModel,
) -> np.ndarray:
    mask = constraints.confidence > 0.0
    if model.feasible is not None or len(model.penalties) > 0:
        mask = np.ones(constraints.n_constraints, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _apply_component_mean_gauge(
    weights: np.ndarray,
    comps: list[list[int]],
    *,
    reference: np.ndarray | None,
) -> np.ndarray:
    aligned = np.asarray(weights, dtype=np.float64).copy()
    ref = None if reference is None else np.asarray(reference, dtype=np.float64)
    for comp in comps:
        idx = np.asarray(comp, dtype=np.int64)
        if idx.size == 0:
            continue
        if ref is None:
            target_mean = 0.0
        else:
            target_mean = float(np.mean(ref[idx]))
        current_mean = float(np.mean(aligned[idx]))
        aligned[idx] += target_mean - current_mean
    return aligned


def _standalone_gauge_policy_description(reg: L2Regularization) -> str:
    if reg.reference is not None:
        return (
            'each effective component is shifted so its mean matches the '
            'reference mean on that component'
        )
    return 'each effective component is centered to mean zero'


def _graph_diagnostics(
    n: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    *,
    n_constraints: int,
) -> ConstraintGraphDiagnostics:
    ii = np.asarray(i_idx, dtype=np.int64)
    jj = np.asarray(j_idx, dtype=np.int64)
    degree = np.zeros(n, dtype=np.int64)
    if ii.size:
        np.add.at(degree, ii, 1)
        np.add.at(degree, jj, 1)
    isolated = tuple(np.flatnonzero(degree == 0).tolist())
    components = tuple(
        tuple(int(node) for node in comp)
        for comp in _connected_components(n, ii, jj)
    )
    edges = {
        (int(min(i, j)), int(max(i, j)))
        for i, j in zip(ii.tolist(), jj.tolist())
    }
    return ConstraintGraphDiagnostics(
        n_points=int(n),
        n_constraints=int(n_constraints),
        n_edges=int(len(edges)),
        isolated_points=isolated,
        connected_components=components,
        fully_connected=bool((n <= 1) or len(components) == 1),
    )


def _format_component_counts(graph: ConstraintGraphDiagnostics) -> str:
    n_components = graph.n_components
    return (
        '1 connected component'
        if n_components == 1
        else f'{n_components} connected components'
    )


def _format_point_list(points: tuple[int, ...]) -> str:
    return '[' + ', '.join(str(int(v)) for v in points) + ']'


def _build_fit_connectivity_diagnostics(
    constraints: PairBisectorConstraints,
    *,
    model: FitModel,
    gauge_policy: str,
) -> ConnectivityDiagnostics:
    n = int(constraints.n_points)
    candidate_graph = _graph_diagnostics(
        n,
        constraints.i,
        constraints.j,
        n_constraints=constraints.n_constraints,
    )
    effective_mask = _difference_identifying_mask(constraints, model)
    effective_graph = _graph_diagnostics(
        n,
        constraints.i[effective_mask],
        constraints.j[effective_mask],
        n_constraints=int(np.count_nonzero(effective_mask)),
    )

    messages: list[str] = []
    if candidate_graph.isolated_points:
        messages.append(
            'candidate graph leaves unconstrained points '
            f'{_format_point_list(candidate_graph.isolated_points)}'
        )
    if candidate_graph.n_components > 1:
        messages.append(
            'candidate graph has ' f'{_format_component_counts(candidate_graph)}'
        )
    if np.any(~effective_mask):
        messages.append(
            'zero-confidence candidate rows do not identify pair differences '
            'in the current objective and are ignored for '
            'connectivity/gauge diagnostics'
        )
    if effective_graph.n_components > 1:
        messages.append(
            'pairwise data identify only '
            f'{_format_component_counts(effective_graph)}; relative component '
            'offsets are not identified by the data'
        )

    return ConnectivityDiagnostics(
        unconstrained_points=candidate_graph.isolated_points,
        candidate_graph=candidate_graph,
        effective_graph=effective_graph,
        candidate_offsets_identified_by_data=bool(effective_graph.fully_connected),
        active_offsets_identified_by_data=None,
        offsets_identified_in_objective=bool(
            effective_graph.fully_connected
            or float(model.regularization.strength) > 0.0
        ),
        gauge_policy=gauge_policy,
        messages=tuple(messages),
    )


def _build_active_set_connectivity_diagnostics(
    constraints: PairBisectorConstraints,
    active_mask: np.ndarray,
    *,
    model: FitModel,
    gauge_policy: str,
) -> ConnectivityDiagnostics:
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (constraints.n_constraints,):
        raise ValueError('active_mask must have shape (m,)')

    n = int(constraints.n_points)
    candidate_graph = _graph_diagnostics(
        n,
        constraints.i,
        constraints.j,
        n_constraints=constraints.n_constraints,
    )
    effective_mask = _difference_identifying_mask(constraints, model)
    effective_graph = _graph_diagnostics(
        n,
        constraints.i[effective_mask],
        constraints.j[effective_mask],
        n_constraints=int(np.count_nonzero(effective_mask)),
    )

    active_constraints = constraints.subset(mask)
    active_graph = _graph_diagnostics(
        n,
        active_constraints.i,
        active_constraints.j,
        n_constraints=active_constraints.n_constraints,
    )
    active_effective_mask = _difference_identifying_mask(active_constraints, model)
    active_effective_graph = _graph_diagnostics(
        n,
        active_constraints.i[active_effective_mask],
        active_constraints.j[active_effective_mask],
        n_constraints=int(np.count_nonzero(active_effective_mask)),
    )

    messages: list[str] = []
    if candidate_graph.isolated_points:
        messages.append(
            'candidate graph leaves unconstrained points '
            f'{_format_point_list(candidate_graph.isolated_points)}'
        )
    if candidate_graph.n_components > 1:
        messages.append(
            'candidate graph has ' f'{_format_component_counts(candidate_graph)}'
        )
    if np.any(~effective_mask):
        messages.append(
            'zero-confidence candidate rows do not identify pair differences '
            'in the current objective and are ignored for '
            'connectivity/gauge diagnostics'
        )
    if effective_graph.n_components > 1:
        messages.append(
            'candidate pairwise data identify only '
            f'{_format_component_counts(effective_graph)}; relative component '
            'offsets are not identified by the data'
        )
    if active_graph.n_components > 1:
        messages.append(
            'final active graph has ' f'{_format_component_counts(active_graph)}'
        )
    if np.any(mask) and np.any(~active_effective_mask):
        messages.append(
            'zero-confidence active rows do not identify pair differences in '
            'the current objective and are ignored for active-component gauge '
            'alignment'
        )
    if active_effective_graph.n_components > 1:
        messages.append(
            'final active pairwise data identify only '
            f'{_format_component_counts(active_effective_graph)}; relative '
            'component offsets are preserved by the self-consistent gauge '
            'policy rather than identified by the data'
        )

    return ConnectivityDiagnostics(
        unconstrained_points=candidate_graph.isolated_points,
        candidate_graph=candidate_graph,
        effective_graph=effective_graph,
        active_graph=active_graph,
        active_effective_graph=active_effective_graph,
        candidate_offsets_identified_by_data=bool(effective_graph.fully_connected),
        active_offsets_identified_by_data=bool(
            active_effective_graph.fully_connected
        ),
        offsets_identified_in_objective=bool(
            active_effective_graph.fully_connected
            or float(model.regularization.strength) > 0.0
        ),
        gauge_policy=gauge_policy,
        messages=tuple(messages),
    )


def _apply_connectivity_policy(
    policy: Literal['none', 'diagnose', 'warn', 'raise'],
    diagnostics: ConnectivityDiagnostics,
    warnings_list: list[str],
) -> None:
    if policy in ('none', 'diagnose') or not diagnostics.messages:
        return
    if policy == 'warn':
        warnings_list.extend(diagnostics.messages)
        return
    if policy == 'raise':
        raise ConnectivityDiagnosticsError(
            '; '.join(diagnostics.messages),
            diagnostics,
        )
    raise ValueError('unsupported connectivity policy')


def _hard_constraint_measurement_bounds(
    feasible: HardConstraint | None,
    n_constraints: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if feasible is None:
        return None
    if isinstance(feasible, Interval):
        lower = np.full(n_constraints, float(feasible.lower), dtype=np.float64)
        upper = np.full(n_constraints, float(feasible.upper), dtype=np.float64)
        return lower, upper
    if isinstance(feasible, FixedValue):
        lower = np.full(n_constraints, float(feasible.value), dtype=np.float64)
        return lower, lower.copy()
    raise TypeError(f'unsupported hard constraint: {type(feasible)!r}')


def _hard_constraint_bounds(
    feasible: HardConstraint | None,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if feasible is None:
        return None
    if isinstance(feasible, Interval):
        lower = np.full_like(alpha, float(feasible.lower))
        upper = np.full_like(alpha, float(feasible.upper))
    elif isinstance(feasible, FixedValue):
        lower = np.full_like(alpha, float(feasible.value))
        upper = lower.copy()
    else:  # pragma: no cover - defensive
        raise TypeError(f'unsupported hard constraint: {type(feasible)!r}')
    z_lo = (lower - beta) / alpha
    z_hi = (upper - beta) / alpha
    lo = np.minimum(z_lo, z_hi)
    hi = np.maximum(z_lo, z_hi)
    return lo.astype(np.float64), hi.astype(np.float64)


def _requires_admm(model: FitModel) -> bool:
    if model.feasible is not None:
        return True
    if model.penalties:
        return True
    return not isinstance(model.mismatch, SquaredLoss)


def _connected_components(
    n: int, i_idx: np.ndarray, j_idx: np.ndarray
) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        adj[i].append(j)
        adj[j].append(i)
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        if len(adj[start]) == 0:
            seen[start] = True
            comps.append([start])
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        comps.append(sorted(comp))
    return comps


def _check_hard_feasibility(
    n: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
) -> tuple[bool, HardConstraintConflict | None]:
    """Check feasibility of difference constraints via Bellman-Ford."""

    edges: list[_DifferenceEdge] = []
    for k, (i, j, lo, hi) in enumerate(
        zip(i_idx.tolist(), j_idx.tolist(), z_lo.tolist(), z_hi.tolist())
    ):
        # w_i - w_j <= hi  ->  w_i <= w_j + hi : edge j -> i with weight hi
        edges.append(
            _DifferenceEdge(
                source=int(j),
                target=int(i),
                weight=float(hi),
                constraint_index=int(k),
                site_i=int(i),
                site_j=int(j),
                relation='<=',
                bound_value=float(hi),
            )
        )
        # w_i - w_j >= lo  ->  w_j - w_i <= -lo: edge i -> j with weight -lo
        edges.append(
            _DifferenceEdge(
                source=int(i),
                target=int(j),
                weight=float(-lo),
                constraint_index=int(k),
                site_i=int(i),
                site_j=int(j),
                relation='>=',
                bound_value=float(lo),
            )
        )

    dist = np.zeros(n, dtype=np.float64)
    pred_node = np.full(n, -1, dtype=np.int64)
    pred_edge = np.full(n, -1, dtype=np.int64)
    last_updated = -1
    tol = 1e-12

    for _ in range(n):
        updated = False
        last_updated = -1
        for edge_index, edge in enumerate(edges):
            cand = dist[edge.source] + edge.weight
            if cand < dist[edge.target] - tol:
                dist[edge.target] = cand
                pred_node[edge.target] = edge.source
                pred_edge[edge.target] = edge_index
                updated = True
                last_updated = edge.target
        if not updated:
            return True, None

    if last_updated < 0:
        return True, None

    y = int(last_updated)
    for _ in range(n):
        y = int(pred_node[y])
        if y < 0:
            return False, None

    cycle_edges_rev: list[_DifferenceEdge] = []
    cur = y
    while True:
        edge_index = int(pred_edge[cur])
        if edge_index < 0:
            return False, None
        edge = edges[edge_index]
        cycle_edges_rev.append(edge)
        cur = edge.source
        if cur == y:
            break

    cycle_edges = tuple(reversed(cycle_edges_rev))
    cycle_nodes_list: list[int] = []
    if cycle_edges:
        cycle_nodes_list.append(cycle_edges[0].source)
        cycle_nodes_list.extend(edge.target for edge in cycle_edges)
        if len(cycle_nodes_list) >= 2 and cycle_nodes_list[0] == cycle_nodes_list[-1]:
            cycle_nodes_list.pop()

    cycle_node_set = set(cycle_nodes_list)
    component_nodes: tuple[int, ...] = ()
    for comp in _connected_components(n, i_idx, j_idx):
        if any(node in cycle_node_set for node in comp):
            component_nodes = tuple(int(node) for node in comp)
            break

    terms = tuple(
        HardConstraintConflictTerm(
            constraint_index=edge.constraint_index,
            site_i=edge.site_i,
            site_j=edge.site_j,
            relation=edge.relation,
            bound_value=edge.bound_value,
        )
        for edge in cycle_edges
    )
    unique_constraints = tuple(sorted({term.constraint_index for term in terms}))
    component_label = (
        '[' + ', '.join(str(v) for v in component_nodes) + ']'
        if component_nodes
        else '[]'
    )
    cycle_label = '[' + ', '.join(str(v) for v in unique_constraints) + ']'
    conflict = HardConstraintConflict(
        component_nodes=component_nodes,
        cycle_nodes=tuple(int(v) for v in cycle_nodes_list),
        terms=terms,
        message=(
            'inconsistent hard separator restrictions on connected component '
            f'{component_label}; contradiction cycle uses constraint rows {cycle_label}'
        ),
    )
    return False, conflict


def _solve_component_analytic(
    I: np.ndarray,
    J: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    lambda_regularize: float,
) -> np.ndarray:
    n_c = int(np.max(np.maximum(I, J))) + 1
    if w0.shape != (n_c,):
        w0 = np.asarray(w0, dtype=float).reshape(n_c)
    lam = float(lambda_regularize)
    L = np.zeros((n_c, n_c), dtype=np.float64)
    rhs = np.zeros(n_c, dtype=np.float64)
    for i, j, ak, bk in zip(I.tolist(), J.tolist(), a.tolist(), b.tolist()):
        L[i, i] += ak
        L[j, j] += ak
        L[i, j] -= ak
        L[j, i] -= ak
        rhs[i] += ak * bk
        rhs[j] -= ak * bk
    if lam > 0:
        L += lam * np.eye(n_c)
        rhs += lam * w0

    if n_c == 1:
        if lam > 0:
            return w0.astype(np.float64, copy=True)
        return np.zeros(1, dtype=np.float64)

    if lam > 0:
        return np.linalg.solve(L, rhs).astype(np.float64)

    free = np.arange(1, n_c, dtype=np.int64)
    Lf = L[np.ix_(free, free)]
    rhsf = rhs[free]
    wf = np.linalg.solve(Lf, rhsf)
    w = np.zeros(n_c, dtype=np.float64)
    w[free] = wf
    return w


def _positive_confidence_connects_component(
    n_c: int,
    I: np.ndarray,
    J: np.ndarray,
    confidence: np.ndarray,
) -> bool:
    mask = np.asarray(confidence, dtype=np.float64) > 0.0
    if not np.any(mask):
        return False
    comps = _connected_components(n_c, I[mask], J[mask])
    return len(comps) == 1


def _admm_warm_start_weights(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    w0: np.ndarray,
    *,
    lambda_regularize: float,
) -> np.ndarray:
    n_c = int(np.max(np.maximum(I, J))) + 1
    lam = float(lambda_regularize)
    if lam > 0.0 or _positive_confidence_connects_component(
        n_c,
        I,
        J,
        confidence,
    ):
        try:
            return _solve_component_analytic(
                I,
                J,
                np.asarray(confidence, dtype=np.float64) * (alpha * alpha),
                (target - beta) / alpha,
                w0,
                lam,
            )
        except np.linalg.LinAlgError:
            pass
    if lam > 0.0:
        return np.asarray(w0, dtype=np.float64).copy()
    return np.zeros(n_c, dtype=np.float64)


def _solve_component_admm(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    w0: np.ndarray,
    *,
    model: FitModel,
    lambda_regularize: float,
    rho: float,
    max_iter: int,
    tol_abs: float,
    tol_rel: float,
    y_lo: np.ndarray | None,
    y_hi: np.ndarray | None,
) -> tuple[np.ndarray, int, bool]:
    n_c = int(np.max(np.maximum(I, J))) + 1
    m_c = I.shape[0]
    lam = float(lambda_regularize)

    if lam > 0.0:
        anchor: int | None = None
        free = np.arange(n_c, dtype=np.int64)
    else:
        anchor = 0
        free = np.arange(1, n_c, dtype=np.int64)

    edge_scale = alpha * alpha
    L = np.zeros((n_c, n_c), dtype=np.float64)
    for i, j, scale in zip(I.tolist(), J.tolist(), edge_scale.tolist()):
        L[i, i] += scale
        L[j, j] += scale
        L[i, j] -= scale
        L[j, i] -= scale

    M = rho * L + lam * np.eye(n_c)
    Mf = M[np.ix_(free, free)]
    if free.size and not np.all(np.isfinite(Mf)):
        raise _NumericalFailure('ADMM system matrix contains non-finite values')
    try:
        chol = (
            np.linalg.cholesky(Mf)
            if free.size
            else np.zeros((0, 0), dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        Mf2 = Mf + 1e-12 * np.eye(Mf.shape[0])
        try:
            chol = np.linalg.cholesky(Mf2)
        except np.linalg.LinAlgError as exc:
            raise _NumericalFailure(
                'ADMM system matrix is not numerically positive definite'
            ) from exc
        Mf = Mf2

    def solve_M(rhs_free: np.ndarray) -> np.ndarray:
        if rhs_free.size == 0:
            return np.zeros(0, dtype=np.float64)
        y = np.linalg.solve(chol, rhs_free)
        x = np.linalg.solve(chol.T, y)
        if not np.all(np.isfinite(x)):
            raise _NumericalFailure('ADMM linear solve produced non-finite values')
        return x

    w = _admm_warm_start_weights(
        I,
        J,
        alpha,
        beta,
        target,
        confidence,
        w0,
        lambda_regularize=lam,
    )
    if not np.all(np.isfinite(w)):
        raise _NumericalFailure('ADMM warm start produced non-finite values')
    y = beta + alpha * (w[I] - w[J])
    if y_lo is not None:
        y = np.maximum(y, y_lo)
    if y_hi is not None:
        y = np.minimum(y, y_hi)
    u = np.zeros(m_c, dtype=np.float64)
    converged = False

    for _it in range(1, max_iter + 1):
        rhs = np.zeros(n_c, dtype=np.float64)
        edge_rhs = rho * alpha * (y - u - beta)
        np.add.at(rhs, I, edge_rhs)
        np.add.at(rhs, J, -edge_rhs)
        if lam > 0.0:
            rhs += lam * w0

        w_free = solve_M(rhs[free])
        if anchor is not None:
            w[anchor] = 0.0
        w[free] = w_free
        if not np.all(np.isfinite(w)):
            raise _NumericalFailure('ADMM primal iterate became non-finite')

        predicted_y = beta + alpha * (w[I] - w[J])
        y_prev = y.copy()
        y = _prox_measurement_objective(
            predicted_y + u,
            target,
            confidence,
            model=model,
            rho=rho,
            y_lo=y_lo,
            y_hi=y_hi,
        )
        r = predicted_y - y
        u = u + r
        if not (
            np.all(np.isfinite(predicted_y))
            and np.all(np.isfinite(y))
            and np.all(np.isfinite(r))
            and np.all(np.isfinite(u))
        ):
            raise _NumericalFailure('ADMM iterates became non-finite')

        r_norm = float(np.linalg.norm(r))
        predicted_norm = float(np.linalg.norm(predicted_y))
        y_norm = float(np.linalg.norm(y))
        eps_pri = np.sqrt(m_c) * tol_abs + tol_rel * max(predicted_norm, y_norm)

        dy = y - y_prev
        s_vec = np.zeros(n_c, dtype=np.float64)
        np.add.at(s_vec, I, rho * alpha * dy)
        np.add.at(s_vec, J, -rho * alpha * dy)
        s_norm = float(np.linalg.norm(s_vec[free])) if free.size else 0.0

        dual_vec = np.zeros(n_c, dtype=np.float64)
        np.add.at(dual_vec, I, rho * alpha * u)
        np.add.at(dual_vec, J, -rho * alpha * u)
        dual_norm = float(np.linalg.norm(dual_vec[free])) if free.size else 0.0
        eps_dual = np.sqrt(len(free)) * tol_abs + tol_rel * dual_norm

        if r_norm <= eps_pri and s_norm <= eps_dual:
            converged = True
            break

    return w, _it, converged


def _prox_measurement_mismatch_only(
    v: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
    rho: float,
) -> np.ndarray:
    if isinstance(mismatch, SquaredLoss):
        denom = rho + (2.0 * confidence)
        return (rho * v + (2.0 * confidence * target)) / denom
    if isinstance(mismatch, HuberLoss):
        delta = float(mismatch.delta)
        y_quad = (rho * v + confidence * target) / (rho + confidence)
        lower = target - delta
        upper = target + delta
        y_lower = v + (confidence * delta) / rho
        y_upper = v - (confidence * delta) / rho
        return np.where(
            y_quad < lower,
            y_lower,
            np.where(y_quad > upper, y_upper, y_quad),
        )
    raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')


def _prox_measurement_objective(
    v: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    *,
    model: FitModel,
    rho: float,
    y_lo: np.ndarray | None,
    y_hi: np.ndarray | None,
) -> np.ndarray:
    y = _prox_measurement_mismatch_only(
        v,
        target,
        confidence,
        model.mismatch,
        rho,
    )
    if y_lo is not None:
        y = np.maximum(y, y_lo)
    if y_hi is not None:
        y = np.minimum(y, y_hi)
    if not model.penalties:
        return y

    for _ in range(60):
        fp_y, fpp_y = _mismatch_derivatives(y, target, confidence, model.mismatch)
        for penalty in model.penalties:
            p_fp_y, p_fpp_y = _penalty_derivatives(y, penalty)
            fp_y = fp_y + p_fp_y
            fpp_y = fpp_y + p_fpp_y

        g = fp_y + rho * (y - v)
        gp = fpp_y + rho
        if not np.all(np.isfinite(gp)) or np.any(np.abs(gp) < 1e-18):
            raise _NumericalFailure(
                'prox Newton derivative became singular or non-finite'
            )
        step = g / gp
        if not np.all(np.isfinite(step)):
            raise _NumericalFailure('prox Newton step became non-finite')
        y_new = y - step
        if y_lo is not None:
            y_new = np.maximum(y_new, y_lo)
        if y_hi is not None:
            y_new = np.minimum(y_new, y_hi)
        if float(np.max(np.abs(step))) < 1e-12:
            y = y_new
            break
        y = y_new
    return y


def _mismatch_derivatives(
    y: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> tuple[np.ndarray, np.ndarray]:
    r = y - target
    if isinstance(mismatch, SquaredLoss):
        fp_y = 2.0 * confidence * r
        fpp_y = 2.0 * confidence
        return fp_y, fpp_y
    if isinstance(mismatch, HuberLoss):
        delta = float(mismatch.delta)
        abs_r = np.abs(r)
        quad = abs_r <= delta
        fp_y = np.where(quad, confidence * r, confidence * delta * np.sign(r))
        fpp_y = np.where(quad, confidence, 0.0)
        return fp_y, fpp_y
    raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')


def _penalty_derivatives(
    y: np.ndarray,
    penalty: SoftIntervalPenalty
    | ExponentialBoundaryPenalty
    | ReciprocalBoundaryPenalty,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(penalty, SoftIntervalPenalty):
        lower = float(penalty.lower)
        upper = float(penalty.upper)
        strength = float(penalty.strength)
        fp = np.zeros_like(y)
        fpp = np.zeros_like(y)
        lo_mask = y < lower
        hi_mask = y > upper
        if np.any(lo_mask):
            fp[lo_mask] += 2.0 * strength * (y[lo_mask] - lower)
            fpp[lo_mask] += 2.0 * strength
        if np.any(hi_mask):
            fp[hi_mask] += 2.0 * strength * (y[hi_mask] - upper)
            fpp[hi_mask] += 2.0 * strength
        return fp, fpp

    if isinstance(penalty, ExponentialBoundaryPenalty):
        lower = float(penalty.lower)
        upper = float(penalty.upper)
        margin = float(penalty.margin)
        strength = float(penalty.strength)
        tau = float(penalty.tau)
        left = lower + margin
        right = upper - margin
        A = np.exp((left - y) / tau)
        B = np.exp((y - right) / tau)
        fp = strength * (-A + B) / tau
        fpp = strength * (A + B) / (tau * tau)
        return fp, fpp

    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower = float(penalty.lower)
        upper = float(penalty.upper)
        margin = float(penalty.margin)
        strength = float(penalty.strength)
        eps = float(penalty.epsilon)
        left = lower + margin
        right = upper - margin
        fp = np.zeros_like(y)
        fpp = np.zeros_like(y)
        lo_mask = y < left
        if np.any(lo_mask):
            denom = np.maximum(y[lo_mask] - lower, eps)
            fp[lo_mask] += -strength / (denom**2)
            fpp[lo_mask] += 2.0 * strength / (denom**3)
        hi_mask = y > right
        if np.any(hi_mask):
            denom = np.maximum(upper - y[hi_mask], eps)
            fp[hi_mask] += strength / (denom**2)
            fpp[hi_mask] += 2.0 * strength / (denom**3)
        return fp, fpp

    raise TypeError(f'unsupported penalty: {type(penalty)!r}')
