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
    weights: np.ndarray, *, r_min: float = 0.0
) -> tuple[np.ndarray, float]:
    """Convert power weights to radii using a global gauge shift."""

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError('weights must be 1D')
    if not np.all(np.isfinite(w)):
        raise ValueError('weights must contain only finite values')
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
    solver: Literal['auto', 'analytic', 'admm'] = 'auto',
    max_iter: int = 2000,
    rho: float = 1.0,
    tol_abs: float = 1e-6,
    tol_rel: float = 1e-5,
) -> PowerWeightFitResult:
    """Fit power weights from resolved pairwise separator constraints.

    The raw constraint tuples are ``(i, j, value[, shift])`` where ``shift`` is
    the integer lattice image applied to site ``j``.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')

    if model is None:
        model = FitModel()

    if isinstance(constraints, PairBisectorConstraints):
        resolved = constraints
        if resolved.n_points != pts.shape[0]:
            raise ValueError('resolved constraints do not match the number of points')
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
        solver=solver,
        max_iter=max_iter,
        rho=rho,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
    )


def _fit_power_weights_resolved(
    constraints: PairBisectorConstraints,
    *,
    model: FitModel,
    r_min: float,
    solver: Literal['auto', 'analytic', 'admm'],
    max_iter: int,
    rho: float,
    tol_abs: float,
    tol_rel: float,
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

    reg = model.regularization
    lam = float(reg.strength)
    w0 = _regularization_reference(reg, n)

    geom = _measurement_geometry(constraints)
    z_target = (geom.target - geom.beta) / geom.alpha
    a = constraints.confidence * (geom.alpha**2)

    hard = _hard_constraint_bounds(model.feasible, geom.alpha, geom.beta)
    z_lo = hard[0] if hard is not None else None
    z_hi = hard[1] if hard is not None else None

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
            )
    else:
        conflict = None

    if m == 0:
        if lam > 0.0:
            weights = w0.copy()
            warnings_list.append(
                'empty constraint set; using the regularization-only solution'
            )
        else:
            weights = np.zeros(n, dtype=np.float64)
            warnings_list.append('empty constraint set; returning zero weights')
        radii, shift = weights_to_radii(weights, r_min=r_min)
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
        )

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

    if solver_eff == 'analytic' and lam == 0.0:
        effective_mask = a > 0.0
        comps = _connected_components(
            n,
            constraints.i[effective_mask],
            constraints.j[effective_mask],
        )
        if np.any(~effective_mask):
            warnings_list.append(
                'zero-confidence constraints do not affect the quadratic fit '
                'objective and are ignored for gauge connectivity'
            )
    else:
        comps = _connected_components(n, constraints.i, constraints.j)
    if len(comps) > 1 and lam == 0.0:
        warnings_list.append(
            'effective constraint graph has multiple connected components; '
            'each component is gauge-fixed independently'
        )

    weights = np.zeros(n, dtype=np.float64)
    converged_all = True
    n_iter_max = 0

    for nodes in comps:
        if len(nodes) <= 1:
            if lam > 0 and len(nodes) == 1:
                weights[nodes[0]] = w0[nodes[0]]
            continue

        node_set = set(nodes)
        mask = np.array(
            [
                (int(i) in node_set) and (int(j) in node_set)
                for i, j in zip(constraints.i, constraints.j)
            ],
            dtype=bool,
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
        w0_c = w0[np.array(nodes, dtype=np.int64)]
        z_lo_c = None if z_lo is None else z_lo[mask]
        z_hi_c = None if z_hi is None else z_hi[mask]

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
                z_lo=z_lo_c,
                z_hi=z_hi_c,
            )
        weights[np.array(nodes, dtype=np.int64)] = w_c
        converged_all = converged_all and conv
        n_iter_max = max(n_iter_max, iters)

    radii, shift = weights_to_radii(weights, r_min=r_min)
    pred_fraction, pred_position, pred = _predict_measurements(weights, constraints)
    residuals = pred - geom.target
    rms = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0
    mx = float(np.max(np.abs(residuals))) if residuals.size else 0.0

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


def _regularization_reference(reg: L2Regularization, n: int) -> np.ndarray:
    if reg.reference is None:
        return np.zeros(n, dtype=np.float64)
    w0 = np.asarray(reg.reference, dtype=float)
    if w0.shape != (n,):
        raise ValueError('regularization.reference must have shape (n,)')
    return w0.astype(np.float64)


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
    z_lo: np.ndarray | None,
    z_hi: np.ndarray | None,
) -> tuple[np.ndarray, int, bool]:
    n_c = int(np.max(np.maximum(I, J))) + 1
    m_c = I.shape[0]
    lam = float(lambda_regularize)

    if lam > 0:
        anchor: int | None = None
        free = np.arange(n_c, dtype=np.int64)
    else:
        anchor = 0
        free = np.arange(1, n_c, dtype=np.int64)

    L = np.zeros((n_c, n_c), dtype=np.float64)
    for i, j in zip(I.tolist(), J.tolist()):
        L[i, i] += 1.0
        L[j, j] += 1.0
        L[i, j] -= 1.0
        L[j, i] -= 1.0

    M = rho * L + lam * np.eye(n_c)
    Mf = M[np.ix_(free, free)]
    try:
        chol = np.linalg.cholesky(Mf)
    except np.linalg.LinAlgError:
        Mf2 = Mf + 1e-12 * np.eye(Mf.shape[0])
        chol = np.linalg.cholesky(Mf2)
        Mf = Mf2

    def solve_M(rhs_free: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(chol, rhs_free)
        x = np.linalg.solve(chol.T, y)
        return x

    # Initialize at the target z implied by the chosen measurement.
    z = (target - beta) / alpha
    if z_lo is not None and z_hi is not None:
        z = np.clip(z, z_lo, z_hi)
    u = np.zeros(m_c, dtype=np.float64)
    w = np.zeros(n_c, dtype=np.float64)
    converged = False

    for _it in range(1, max_iter + 1):
        y = z - u
        rhs = np.zeros(n_c, dtype=np.float64)
        np.add.at(rhs, I, rho * y)
        np.add.at(rhs, J, -rho * y)
        if lam > 0:
            rhs += lam * w0

        rhs_free = rhs[free]
        w_free = solve_M(rhs_free)
        if anchor is not None:
            w[anchor] = 0.0
        w[free] = w_free

        v = (w[I] - w[J]) + u
        z_prev = z.copy()
        z = _prox_edge_objective(
            v,
            alpha,
            beta,
            target,
            confidence,
            model=model,
            rho=rho,
            z_lo=z_lo,
            z_hi=z_hi,
        )

        Aw = w[I] - w[J]
        r = Aw - z
        u = u + r

        r_norm = float(np.linalg.norm(r))
        z_norm = float(np.linalg.norm(z))
        Aw_norm = float(np.linalg.norm(Aw))
        eps_pri = np.sqrt(m_c) * tol_abs + tol_rel * max(Aw_norm, z_norm)

        dz = z - z_prev
        s_vec = np.zeros(n_c, dtype=np.float64)
        np.add.at(s_vec, I, rho * dz)
        np.add.at(s_vec, J, -rho * dz)
        s_norm = float(np.linalg.norm(s_vec[free])) if free.size else 0.0
        u_norm = float(np.linalg.norm(u))
        eps_dual = np.sqrt(len(free)) * tol_abs + tol_rel * rho * u_norm

        if r_norm <= eps_pri and s_norm <= eps_dual:
            converged = True
            break

    return w, _it, converged


def _prox_edge_objective(
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    *,
    model: FitModel,
    rho: float,
    z_lo: np.ndarray | None,
    z_hi: np.ndarray | None,
) -> np.ndarray:
    z = v.copy()
    if z_lo is not None and z_hi is not None:
        z = np.clip(z, z_lo, z_hi)

    for _ in range(60):
        y = beta + alpha * z
        fp_y, fpp_y = _mismatch_derivatives(y, target, confidence, model.mismatch)
        for penalty in model.penalties:
            p_fp_y, p_fpp_y = _penalty_derivatives(y, penalty)
            fp_y = fp_y + p_fp_y
            fpp_y = fpp_y + p_fpp_y

        g = fp_y * alpha + rho * (z - v)
        gp = fpp_y * (alpha**2) + rho
        step = g / gp
        z_new = z - step
        if z_lo is not None and z_hi is not None:
            z_new = np.clip(z_new, z_lo, z_hi)
        if float(np.max(np.abs(step))) < 1e-12:
            z = z_new
            break
        z = z_new
    return z


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
