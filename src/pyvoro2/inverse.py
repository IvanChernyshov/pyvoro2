"""Inverse utilities for power (Laguerre) tessellations.

This module provides helpers to *fit* per-site power weights (and radii) from
user-specified desired locations of separating planes between selected pairs.

The fitted result is always a valid *power diagram* (a.k.a. Laguerre / radical
Voronoi tessellation) because it returns weights/radii to be used with
``mode='power'``.

The core quantity in a power diagram is the per-site weight ``w_i``. Voro++
accepts radii ``r_i`` and internally uses ``w_i = r_i**2``.

For two sites ``i`` and an (optional periodic) image of ``j`` at distance ``d``,
the separating plane intersects the line segment at a fraction ``t`` (measured
from ``i`` toward ``j``) given by:

    t = 1/2 + (w_i - w_j) / (2 d^2)

This means that desired plane locations correspond to constraints on weight
differences, and fitting can be posed as a convex optimization problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell


@dataclass(frozen=True, slots=True)
class FitWeightsResult:
    """Result object returned by the inverse fitting routines."""

    weights: np.ndarray
    radii: np.ndarray
    weight_shift: float

    # Per-constraint diagnostics (order matches input constraints)
    t_target: np.ndarray
    t_pred: np.ndarray
    residuals: np.ndarray  # t_pred - t_target
    rms_residual: float
    max_residual: float

    used_shifts: np.ndarray  # (m, 3) integer lattice shifts applied to j

    # Optional adjacency check (requires a tessellation compute)
    is_contact: np.ndarray | None
    inactive_constraints: tuple[int, ...] | None

    # Solver metadata
    solver: str
    n_iter: int
    converged: bool
    warnings: tuple[str, ...]


def radii_to_weights(radii: np.ndarray) -> np.ndarray:
    """Convert radii to power weights (w = r^2)."""

    r = np.asarray(radii, dtype=float)
    if r.ndim != 1:
        raise ValueError('radii must be 1D')
    if np.any(r < 0):
        raise ValueError('radii must be non-negative')
    return r * r


def weights_to_radii(
    weights: np.ndarray, *, r_min: float = 0.0
) -> tuple[np.ndarray, float]:
    """Convert weights to radii by applying a global weight shift.

    Power diagrams are invariant under adding a constant ``C`` to all weights.
    Voro++ requires radii ``r`` with ``w = r^2 >= 0``.

    This helper chooses a shift ``C`` so that:

        min_i sqrt(w_i + C) == r_min

    Args:
        weights: Array of weights (n,).
        r_min: Minimum radius in the returned array.

    Returns:
        (radii, C) where ``radii = sqrt(weights + C)``.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError('weights must be 1D')
    r_min = float(r_min)
    if r_min < 0:
        raise ValueError('r_min must be >= 0')

    w_min = float(np.min(w)) if w.size else 0.0
    C = (r_min * r_min) - w_min
    # C can be negative; that is fine as long as w + C >= r_min^2 >= 0.
    w_shifted = w + C
    if np.any(w_shifted < -1e-14):
        # Numerical guard: in theory this should not happen.
        raise ValueError('weight shift produced negative values (numerical issue)')
    w_shifted = np.maximum(w_shifted, 0.0)
    return np.sqrt(w_shifted), float(C)


def fit_power_weights_from_plane_positions(
    points: np.ndarray,
    constraints: Sequence[
        tuple[int, int, float] | tuple[int, int, float, tuple[int, int, int]]
    ],
    *,
    domain: Box | OrthorhombicCell | PeriodicCell | None = None,
    ids: Sequence[int] | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    constraint_weights: Sequence[float] | None = None,
    # Predicted t(w) restrictions/penalties
    t_bounds: tuple[float, float] | None = (0.0, 1.0),
    t_bounds_mode: Literal['none', 'soft_quadratic', 'hard'] = 'none',
    alpha_out: float = 0.0,
    t_near_penalty: Literal['none', 'exp'] = 'none',
    beta_near: float = 0.0,
    t_margin: float = 0.02,
    t_tau: float = 0.01,
    # Regularization (optional)
    regularize_to: np.ndarray | None = None,
    lambda_regularize: float = 0.0,
    # Radii gauge
    r_min: float = 0.0,
    # Solver controls
    solver: Literal['auto', 'analytic', 'admm'] = 'auto',
    max_iter: int = 2000,
    rho: float = 1.0,
    tol_abs: float = 1e-6,
    tol_rel: float = 1e-5,
    # Optional adjacency check
    check_contacts: bool = False,
) -> FitWeightsResult:
    """Fit power weights from desired *absolute* plane positions.

    Each constraint specifies the desired plane intersection distance ``x`` from
    site ``i`` toward an (optional periodic) image of site ``j``.

    This is converted to a fraction ``t = x / d`` where ``d`` is the distance
    between ``p_i`` and the chosen image ``p_j*``.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')

    i_idx, j_idx, x_target, shifts, shift_given, warnings = _parse_constraints(
        constraints,
        n_points=pts.shape[0],
        ids=ids,
        index_mode=index_mode,
    )

    pts2 = _maybe_remap_points(pts, domain)

    # Resolve periodic image shifts (for constraints that didn't specify them)
    shifts_used, warnings2 = _resolve_constraint_shifts(
        pts2,
        i_idx,
        j_idx,
        shifts,
        shift_given,
        domain=domain,
        image=image,
        image_search=image_search,
    )
    warnings = warnings + warnings2

    # Compute distances d and convert to t targets.
    pj_star = pts2[j_idx] + _shift_to_cart(shifts_used, domain)
    delta = pj_star - pts2[i_idx]
    d = np.linalg.norm(delta, axis=1)
    if np.any(d == 0):
        raise ValueError(
            'some constraints have zero distance (coincident points/image)'
        )
    t_target = x_target / d

    return _fit_power_weights_core(
        pts2,
        i_idx,
        j_idx,
        t_target,
        shifts_used,
        domain=domain,
        constraint_weights=constraint_weights,
        t_bounds=t_bounds,
        t_bounds_mode=t_bounds_mode,
        alpha_out=alpha_out,
        t_near_penalty=t_near_penalty,
        beta_near=beta_near,
        t_margin=t_margin,
        t_tau=t_tau,
        regularize_to=regularize_to,
        lambda_regularize=lambda_regularize,
        r_min=r_min,
        solver=solver,
        max_iter=max_iter,
        rho=rho,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        check_contacts=check_contacts,
        warnings=warnings,
    )


def fit_power_weights_from_plane_fractions(
    points: np.ndarray,
    constraints: Sequence[
        tuple[int, int, float] | tuple[int, int, float, tuple[int, int, int]]
    ],
    *,
    domain: Box | OrthorhombicCell | PeriodicCell | None = None,
    ids: Sequence[int] | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    constraint_weights: Sequence[float] | None = None,
    # Predicted t(w) restrictions/penalties
    t_bounds: tuple[float, float] | None = (0.0, 1.0),
    t_bounds_mode: Literal['none', 'soft_quadratic', 'hard'] = 'none',
    alpha_out: float = 0.0,
    t_near_penalty: Literal['none', 'exp'] = 'none',
    beta_near: float = 0.0,
    t_margin: float = 0.02,
    t_tau: float = 0.01,
    # Regularization (optional)
    regularize_to: np.ndarray | None = None,
    lambda_regularize: float = 0.0,
    # Radii gauge
    r_min: float = 0.0,
    # Solver controls
    solver: Literal['auto', 'analytic', 'admm'] = 'auto',
    max_iter: int = 2000,
    rho: float = 1.0,
    tol_abs: float = 1e-6,
    tol_rel: float = 1e-5,
    # Optional adjacency check
    check_contacts: bool = False,
) -> FitWeightsResult:
    """Fit power weights from desired plane fractions t_ij.

    Each constraint specifies a desired separating plane position as a fraction
    ``t`` along the line segment from site ``i`` toward an (optional periodic)
    image of site ``j``.

    Notes:
        - Values ``t < 0`` and ``t > 1`` are allowed.
        - Additional options can *constrain* or *penalize* the predicted
          plane position ``t(w)`` to lie within or away from the [0, 1] segment.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')

    i_idx, j_idx, t_target, shifts, shift_given, warnings = _parse_constraints(
        constraints,
        n_points=pts.shape[0],
        ids=ids,
        index_mode=index_mode,
    )

    pts2 = _maybe_remap_points(pts, domain)

    shifts_used, warnings2 = _resolve_constraint_shifts(
        pts2,
        i_idx,
        j_idx,
        shifts,
        shift_given,
        domain=domain,
        image=image,
        image_search=image_search,
    )
    warnings = warnings + warnings2

    return _fit_power_weights_core(
        pts2,
        i_idx,
        j_idx,
        t_target,
        shifts_used,
        domain=domain,
        constraint_weights=constraint_weights,
        t_bounds=t_bounds,
        t_bounds_mode=t_bounds_mode,
        alpha_out=alpha_out,
        t_near_penalty=t_near_penalty,
        beta_near=beta_near,
        t_margin=t_margin,
        t_tau=t_tau,
        regularize_to=regularize_to,
        lambda_regularize=lambda_regularize,
        r_min=r_min,
        solver=solver,
        max_iter=max_iter,
        rho=rho,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        check_contacts=check_contacts,
        warnings=warnings,
    )


# ---------------------------- internal helpers ----------------------------


def _parse_constraints(
    constraints: Sequence[tuple],
    *,
    n_points: int,
    ids: Sequence[int] | None,
    index_mode: Literal['index', 'id'],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Parse constraint tuples.

    Accepted forms:
        (i, j, value)
        (i, j, value, shift)

    where shift is a 3-tuple of ints.
    """

    if index_mode not in ('index', 'id'):
        raise ValueError('index_mode must be \'index\' or \'id\'')
    if index_mode == 'id':
        if ids is None:
            raise ValueError('ids must be provided when index_mode="id"')
        id_to_index = {int(v): k for k, v in enumerate(ids)}
    else:
        id_to_index = None

    m = len(constraints)
    if m == 0:
        raise ValueError('constraints must be non-empty')

    i_idx = np.empty(m, dtype=np.int64)
    j_idx = np.empty(m, dtype=np.int64)
    val = np.empty(m, dtype=np.float64)
    shifts = np.zeros((m, 3), dtype=np.int64)
    shift_given = np.zeros(m, dtype=bool)
    warnings: list[str] = []

    for k, c in enumerate(constraints):
        if not isinstance(c, tuple) and not isinstance(c, list):
            raise ValueError(f'constraint {k} must be a tuple/list')
        if len(c) not in (3, 4):
            raise ValueError(
                f'constraint {k} must have length 3 or 4: (i, j, value[, shift])'
            )
        ii = int(c[0])
        jj = int(c[1])
        if id_to_index is not None:
            if ii not in id_to_index or jj not in id_to_index:
                raise ValueError(f'constraint {k} uses id not present in ids')
            ii = id_to_index[ii]
            jj = id_to_index[jj]
        if not (0 <= ii < n_points and 0 <= jj < n_points):
            raise ValueError(f'constraint {k} index out of range')
        if ii == jj:
            raise ValueError(f'constraint {k} has i == j (degenerate)')
        i_idx[k] = ii
        j_idx[k] = jj
        val[k] = float(c[2])

        if len(c) == 4:
            sh = c[3]
            if not (isinstance(sh, tuple) or isinstance(sh, list)) or len(sh) != 3:
                raise ValueError(f'constraint {k} shift must be a length-3 tuple')
            shifts[k] = (int(sh[0]), int(sh[1]), int(sh[2]))
            shift_given[k] = True

    return i_idx, j_idx, val, shifts, shift_given, tuple(warnings)


def _maybe_remap_points(
    points: np.ndarray, domain: Box | OrthorhombicCell | PeriodicCell | None
) -> np.ndarray:
    """Optionally remap points into a primary periodic domain.

    This improves stability of the "nearest image" logic and makes results
    deterministic with respect to lattice translations.
    """

    if domain is None:
        return np.asarray(points, dtype=float)
    if isinstance(domain, PeriodicCell):
        return domain.remap_cart(points, return_shifts=False)
    if isinstance(domain, OrthorhombicCell):
        return domain.remap_cart(points, return_shifts=False)
    return np.asarray(points, dtype=float)


def _resolve_constraint_shifts(
    points: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    shifts: np.ndarray,
    shift_given: np.ndarray,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell | None,
    image: Literal['nearest', 'given_only'],
    image_search: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return per-constraint integer shifts to apply to site j."""

    m = i_idx.shape[0]
    warnings: list[str] = []

    shifts = np.asarray(shifts, dtype=np.int64)
    if shifts.shape != (m, 3):
        raise ValueError('shifts must have shape (m,3)')
    shift_given = np.asarray(shift_given, dtype=bool)
    if shift_given.shape != (m,):
        raise ValueError('shift_given must have shape (m,)')

    # If no domain, shifts must be zero.
    if domain is None:
        if np.any(shifts[shift_given] != 0):
            raise ValueError('constraint shifts require a periodic domain')
        return np.zeros((m, 3), dtype=np.int64), tuple(warnings)

    # Box is non-periodic.
    if isinstance(domain, Box):
        if np.any(shifts[shift_given] != 0):
            raise ValueError('Box domain does not support periodic shifts')
        return np.zeros((m, 3), dtype=np.int64), tuple(warnings)

    shifts2 = shifts.copy()
    provided_mask = shift_given.copy()

    if image == 'given_only':
        # Missing shifts are not allowed in given_only mode.
        if np.any(~provided_mask):
            raise ValueError('some constraints are missing shifts (image="given_only")')
        _validate_shifts_against_domain(shifts2, domain)
        return shifts2, tuple(warnings)

    if image != 'nearest':
        raise ValueError('image must be "nearest" or "given_only"')
    if image_search < 0:
        raise ValueError('image_search must be >= 0')

    # Compute missing shifts by nearest-image search.
    missing = ~provided_mask
    if np.any(missing):
        if isinstance(domain, OrthorhombicCell):
            shifts2[missing] = _nearest_image_shifts_orthorhombic(
                points[i_idx[missing]],
                points[j_idx[missing]],
                domain,
            )
        elif isinstance(domain, PeriodicCell):
            shifts2[missing] = _nearest_image_shifts_triclinic(
                points[i_idx[missing]],
                points[j_idx[missing]],
                domain,
                search=image_search,
            )
        else:
            raise ValueError('unsupported domain type')
        warnings.append(
            'some constraints did not specify shifts; using nearest-image shifts'
        )

    _validate_shifts_against_domain(shifts2, domain)
    return shifts2, tuple(warnings)


def _validate_shifts_against_domain(
    shifts: np.ndarray, domain: Box | OrthorhombicCell | PeriodicCell
) -> None:
    if isinstance(domain, OrthorhombicCell):
        per = domain.periodic
        for ax in range(3):
            if not per[ax] and np.any(shifts[:, ax] != 0):
                raise ValueError(
                    'shifts on non-periodic axes must be 0 for OrthorhombicCell'
                )


def _nearest_image_shifts_orthorhombic(
    pi: np.ndarray, pj: np.ndarray, cell: OrthorhombicCell
) -> np.ndarray:
    """Nearest-image shifts for an orthorhombic cell."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = cell.bounds
    L = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
    per = np.array(cell.periodic, dtype=bool)
    delta = pj - pi
    s = np.zeros_like(delta, dtype=np.int64)
    for ax in range(3):
        if not per[ax]:
            continue
        # Choose shift to bring delta into [-L/2, L/2)
        s[:, ax] = (-np.round(delta[:, ax] / L[ax])).astype(np.int64)
    return s


def _nearest_image_shifts_triclinic(
    pi: np.ndarray, pj: np.ndarray, cell: PeriodicCell, *, search: int = 1
) -> np.ndarray:
    """Nearest-image shifts via brute-force search in [-S,S]^3.

    This is robust for typical cell shapes and avoids subtle issues with
    fractional rounding in highly skewed cells.
    """

    a, b, c = (np.asarray(v, dtype=float) for v in cell.vectors)
    # Build candidate shifts
    rng = np.arange(-search, search + 1, dtype=np.int64)
    cand = (
        np.array(np.meshgrid(rng, rng, rng, indexing='ij')).reshape(3, -1).T
    )  # (n_candidates, 3)

    # Compute deltas for each pair in batch: choose minimal norm.
    # pi/pj are (m,3).
    base = pj - pi  # (m,3)
    # precompute translations for candidates
    trans = (
        cand[:, 0:1] * a[None, :]
        + cand[:, 1:2] * b[None, :]
        + cand[:, 2:3] * c[None, :]
    )
    # Evaluate squared norms: for each pair (m) and each candidate.
    # Use broadcasting: (m,1,3) + (1,n,3) -> (m,n,3)
    diff = base[:, None, :] + trans[None, :, :]
    d2 = np.einsum('mki,mki->mk', diff, diff)
    best = np.argmin(d2, axis=1)
    return cand[best].astype(np.int64)


def _shift_to_cart(
    shifts: np.ndarray, domain: Box | OrthorhombicCell | PeriodicCell | None
) -> np.ndarray:
    """Convert integer shifts to Cartesian translation vectors."""

    sh = np.asarray(shifts, dtype=np.int64)
    if sh.ndim != 2 or sh.shape[1] != 3:
        raise ValueError('shifts must have shape (m,3)')
    if domain is None:
        return np.zeros((sh.shape[0], 3), dtype=np.float64)
    if isinstance(domain, Box):
        return np.zeros((sh.shape[0], 3), dtype=np.float64)
    if isinstance(domain, OrthorhombicCell):
        a, b, c = domain.lattice_vectors
        return (
            sh[:, 0:1] * a[None, :] + sh[:, 1:2] * b[None, :] + sh[:, 2:3] * c[None, :]
        )
    if isinstance(domain, PeriodicCell):
        a, b, c = (np.asarray(v, dtype=float) for v in domain.vectors)
        return (
            sh[:, 0:1] * a[None, :] + sh[:, 1:2] * b[None, :] + sh[:, 2:3] * c[None, :]
        )
    raise ValueError('unsupported domain')


def _fit_power_weights_core(
    points: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    t_target: np.ndarray,
    shifts_used: np.ndarray,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell | None,
    constraint_weights: Sequence[float] | None,
    t_bounds: tuple[float, float] | None,
    t_bounds_mode: Literal['none', 'soft_quadratic', 'hard'],
    alpha_out: float,
    t_near_penalty: Literal['none', 'exp'],
    beta_near: float,
    t_margin: float,
    t_tau: float,
    regularize_to: np.ndarray | None,
    lambda_regularize: float,
    r_min: float,
    solver: Literal['auto', 'analytic', 'admm'],
    max_iter: int,
    rho: float,
    tol_abs: float,
    tol_rel: float,
    check_contacts: bool,
    warnings: tuple[str, ...],
) -> FitWeightsResult:
    """Shared implementation for the two public entry points."""

    n = points.shape[0]
    m = i_idx.shape[0]

    # Resolve constraint weights
    if constraint_weights is None:
        omega = np.ones(m, dtype=np.float64)
    else:
        omega = np.asarray(constraint_weights, dtype=float)
        if omega.shape != (m,):
            raise ValueError('constraint_weights must have shape (m,)')
        if np.any(omega < 0):
            raise ValueError('constraint_weights must be non-negative')

    # Distance squared for each constraint (using chosen periodic image).
    pj_star = points[j_idx] + _shift_to_cart(shifts_used, domain)
    delta = pj_star - points[i_idx]
    d2 = np.einsum('mi,mi->m', delta, delta)
    if np.any(d2 <= 0):
        raise ValueError(
            'some constraints have zero distance (coincident points/image)'
        )

    # Convert target t to target weight differences b = d^2(2t-1)
    b = d2 * (2.0 * t_target - 1.0)
    # Quadratic coefficient for mismatch in *t* space:
    # (t(z)-t_target)^2 = (z-b)^2 / (4 d^4)
    a = omega / (4.0 * d2 * d2)

    # Bounds handling
    if t_bounds is None:
        t_lo, t_hi = (0.0, 1.0)
        bounds_enabled = False
    else:
        t_lo = float(t_bounds[0])
        t_hi = float(t_bounds[1])
        if not t_hi > t_lo:
            raise ValueError('t_bounds must satisfy hi > lo')
        bounds_enabled = True

    t_bounds_mode = str(t_bounds_mode)
    if t_bounds_mode not in ('none', 'soft_quadratic', 'hard'):
        raise ValueError('t_bounds_mode must be one of: none, soft_quadratic, hard')
    if not bounds_enabled and t_bounds_mode != 'none':
        raise ValueError('t_bounds_mode requires t_bounds')

    alpha_out = float(alpha_out)
    beta_near = float(beta_near)
    lambda_regularize = float(lambda_regularize)
    rho = float(rho)
    if alpha_out < 0 or beta_near < 0 or lambda_regularize < 0:
        raise ValueError('alpha_out/beta_near/lambda_regularize must be >= 0')
    if rho <= 0:
        raise ValueError('rho must be > 0')
    if max_iter <= 0:
        raise ValueError('max_iter must be > 0')
    if tol_abs <= 0 or tol_rel <= 0:
        raise ValueError('tol_abs and tol_rel must be > 0')
    if t_margin < 0:
        raise ValueError('t_margin must be >= 0')
    if t_tau <= 0:
        raise ValueError('t_tau must be > 0')
    if t_near_penalty not in ('none', 'exp'):
        raise ValueError('t_near_penalty must be "none" or "exp"')
    if (t_near_penalty == 'exp' and beta_near > 0) and (not bounds_enabled):
        raise ValueError('t_near_penalty requires t_bounds (for boundary definitions)')

    # Regularization target weights
    if regularize_to is not None:
        w0 = np.asarray(regularize_to, dtype=float)
        if w0.shape != (n,):
            raise ValueError('regularize_to must have shape (n,)')
    else:
        w0 = np.zeros(n, dtype=np.float64)

    # Determine whether we can use the analytic (quadratic) solver.
    nonquadratic = False
    if bounds_enabled and t_bounds_mode == 'hard':
        # Hard bounds are explicit constraints.
        nonquadratic = True
    if bounds_enabled and t_bounds_mode == 'soft_quadratic' and alpha_out > 0:
        # The hinge makes the objective piecewise.
        nonquadratic = True
    if bounds_enabled and t_near_penalty == 'exp' and beta_near > 0:
        nonquadratic = True

    if solver == 'auto':
        solver_eff = 'analytic' if (not nonquadratic) else 'admm'
    else:
        solver_eff = solver
    if solver_eff not in ('analytic', 'admm'):
        raise ValueError('solver must be auto, analytic, or admm')
    if solver_eff == 'analytic' and nonquadratic:
        raise ValueError(
            'analytic solver cannot be used with bounds/near-boundary penalties'
        )

    # Build connected components on the constraint graph.
    comps = _connected_components(n, i_idx, j_idx)
    weights = np.zeros(n, dtype=np.float64)
    converged_all = True
    n_iter_max = 0

    # Solve each component independently (gauge freedom is per component).
    for nodes in comps:
        if len(nodes) <= 1:
            # isolated node: keep at 0 (or regularization target if lambda > 0?)
            if lambda_regularize > 0 and len(nodes) == 1:
                weights[nodes[0]] = w0[nodes[0]]
            continue

        node_set = set(nodes)
        mask = np.array(
            [
                (int(i) in node_set) and (int(j) in node_set)
                for i, j in zip(i_idx, j_idx)
            ],
            dtype=bool,
        )
        # Local mapping
        local_index = {int(node): k for k, node in enumerate(nodes)}
        ii = np.array([local_index[int(i)] for i in i_idx[mask]], dtype=np.int64)
        jj = np.array([local_index[int(j)] for j in j_idx[mask]], dtype=np.int64)
        d2_c = d2[mask]
        b_c = b[mask]
        a_c = a[mask]

        # Bounds in z-space for hard constraints
        if bounds_enabled:
            z_lo = d2_c * (2.0 * t_lo - 1.0)
            z_hi = d2_c * (2.0 * t_hi - 1.0)
        else:
            z_lo = None
            z_hi = None

        w0_c = w0[np.array(nodes, dtype=np.int64)]

        if solver_eff == 'analytic':
            w_c = _solve_component_analytic(ii, jj, a_c, b_c, w0_c, lambda_regularize)
            iters = 1
            conv = True
        else:
            w_c, iters, conv = _solve_component_admm(
                ii,
                jj,
                d2_c,
                a_c,
                b_c,
                w0_c,
                lambda_regularize=lambda_regularize,
                rho=rho,
                max_iter=max_iter,
                tol_abs=tol_abs,
                tol_rel=tol_rel,
                # penalties
                bounds_enabled=bounds_enabled,
                t_lo=t_lo,
                t_hi=t_hi,
                t_bounds_mode=t_bounds_mode,
                alpha_out=alpha_out,
                t_near_penalty=t_near_penalty,
                beta_near=beta_near,
                t_margin=t_margin,
                t_tau=t_tau,
                z_lo=z_lo,
                z_hi=z_hi,
            )

        # Write back (anchor is internal; weights are gauge-fixed per component)
        weights[np.array(nodes, dtype=np.int64)] = w_c
        converged_all = converged_all and conv
        n_iter_max = max(n_iter_max, iters)

    # Convert weights to radii with requested minimum.
    radii, C = weights_to_radii(weights, r_min=r_min)

    # Predict t for all constraints
    z_pred = weights[i_idx] - weights[j_idx]
    t_pred = 0.5 + z_pred / (2.0 * d2)
    residuals = t_pred - t_target
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    mx = float(np.max(np.abs(residuals)))

    is_contact = None
    inactive: tuple[int, ...] | None = None
    warnings_list = list(warnings)

    if check_contacts:
        if domain is None:
            warnings_list.append(
                'check_contacts=True requires a domain; skipping contact check'
            )
        else:
            try:
                is_contact, inactive = _check_contacts(
                    points, domain, radii, i_idx, j_idx, shifts_used
                )
                if inactive and len(inactive) > 0:
                    warnings_list.append(
                        f'{len(inactive)}/{m} constraints did not correspond to a '
                        'tessellation face (inactive)'
                    )
            except Exception as e:  # pragma: no cover
                warnings_list.append(f'contact check failed: {e!r}')

    return FitWeightsResult(
        weights=weights,
        radii=radii,
        weight_shift=C,
        t_target=np.asarray(t_target, dtype=np.float64),
        t_pred=np.asarray(t_pred, dtype=np.float64),
        residuals=np.asarray(residuals, dtype=np.float64),
        rms_residual=rms,
        max_residual=mx,
        used_shifts=np.asarray(shifts_used, dtype=np.int64),
        is_contact=is_contact,
        inactive_constraints=inactive,
        solver=solver_eff,
        n_iter=int(n_iter_max),
        converged=bool(converged_all),
        warnings=tuple(warnings_list),
    )


def _connected_components(
    n: int, i_idx: np.ndarray, j_idx: np.ndarray
) -> list[list[int]]:
    """Connected components of an undirected graph given by edge list."""
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


def _solve_component_analytic(
    I: np.ndarray,
    J: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    lambda_regularize: float,
) -> np.ndarray:
    """Analytic weighted least squares for a connected component.

    Solves:
        min_w sum_k a_k ( (w_i - w_j) - b_k )^2 + (lambda/2)||w-w0||^2

    with gauge fixed by setting w[0] = 0.
    """

    n_c = int(np.max(np.maximum(I, J))) + 1
    if w0.shape != (n_c,):
        w0 = np.asarray(w0, dtype=float).reshape(n_c)
    lam = float(lambda_regularize)
    # Build weighted Laplacian
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
        # Regularization makes the system strictly convex, so we can solve
        # without anchoring a node.
        return np.linalg.solve(L, rhs).astype(np.float64)

    # Gauge: anchor node 0 to 0.
    free = np.arange(1, n_c, dtype=np.int64)
    Lf = L[np.ix_(free, free)]
    rhsf = rhs[free]
    wf = np.linalg.solve(Lf, rhsf)
    w = np.zeros(n_c, dtype=np.float64)
    w[free] = wf
    w[0] = 0.0
    return w


def _solve_component_admm(
    I: np.ndarray,
    J: np.ndarray,
    d2: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    *,
    lambda_regularize: float,
    rho: float,
    max_iter: int,
    tol_abs: float,
    tol_rel: float,
    # penalties
    bounds_enabled: bool,
    t_lo: float,
    t_hi: float,
    t_bounds_mode: str,
    alpha_out: float,
    t_near_penalty: str,
    beta_near: float,
    t_margin: float,
    t_tau: float,
    z_lo: np.ndarray | None,
    z_hi: np.ndarray | None,
) -> tuple[np.ndarray, int, bool]:
    """ADMM solver for a connected component."""

    n_c = int(np.max(np.maximum(I, J))) + 1
    m_c = I.shape[0]
    lam = float(lambda_regularize)

    # Gauge handling:
    # - if lam == 0, the objective is invariant to adding a constant to all
    #   weights, so we fix the gauge by anchoring node 0 to 0.
    # - if lam > 0, the regularization makes the system strictly convex, so we
    #   do not anchor a node.
    if lam > 0:
        anchor: int | None = None
        free = np.arange(n_c, dtype=np.int64)
    else:
        anchor = 0
        free = np.arange(1, n_c, dtype=np.int64)

    # Build (unweighted) Laplacian for augmented term.
    L = np.zeros((n_c, n_c), dtype=np.float64)
    for i, j in zip(I.tolist(), J.tolist()):
        L[i, i] += 1.0
        L[j, j] += 1.0
        L[i, j] -= 1.0
        L[j, i] -= 1.0

    M = rho * L + lam * np.eye(n_c)
    Mf = M[np.ix_(free, free)]
    # Pre-factorize
    try:
        chol = np.linalg.cholesky(Mf)
    except np.linalg.LinAlgError:
        # As a fallback, add a tiny diagonal and retry.
        Mf2 = Mf + 1e-12 * np.eye(Mf.shape[0])
        chol = np.linalg.cholesky(Mf2)
        Mf = Mf2

    def solve_M(rhs_free: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(chol, rhs_free)
        x = np.linalg.solve(chol.T, y)
        return x

    w = np.zeros(n_c, dtype=np.float64)
    # Initialize z to target differences b (clipped for hard bounds)
    z = b.copy()
    if (
        bounds_enabled
        and t_bounds_mode == 'hard'
        and z_lo is not None
        and z_hi is not None
    ):
        z = np.clip(z, z_lo, z_hi)
    u = np.zeros(m_c, dtype=np.float64)

    # Precompute some constants
    dt_dz = 1.0 / (2.0 * d2)

    left_near = t_lo + t_margin
    right_near = t_hi - t_margin

    converged = False
    z_prev = z.copy()

    for it in range(1, max_iter + 1):
        # w-update: solve (rho L + lam I) w = rho A^T(z - u) + lam w0
        y = z - u
        rhs = np.zeros(n_c, dtype=np.float64)
        # A^T y
        # edge k: +y_k to I[k], -y_k to J[k]
        np.add.at(rhs, I, rho * y)
        np.add.at(rhs, J, -rho * y)
        if lam > 0:
            rhs += lam * w0

        rhs_free = rhs[free]
        w_free = solve_M(rhs_free)
        if anchor is not None:
            w[anchor] = 0.0
        w[free] = w_free

        # z-update: prox over edges
        v = (w[I] - w[J]) + u
        z_prev = z
        z = _prox_edge_objective(
            v,
            d2,
            a,
            b,
            rho=rho,
            dt_dz=dt_dz,
            # bounds/penalties
            bounds_enabled=bounds_enabled,
            t_lo=t_lo,
            t_hi=t_hi,
            t_bounds_mode=t_bounds_mode,
            alpha_out=alpha_out,
            t_near_penalty=t_near_penalty,
            beta_near=beta_near,
            left_near=left_near,
            right_near=right_near,
            t_tau=t_tau,
            z_lo=z_lo,
            z_hi=z_hi,
        )

        # u-update
        Aw = w[I] - w[J]
        r = Aw - z
        u = u + r

        # Convergence check
        r_norm = float(np.linalg.norm(r))
        z_norm = float(np.linalg.norm(z))
        Aw_norm = float(np.linalg.norm(Aw))
        eps_pri = np.sqrt(m_c) * tol_abs + tol_rel * max(Aw_norm, z_norm)

        # Dual residual: rho * A^T (z - z_prev)
        dz = z - z_prev
        s_vec = np.zeros(n_c, dtype=np.float64)
        np.add.at(s_vec, I, rho * dz)
        np.add.at(s_vec, J, -rho * dz)
        s_norm = float(np.linalg.norm(s_vec[free]))
        u_norm = float(np.linalg.norm(u))
        eps_dual = np.sqrt(len(free)) * tol_abs + tol_rel * rho * u_norm

        if r_norm <= eps_pri and s_norm <= eps_dual:
            converged = True
            break

    return w, it, converged


def _prox_edge_objective(
    v: np.ndarray,
    d2: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    rho: float,
    dt_dz: np.ndarray,
    bounds_enabled: bool,
    t_lo: float,
    t_hi: float,
    t_bounds_mode: str,
    alpha_out: float,
    t_near_penalty: str,
    beta_near: float,
    left_near: float,
    right_near: float,
    t_tau: float,
    z_lo: np.ndarray | None,
    z_hi: np.ndarray | None,
) -> np.ndarray:
    """Vectorized proximal operator for per-edge objectives."""

    z = v.copy()
    if (
        bounds_enabled
        and t_bounds_mode == 'hard'
        and z_lo is not None
        and z_hi is not None
    ):
        z = np.clip(z, z_lo, z_hi)

    # Newton iterations (vectorized)
    for _ in range(50):
        t = 0.5 + z / (2.0 * d2)

        # f'(z): mismatch term
        fp = 2.0 * a * (z - b)
        fpp = 2.0 * a

        # Soft out-of-range quadratic penalty
        if bounds_enabled and t_bounds_mode == 'soft_quadratic' and alpha_out > 0:
            # Below lower bound
            m_lo = t < t_lo
            if np.any(m_lo):
                fp[m_lo] += (2.0 * alpha_out * (t[m_lo] - t_lo)) * dt_dz[m_lo]
                fpp[m_lo] += (2.0 * alpha_out) * (dt_dz[m_lo] ** 2)
            # Above upper bound
            m_hi = t > t_hi
            if np.any(m_hi):
                fp[m_hi] += (2.0 * alpha_out * (t[m_hi] - t_hi)) * dt_dz[m_hi]
                fpp[m_hi] += (2.0 * alpha_out) * (dt_dz[m_hi] ** 2)

        # Near-boundary exponential penalty
        if bounds_enabled and t_near_penalty == 'exp' and beta_near > 0:
            # exp((left - t)/tau) + exp((t - right)/tau)
            A = np.exp((left_near - t) / t_tau)
            B = np.exp((t - right_near) / t_tau)
            fp += (beta_near * (-A + B) / t_tau) * dt_dz
            fpp += (beta_near * (A + B) / (t_tau * t_tau)) * (dt_dz**2)

        # Full derivative of objective: f'(z) + rho(z - v)
        g = fp + rho * (z - v)
        gp = fpp + rho
        step = g / gp

        z_new = z - step
        if (
            bounds_enabled
            and t_bounds_mode == 'hard'
            and z_lo is not None
            and z_hi is not None
        ):
            z_new = np.clip(z_new, z_lo, z_hi)

        # Stop criterion
        if float(np.max(np.abs(step))) < 1e-12:
            z = z_new
            break
        z = z_new

    return z


def _check_contacts(
    points: np.ndarray,
    domain: Box | OrthorhombicCell | PeriodicCell,
    radii: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    shifts: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Check which constraints correspond to actual faces in the tessellation."""
    from .api import compute

    periodic = isinstance(domain, PeriodicCell) or (
        isinstance(domain, OrthorhombicCell) and any(domain.periodic)
    )
    cells = compute(
        points,
        domain=domain,
        mode='power',
        radii=radii,
        return_vertices=True,
        return_faces=True,
        return_adjacency=False,
        return_face_shifts=bool(periodic),
        include_empty=True,
    )
    # Build neighbor set
    neigh: set[tuple[int, int, int, int, int]] = set()
    for cell in cells:
        ci = int(cell['id'])
        for face in cell.get('faces', []):
            cj = int(face['adjacent_cell'])
            if cj < 0:
                continue
            sh = face.get('adjacent_shift', (0, 0, 0))
            neigh.add((ci, cj, int(sh[0]), int(sh[1]), int(sh[2])))

    m = i_idx.shape[0]
    is_contact = np.zeros(m, dtype=bool)
    inactive: list[int] = []
    for k in range(m):
        key = (
            int(i_idx[k]),
            int(j_idx[k]),
            int(shifts[k, 0]),
            int(shifts[k, 1]),
            int(shifts[k, 2]),
        )
        rev = (
            int(j_idx[k]),
            int(i_idx[k]),
            int(-shifts[k, 0]),
            int(-shifts[k, 1]),
            int(-shifts[k, 2]),
        )
        ok = (key in neigh) or (rev in neigh)
        is_contact[k] = ok
        if not ok:
            inactive.append(k)
    return is_contact, tuple(inactive)
