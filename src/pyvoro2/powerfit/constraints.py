"""Constraint parsing and geometric normalization for inverse power fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ..domains import Box, OrthorhombicCell, PeriodicCell

ConstraintInput = Sequence[
    tuple[int, int, float] | tuple[int, int, float, tuple[int, int, int]]
]


@dataclass(frozen=True, slots=True)
class PairBisectorConstraints:
    """Resolved pairwise separator constraints.

    This object is the stable boundary between downstream pair-selection logic
    and pyvoro2's inverse solver. Each row refers to a specific ordered pair
    ``(i, j, shift)`` where ``shift`` is the lattice image applied to site ``j``.
    """

    n_points: int
    i: np.ndarray
    j: np.ndarray
    shifts: np.ndarray
    target: np.ndarray
    confidence: np.ndarray
    measurement: Literal['fraction', 'position']
    distance: np.ndarray
    distance2: np.ndarray
    delta: np.ndarray
    target_fraction: np.ndarray
    target_position: np.ndarray
    input_index: np.ndarray
    explicit_shift: np.ndarray
    ids: np.ndarray | None
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        m = int(self.i.shape[0])
        if self.i.shape != (m,) or self.j.shape != (m,):
            raise ValueError('PairBisectorConstraints.i/j must have shape (m,)')
        if self.shifts.shape != (m, 3):
            raise ValueError('PairBisectorConstraints.shifts must have shape (m, 3)')
        for name in (
            'target',
            'confidence',
            'distance',
            'distance2',
            'target_fraction',
            'target_position',
            'input_index',
            'explicit_shift',
        ):
            arr = getattr(self, name)
            if arr.shape != (m,):
                raise ValueError(f'PairBisectorConstraints.{name} must have shape (m,)')
        if self.delta.shape != (m, 3):
            raise ValueError('PairBisectorConstraints.delta must have shape (m, 3)')
        if self.measurement not in ('fraction', 'position'):
            raise ValueError('measurement must be "fraction" or "position"')

    @property
    def n_constraints(self) -> int:
        return int(self.i.shape[0])

    def subset(self, mask: np.ndarray) -> PairBisectorConstraints:
        """Return a subset with row order preserved."""

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_constraints,):
            raise ValueError('mask must have shape (m,)')
        return PairBisectorConstraints(
            n_points=self.n_points,
            i=self.i[mask].copy(),
            j=self.j[mask].copy(),
            shifts=self.shifts[mask].copy(),
            target=self.target[mask].copy(),
            confidence=self.confidence[mask].copy(),
            measurement=self.measurement,
            distance=self.distance[mask].copy(),
            distance2=self.distance2[mask].copy(),
            delta=self.delta[mask].copy(),
            target_fraction=self.target_fraction[mask].copy(),
            target_position=self.target_position[mask].copy(),
            input_index=self.input_index[mask].copy(),
            explicit_shift=self.explicit_shift[mask].copy(),
            ids=None if self.ids is None else self.ids.copy(),
            warnings=self.warnings,
        )


def resolve_pair_bisector_constraints(
    points: np.ndarray,
    constraints: ConstraintInput,
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: Box | OrthorhombicCell | PeriodicCell | None = None,
    ids: Sequence[int] | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: Sequence[float] | None = None,
    allow_empty: bool = False,
) -> PairBisectorConstraints:
    """Parse and resolve pairwise separator constraints.

    Args:
        points: Site coordinates with shape ``(n, 3)``.
        constraints: Raw constraint tuples ``(i, j, value[, shift])``.
        measurement: Whether ``value`` is interpreted as a normalized fraction
            in ``[0, 1]`` or as an absolute position along the connector.
        domain: Optional non-periodic or periodic domain.
        ids: External ids used when ``index_mode='id'``.
        index_mode: Interpret the first two tuple entries as internal indices or
            external ids.
        image: Shift resolution policy for tuples that do not specify a shift.
        image_search: Search radius for nearest-image resolution in triclinic
            periodic cells.
        confidence: Optional non-negative per-constraint weights.
        allow_empty: Allow zero constraints and return an empty resolved object.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if measurement not in ('fraction', 'position'):
        raise ValueError('measurement must be "fraction" or "position"')

    i_idx, j_idx, target, shifts, shift_given, warnings = _parse_constraints(
        constraints,
        n_points=pts.shape[0],
        ids=ids,
        index_mode=index_mode,
        allow_empty=allow_empty,
    )

    m = int(i_idx.shape[0])
    if confidence is None:
        omega = np.ones(m, dtype=np.float64)
    else:
        omega = np.asarray(confidence, dtype=float)
        if omega.shape != (m,):
            raise ValueError('confidence must have shape (m,)')
        if np.any(omega < 0):
            raise ValueError('confidence must be non-negative')

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

    if m == 0:
        ids_arr = None if ids is None else np.asarray(ids)
        zeros_i = np.zeros(0, dtype=np.int64)
        zeros_f = np.zeros(0, dtype=np.float64)
        zeros_s = np.zeros((0, 3), dtype=np.int64)
        zeros_b = np.zeros(0, dtype=bool)
        return PairBisectorConstraints(
            n_points=int(pts.shape[0]),
            i=zeros_i,
            j=zeros_i.copy(),
            shifts=zeros_s,
            target=zeros_f,
            confidence=zeros_f,
            measurement=measurement,
            distance=zeros_f,
            distance2=zeros_f,
            delta=np.zeros((0, 3), dtype=np.float64),
            target_fraction=zeros_f,
            target_position=zeros_f,
            input_index=zeros_i,
            explicit_shift=zeros_b,
            ids=ids_arr,
            warnings=warnings,
        )

    pj_star = pts2[j_idx] + shift_to_cart(shifts_used, domain)
    delta = pj_star - pts2[i_idx]
    d2 = np.einsum('mi,mi->m', delta, delta)
    if np.any(d2 <= 0.0):
        raise ValueError('some constraints have zero distance (coincident points/image)')
    d = np.sqrt(d2)

    target_arr = np.asarray(target, dtype=np.float64)
    if measurement == 'fraction':
        target_fraction = target_arr.copy()
        target_position = target_fraction * d
    else:
        target_position = target_arr.copy()
        target_fraction = target_position / d

    ids_arr = None if ids is None else np.asarray(ids)

    return PairBisectorConstraints(
        n_points=int(pts.shape[0]),
        i=np.asarray(i_idx, dtype=np.int64),
        j=np.asarray(j_idx, dtype=np.int64),
        shifts=np.asarray(shifts_used, dtype=np.int64),
        target=target_arr,
        confidence=omega,
        measurement=measurement,
        distance=np.asarray(d, dtype=np.float64),
        distance2=np.asarray(d2, dtype=np.float64),
        delta=np.asarray(delta, dtype=np.float64),
        target_fraction=np.asarray(target_fraction, dtype=np.float64),
        target_position=np.asarray(target_position, dtype=np.float64),
        input_index=np.arange(m, dtype=np.int64),
        explicit_shift=np.asarray(shift_given, dtype=bool),
        ids=ids_arr,
        warnings=warnings,
    )


# ---------------------------- internal helpers ----------------------------


def _parse_constraints(
    constraints: ConstraintInput,
    *,
    n_points: int,
    ids: Sequence[int] | None,
    index_mode: Literal['index', 'id'],
    allow_empty: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Parse raw tuple/list constraints.

    Accepted forms:
        ``(i, j, value)``
        ``(i, j, value, shift)``
    """

    if index_mode not in ('index', 'id'):
        raise ValueError('index_mode must be "index" or "id"')
    if index_mode == 'id':
        if ids is None:
            raise ValueError('ids must be provided when index_mode="id"')
        id_to_index = {int(v): k for k, v in enumerate(ids)}
    else:
        id_to_index = None

    m = len(constraints)
    if m == 0 and not allow_empty:
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


def maybe_remap_points(
    points: np.ndarray, domain: Box | OrthorhombicCell | PeriodicCell | None
) -> np.ndarray:
    return _maybe_remap_points(points, domain)



def _maybe_remap_points(
    points: np.ndarray, domain: Box | OrthorhombicCell | PeriodicCell | None
) -> np.ndarray:
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

    if domain is None:
        if np.any(shifts[shift_given] != 0):
            raise ValueError('constraint shifts require a periodic domain')
        return np.zeros((m, 3), dtype=np.int64), tuple(warnings)

    if isinstance(domain, Box):
        if np.any(shifts[shift_given] != 0):
            raise ValueError('Box domain does not support periodic shifts')
        return np.zeros((m, 3), dtype=np.int64), tuple(warnings)

    shifts2 = shifts.copy()
    provided_mask = shift_given.copy()

    if image == 'given_only':
        if np.any(~provided_mask):
            raise ValueError('some constraints are missing shifts (image="given_only")')
        _validate_shifts_against_domain(shifts2, domain)
        return shifts2, tuple(warnings)

    if image != 'nearest':
        raise ValueError('image must be "nearest" or "given_only"')
    if image_search < 0:
        raise ValueError('image_search must be >= 0')

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
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = cell.bounds
    L = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
    per = np.array(cell.periodic, dtype=bool)
    delta = pj - pi
    s = np.zeros_like(delta, dtype=np.int64)
    for ax in range(3):
        if not per[ax]:
            continue
        s[:, ax] = (-np.round(delta[:, ax] / L[ax])).astype(np.int64)
    return s


def _nearest_image_shifts_triclinic(
    pi: np.ndarray, pj: np.ndarray, cell: PeriodicCell, *, search: int = 1
) -> np.ndarray:
    a, b, c = (np.asarray(v, dtype=float) for v in cell.vectors)
    rng = np.arange(-search, search + 1, dtype=np.int64)
    cand = np.array(np.meshgrid(rng, rng, rng, indexing='ij')).reshape(3, -1).T
    base = pj - pi
    trans = (
        cand[:, 0:1] * a[None, :]
        + cand[:, 1:2] * b[None, :]
        + cand[:, 2:3] * c[None, :]
    )
    diff = base[:, None, :] + trans[None, :, :]
    d2 = np.einsum('mki,mki->mk', diff, diff)
    best = np.argmin(d2, axis=1)
    return cand[best].astype(np.int64)


def shift_to_cart(
    shifts: np.ndarray, domain: Box | OrthorhombicCell | PeriodicCell | None
) -> np.ndarray:
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
            sh[:, 0:1] * a[None, :]
            + sh[:, 1:2] * b[None, :]
            + sh[:, 2:3] * c[None, :]
        )
    if isinstance(domain, PeriodicCell):
        a, b, c = (np.asarray(v, dtype=float) for v in domain.vectors)
        return (
            sh[:, 0:1] * a[None, :]
            + sh[:, 1:2] * b[None, :]
            + sh[:, 2:3] * c[None, :]
        )
    raise ValueError('unsupported domain')
