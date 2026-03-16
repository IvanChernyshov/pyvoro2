"""Constraint parsing and geometric normalization for inverse power fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .._domain_geometry import geometry3d
from ..domains import Box as Box3D, OrthorhombicCell, PeriodicCell
from ..planar._domain_geometry import geometry2d
from ..planar.domains import Box as Box2D, RectangularCell

ConstraintRow = tuple[int, int, float] | tuple[int, int, float, Sequence[int]]
ConstraintInput = Sequence[ConstraintRow]
Domain3D = Box3D | OrthorhombicCell | PeriodicCell
Domain2D = Box2D | RectangularCell
DomainAny = Domain2D | Domain3D


def _plain_value(value: object) -> object:
    return value.item() if hasattr(value, 'item') else value


def _validated_ids_array(ids: Sequence[int] | np.ndarray, n_points: int) -> np.ndarray:
    """Return validated external ids as a 1D NumPy array.

    The power-fit layer uses ids only as external labels and for mapping raw
    constraint tuples when ``index_mode='id'``. The ids must therefore match
    the point array length and be unique.
    """

    if len(ids) != n_points:
        raise ValueError('ids must have length n_points')
    ids_arr = np.asarray(ids)
    if ids_arr.shape != (n_points,):
        raise ValueError('ids must be a 1D sequence of length n_points')
    if np.unique(ids_arr).size != n_points:
        raise ValueError('ids must be unique')
    return ids_arr


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
        if self.shifts.ndim != 2 or self.shifts.shape[0] != m:
            raise ValueError('PairBisectorConstraints.shifts must have shape (m, d)')
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
        if self.delta.ndim != 2 or self.delta.shape[0] != m:
            raise ValueError('PairBisectorConstraints.delta must have shape (m, d)')
        if self.delta.shape[1] != self.shifts.shape[1]:
            raise ValueError(
                'PairBisectorConstraints.delta and shifts must use the same dimension'
            )
        if self.measurement not in ('fraction', 'position'):
            raise ValueError('measurement must be "fraction" or "position"')
        for name in (
            'target',
            'confidence',
            'distance',
            'distance2',
            'delta',
            'target_fraction',
            'target_position',
        ):
            arr = np.asarray(getattr(self, name))
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f'PairBisectorConstraints.{name} must contain only finite values'
                )
        if np.any(self.confidence < 0.0):
            raise ValueError('PairBisectorConstraints.confidence must be non-negative')
        if np.any(self.distance <= 0.0) or np.any(self.distance2 <= 0.0):
            raise ValueError(
                'PairBisectorConstraints distances must be strictly positive'
            )
        if self.ids is not None:
            ids_arr = np.asarray(self.ids)
            if ids_arr.shape != (int(self.n_points),):
                raise ValueError(
                    'PairBisectorConstraints.ids must have shape (n_points,)'
                )
            if np.unique(ids_arr).size != int(self.n_points):
                raise ValueError('PairBisectorConstraints.ids must be unique')

    @property
    def n_constraints(self) -> int:
        return int(self.i.shape[0])

    @property
    def dim(self) -> int:
        return int(self.shifts.shape[1])

    def pair_labels(self, *, use_ids: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return the left/right pair labels as indices or external ids."""

        if use_ids:
            if self.ids is None:
                raise ValueError(
                    'use_ids=True requires ids on the resolved constraint set'
                )
            return self.ids[self.i].copy(), self.ids[self.j].copy()
        return self.i.copy(), self.j.copy()

    def to_records(self, *, use_ids: bool = False) -> tuple[dict[str, object], ...]:
        """Return one plain-Python record per constraint row."""

        left, right = self.pair_labels(use_ids=use_ids)
        rows: list[dict[str, object]] = []
        left_is_int = np.issubdtype(np.asarray(left).dtype, np.integer)
        right_is_int = np.issubdtype(np.asarray(right).dtype, np.integer)
        for k in range(self.n_constraints):
            site_i = int(left[k]) if left_is_int else _plain_value(left[k])
            site_j = int(right[k]) if right_is_int else _plain_value(right[k])
            rows.append(
                {
                    'constraint_index': int(k),
                    'site_i': site_i,
                    'site_j': site_j,
                    'shift': tuple(int(v) for v in self.shifts[k]),
                    'target': float(self.target[k]),
                    'confidence': float(self.confidence[k]),
                    'measurement': self.measurement,
                    'distance': float(self.distance[k]),
                    'target_fraction': float(self.target_fraction[k]),
                    'target_position': float(self.target_position[k]),
                    'input_index': int(self.input_index[k]),
                    'explicit_shift': bool(self.explicit_shift[k]),
                }
            )
        return tuple(rows)

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
    domain: DomainAny | None = None,
    ids: Sequence[int] | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: Sequence[float] | None = None,
    allow_empty: bool = False,
) -> PairBisectorConstraints:
    """Parse and resolve pairwise separator constraints.

    Args:
        points: Site coordinates with shape ``(n, d)`` where ``d`` is currently
            supported for planar (2D) and spatial (3D) workflows.
        constraints: Raw constraint tuples ``(i, j, value[, shift])``.
        measurement: Whether ``value`` is interpreted as a normalized fraction
            in ``[0, 1]`` or as an absolute position along the connector.
        domain: Optional non-periodic or periodic domain.
        ids: External ids used when ``index_mode='id'``.
        index_mode: Interpret the first two tuple entries as internal indices or
            external ids.
        image: Shift resolution policy for tuples that do not specify a shift.
        image_search: Search radius for nearest-image resolution in triclinic
            periodic 3D cells. It is ignored for the current planar backend.
        confidence: Optional non-negative per-constraint weights.
        allow_empty: Allow zero constraints and return an empty resolved object.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError('points must have shape (n, d) with d in {2, 3}')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')
    if measurement not in ('fraction', 'position'):
        raise ValueError('measurement must be "fraction" or "position"')

    ids_arr = None if ids is None else _validated_ids_array(ids, int(pts.shape[0]))

    i_idx, j_idx, target, shifts, shift_given, warnings = _parse_constraints(
        constraints,
        n_points=pts.shape[0],
        ids=ids_arr,
        index_mode=index_mode,
        allow_empty=allow_empty,
        shift_dim=pts.shape[1],
    )

    target_arr = np.asarray(target, dtype=np.float64)
    if not np.all(np.isfinite(target_arr)):
        raise ValueError('constraint values must contain only finite values')

    m = int(i_idx.shape[0])
    if confidence is None:
        omega = np.ones(m, dtype=np.float64)
    else:
        omega = np.asarray(confidence, dtype=float)
        if omega.shape != (m,):
            raise ValueError('confidence must have shape (m,)')
        if not np.all(np.isfinite(omega)):
            raise ValueError('confidence must contain only finite values')
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
        zeros_i = np.zeros(0, dtype=np.int64)
        zeros_f = np.zeros(0, dtype=np.float64)
        zeros_s = np.zeros((0, pts.shape[1]), dtype=np.int64)
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
            delta=np.zeros((0, pts.shape[1]), dtype=np.float64),
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
        raise ValueError(
            'some constraints have zero distance (coincident points/image)'
        )
    d = np.sqrt(d2)

    if measurement == 'fraction':
        target_fraction = target_arr.copy()
        target_position = target_fraction * d
    else:
        target_position = target_arr.copy()
        target_fraction = target_position / d

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
    shift_dim: int,
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
    shifts = np.zeros((m, shift_dim), dtype=np.int64)
    shift_given = np.zeros(m, dtype=bool)
    warnings: list[str] = []

    for k, c in enumerate(constraints):
        if not isinstance(c, (tuple, list)):
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
            if (
                not isinstance(sh, (tuple, list))
                or len(sh) != shift_dim
            ):
                raise ValueError(
                    f'constraint {k} shift must be a length-{shift_dim} tuple'
                )
            shifts[k] = tuple(int(v) for v in sh)
            shift_given[k] = True

    return i_idx, j_idx, val, shifts, shift_given, tuple(warnings)


def maybe_remap_points(points: np.ndarray, domain: DomainAny | None) -> np.ndarray:
    return _maybe_remap_points(points, domain)


def _geometry_for_dim(dim: int, domain: DomainAny | None):
    if dim == 2:
        if domain is not None and not isinstance(domain, (Box2D, RectangularCell)):
            raise ValueError(
                '2D points require domain=None or a planar domain '
                '(pyvoro2.planar.Box or RectangularCell)'
            )
        return geometry2d(domain)
    if dim == 3:
        if domain is not None and not isinstance(
            domain, (Box3D, OrthorhombicCell, PeriodicCell)
        ):
            raise ValueError(
                '3D points require domain=None or a 3D domain '
                '(Box, OrthorhombicCell, or PeriodicCell)'
            )
        return geometry3d(domain)
    raise ValueError('only 2D and 3D points are supported')


def _maybe_remap_points(points: np.ndarray, domain: DomainAny | None) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError('points must have shape (n, d)')
    return _geometry_for_dim(int(pts.shape[1]), domain).remap_cart(pts)


def _resolve_constraint_shifts(
    points: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    shifts: np.ndarray,
    shift_given: np.ndarray,
    *,
    domain: DomainAny | None,
    image: Literal['nearest', 'given_only'],
    image_search: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return per-constraint integer shifts to apply to site j."""

    m = i_idx.shape[0]
    warnings: list[str] = []
    dim = int(points.shape[1])
    geom = _geometry_for_dim(dim, domain)

    shifts = np.asarray(shifts, dtype=np.int64)
    if shifts.shape != (m, dim):
        raise ValueError(f'shifts must have shape (m,{dim})')
    shift_given = np.asarray(shift_given, dtype=bool)
    if shift_given.shape != (m,):
        raise ValueError('shift_given must have shape (m,)')

    if not geom.has_any_periodic_axis:
        geom.validate_shifts(shifts[shift_given])
        return np.zeros((m, dim), dtype=np.int64), tuple(warnings)

    shifts2 = shifts.copy()
    provided_mask = shift_given.copy()

    if image == 'given_only':
        if np.any(~provided_mask):
            raise ValueError('some constraints are missing shifts (image="given_only")')
        geom.validate_shifts(shifts2)
        return shifts2, tuple(warnings)

    if image != 'nearest':
        raise ValueError('image must be "nearest" or "given_only"')
    if image_search < 0:
        raise ValueError('image_search must be >= 0')

    missing = ~provided_mask
    if np.any(missing):
        if dim == 2:
            resolved = geom.nearest_image_shifts(
                points[i_idx[missing]],
                points[j_idx[missing]],
            )
            boundary_hits = np.zeros(resolved.shape[0], dtype=bool)
        else:
            resolved, boundary_hits = geom.nearest_image_shifts(
                points[i_idx[missing]],
                points[j_idx[missing]],
                search=image_search,
            )
        shifts2[missing] = resolved
        warnings.append(
            'some constraints did not specify shifts; using nearest-image shifts'
        )
        if dim == 3 and geom.is_triclinic and np.any(boundary_hits):
            warnings.append(
                'some nearest-image shifts touch the image_search boundary; '
                'increase image_search for extra safety in skewed triclinic cells'
            )

    geom.validate_shifts(shifts2)
    return shifts2, tuple(warnings)


def shift_to_cart(shifts: np.ndarray, domain: DomainAny | None) -> np.ndarray:
    sh = np.asarray(shifts, dtype=np.int64)
    if sh.ndim != 2:
        raise ValueError('shifts must have shape (m, d)')
    return _geometry_for_dim(int(sh.shape[1]), domain).shift_to_cart(sh)
