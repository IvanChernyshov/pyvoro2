"""Domain specifications for Voronoi tessellation.

pyvoro2 currently supports:
- Box: orthogonal bounding box (non-periodic, for 0D systems)
- OrthorhombicCell: orthogonal cell with optional per-axis periodicity
  (1D/2D/3D periodic)
- PeriodicCell: fully periodic triclinic cell (3D crystals), implemented via
  a coordinate transform into Voro++'s lower-triangular representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np


def _default_snap_eps(L: float, *, rel: float = 1e-12) -> float:
    """Return a scale-relative snapping epsilon for remapping.

    The returned value scales with ``L`` and has **no** hard absolute floor.
    A small machine-epsilon-based lower bound keeps it from becoming
    numerically ineffective for typical floating-point ranges.
    """

    Lf = float(L)
    if not np.isfinite(Lf) or Lf <= 0.0:
        return 0.0
    epsf = float(np.finfo(float).eps)
    # Keep everything scale-relative: both terms are proportional to L.
    return float(max(rel * Lf, 64.0 * epsf * Lf))


@dataclass(frozen=True, slots=True)
class Box:
    """Orthogonal bounding box domain.

    Args:
        bounds: Three (min, max) pairs for x, y, z.

    Raises:
        ValueError: If bounds are malformed or degenerate.
    """

    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.bounds) != 3:
            raise ValueError('bounds must have length 3')
        for lo, hi in self.bounds:
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError('bounds must be finite')
            if not hi > lo:
                raise ValueError('each bound must satisfy hi > lo')

    @classmethod
    def from_points(cls, points: np.ndarray, padding: float = 2.0) -> 'Box':
        """Create a box that encloses points with optional padding.

        Args:
            points: Array of shape (n, 3).
            padding: Padding added on each side, in the same units as points.

        Returns:
            Box: Bounding box.

        Raises:
            ValueError: If points shape is invalid.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError('points must have shape (n, 3)')
        mins = pts.min(axis=0) - padding
        maxs = pts.max(axis=0) + padding
        return cls(
            bounds=(
                (float(mins[0]), float(maxs[0])),
                (float(mins[1]), float(maxs[1])),
                (float(mins[2]), float(maxs[2])),
            )
        )


@dataclass(frozen=True, slots=True)
class OrthorhombicCell:
    """Orthorhombic cell with optional periodicity along each axis.

    This domain corresponds to Voro++'s rectangular containers (`container` and
    `container_poly`) with per-axis periodic flags.

    Args:
        bounds: Three (min, max) pairs for x, y, z.
        periodic: Tuple of three booleans (px, py, pz). If an axis is periodic,
            points may lie outside the corresponding bounds and will be remapped
            by Voro++ into the primary domain.

    Notes:
        - For periodic axes, the primary domain uses a half-open convention:
          x in [xmin, xmax), etc.
        - For non-periodic axes, the container has walls at the bounds.
    """

    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    periodic: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        if len(self.bounds) != 3:
            raise ValueError('bounds must have length 3')
        for lo, hi in self.bounds:
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError('bounds must be finite')
            if not hi > lo:
                raise ValueError('each bound must satisfy hi > lo')
        if len(self.periodic) != 3:
            raise ValueError('periodic must have length 3')
        # Normalize to a plain tuple[bool,bool,bool]
        object.__setattr__(
            self,
            'periodic',
            (bool(self.periodic[0]), bool(self.periodic[1]), bool(self.periodic[2])),
        )

    @property
    def lattice_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return lattice vectors (a, b, c) for this orthorhombic cell.

        Vectors are returned in Cartesian coordinates and correspond to
        translations that map the cell onto itself.
        """
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        a = np.array([xmax - xmin, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, ymax - ymin, 0.0], dtype=np.float64)
        c = np.array([0.0, 0.0, zmax - zmin], dtype=np.float64)
        return a, b, c

    def remap_cart(
        self,
        points: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Remap Cartesian points into the primary orthorhombic domain.

        For each periodic axis, points are wrapped into the half-open interval
        [min, max). For non-periodic axes, coordinates are left unchanged.

        Args:
            points: Array of shape (n, 3).
            return_shifts: If True, also return integer shifts (nx, ny, nz)
                such that:

                    p_original ~= p_remapped + nx*a + ny*b + nz*c

                where a/b/c are the cell lattice vectors.
            eps: Optional snapping tolerance. If None, defaults to
                1e-12 * L where L is the maximum periodic axis length.

        Returns:
            If return_shifts is False:
                Remapped coordinates, shape (n, 3).
            If return_shifts is True:
                (remapped_points, shifts) where shifts has shape (n, 3)
                and contains integer (nx, ny, nz).
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError('points must have shape (n, 3)')

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        Lx = float(xmax - xmin)
        Ly = float(ymax - ymin)
        Lz = float(zmax - zmin)

        if eps is None:
            # Use the maximum length among periodic axes (no hard floor).
            Lp = 0.0
            if bool(self.periodic[0]):
                Lp = max(Lp, Lx)
            if bool(self.periodic[1]):
                Lp = max(Lp, Ly)
            if bool(self.periodic[2]):
                Lp = max(Lp, Lz)
            eps_val = _default_snap_eps(Lp)
        else:
            eps_val = float(eps)
            if eps_val < 0:
                raise ValueError('eps must be >= 0')

        x = pts[:, 0].astype(float, copy=True)
        y = pts[:, 1].astype(float, copy=True)
        z = pts[:, 2].astype(float, copy=True)

        shifts = np.zeros((pts.shape[0], 3), dtype=np.int64)

        for axis, (lo, hi, L, is_per) in enumerate(
            (
                (xmin, xmax, Lx, self.periodic[0]),
                (ymin, ymax, Ly, self.periodic[1]),
                (zmin, zmax, Lz, self.periodic[2]),
            )
        ):
            if not is_per:
                continue
            coord = x if axis == 0 else y if axis == 1 else z
            # Wrap into [lo, hi) using floor.
            s = np.floor((coord - lo) / L).astype(np.int64)
            coord -= s * L
            shifts[:, axis] = s

            if eps_val > 0.0:
                # Snap near the lower boundary to lo.
                m0 = np.abs(coord - lo) < eps_val
                if np.any(m0):
                    coord[m0] = lo
                # Snap near the upper boundary to lo with shift increment.
                m1 = coord >= (hi - eps_val)
                if np.any(m1):
                    coord[m1] = lo
                    shifts[m1, axis] += 1

            if axis == 0:
                x = coord
            elif axis == 1:
                y = coord
            else:
                z = coord

        out = np.stack([x, y, z], axis=1).astype(np.float64)
        if return_shifts:
            return out, shifts
        return out


@dataclass(frozen=True, slots=True)
class PeriodicCell:
    """Fully periodic triclinic cell for 3D crystals.

    The user provides three lattice vectors in Cartesian coordinates. Internally,
    pyvoro2 converts them into the Voro++ periodic container representation:
        a = (bx, 0, 0)
        b = (bxy, by, 0)
        c = (bxz, byz, bz)

    and transforms points into that coordinate system before tessellation.

    Args:
        vectors: Three lattice vectors (a, b, c), each length-3.
        origin: Origin of the unit cell in Cartesian coordinates.

    Raises:
        ValueError: If vectors are malformed or degenerate.
    """

    vectors: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        vec = np.asarray(self.vectors, dtype=float)
        if vec.shape != (3, 3):
            raise ValueError('vectors must have shape (3, 3)')
        if not np.all(np.isfinite(vec)):
            raise ValueError('vectors must contain only finite values')

        org = np.asarray(self.origin, dtype=float)
        if org.shape != (3,):
            raise ValueError('origin must be a length-3 vector')
        if not np.all(np.isfinite(org)):
            raise ValueError('origin must contain only finite values')

        # Basic non-degeneracy checks.
        norms = np.linalg.norm(vec, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError('cell vectors must have positive finite lengths')

        det = float(np.linalg.det(vec))
        if not np.isfinite(det):
            raise ValueError('cell vectors produce a non-finite determinant')
        if det == 0.0:
            raise ValueError('cell vectors are degenerate (det == 0)')

        # Near-degeneracy detection:
        #   - relvol ~ 0 indicates near-coplanar / almost-degenerate cells.
        #   - a huge condition number indicates numerical instability for
        #     matrix inversions and image bookkeeping.
        s = np.linalg.svd(vec, compute_uv=False)
        if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError('cell vectors are singular or ill-defined')
        smax = float(np.max(s))
        smin = float(np.min(s))
        cond = float(smax / smin) if smin > 0 else float('inf')

        relvol = float(abs(det) / float(norms[0] * norms[1] * norms[2]))

        # Raise for truly near-coplanar / nearly singular cells.
        if relvol < 1e-12 or cond > 1e15:
            raise ValueError(
                'cell vectors are nearly degenerate (poorly conditioned). '
                f'relvol={relvol:.3g}, cond={cond:.3g}. '
                'Use a well-conditioned 3D cell (non-coplanar vectors) or '
                'rescale/re-parameterize your lattice.'
            )

        # Warn for very ill-conditioned (but not nearly singular) cells.
        # This can happen for extreme aspect-ratio boxes/slabs.
        if cond > 1e10:
            warnings.warn(
                'PeriodicCell lattice vectors are very ill-conditioned '
                f'(cond≈{cond:.3g}). Numerical accuracy and periodic image '
                'bookkeeping may be unstable; consider rescaling or using a '
                'better-conditioned basis.',
                RuntimeWarning,
                stacklevel=2,
            )

    @classmethod
    def from_params(
        cls,
        bx: float,
        bxy: float,
        by: float,
        bxz: float,
        byz: float,
        bz: float,
        *,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> 'PeriodicCell':
        """Create a PeriodicCell from Voro++ internal cell parameters.

        Voro++ represents a triclinic periodic cell via the lower-triangular
        lattice vectors:

            a = (bx,  0,  0)
            b = (bxy, by, 0)
            c = (bxz, byz, bz)

        This constructor allows users to initialize a :class:`PeriodicCell`
        directly using those parameters.

        Args:
            bx: x component of vector a.
            bxy: x component of vector b.
            by: y component of vector b.
            bxz: x component of vector c.
            byz: y component of vector c.
            bz: z component of vector c.
            origin: Origin of the unit cell in Cartesian coordinates.

        Returns:
            PeriodicCell: A fully periodic triclinic cell.

        Notes:
            This constructor does not impose additional validation beyond what
            is performed by :meth:`to_internal_params`. In particular,
            :meth:`to_internal_params` will raise a ValueError if bx, by, or bz
            are non-positive (which would indicate an invalid handedness).
        """
        a = (float(bx), 0.0, 0.0)
        b = (float(bxy), float(by), 0.0)
        c = (float(bxz), float(byz), float(bz))
        return cls(vectors=(a, b, c), origin=origin)

    def _rotation_to_internal(self) -> np.ndarray:
        """Return the 3x3 rotation that maps Cartesian -> internal basis."""
        a, b, _c = np.asarray(self.vectors, dtype=float)
        e1 = a / np.linalg.norm(a)
        b_perp = b - np.dot(b, e1) * e1
        nb = np.linalg.norm(b_perp)
        if nb == 0:
            raise ValueError('vectors a and b are colinear')
        e2 = b_perp / nb
        e3 = np.cross(e1, e2)
        r = np.vstack([e1, e2, e3])
        return r

    def to_internal_params(self) -> tuple[float, float, float, float, float, float]:
        """Convert lattice vectors into Voro++ periodic cell parameters.

        Returns:
            Tuple of (bx, bxy, by, bxz, byz, bz).
        """
        r = self._rotation_to_internal()
        a, b, c = (r @ np.asarray(self.vectors, dtype=float).T).T
        bx = float(a[0])
        bxy = float(b[0])
        by = float(b[1])
        bxz = float(c[0])
        byz = float(c[1])
        bz = float(c[2])
        if bx <= 0 or by <= 0 or bz <= 0:
            raise ValueError(
                'internal cell parameters must be positive (check handedness)'
            )
        return bx, bxy, by, bxz, byz, bz

    def cart_to_internal(self, points: np.ndarray) -> np.ndarray:
        """Transform Cartesian points into the internal coordinate system."""
        r = self._rotation_to_internal()
        origin = np.asarray(self.origin, dtype=float)
        pts = np.asarray(points, dtype=float) - origin[None, :]
        return (r @ pts.T).T

    def internal_to_cart(self, points_internal: np.ndarray) -> np.ndarray:
        """Transform internal points back into Cartesian coordinates."""
        r = self._rotation_to_internal()
        origin = np.asarray(self.origin, dtype=float)
        pts = (r.T @ np.asarray(points_internal, dtype=float).T).T + origin[None, :]
        return pts

    def remap_internal(
        self,
        points_internal: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Remap internal coordinates into the primary periodic domain.

        Voro++ stores a triclinic periodic domain using the lower-triangular
        vectors:

            a = (bx,  0,  0)
            b = (bxy, by, 0)
            c = (bxz, byz, bz)

        Remapping into the primary domain is *not* an independent modulo on
        x/y/z when the cell is sheared (bxy/bxz/byz != 0). In particular,
        wrapping in z shifts x and y, and wrapping in y shifts x.

        This routine performs a lattice-consistent remap equivalent to applying
        integer translations along c, then b, then a. It enforces a half-open
        convention for the primary cell:

            x in [0, bx), y in [0, by), z in [0, bz)

        To make the mapping deterministic at boundaries (and stable under
        repeated remapping), an epsilon snapping rule is applied:

            - values within `eps` of 0 are set to 0
            - values within `eps` of the upper boundary are wrapped to 0 and the
              corresponding lattice shift is incremented

        Notes:
            - This method is provided as an explicit helper.
            - The main `compute(...)` API does **not** pre-wrap points for
              periodic domains; Voro++ remaps points internally.

        Args:
            points_internal: Points in the internal coordinate system,
                shape (n, 3).
            return_shifts: If True, also return integer lattice shifts
                (na, nb, nc) applied to each point.
            eps: Snapping tolerance in internal distance units. If None,
                defaults to 1e-12 * L where L = max(bx, by, bz).

        Returns:
            If return_shifts is False:
                Remapped coordinates, shape (n, 3).
            If return_shifts is True:
                (remapped_points, shifts) where shifts has shape (n, 3)
                and contains integer (na, nb, nc).
        """
        bx, bxy, by, bxz, byz, bz = self.to_internal_params()
        pts = np.asarray(points_internal, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError('points_internal must have shape (n, 3)')

        x = pts[:, 0].astype(float, copy=True)
        y = pts[:, 1].astype(float, copy=True)
        z = pts[:, 2].astype(float, copy=True)

        if eps is None:
            eps_val = _default_snap_eps(max(bx, by, bz))
        else:
            eps_val = float(eps)
            if eps_val < 0:
                raise ValueError('eps must be >= 0')

        na = np.zeros_like(x, dtype=np.int64)
        nb = np.zeros_like(x, dtype=np.int64)
        nc = np.zeros_like(x, dtype=np.int64)

        # Iterate several times: remap -> (optional) snap upper boundary.
        # In normal cases, this converges in 1 iteration. Extra iterations
        # handle points that land extremely close to periodic boundaries
        # and would otherwise flip images due to floating-point round-off.
        for _ in range(3):
            # Remap into [0,b) using lower-triangular lattice steps.
            dc = np.floor(z / bz).astype(np.int64)
            z -= dc * bz
            y -= dc * byz
            x -= dc * bxz

            db = np.floor(y / by).astype(np.int64)
            y -= db * by
            x -= db * bxy

            da = np.floor(x / bx).astype(np.int64)
            x -= da * bx

            na += da
            nb += db
            nc += dc

            if eps_val == 0.0:
                break

            # Snap tiny values to 0 (does not change shifts).
            x[np.abs(x) < eps_val] = 0.0
            y[np.abs(y) < eps_val] = 0.0
            z[np.abs(z) < eps_val] = 0.0

            # Snap near upper boundaries to 0 with the corresponding shift increment.
            changed = False

            mz = z >= (bz - eps_val)
            if np.any(mz):
                z[mz] = 0.0
                y[mz] -= byz
                x[mz] -= bxz
                nc[mz] += 1
                changed = True

            my = y >= (by - eps_val)
            if np.any(my):
                y[my] = 0.0
                x[my] -= bxy
                nb[my] += 1
                changed = True

            mx = x >= (bx - eps_val)
            if np.any(mx):
                x[mx] = 0.0
                na[mx] += 1
                changed = True

            if not changed:
                break

        # Final remap to guarantee we are inside the primary cell after any snapping.
        dc = np.floor(z / bz).astype(np.int64)
        z -= dc * bz
        y -= dc * byz
        x -= dc * bxz

        db = np.floor(y / by).astype(np.int64)
        y -= db * by
        x -= db * bxy

        da = np.floor(x / bx).astype(np.int64)
        x -= da * bx

        na += da
        nb += db
        nc += dc

        # Snap tiny values to 0 again for cleanliness.
        if eps_val > 0.0:
            x[np.abs(x) < eps_val] = 0.0
            y[np.abs(y) < eps_val] = 0.0
            z[np.abs(z) < eps_val] = 0.0

        remapped = np.stack([x, y, z], axis=1)

        if not return_shifts:
            return remapped
        shifts = np.stack([na, nb, nc], axis=1).astype(np.int64)
        return remapped, shifts

    def wrap_internal(self, points_internal: np.ndarray) -> np.ndarray:
        """Alias for :meth:`remap_internal`.

        This name existed in early versions of pyvoro2 but previously used an
        incorrect independent modulo for sheared cells.
        """
        return self.remap_internal(points_internal, return_shifts=False)

    def remap_cart(
        self,
        points: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Remap Cartesian points into the primary cell.

        This is a convenience wrapper around :meth:`cart_to_internal`,
        :meth:`remap_internal`, and :meth:`internal_to_cart`.

        Args:
            points: Cartesian coordinates, shape (n, 3).
            return_shifts: If True, also return integer lattice shifts
                (na, nb, nc) applied to each point.
            eps: Snapping tolerance passed through to :meth:`remap_internal`.
        """
        pts_i = self.cart_to_internal(points)
        if return_shifts:
            pts_i2, shifts = self.remap_internal(pts_i, return_shifts=True, eps=eps)
            return self.internal_to_cart(pts_i2), shifts
        pts_i2 = self.remap_internal(pts_i, return_shifts=False, eps=eps)
        return self.internal_to_cart(pts_i2)
