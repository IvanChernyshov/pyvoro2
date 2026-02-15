"""Face-level geometric properties.

This module adds optional post-processing utilities for Voronoi cells returned
by :func:`pyvoro2.compute`.

The core computation in Voro++ is fast and focuses on topology/geometry of the
cells. Many chemistry workflows benefit from extra per-face descriptors (face
centroid, oriented normals, and a few contact heuristics). These can be
expensive, so they are provided as an explicit, opt-in post-processing step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from .diagnostics import TessellationDiagnostics


def _poly_centroid_area_normal(
    v: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Return (centroid, area, unit_normal) for a planar polygon.

    The polygon is assumed to be convex and vertices are assumed to be ordered
    around the face (either CW or CCW). If the polygon is degenerate, returns
    None.
    """
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
        return None

    v0 = v[0]
    area_sum = 0.0
    centroid_sum = np.zeros(3, dtype=np.float64)
    normal_sum = np.zeros(3, dtype=np.float64)

    for i in range(1, v.shape[0] - 1):
        a = v[i] - v0
        b = v[i + 1] - v0
        cr = np.cross(a, b)
        nn = float(np.linalg.norm(cr))
        if nn == 0.0:
            continue
        tri_area = 0.5 * nn
        tri_centroid = (v0 + v[i] + v[i + 1]) / 3.0
        area_sum += tri_area
        centroid_sum += tri_area * tri_centroid
        normal_sum += cr

    if area_sum == 0.0:
        return None

    centroid = centroid_sum / area_sum
    nrm = float(np.linalg.norm(normal_sum))
    if nrm == 0.0:
        return None
    unit = normal_sum / nrm
    area = 0.5 * nrm
    return centroid, float(area), unit


def _point_in_convex_polygon_2d(poly: np.ndarray, p: np.ndarray, eps: float) -> bool:
    """Return True if point p is inside/on boundary of convex polygon poly.

    Args:
        poly: (m,2) vertices in cyclic order.
        p: (2,) point.
        eps: tolerance.
    """
    m = int(poly.shape[0])
    if m < 3:
        return False
    pos = False
    neg = False
    for i in range(m):
        j = (i + 1) % m
        e = poly[j] - poly[i]
        w = p - poly[i]
        cross = float(e[0] * w[1] - e[1] * w[0])
        if cross > eps:
            pos = True
        elif cross < -eps:
            neg = True
        if pos and neg:
            return False
    return True


def _dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point p to segment [a,b]."""
    ab = b - a
    den = float(np.dot(ab, ab))
    if den == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / den)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def annotate_face_properties(
    cells: list[dict[str, Any]],
    domain: Box | OrthorhombicCell | PeriodicCell,
    *,
    diagnostics: TessellationDiagnostics | None = None,
    tol: float = 1e-10,
) -> None:
    """Annotate faces with additional geometric properties.

    This function mutates the provided cell dictionaries in-place.

    Added face fields (when computable):
      - centroid: [x,y,z]
      - normal: [nx,ny,nz] unit vector oriented from site -> face
      - area: float
      - other_site: [x,y,z] (neighbor site position, including periodic shift)
      - intersection: [x,y,z] intersection of (site->other_site) line with face plane
      - intersection_inside: bool
      - intersection_centroid_dist: float
      - intersection_edge_min_dist: float

    Policy note:
        We preserve as much information as possible. `other_site` is only set
        to None when it cannot be determined (missing neighbor id/shift). Global
        diagnostics (gap/overlap) are kept external via `diagnostics`.

    Args:
        cells: Output list from :func:`pyvoro2.compute`.
        domain: Domain used for computation.
        diagnostics: Optional diagnostics object (not required). If provided,
            and `diagnostics` marked faces with local flags (e.g. orphan), those
            flags are preserved but do not suppress property computation.
        tol: Numerical tolerance.
    """
    _ = diagnostics  # reserved for future policy hooks

    # Map id -> site (Cartesian)
    site_by_id: dict[int, np.ndarray] = {}
    for c in cells:
        cid = int(c.get('id', -1))
        s = np.asarray(c.get('site', []), dtype=np.float64)
        if cid >= 0 and s.size == 3:
            site_by_id[cid] = s.reshape(3)

    domain_periodic = isinstance(domain, PeriodicCell) or (
        isinstance(domain, OrthorhombicCell) and any(domain.periodic)
    )

    if domain_periodic:
        if isinstance(domain, PeriodicCell):
            vec = np.asarray(domain.vectors, dtype=np.float64)
            a, b, cvec = vec[0], vec[1], vec[2]
        else:
            # OrthorhombicCell
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
            a = np.array([xmax - xmin, 0.0, 0.0], dtype=np.float64)
            b = np.array([0.0, ymax - ymin, 0.0], dtype=np.float64)
            cvec = np.array([0.0, 0.0, zmax - zmin], dtype=np.float64)

        def _other_site(i: int, f: dict[str, Any]) -> np.ndarray | None:
            j = int(f.get('adjacent_cell', -999999))
            if j < 0:
                return None
            sj = site_by_id.get(j)
            if sj is None:
                return None
            if 'adjacent_shift' not in f:
                return None
            s = f.get('adjacent_shift', (0, 0, 0))
            try:
                na, nb, nc = int(s[0]), int(s[1]), int(s[2])
            except Exception:
                return None
            return sj + na * a + nb * b + nc * cvec

    else:

        def _other_site(i: int, f: dict[str, Any]) -> np.ndarray | None:
            j = int(f.get('adjacent_cell', -999999))
            if j < 0:
                return None
            return site_by_id.get(j)

    eps2d = float(max(tol, 1e-12))

    for c in cells:
        cid = int(c.get('id', -1))
        site = site_by_id.get(cid)
        faces = c.get('faces')
        if site is None or faces is None:
            continue
        verts = np.asarray(c.get('vertices', []), dtype=np.float64)
        if verts.size == 0:
            continue

        for f in faces:
            idx = np.asarray(f.get('vertices', []), dtype=np.int64)
            if idx.size < 3:
                # Degenerate face
                f.setdefault('centroid', None)
                f.setdefault('normal', None)
                f.setdefault('area', 0.0)
                f.setdefault('other_site', None)
                f.setdefault('intersection', None)
                f.setdefault('intersection_inside', None)
                f.setdefault('intersection_centroid_dist', None)
                f.setdefault('intersection_edge_min_dist', None)
                continue

            vv = verts[idx]
            props = _poly_centroid_area_normal(vv)
            if props is None:
                f['centroid'] = None
                f['normal'] = None
                f['area'] = 0.0
                # still try other_site
                other = _other_site(cid, f)
                f['other_site'] = other.tolist() if other is not None else None
                f['intersection'] = None
                f['intersection_inside'] = None
                f['intersection_centroid_dist'] = None
                f['intersection_edge_min_dist'] = None
                continue

            centroid, area, n = props
            # Orient normal from site -> face
            if float(np.dot(n, centroid - site)) < 0.0:
                n = -n

            f['centroid'] = centroid.tolist()
            f['area'] = float(area)
            f['normal'] = n.tolist()

            other = _other_site(cid, f)
            f['other_site'] = other.tolist() if other is not None else None

            # Intersection-based descriptors (only if other_site is known)
            if other is None:
                f['intersection'] = None
                f['intersection_inside'] = None
                f['intersection_centroid_dist'] = None
                f['intersection_edge_min_dist'] = None
                continue

            d = other - site
            denom = float(np.dot(n, d))
            if abs(denom) < float(tol):
                f['intersection'] = None
                f['intersection_inside'] = None
                f['intersection_centroid_dist'] = None
                f['intersection_edge_min_dist'] = None
                continue

            # Plane point from any vertex (vv[0])
            t = float(np.dot(n, (vv[0] - site)) / denom)
            x = site + t * d

            # Project to 2D basis to test inside.
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(n[0])) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            u = np.cross(n, ref)
            un = float(np.linalg.norm(u))
            if un == 0.0:
                f['intersection'] = x.tolist()
                f['intersection_inside'] = None
                f['intersection_centroid_dist'] = float(np.linalg.norm(x - centroid))
                f['intersection_edge_min_dist'] = None
                continue
            u = u / un
            v = np.cross(n, u)

            poly2 = np.stack([vv @ u, vv @ v], axis=1)
            p2 = np.array([float(np.dot(x, u)), float(np.dot(x, v))], dtype=np.float64)
            inside = _point_in_convex_polygon_2d(poly2, p2, eps2d)

            # Edge distances
            dmin = float('inf')
            for i in range(vv.shape[0]):
                j = (i + 1) % vv.shape[0]
                dmin = min(dmin, _dist_point_to_segment(x, vv[i], vv[j]))
            if not np.isfinite(dmin):
                dmin = None

            f['intersection'] = x.tolist()
            f['intersection_inside'] = bool(inside)
            f['intersection_centroid_dist'] = float(np.linalg.norm(x - centroid))
            f['intersection_edge_min_dist'] = float(dmin) if dmin is not None else None
