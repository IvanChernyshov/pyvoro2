# SPDX-License-Identifier: MIT
"""Optional 3D visualization helpers.

This module provides a small set of convenience functions for visualizing
pyvoro2 outputs using **py3Dmol** (a Python wrapper around 3Dmol.js).

The dependency is intentionally optional to keep pyvoro2 lightweight.

Install with:

.. code-block:: bash

    pip install "pyvoro2[viz]"

or:

.. code-block:: bash

    pip install py3Dmol

Notes
-----
* The functions here are best-effort helpers meant for interactive exploration.
  They are not a full-featured rendering pipeline.
* The helpers are designed around the dictionaries returned by
  :func:`pyvoro2.compute` and :func:`pyvoro2.ghost_cells`.
* All coordinates are treated as **Cartesian**. For periodic domains, you can
  optionally wrap each cell geometry so that its site lies inside the primary
  domain (see :func:`view_tessellation`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Literal

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from .normalize import NormalizedTopology, NormalizedVertices
from ._util import (
    domain_length_scale,
    domain_lattice_vectors,
    domain_origin,
    is_periodic_domain,
)

try:  # optional dependency
    import py3Dmol as _py3Dmol  # type: ignore
except Exception:  # pragma: no cover
    _py3Dmol = None


def _require_py3dmol() -> Any:
    """Return the imported `py3Dmol` module or raise a helpful ImportError."""

    if _py3Dmol is None:
        raise ImportError(
            'py3Dmol is required for visualization. Install with '
            '`pip install "pyvoro2[viz]"` (or `pip install py3Dmol`).'
        )
    return _py3Dmol


def _xyz(p: Sequence[float]) -> dict[str, float]:
    return {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}


@dataclass(frozen=True, slots=True)
class VizStyle:
    """Styling options for visualization helpers."""

    background: str = '0xffffff'

    # Sites (generators)
    site_color: str = '0x777777'
    # Defaults are intentionally fairly small; 3Dmol scenes get cluttered fast.
    site_radius: float = 0.093
    site_label_color: str = '0x000000'
    site_label_background: str = '0xffffff'
    site_label_font_size: int = 8

    # Cell wireframes
    edge_color: str = '0x1f77b4'
    edge_line_width: float = 2.5

    # Domain wireframe
    domain_color: str = '0x000000'
    domain_line_width: float = 2.5

    # Vertices
    vertex_color: str = '0xff7f0e'
    vertex_radius: float = 0.04
    vertex_label_color: str = '0x000000'
    vertex_label_background: str = '0xffffff'
    vertex_label_font_size: int = 7

    # Axes
    axes_line_width: float = 2.0
    axes_label_font_size: int = 12
    axes_color_x: str = '0xff0000'
    axes_color_y: str = '0x00aa00'
    axes_color_z: str = '0x0000ff'


def make_view(
    *, width: int = 640, height: int = 480, background: str = '0xffffff'
) -> Any:
    """Create a py3Dmol view."""

    py3Dmol = _require_py3dmol()
    v = py3Dmol.view(width=width, height=height)
    v.setBackgroundColor(background)
    return v


def add_axes(
    view: Any,
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    length: float = 1.0,
    line_width: float = 2.0,
    label_font_size: int = 12,
    color_x: str = '0xff0000',
    color_y: str = '0x00aa00',
    color_z: str = '0x0000ff',
) -> Any:
    """Add simple Cartesian axes (x, y, z) as three lines + labels."""

    _require_py3dmol()
    o = np.asarray(origin, dtype=float)
    L = float(length)

    # Axis lines
    view.addLine(
        {
            'start': _xyz(o),
            'end': _xyz(o + np.array([L, 0, 0])),
            'color': color_x,
            'lineWidth': float(line_width),
        }
    )
    view.addLine(
        {
            'start': _xyz(o),
            'end': _xyz(o + np.array([0, L, 0])),
            'color': color_y,
            'lineWidth': float(line_width),
        }
    )
    view.addLine(
        {
            'start': _xyz(o),
            'end': _xyz(o + np.array([0, 0, L])),
            'color': color_z,
            'lineWidth': float(line_width),
        }
    )

    # Labels at the endpoints
    view.addLabel(
        'x',
        {
            'position': _xyz(o + np.array([L, 0, 0])),
            'fontColor': color_x,
            'backgroundColor': '0xffffff',
            'fontSize': int(label_font_size),
        },
    )
    view.addLabel(
        'y',
        {
            'position': _xyz(o + np.array([0, L, 0])),
            'fontColor': color_y,
            'backgroundColor': '0xffffff',
            'fontSize': int(label_font_size),
        },
    )
    view.addLabel(
        'z',
        {
            'position': _xyz(o + np.array([0, 0, L])),
            'fontColor': color_z,
            'backgroundColor': '0xffffff',
            'fontSize': int(label_font_size),
        },
    )

    return view


def add_sites(
    view: Any,
    points: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    color: str = '0x777777',
    radius: float = 0.093,
    label_color: str = '0x000000',
    label_background: str = '0xffffff',
    label_font_size: int = 8,
) -> Any:
    """Add generator sites as spheres, optionally with text labels."""

    _require_py3dmol()
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if labels is not None and len(labels) != len(pts):
        raise ValueError('labels must have the same length as points')

    for i, (x, y, z) in enumerate(pts):
        view.addSphere(
            {
                'center': {'x': float(x), 'y': float(y), 'z': float(z)},
                'radius': float(radius),
                'color': color,
            }
        )
        if labels is not None:
            view.addLabel(
                str(labels[i]),
                {
                    'position': {'x': float(x), 'y': float(y), 'z': float(z)},
                    'fontColor': label_color,
                    'backgroundColor': label_background,
                    'fontSize': int(label_font_size),
                },
            )
    return view


def add_vertices(
    view: Any,
    vertices: np.ndarray,
    *,
    labels: Sequence[str | None] | None = None,
    color: str = '0xff7f0e',
    radius: float = 0.04,
    label_color: str = '0x000000',
    label_background: str = '0xffffff',
    label_font_size: int = 7,
) -> Any:
    """Add vertex markers as small spheres, optionally with labels."""

    _require_py3dmol()
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError('vertices must have shape (m, 3)')
    if labels is not None and len(labels) != len(v):
        raise ValueError('labels must have the same length as vertices')

    for i, (x, y, z) in enumerate(v):
        view.addSphere(
            {
                'center': {'x': float(x), 'y': float(y), 'z': float(z)},
                'radius': float(radius),
                'color': color,
            }
        )
        if labels is not None and labels[i] is not None:
            view.addLabel(
                str(labels[i]),
                {
                    'position': {'x': float(x), 'y': float(y), 'z': float(z)},
                    'fontColor': label_color,
                    'backgroundColor': label_background,
                    'fontSize': int(label_font_size),
                },
            )
    return view


def add_domain_wireframe(
    view: Any,
    domain: Box | OrthorhombicCell | PeriodicCell,
    *,
    color: str = '0x000000',
    line_width: float = 2.5,
) -> Any:
    """Draw the domain boundary as a wireframe."""

    _require_py3dmol()

    if isinstance(domain, (Box, OrthorhombicCell)):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
        o = np.array([xmin, ymin, zmin], dtype=float)
        a = np.array([xmax - xmin, 0.0, 0.0])
        b = np.array([0.0, ymax - ymin, 0.0])
        c = np.array([0.0, 0.0, zmax - zmin])
    elif isinstance(domain, PeriodicCell):
        o = np.array(domain.origin, dtype=float)
        a, b, c = (np.array(v, dtype=float) for v in domain.vectors)
    else:  # pragma: no cover
        raise TypeError(f'Unsupported domain type: {type(domain)!r}')

    corners = [
        o,
        o + a,
        o + b,
        o + c,
        o + a + b,
        o + a + c,
        o + b + c,
        o + a + b + c,
    ]

    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    for i, j in edges:
        view.addLine(
            {
                'start': _xyz(corners[i]),
                'end': _xyz(corners[j]),
                'color': color,
                'lineWidth': float(line_width),
            }
        )
    return view


def add_cell_wireframe(
    view: Any,
    cell: dict[str, Any],
    *,
    color: str = '0x1f77b4',
    line_width: float = 2.5,
) -> Any:
    """Add a single cell wireframe (edges of polygonal faces)."""

    _require_py3dmol()
    verts = cell.get('vertices')
    faces = cell.get('faces')
    if verts is None or faces is None:
        return view
    # Be robust to both list and numpy inputs.
    if len(faces) == 0:
        return view
    v = np.asarray(verts, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or v.size == 0:
        return view

    edges: set[tuple[int, int]] = set()
    for f in faces:
        idx = f.get('vertices')
        if idx is None or len(idx) == 0:
            continue
        m = len(idx)
        for k in range(m):
            a = int(idx[k])
            b = int(idx[(k + 1) % m])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))

    for a, b in edges:
        if a < 0 or b < 0 or a >= len(v) or b >= len(v):
            continue
        view.addLine(
            {
                'start': _xyz(v[a]),
                'end': _xyz(v[b]),
                'color': color,
                'lineWidth': float(line_width),
            }
        )
    return view


def add_tessellation_wireframe(
    view: Any,
    cells: Iterable[dict[str, Any]],
    *,
    color: str = '0x1f77b4',
    line_width: float = 2.5,
    cell_ids: set[int] | None = None,
) -> Any:
    """Add wireframes for multiple cells."""

    _require_py3dmol()
    for c in cells:
        cid = c.get('id')
        if cell_ids is not None and cid not in cell_ids:
            continue
        add_cell_wireframe(view, c, color=color, line_width=line_width)
    return view


def _shift_cell_geometry(cell: dict[str, Any], delta: np.ndarray) -> dict[str, Any]:
    """Return a shallow copy of `cell` with site/vertices shifted by `-delta`."""

    out = dict(cell)
    if out.get('site') is not None:
        s = np.asarray(out['site'], dtype=float)
        out['site'] = (s - delta).tolist()
    if out.get('vertices') is not None:
        v = np.asarray(out['vertices'], dtype=float)
        if v.ndim == 2 and v.shape[1] == 3:
            out['vertices'] = (v - delta).tolist()
    return out


def _dedup_vertices(vertices: np.ndarray, *, tol: float) -> np.ndarray:
    """Deduplicate vertices by quantized coordinate tuples.

    This is intentionally simple and collision-free (unlike hash-mixing).
    The function preserves the first occurrence order.
    """

    v = np.asarray(vertices, dtype=float)
    if v.size == 0:
        return v.reshape((0, 3)).astype(np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError('vertices must have shape (n, 3)')

    tol_f = float(tol)
    if tol_f <= 0:
        raise ValueError('tol must be positive')

    q = np.rint(v / tol_f).astype(np.int64)
    seen: set[tuple[int, int, int]] = set()
    uniq: list[np.ndarray] = []
    for i in range(len(v)):
        key = (int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v[i])
    return np.asarray(uniq, dtype=np.float64)


def _collect_vertex_markers_from_cells(
    cells: Sequence[dict[str, Any]],
    *,
    tol: float,
    cell_ids: set[int] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Collect vertex marker positions and optional labels.

    This helper exists to keep vertex markers consistent with the *drawn*
    wireframes.

    When the user passes a :class:`~pyvoro2.normalize.NormalizedVertices` or
    :class:`~pyvoro2.normalize.NormalizedTopology` object into
    :func:`view_tessellation`, each cell contains a per-local-vertex mapping to
    a **global vertex id** (``vertex_global_id``).

    A single global vertex may appear in multiple periodic images; therefore we
    collect markers in **Cartesian** coordinates from each cell's local vertex
    list and label them with the global id. Markers are deduplicated by
    quantized coordinates *and* global id.
    """

    coords: list[np.ndarray] = []
    labels: list[str] = []
    seen: set[tuple[int, int, int, int]] = set()

    for c in cells:
        cid = c.get('id')
        if cell_ids is not None and cid not in cell_ids:
            continue

        vv = c.get('vertices')
        gids = c.get('vertex_global_id')
        if vv is None or gids is None:
            continue

        v = np.asarray(vv, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or v.size == 0:
            continue
        if len(gids) != len(v):
            continue

        q = np.rint(v / float(tol)).astype(np.int64)
        for i in range(len(v)):
            try:
                gid = int(gids[i])
            except Exception:
                continue
            key = (gid, int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))
            if key in seen:
                continue
            seen.add(key)
            coords.append(v[i])
            labels.append(f'v{gid}')

    if not coords:
        return np.zeros((0, 3), dtype=np.float64), None

    return np.asarray(coords, dtype=np.float64), labels


def view_tessellation(
    cells: Sequence[dict[str, Any]] | NormalizedVertices | NormalizedTopology,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell | None = None,
    show_sites: bool = True,
    show_site_labels: bool = True,
    max_site_labels: int = 200,
    show_domain: bool = True,
    show_axes: bool = True,
    axes_length: float | None = None,
    wrap_cells: bool = False,
    cell_ids: set[int] | None = None,
    show_vertices: bool = False,
    show_vertex_labels: Literal['off', 'auto', 'on'] = 'auto',
    max_vertex_labels: int = 200,
    style: VizStyle | None = None,
    width: int = 640,
    height: int = 480,
    zoom: bool = True,
) -> Any:
    """Create a py3Dmol view of a tessellation.

    Parameters
    ----------
    cells:
        Output of :func:`pyvoro2.compute` or :func:`pyvoro2.ghost_cells`.
        For wireframe rendering, cells must include both `vertices` and `faces`.

        You may also pass the result of :func:`pyvoro2.normalize_vertices` or
        :func:`pyvoro2.normalize_topology` to enable vertex markers/labels using
        global vertex ids.
    domain:
        If provided, draws the domain wireframe.
    wrap_cells:
        If True and `domain` is periodic, translate each cell by an integer
        lattice vector so that its site lies in the primary domain.
        This is purely a visualization convenience.
    show_vertex_labels:
        One of ``"off"``, ``"auto"`` (default), or ``"on"``.

        Vertex labels require **global vertex ids**, which are only available
        when you pass a :class:`~pyvoro2.normalize.NormalizedVertices` or
        :class:`~pyvoro2.normalize.NormalizedTopology` object.

        * ``"off"``: never show vertex labels.
        * ``"auto"``: show labels only when global ids are available; if there
          are many vertices, labels are truncated silently to keep the scene readable.
        * ``"on"``: show labels when global ids are available; warns when labels
          are truncated.
    """

    import warnings

    py3Dmol = _require_py3dmol()
    if show_vertex_labels not in ('off', 'auto', 'on'):
        raise ValueError('show_vertex_labels must be one of: \'off\', \'auto\', \'on\'')
    st = style or VizStyle()
    v = py3Dmol.view(width=width, height=height)
    v.setBackgroundColor(st.background)

    # Unpack normalized inputs if provided
    global_vertices: np.ndarray | None = None
    cell_list: list[dict[str, Any]]
    if isinstance(cells, (NormalizedVertices, NormalizedTopology)):
        global_vertices = np.asarray(cells.global_vertices, dtype=float)
        cell_list = list(cells.cells)
    else:
        cell_list = list(cells)

    # Optionally shift each cell so its site is wrapped into the primary domain
    draw_cells = cell_list
    if wrap_cells and domain is not None and is_periodic_domain(domain):
        # This is purely a visualization convenience:
        # translate each cell by an integer lattice vector so that its *site*
        # lies in the primary domain.
        if not isinstance(domain, Box):
            shifted: list[dict[str, Any]] = []

            # For PeriodicCell we intentionally wrap into the *geometric*
            # parallelepiped spanned by the user-provided vectors.
            # Precompute the inverse once (important for interactive use).
            if isinstance(domain, PeriodicCell):
                o = np.asarray(domain.origin, dtype=float)
                a, b, c = (np.asarray(vv, dtype=float) for vv in domain.vectors)
                A = np.column_stack([a, b, c])
                Ainv = np.linalg.inv(A)

            for cc in cell_list:
                site = cc.get('site')
                if site is None:
                    shifted.append(cc)
                    continue

                s = np.asarray(site, dtype=float).reshape((3,))

                if isinstance(domain, PeriodicCell):
                    # Wrap into origin + u*a + v*b + w*c with u,v,w in [0,1).
                    frac = Ainv @ (s - o)
                    sh = np.floor(frac).astype(np.int64)
                    frac2 = frac - sh.astype(float)

                    # Deterministic half-open convention [0,1).
                    eps = 1e-12
                    if eps > 0.0:
                        for k in range(3):
                            if abs(frac2[k]) < eps:
                                frac2[k] = 0.0
                            if frac2[k] >= 1.0 - eps:
                                frac2[k] = 0.0
                                sh[k] += 1

                    delta = sh[0] * a + sh[1] * b + sh[2] * c
                    shifted.append(_shift_cell_geometry(cc, delta))

                else:
                    # OrthorhombicCell wrapping uses the same definition as
                    # the domain wireframe (half-open periodic axes).
                    a, b, c = domain_lattice_vectors(domain)  # type: ignore[arg-type]
                    _s2, shifts = domain.remap_cart(  # type: ignore[union-attr]
                        s.reshape((1, 3)), return_shifts=True
                    )
                    sh = shifts[0].astype(np.int64)
                    delta = sh[0] * a + sh[1] * b + sh[2] * c
                    shifted.append(_shift_cell_geometry(cc, delta))

            draw_cells = shifted

    # Domain wireframe
    if show_domain and domain is not None:
        add_domain_wireframe(
            v, domain, color=st.domain_color, line_width=st.domain_line_width
        )

    # Axes
    if show_axes:
        if domain is not None:
            o = domain_origin(domain)
            L = domain_length_scale(domain)
        else:
            o = np.zeros(3, dtype=float)
            L = 1.0
        ax_len = (
            float(axes_length) if axes_length is not None else 0.25 * float(max(L, 1.0))
        )
        add_axes(
            v,
            origin=o,
            length=ax_len,
            line_width=st.axes_line_width,
            label_font_size=st.axes_label_font_size,
            color_x=st.axes_color_x,
            color_y=st.axes_color_y,
            color_z=st.axes_color_z,
        )

    # Sites + labels
    if show_sites:
        pts = []
        lbls: list[str] = []
        for idx, c in enumerate(draw_cells):
            site = c.get('site')
            if site is None:
                continue
            pts.append(site)
            lab = c.get('id', c.get('query_index', idx))
            lbls.append(f'p{lab}')
        if pts:
            pts_arr = np.asarray(pts, dtype=float)
            if show_site_labels and len(lbls) <= int(max_site_labels):
                add_sites(
                    v,
                    pts_arr,
                    labels=lbls,
                    color=st.site_color,
                    radius=st.site_radius,
                    label_color=st.site_label_color,
                    label_background=st.site_label_background,
                    label_font_size=st.site_label_font_size,
                )
            else:
                if show_site_labels and len(lbls) > int(max_site_labels):
                    msg = (
                        f'Skipping site labels because n={len(lbls)} exceeds '
                        f'max_site_labels={max_site_labels}.'
                    )
                    warnings.warn(msg)
                add_sites(
                    v, pts_arr, labels=None, color=st.site_color, radius=st.site_radius
                )

    # Cell wireframes
    add_tessellation_wireframe(
        v,
        draw_cells,
        color=st.edge_color,
        line_width=st.edge_line_width,
        cell_ids=cell_ids,
    )

    # Vertex markers (optional)
    if show_vertices:
        # If the user passed a normalized vertex/topology object, prefer to
        # place markers at the *drawn* (local) vertex positions while labeling
        # them with global vertex ids. This avoids a confusing mismatch where
        # wireframes are drawn in one periodic image but global vertices are
        # remapped into the 000 cell.
        used = False
        if global_vertices is not None:
            tol = 1e-8 * (
                float(domain_length_scale(domain)) if domain is not None else 1.0
            )
            vtx, vlabels = _collect_vertex_markers_from_cells(
                draw_cells, tol=tol, cell_ids=cell_ids
            )
            if vtx.size:
                # Decide whether to show labels.
                if show_vertex_labels == 'off':
                    vlabels_use: list[str | None] | None = None
                else:
                    auto = show_vertex_labels == 'auto'
                    if len(vtx) > int(max_vertex_labels):
                        if auto:
                            # In auto mode, keep the scene readable by
                            # truncating the number of labels.
                            m = int(max_vertex_labels)
                            vlabels_use = [
                                lab if i < m else None for i, lab in enumerate(vlabels)
                            ]
                        else:
                            msg = (
                                'Truncating vertex labels to the first '
                                f'{max_vertex_labels} of {len(vtx)}. '
                                'Increase max_vertex_labels to label more.'
                            )
                            warnings.warn(msg)
                            # Keep spheres for all vertices, but only label the
                            # first `max_vertex_labels` to avoid overwhelming
                            # the viewer.
                            m = int(max_vertex_labels)
                            vlabels_use = [
                                lab if i < m else None for i, lab in enumerate(vlabels)
                            ]
                    else:
                        vlabels_use = list(vlabels)
                add_vertices(
                    v,
                    vtx,
                    labels=vlabels_use,
                    color=st.vertex_color,
                    radius=st.vertex_radius,
                    label_color=st.vertex_label_color,
                    label_background=st.vertex_label_background,
                    label_font_size=st.vertex_label_font_size,
                )
                used = True

        if not used:
            if show_vertex_labels == 'on':
                msg = (
                    'show_vertex_labels="on" requested, but global vertex ids are '
                    'not available. Pass the result of '
                    'normalize_vertices/normalize_topology into '
                    'view_tessellation to enable labeled global vertices.'
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            # Fall back to deduplicating local vertices (no global ids).
            all_v = []
            for c in draw_cells:
                cid = c.get('id')
                if cell_ids is not None and cid not in cell_ids:
                    continue
                vv = c.get('vertices')
                if vv is None:
                    continue
                arr = np.asarray(vv, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.size:
                    all_v.append(arr)
            if all_v:
                vv = np.concatenate(all_v, axis=0)
                tol = 1e-8 * (
                    float(domain_length_scale(domain)) if domain is not None else 1.0
                )
                vtx = _dedup_vertices(vv, tol=tol)
                # No labels in this fallback mode (no global ids).
                add_vertices(
                    v, vtx, labels=None, color=st.vertex_color, radius=st.vertex_radius
                )

    if zoom:
        v.zoomTo()
    return v


__all__ = [
    'VizStyle',
    'make_view',
    'add_axes',
    'add_sites',
    'add_vertices',
    'add_domain_wireframe',
    'add_cell_wireframe',
    'add_tessellation_wireframe',
    'view_tessellation',
]
