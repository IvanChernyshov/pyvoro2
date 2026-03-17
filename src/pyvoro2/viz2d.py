"""Optional matplotlib-based visualization helpers for planar tessellations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class _SupportsPlanarBounds(Protocol):
    """Protocol for simple 2D domains that expose rectangular bounds."""

    bounds: tuple[tuple[float, float], tuple[float, float]]


def plot_tessellation(
    cells: Iterable[dict],
    *,
    ax: Axes | None = None,
    domain: _SupportsPlanarBounds | None = None,
    show_sites: bool = False,
    annotate_ids: bool = False,
) -> tuple[Figure, Axes]:
    """Plot planar cells using matplotlib.

    Args:
        cells: Iterable of raw 2D cell dictionaries as returned by
            ``pyvoro2.planar.compute`` or ``pyvoro2.planar.ghost_cells``.
        ax: Optional existing matplotlib axes.
        domain: Optional planar domain. When it exposes ``bounds``, the domain
            rectangle is drawn as a simple outline.
        show_sites: If True, draw the reported cell sites.
        annotate_ids: If True, label cell IDs at their reported sites.

    Returns:
        ``(fig, ax)``.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if domain is not None and hasattr(domain, 'bounds'):
        try:
            (xmin, xmax), (ymin, ymax) = domain.bounds
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
        else:
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
            )

    for cell in cells:
        vertices = cell.get('vertices') or []
        edges = cell.get('edges') or []
        if vertices and edges:
            for edge in edges:
                vids = edge.get('vertices', ())
                if len(vids) != 2:
                    continue
                i, j = int(vids[0]), int(vids[1])
                if i < 0 or j < 0 or i >= len(vertices) or j >= len(vertices):
                    continue
                vi = vertices[i]
                vj = vertices[j]
                ax.plot([vi[0], vj[0]], [vi[1], vj[1]])

        site = cell.get('site')
        if site is not None and show_sites:
            ax.plot([float(site[0])], [float(site[1])], marker='o', linestyle='None')
        if site is not None and annotate_ids:
            ax.text(float(site[0]), float(site[1]), str(cell.get('id', '?')))

    ax.set_aspect('equal', adjustable='box')
    return fig, ax
