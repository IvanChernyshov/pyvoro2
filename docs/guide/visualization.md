# Visualization (optional)

Tessellations are geometric objects. Even a small mistake in periodic handling
can lead to results that look plausible numerically but are wrong topologically.
For that reason, it is often worth having a lightweight way to **look at the output**.

pyvoro2 intentionally keeps visualization **optional**:

- the core package has no plotting dependencies;
- planar 2D plotting is handled by a lightweight matplotlib helper;
- spatial 3D viewing is handled by the optional `py3Dmol` helper;
- both are aimed at *debugging and exploratory work*, not publication-quality rendering.

## Installing optional visualization helpers

```bash
# 2D plotting only
pip install "pyvoro2[viz2d]"

# both 2D + 3D helpers
pip install "pyvoro2[viz]"
```

## A minimal 2D example

```python
import numpy as np
import pyvoro2.planar as pv2
from pyvoro2.viz2d import plot_tessellation

pts = np.array([[0.2, 0.2], [0.8, 0.25], [0.4, 0.8]], dtype=float)
domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))

cells = pv2.compute(
    pts,
    domain=domain,
    return_vertices=True,
    return_edges=True,
    return_edge_shifts=True,
)

fig, ax = plot_tessellation(cells, domain=domain, show_sites=True)
```

The 2D helper returns `(fig, ax)` and is best suited for inspecting raw planar
output, debugging periodic edge structure, and checking that a normalized or
power-fitted result looks qualitatively right.

## A minimal 3D example

```python
import numpy as np
import pyvoro2 as pv
from pyvoro2.viz3d import view_tessellation

pts = np.random.default_rng(0).uniform(-1, 1, size=(20, 3))
box = pv.Box(((-2, 2), (-2, 2), (-2, 2)))

cells = pv.compute(
    pts,
    domain=box,
    mode='standard',
    return_vertices=True,
    return_faces=True,
)

v = view_tessellation(
    cells,
    domain=box,
    show_site_labels=True,
    show_axes=True,
    show_vertices=True,
)
v
```

The 3D viewer renders:

- sites as small spheres (with optional text labels like `p0`, `p1`, ...),
- cell faces as a simple wireframe,
- the domain boundary as a wireframe,
- optional vertex markers.

Vertex labels are controlled by `show_vertex_labels`:

- `'off'`: never show vertex labels.
- `'auto'` (default): label vertices only when global vertex ids are available
  (i.e. when you pass the output of `normalize_vertices` / `normalize_topology`).
- `'on'`: always label when global ids are available (may get visually noisy).

If you pass the output of `normalize_vertices` / `normalize_topology`, vertex labels use
**global vertex ids** (`v0`, `v1`, ...). Markers are still placed at the same Cartesian
positions as the drawn wireframes (so labels stay consistent even when a vertex appears
in different periodic images).

## Styling

All visual parameters (sphere sizes, line widths, label sizes, and colors) are grouped
in the `VizStyle` dataclass. You can override only the parameters you care about and
keep the rest at their defaults:

```python
from pyvoro2.viz3d import VizStyle

style = VizStyle(site_radius=0.06, vertex_radius=0.03, edge_line_width=4.0)

v = view_tessellation(cells, domain=box, style=style)
```

## Periodic domains and “vertices outside the cell”

In periodic cells, it is normal for some vertices and edges to appear outside the
geometric parallelepiped of the primary cell.
This does **not** mean the tessellation is wrong: it is a consequence of representing
periodic images in a single Cartesian coordinate system.

For ease of interpretation, `view_tessellation(..., wrap_cells=True)` can translate each
cell by a lattice vector so that its *site* lies inside the primary cell.

For `PeriodicCell`, the viewer wraps sites into the **geometric parallelepiped** spanned
by the user-provided vectors. This is slightly different from
`PeriodicCell.remap_cart(...)`, which matches Voro++'s internal wrapping convention.

## Practical tips

- For large systems, drawing every cell is expensive. Use `cell_ids={...}` to view a subset.
- Use the notebooks in the **Examples** section for end-to-end demonstrations.
