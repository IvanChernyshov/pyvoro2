# Planar 2D (`pyvoro2.planar`)

pyvoro2 now ships a dedicated **planar 2D namespace**:

```python
import pyvoro2.planar as pv2
```

This is intentionally separate from the 3D top-level API. The goal is to keep
both surfaces explicit and mathematically honest:

- `pyvoro2` is the 3D package,
- `pyvoro2.planar` is the 2D package.

The current 2D release scope is deliberately limited to the domains that the
vendored legacy backend supports well:

- `pv2.Box`
- `pv2.RectangularCell`

There is **no** planar `PeriodicCell` yet. Rectangular periodic domains can be
periodic in either or both planar axes.

## Basic compute

```python
import numpy as np
import pyvoro2.planar as pv2

pts = np.array([
    [0.2, 0.2],
    [0.8, 0.2],
    [0.5, 0.8],
], dtype=float)

cells = pv2.compute(
    pts,
    domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
    return_vertices=True,
    return_edges=True,
)
```

Raw planar cells are dimension-specific by design:

- `area` instead of `volume`,
- `edges` instead of `faces`,
- `adjacent_shift` is a length-2 periodic image shift when requested.

## Rectangular periodic cells and edge shifts

For periodic rectangular domains, request `return_edge_shifts=True` when you
need the explicit periodic image of the neighboring site:

```python
cell = pv2.RectangularCell(
    ((0.0, 1.0), (0.0, 1.0)),
    periodic=(True, True),
)

cells = pv2.compute(
    pts,
    domain=cell,
    return_vertices=True,
    return_edges=True,
    return_edge_shifts=True,
)
```

The planar wrapper reconstructs these edge shifts in Python and also repairs a
legacy backend quirk where some fully periodic adjacencies can otherwise appear
with negative neighbor ids.

## `locate(...)` and `ghost_cells(...)`

The planar namespace mirrors the 3D operation names:

```python
owners = pv2.locate(pts, [[0.1, 0.2], [0.9, 0.2]], domain=cell)

ghost = pv2.ghost_cells(
    pts,
    [[0.5, 0.5]],
    domain=cell,
    return_vertices=True,
    return_edges=True,
)
```

So the same three high-level questions exist in both dimensions:

1. compute every cell,
2. locate the owner of a query point,
3. compute the hypothetical cell of a query point without inserting it.

## Diagnostics and wrapper-level convenience

Planar `compute(...)` supports the same kind of post-compute convenience that
3D users already expect, but specialized for 2D semantics:

```python
cells, diag = pv2.compute(
    pts,
    domain=cell,
    return_diagnostics=True,
)
```

For periodic domains, the wrapper automatically computes the temporary geometry
needed for reciprocity checks and then strips it back out of the raw returned
cells unless you explicitly requested it.

The same holds for normalization convenience:

```python
result = pv2.compute(
    pts,
    domain=cell,
    return_diagnostics=True,
    normalize='topology',
)
```

This returns a `pv2.PlanarComputeResult` bundling:

- raw `cells`,
- optional tessellation diagnostics,
- optional normalized vertices,
- optional normalized topology.

This keeps the public API structured once the user wants more than a bare list
of raw cells.

## Planar normalization

The dedicated planar normalization helpers are:

- `pv2.normalize_vertices(...)`
- `pv2.normalize_edges(...)`
- `pv2.normalize_topology(...)`
- `pv2.validate_normalized_topology(...)`

In planar topology work, the globally deduplicated boundary objects are
**edges**, not faces.

## Planar plotting

For quick inspection, use the optional matplotlib helper:

```python
from pyvoro2.planar import plot_tessellation

fig, ax = plot_tessellation(cells, annotate_ids=True)
```

Install it with:

```bash
pip install "pyvoro2[viz2d]"
```

or install both 2D and 3D visualization helpers with:

```bash
pip install "pyvoro2[viz]"
```

## Planar power fitting

The generic pairwise-separator `powerfit` API now supports planar domains too.
The solver vocabulary is shared between 2D and 3D; what changes is the meaning
of the realized boundary measure:

- face area in 3D,
- edge length in 2D.

The current planar domain restriction still applies here: rectangular periodic
cells are supported, but there is no planar oblique-periodic `PeriodicCell`
yet.
