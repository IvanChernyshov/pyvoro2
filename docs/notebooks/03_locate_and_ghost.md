<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/03_locate_and_ghost.ipynb)
# Point queries: locate(...) and ghost_cells(...)

A full tessellation (`compute`) gives you all cells at once. In many workflows you only need
**local queries**:

- **Owner lookup**: *which site owns this point?* → `locate(...)`
- **Probe/ghost cell**: *what cell would a query point have if it were inserted?* → `ghost_cells(...)`

Both operations are **stateless** in pyvoro2: each call builds a temporary Voro++ container,
runs the query, and returns plain Python/NumPy outputs.

This notebook demonstrates both operations in a non-periodic `Box`.
```python
import numpy as np
from pprint import pprint

import pyvoro2 as pv

rng = np.random.default_rng(0)

# Generator sites
points = rng.uniform(-1.0, 1.0, size=(25, 3))

box = pv.Box(((-2, 2), (-2, 2), (-2, 2)))
```
## 1) Owner lookup with locate(...)

`locate(points, queries, domain=...)` returns, for each query point, whether it was located and
which generator site owns it.

- For a non-periodic `Box`, queries outside the box are typically reported as `found=False`.
```python
queries = np.array(
    [
        [0.0, 0.0, 0.0],   # inside
        [1.5, 1.5, 1.5],   # inside (near boundary)
        [5.0, 0.0, 0.0],   # outside
    ],
    dtype=float,
)

res = pv.locate(
    points,
    queries,
    domain=box,
    return_owner_position=True,
)

pprint(res)
```
**Output**

```text
{'found': array([ True,  True, False]),
 'owner_id': array([14,  9, -1]),
 'owner_pos': array([[ 0.18860006, -0.32417755, -0.216762  ],
       [ 0.96167068,  0.37108397,  0.30091855],
       [        nan,         nan,         nan]])}
```
## 2) Probe cells with ghost_cells(...)

`ghost_cells(points, queries, domain=...)` computes the Voronoi cell **around each query point**
without inserting it permanently into the point set.

This is useful for:
- sampling free volume at probe points,
- inspecting local environments,
- building “what-if” analyses without recomputing the entire tessellation.

For a non-periodic `Box`, a query outside the box may yield an empty result when `include_empty=True`.
```python
ghost = pv.ghost_cells(
    points,
    queries,
    domain=box,
    include_empty=True,
    return_vertices=True,
    return_faces=True,
)

# Show a compact summary
[(g['query_index'], bool(g.get('empty', False)), float(g.get('volume', 0.0))) for g in ghost]
```
**Output**

```text
[(0, False, 0.21887577215282997),
 (1, False, 3.3710997729938335),
 (2, True, 0.0)]
```
## Notes

- In a periodic domain, `locate` and `ghost_cells` wrap queries into a primary domain.
- In power mode (`mode='power'`), a ghost cell also needs a radius/weight for the query site (`ghost_radius`).
