<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/01_basic_compute.ipynb)
# Basic tessellations in pyvoro2

This notebook is a compact tour of the most common `pyvoro2.compute(...)` workflows.
It is written as a narrative: each section introduces the geometric idea first, and then shows
the minimal code needed to reproduce it.

We cover:
- Voronoi cells in a non-periodic **bounding box** (`Box`)
- Voronoi cells in a **triclinic periodic unit cell** (`PeriodicCell`)
- Power/Laguerre tessellation (`mode='power'`) and the meaning of per-site radii
- What geometry is returned (`vertices`, `faces`, `adjacency`)
- Periodic face shifts (`adjacent_shift`) and basic diagnostics
- Global enumeration utilities (`normalize_topology`) and per-face descriptors

> Tip: If you are new to Voronoi terminology, the short conceptual background is in
> the docs section [Concepts](../guide/concepts.md).
```python
import numpy as np
from pprint import pprint

import pyvoro2 as pv
from pyvoro2 import Box, OrthorhombicCell, PeriodicCell, compute
```
## Voronoi tessellation in a bounding box (Box)

In a non-periodic domain, the Voronoi cell of a site is the region of space that is closer
to that site than to any other site. In practice, we also need a finite *domain* to cut
the unbounded cells — here we use a rectangular `Box`.
```python
pts = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ],
    dtype=float,
)

box = Box(bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))

cells = compute(
    pts,
    domain=box,
    mode='standard',
    return_vertices=True,
    return_faces=True,
    return_adjacency=False,  # keep output small for display
)

print(f'Total number of cells: {len(cells)}\n')
pprint(cells[0])
```
**Output**

```text
Total number of cells: 4

{'faces': [{'adjacent_cell': 1, 'vertices': [1, 5, 7, 3]},
           {'adjacent_cell': -3, 'vertices': [1, 0, 4, 5]},
           {'adjacent_cell': -5, 'vertices': [1, 3, 2, 0]},
           {'adjacent_cell': 2, 'vertices': [2, 3, 7, 6]},
           {'adjacent_cell': -1, 'vertices': [2, 6, 4, 0]},
           {'adjacent_cell': 3, 'vertices': [4, 6, 7, 5]}],
 'id': 0,
 'site': [0.0, 0.0, 0.0],
 'vertices': [[-5.0, -5.0, -5.0],
              [1.0, -5.0, -5.0],
              [-5.0, 1.0, -5.0],
              [1.0, 1.0, -5.0],
              [-5.0, -5.0, 1.0],
              [1.0, -5.0, 1.0],
              [-5.0, 1.0, 1.0],
              [1.0, 1.0, 1.0]],
 'volume': 216.0}
```
## Periodic tessellation in a triclinic unit cell (PeriodicCell)

For crystals and other periodic systems, the natural domain is a unit cell with periodic boundary
conditions. `PeriodicCell` supports fully triclinic (skew) cells by representing the cell with
three lattice vectors.

A useful sanity check: in a fully periodic Voronoi tessellation, the sum of all cell volumes
should equal the unit cell volume (up to numerical tolerance).
```python
cell = PeriodicCell(
    vectors=(
        (10.0, 0.0, 0.0),
        (2.0, 9.5, 0.0),
        (1.0, 0.5, 9.0),
    )
)

pts_pbc = np.array(
    [
        [1.0, 1.0, 1.0],
        [5.0, 5.0, 5.0],
        [8.0, 2.0, 7.0],
        [3.0, 9.0, 4.0],
    ],
    dtype=float,
)

cells_pbc = compute(
    pts_pbc,
    domain=cell,
    mode='standard',
    return_vertices=False,
    return_faces=False,
    return_adjacency=False,
)

# In periodic mode, all Voronoi volumes should sum to the unit cell volume.
cell_volume = abs(np.linalg.det(np.array(cell.vectors, dtype=float)))
sum_vol = float(sum(c['volume'] for c in cells_pbc))
cell_volume, sum_vol
```
**Output**

```text
(855.0000000000013, 855.0)
```
## Power/Laguerre tessellation (mode="power")

A power (Laguerre) tessellation generalizes Voronoi cells by assigning each site a weight.
Voro++ (and pyvoro2) expose this as a per-site **radius** $r_i$, which corresponds to a weight
$w_i = r_i^2$ in the power distance.

Intuitively: increasing a site's radius tends to expand its cell at the expense of neighbors.
Unlike standard Voronoi cells, **empty cells are possible** in power mode.
```python
# Re-define the periodic cell and points (self-contained example)
cell = PeriodicCell(
    vectors=(
        (10.0, 0.0, 0.0),
        (2.0, 9.5, 0.0),
        (1.0, 0.5, 9.0),
    )
)

pts_pbc = np.array(
    [
        [1.0, 1.0, 1.0],
        [5.0, 5.0, 5.0],
        [8.0, 2.0, 7.0],
        [3.0, 9.0, 4.0],
    ],
    dtype=float,
)

radii = np.array([0.0, 0.0, 2.0, 0.0], dtype=float)

cells_std = compute(
    pts_pbc,
    domain=cell,
    mode='standard',
    return_vertices=False,
    return_faces=False,
    return_adjacency=False,
)

cells_pow = compute(
    pts_pbc,
    domain=cell,
    mode='power',
    radii=radii,
    return_vertices=False,
    return_faces=False,
    return_adjacency=False,
)

vols_std = [c['volume'] for c in cells_std]
vols_pow = [c['volume'] for c in cells_pow]

vols_std, vols_pow
```
**Output**

```text
([204.52350840152917,
  243.35630134069405,
  231.409081979397,
  175.71110827837984],
 [177.66314170369014,
  213.6503389726455,
  307.3025551674562,
  156.38396415620826])
```
## Inspecting geometry: vertices, faces, adjacency

`compute(...)` can return different levels of geometric detail. For downstream analysis, the most
important pieces are:

- `vertices`: coordinates of the cell vertices
- `faces`: polygonal faces (each includes the list of vertex indices and the adjacent cell id)
- `adjacency`: per-vertex adjacency lists (optional)

The cell dictionaries are designed to be plain data (NumPy arrays + Python lists), so you can
serialize them or process them with your own code.
```python
# Re-define the 0D box system (self-contained example)
pts = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ],
    dtype=float,
)

box = Box(bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))

cells_full = compute(
    pts,
    domain=box,
    mode='standard',
    return_vertices=True,
    return_faces=True,
    return_adjacency=True,
)

pprint(cells_full[0])
```
**Output**

```text
{'adjacency': [[1, 4, 2],
               [5, 0, 3],
               [3, 0, 6],
               [7, 1, 2],
               [6, 0, 5],
               [4, 1, 7],
               [7, 2, 4],
               [5, 3, 6]],
 'faces': [{'adjacent_cell': 1, 'vertices': [1, 5, 7, 3]},
           {'adjacent_cell': -3, 'vertices': [1, 0, 4, 5]},
           {'adjacent_cell': -5, 'vertices': [1, 3, 2, 0]},
           {'adjacent_cell': 2, 'vertices': [2, 3, 7, 6]},
           {'adjacent_cell': -1, 'vertices': [2, 6, 4, 0]},
           {'adjacent_cell': 3, 'vertices': [4, 6, 7, 5]}],
 'id': 0,
 'site': [0.0, 0.0, 0.0],
 'vertices': [[-5.0, -5.0, -5.0],
              [1.0, -5.0, -5.0],
              [-5.0, 1.0, -5.0],
              [1.0, 1.0, -5.0],
              [-5.0, -5.0, 1.0],
              [1.0, -5.0, 1.0],
              [-5.0, 1.0, 1.0],
              [1.0, 1.0, 1.0]],
 'volume': 216.0}
```
## Empty cells in power mode (include_empty=True)

In a power diagram, some sites can be dominated by others and end up with **zero volume**.
This is mathematically valid. If you want these cases to appear explicitly in the output,
use `include_empty=True`.
```python
cell_u = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
pts_u = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)
radii_u = np.array([1.0, 2.0], dtype=float)

cells_pow = compute(
    pts_u,
    domain=cell_u,
    mode='power',
    radii=radii_u,
    include_empty=True,
    return_vertices=True,
    return_faces=True,
    return_adjacency=False,
    return_face_shifts=True,
    face_shift_search=1,
)

[(int(c['id']), c.get('empty', False), float(c.get('volume', 0.0))) for c in cells_pow]
```
**Output**

```text
[(0, True, 0.0), (1, False, 0.9999999999999997)]
```
## Periodic face shifts and diagnostics

In periodic domains, an adjacency is not just “site *i* touches site *j*”. The shared face is formed
with a **particular periodic image** of *j*. pyvoro2 can annotate each face with an integer lattice
shift `adjacent_shift = (na, nb, nc)`.

This section also shows how to request diagnostics when you want to actively validate a tessellation.
```python
cell_u = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
pts_u = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

cells_std_u, diag_std_u = compute(
    pts_u,
    domain=cell_u,
    mode='standard',
    return_vertices=True,
    return_faces=True,
    return_adjacency=False,
    return_face_shifts=True,
    face_shift_search=1,
    tessellation_check='diagnose',
    return_diagnostics=True,
)

# Inspect the face between the two sites across the x-boundary.
c0 = next(c for c in cells_std_u if int(c['id']) == 0)
idx = next(i for i, f in enumerate(c0['faces']) if int(f['adjacent_cell']) == 1)
face01 = c0['faces'][idx]

(diag_std_u.ok, diag_std_u.volume_ratio, diag_std_u.n_faces_orphan), face01
```
**Output**

```text
((True, 1.0, 0),
 {'adjacent_cell': 1,
  'vertices': [1, 6, 4, 5],
  'adjacent_shift': (-1, 0, 0),
  'orphan': False,
  'reciprocal_mismatch': False,
  'reciprocal_missing': False})
```
## Normalization: global vertices / edges / faces

When you compute cells, each cell has its own local vertex indexing. For graph and topology work,
it is often helpful to build a **global** pool of vertices/edges/faces with stable IDs that are
consistent across cells.

`normalize_topology(...)` can mutate cell dicts (unless `copy_cells=True`) and adds global-id arrays
such as `vertex_global_id` and `face_global_id`.
```python
from pyvoro2 import normalize_topology

cell_n = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
pts_n = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

cells_n = compute(
    pts_n,
    domain=cell_n,
    mode='standard',
    return_vertices=True,
    return_faces=True,
    return_adjacency=False,
    return_face_shifts=True,
    face_shift_search=1,
)

# Pick the periodic wrap face (0 -> 1 across x-wrap)
c0 = next(c for c in cells_n if int(c['id']) == 0)
idx = next(
    i
    for i, f in enumerate(c0['faces'])
    if int(f['adjacent_cell']) == 1 and tuple(int(x) for x in f['adjacent_shift']) == (-1, 0, 0)
)

# Mutate in place so the original cell dictionaries gain global id fields.
nt = normalize_topology(cells_n, domain=cell_n, copy_cells=False)

n_global = (len(nt.global_vertices), len(nt.global_edges), len(nt.global_faces))

# Example: show the face's global id and its global vertex ids
fid0 = int(c0['face_global_id'][idx])
print(f'Global counts for vertices, edges, and faces: {n_global}')
print('\nGlobal face data:')
pprint(nt.global_faces[fid0])
print('\nUpdated cell:')
pprint(c0)
```
**Output**

```text
Global counts for vertices, edges, and faces: (16, 24, 6)

Global face data:
{'cell_shifts': ((0, 0, 0), (-1, 0, 0)),
 'cells': (0, 1),
 'vertex_shifts': [(0, 0, 0), (0, 1, 0), (0, 1, -1), (0, 0, -1)],
 'vertices': [1, 5, 4, 6]}

Updated cell:
{'edge_global_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
 'edges': [(0, 3),
           (0, 4),
           (0, 7),
           (1, 2),
           (1, 5),
           (1, 6),
           (2, 3),
           (2, 7),
           (3, 5),
           (4, 5),
           (4, 6),
           (6, 7)],
 'face_global_id': [0, 1, 2, 3, 0, 1],
 'faces': [{'adjacent_cell': 0,
            'adjacent_shift': (0, -1, 0),
            'vertices': [1, 2, 7, 6]},
           {'adjacent_cell': 0,
            'adjacent_shift': (0, 0, 1),
            'vertices': [1, 5, 3, 2]},
           {'adjacent_cell': 1,
            'adjacent_shift': (-1, 0, 0),
            'vertices': [1, 6, 4, 5]},
           {'adjacent_cell': 1,
            'adjacent_shift': (0, 0, 0),
            'vertices': [2, 3, 0, 7]},
           {'adjacent_cell': 0,
            'adjacent_shift': (0, 1, 0),
            'vertices': [3, 5, 4, 0]},
           {'adjacent_cell': 0,
            'adjacent_shift': (0, 0, -1),
            'vertices': [4, 6, 7, 0]}],
 'id': 0,
 'site': [0.1, 0.5, 0.5],
 'vertex_global_id': [0, 1, 2, 3, 4, 5, 6, 7],
 'vertex_shift': [(0, 1, 0),
                  (0, 0, 1),
                  (0, 0, 1),
                  (0, 1, 1),
                  (0, 1, 0),
                  (0, 1, 1),
                  (0, 0, 0),
                  (0, 0, 0)],
 'vertices': [[0.5, 1.0, 0.0],
              [-1.3877787807814457e-16, 0.0, 1.0],
              [0.5, 0.0, 1.0],
              [0.5, 1.0, 0.9999999999999998],
              [-1.3877787807814457e-16, 1.0, 0.0],
              [-1.3877787807814457e-16, 1.0, 1.0],
              [-1.3877787807814457e-16, 0.0, 0.0],
              [0.5, 0.0, 1.1102230246251565e-16]],
 'volume': 0.5000000000000001}
```
## Face properties: contact descriptors

`annotate_face_properties(...)` computes per-face descriptors (centroid, normal, and intersection
with the site-to-site line) that are often useful for contact analysis.
```python
from pyvoro2 import annotate_face_properties

cell_f = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
pts_f = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

cells_f, diag_f = compute(
    pts_f,
    domain=cell_f,
    mode='standard',
    return_vertices=True,
    return_faces=True,
    return_adjacency=False,
    return_face_shifts=True,
    face_shift_search=1,
    tessellation_check='diagnose',
    return_diagnostics=True,
)

c0 = next(c for c in cells_f if int(c['id']) == 0)
idx = next(
    i
    for i, f in enumerate(c0['faces'])
    if int(f['adjacent_cell']) == 1 and tuple(int(x) for x in f['adjacent_shift']) == (-1, 0, 0)
)

annotate_face_properties(cells_f, domain=cell_f, diagnostics=diag_f)
f = c0['faces'][idx]
{
    'centroid': f.get('centroid'),
    'normal': f.get('normal'),
    'intersection': f.get('intersection'),
    'intersection_inside': f.get('intersection_inside'),
    'intersection_centroid_dist': f.get('intersection_centroid_dist'),
    'intersection_edge_min_dist': f.get('intersection_edge_min_dist'),
}
```
**Output**

```text
{'centroid': [-1.3877787807814457e-16, 0.5, 0.5],
 'normal': [-1.0, -0.0, -0.0],
 'intersection': [-1.3877787807814457e-16, 0.5, 0.5],
 'intersection_inside': True,
 'intersection_centroid_dist': 0.0,
 'intersection_edge_min_dist': 0.5}
```
