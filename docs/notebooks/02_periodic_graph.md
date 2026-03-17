<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/02_periodic_graph.ipynb)
# Periodic tessellation and neighbor graphs

In non-periodic geometry, a Voronoi tessellation naturally defines a **neighbor graph**:
two sites are neighbors if their cells share a face.

In a **periodic** domain, there is an additional subtlety:

- every site has infinitely many periodic images,
- a face between sites *i* and *j* is formed with a **specific image** of *j*.

If you want a graph that is correct for crystals, you typically need that image information.
pyvoro2 can annotate each face with an integer lattice shift:

- `adjacent_cell`: neighbor id
- `adjacent_shift = (na, nb, nc)`: which periodic image produced the face

This notebook shows a minimal workflow:
1. compute a periodic tessellation in a triclinic cell,
2. extract graph edges `(i, j, shift)`,
3. canonicalize edges into an undirected contact list.
```python
import numpy as np
import pyvoro2 as pv

rng = np.random.default_rng(0)

# Random points in Cartesian coordinates (not necessarily wrapped)
points = rng.random((30, 3))

cell = pv.PeriodicCell(
    vectors=((10.0, 0.0, 0.0), (2.0, 9.0, 0.0), (1.0, 0.5, 8.0)),
    origin=(0.0, 0.0, 0.0),
)

cells = pv.compute(
    points,
    domain=cell,
    return_faces=True,
    return_vertices=True,
    return_face_shifts=True,  # <-- adds `adjacent_shift` to each face
    face_shift_search=2,
)
len(cells)
```
**Output**

```text
30
```
## Inspecting face shifts

For a well-formed periodic tessellation, face shifts should be **reciprocal**:
if cell *i* has a face to neighbor *j* with shift *s*, then cell *j* should have the
corresponding face back to *i* with shift `-s`.

Let's inspect one example face.
```python
# Pick a cell and show its first non-boundary face
c0 = next(c for c in cells if int(c['id']) == 0)
f0 = next(f for f in c0['faces'] if int(f.get('adjacent_cell', -1)) >= 0)

(i, j, shift) = (int(c0['id']), int(f0['adjacent_cell']), tuple(int(x) for x in f0['adjacent_shift']))
(i, j, shift)
```
**Output**

```text
(0, 8, (0, 0, -1))
```
## Extracting a periodic neighbor graph

A simple representation for periodic adjacency is a list of **directed** edges:

- `(i, j, shift)`

meaning: *cell i* touches the image of *cell j* translated by `shift`.

Depending on your application, you may want to:
- keep the graph directed (useful for some algorithms), or
- canonicalize contacts into an **undirected** set by storing only one orientation.

Below we build both.
```python
# 1) Directed edges from faces
directed = []
for c in cells:
    i = int(c['id'])
    for f in c.get('faces', []):
        j = int(f.get('adjacent_cell', -1))
        if j < 0:
            continue
        s = tuple(int(x) for x in f.get('adjacent_shift', (0, 0, 0)))
        directed.append((i, j, s))

print('n_directed:', len(directed))
print('sample:', directed[:5])
```
**Output**

```text
n_directed: 436
sample: [(0, 8, (0, 0, -1)), (0, 20, (0, 0, 0)), (0, 6, (0, 0, 0)), (0, 19, (0, 0, 0)), (0, 10, (0, 0, 0))]
```
```python
# 2) Canonicalize into an undirected contact set
#
# We choose a convention:
# - store edges with i < j
# - if we flip direction, also flip the shift (reciprocity)
undirected = set()
for (i, j, s) in directed:
    if i < j:
        undirected.add((i, j, s))
    elif j < i:
        undirected.add((j, i, (-s[0], -s[1], -s[2])))

print('n_undirected:', len(undirected))
print('sample:', list(sorted(undirected))[:5])
```
**Output**

```text
n_undirected: 218
sample: [(0, 3, (0, 0, 0)), (0, 6, (0, 0, 0)), (0, 8, (0, 0, -1)), (0, 10, (0, 0, 0)), (0, 14, (0, 0, 0))]
```
## Building an adjacency list

Many downstream workflows prefer an adjacency list:

- `adj[i] = [(j, shift), ...]`

Here we build it from the directed edges.
```python
from collections import defaultdict

adj = defaultdict(list)
for (i, j, s) in directed:
    adj[i].append((j, s))

# Show the neighbors of site 0
adj[0][:10]
```
**Output**

```text
[(8, (0, 0, -1)),
 (20, (0, 0, 0)),
 (6, (0, 0, 0)),
 (19, (0, 0, 0)),
 (10, (0, 0, 0)),
 (14, (0, 0, 0)),
 (3, (0, 0, 0))]
```
## Notes

- For `OrthorhombicCell` with only partial periodicity, shifts on non-periodic axes are always zero.
- If you plan to compute a graph repeatedly (e.g., for many frames), consider:
  - keeping your inputs in a consistent wrapped form, and
  - using `tessellation_check='warn'` or `'diagnose'` during development.
