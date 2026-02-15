# Topology and neighbor graphs

A very common reason to compute a tessellation in a periodic system is not the
polyhedra themselves, but the **neighbor graph**:

- Which sites are adjacent?
- Which periodic image produces that adjacency?
- Can we build a reproducible graph representation for downstream analysis?

This page explains why periodic graphs are subtle, and how pyvoro2 supports this workflow.

## Why periodic adjacency needs an image shift

In a periodic cell, each site has infinitely many periodic images.
A face between site $i$ and site $j$ therefore corresponds to **one specific image** of $j$.

If you record only “$i$ neighbors $j$”, you lose information. The edge is ambiguous.

pyvoro2 can annotate each face with:

- `adjacent_cell`: the neighbor site id
- `adjacent_shift`: an integer lattice shift `(na, nb, nc)` describing which image produced the face

You enable this with `return_face_shifts=True`:

```python
cells = pyvoro2.compute(points, domain=cell, return_faces=True, return_face_shifts=True)
```

For `PeriodicCell`, the shift is expressed in the $(a,b,c)$ lattice basis.
For `OrthorhombicCell`, the shift is expressed in axis-aligned lattice units.

## Building a periodic graph in practice

A minimal workflow looks like this:

1) Compute a tessellation with face shifts
2) Extract edges `(i, j, shift)` from faces

```python
edges = []
for c in cells:
    i = c['id']
    for f in c.get('faces', []):
        j = f['adjacent_cell']
        if j < 0:
            # boundary face (in a non-periodic domain)
            continue
        shift = f.get('adjacent_shift', (0, 0, 0))
        edges.append((i, j, shift))
```

In many scientific applications you will then:

- merge duplicate edges,
- choose an orientation convention (e.g., keep only `i < j`), and
- use `shift` to translate neighbor positions consistently.

## Normalization utilities

When you want to build a **reproducible** periodic graph, it is often helpful to normalize
geometric entities (vertices, edges, faces) so that they have a global indexing.
This makes it easier to compare results across different runs or different point orders.

pyvoro2 provides:

- `normalize_vertices(...)`
- `normalize_edges_faces(...)`
- `normalize_topology(...)`

These utilities are most useful for periodic settings.

## Diagnostics: catching subtle issues early

When building graphs, you typically want a few simple consistency checks.
pyvoro2 provides `analyze_tessellation(...)`, and `compute(..., return_diagnostics=True)`.

Diagnostics can check, for example:

- whether cell volumes sum to the domain volume (within tolerance),
- whether periodic face reciprocity holds (if face shifts are enabled).

This is not “proving correctness”, but it is extremely effective at catching mistakes
in downstream graph code.
