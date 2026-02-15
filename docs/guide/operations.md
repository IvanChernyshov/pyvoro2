# Operations

pyvoro2 exposes three high-level operations. They correspond to three common
questions you may ask about a set of sites:

1. **What does the full tessellation look like?**  
   (Compute every Voronoi/power cell.)
2. **Which site owns this location in space?**  
   (Assign arbitrary query points to sites.)
3. **What would the cell of a hypothetical point be?**  
   (Compute a “probe” cell without inserting the point.)

All operations are **stateless**: pyvoro2 creates a Voro++ container in C++, inserts the sites, performs the computation, and returns Python data structures. There is no persistent container object that you need to manage.

## Coordinate scale and numerical safety

Voro++ uses a few **fixed absolute tolerances** internally (notably a hard
near-duplicate check around `~1e-5` in the coordinate units of the container).
This is fast and robust for “order-1” coordinate systems, but it means that
very small unit systems can be problematic.

If your coordinates are in SI meters for atomistic systems (typical distances
around `1e-10`), Voro++ may treat distinct sites as “too close” and terminate
the process.

pyvoro2 intentionally does **not** rescale inputs automatically.
If you work in very small or very large units, **rescale explicitly** before
calling `compute`, `locate`, or `ghost_cells` (for example, multiply all
coordinates and domain vectors by a constant).

As an additional safety net, you can ask pyvoro2 to run a fast **Python-side**
near-duplicate pre-check before entering the C++ layer:

```python
cells = pyvoro2.compute(
    points,
    domain=cell,
    duplicate_check='raise',  # recommended ("warn" is diagnostic only)
)
```

This checks for point pairs closer than ~`1e-5` (in your coordinate units).
Using `duplicate_check='raise'` prevents Voro++ from terminating the process.

Note: `duplicate_check='warn'` only reports the issue and still enters the C++ layer.
If your points truly violate Voro++'s hard threshold, the process may still terminate.

## 1) `compute(...)`: tessellate all sites

`compute` computes the Voronoi (standard) or power/Laguerre (weighted) cell for each site.

### Standard Voronoi

```python
cells = pyvoro2.compute(points, domain=box, mode='standard')
```

This is the classic “midplane” Voronoi construction.

### Power/Laguerre (weighted)

```python
cells = pyvoro2.compute(
    points,
    domain=box,
    mode='power',
    radii=radii,
    include_empty=True,
)
```

Power diagrams can produce **empty cells** (volume 0). Voro++ omits those in its iteration;
pyvoro2 can reinsert explicit empty-cell records when `include_empty=True`.

### Periodic neighbor image shifts

In periodic domains, a face between $i$ and $j$ corresponds to a specific periodic image of $j$.
If your goal is a periodic neighbor graph, this image information is essential.

Request it with:

```python
cells = pyvoro2.compute(
    points,
    domain=cell,
    return_faces=True,
    return_vertices=True,
    return_face_shifts=True,
)
```

Each face can then include:

- `adjacent_cell`: neighbor id
- `adjacent_shift`: integer shift `(na, nb, nc)` describing which neighbor image produced the face

## 2) `locate(...)`: assign query points to sites

`locate` answers a simpler question than full tessellation:

> Given a query point $q$, which site owns it?

This wraps the Voro++ routine `find_voronoi_cell`.

```python
out = pyvoro2.locate(points, queries, domain=cell, return_owner_position=True)
owner_ids = out['owner_id']
```

If `return_owner_position=True`, the output also contains `owner_pos`.
In periodic domains this position may be a **periodic image** of the generator, chosen
consistently with the query point.

## 3) `ghost_cells(...)`: compute probe (ghost) cells

`ghost_cells` asks a slightly different question:

> What would the cell of $q$ look like if $q$ were inserted as an additional site?

This wraps the Voro++ routine `compute_ghost_cell`.

```python
ghost = pyvoro2.ghost_cells(points, queries, domain=cell)
```

Each returned record describes the polyhedron of the ghost cell.

### `query` vs `site` in periodic domains

For periodic domains, pyvoro2 wraps each query into the primary cell before calling Voro++.
Therefore each record contains:

- `query`: the original coordinate you supplied
- `site`: the coordinate actually used for the computation (a wrapped periodic representative)

They are in the same coordinate system and differ only by an integer lattice translation.
This wrapping makes results easier to compare and visualize.
