# Concepts

This section introduces the geometric objects that pyvoro2 computes. The goal is
not to be encyclopedic, but to give you the minimum vocabulary needed to use the
library confidently.

## The problem being solved

Suppose you have a finite set of points in three dimensions, which we will call
**sites** (or *generators*),

- $p_1, p_2, \dots, p_n \in \mathbb{R}^3$.

A *tessellation* assigns every point $x$ in your domain (a box or a periodic unit cell)
to one of these sites, producing a set of convex polyhedra. These polyhedra can be
used as a spatial partition, a neighbor definition, or a geometric model of “regions
of influence” around sites.

Two related tessellations appear again and again in scientific computing:

- the **standard Voronoi tessellation** (unweighted), and
- the **power / Laguerre tessellation** (weighted Voronoi).

pyvoro2 computes both, using the proven C++ library **Voro++**.

## Standard Voronoi tessellation

In the standard Voronoi tessellation, the cell of site $p_i$ is the set of points
that are closer to $p_i$ than to any other site:

$$
V_i = \{x \in \mathbb{R}^3 \mid \lVert x - p_i \rVert^2 \le \lVert x - p_j \rVert^2 \;\; \forall j\}.
$$

Geometrically, each boundary between two neighboring sites is a plane located at
the midpoint between them (an “equidistance plane”). The resulting cells are
convex polyhedra.

In pyvoro2 this corresponds to:

- `mode='standard'`

## Power / Laguerre tessellation

Many scientific problems need something slightly more flexible than “closest site”.
For example, you may want larger atoms to occupy more volume, or you may want to
approximate a reference partition that is not purely distance-based.

A **power diagram** (also called **Laguerre** or **radical Voronoi**) introduces a per-site
weight $w_i$ and compares *power distances* instead of Euclidean distances:

$$
\pi_i(x) = \lVert x - p_i \rVert^2 - w_i.
$$

The cell of site $i$ becomes:

$$
V_i = \{x \in \mathbb{R}^3 \mid \pi_i(x) \le \pi_j(x) \;\; \forall j\}.
$$

A convenient interpretation is to write weights as squared “radii”, $w_i = r_i^2$.
This is the convention used by Voro++ (and therefore by pyvoro2), where you pass
`radii=...`.

**Important consequences (and common surprises):**

- The separating plane between two sites depends only on the **difference**
  $w_i - w_j$.
- Some sites can acquire **empty cells** (zero volume). This is not a failure: it
  is a normal feature of power diagrams.

In pyvoro2 this corresponds to:

- `mode='power'`
- `radii=...` (per-site)
- `include_empty=True` if you want explicit records for empty cells

## Periodicity and “which neighbor image”

In periodic domains, every site has infinitely many periodic images. A face between
two sites therefore corresponds not just to “$i$ is adjacent to $j$”, but to a
**specific periodic image** of $j$.

If your goal is a neighbor graph (e.g., for crystals), this image information is
essential.

pyvoro2 can annotate each face with:

- `adjacent_cell`: the neighbor site id
- `adjacent_shift`: an integer lattice shift `(na, nb, nc)` describing *which image*
  of the neighbor produced the face

You enable this with `return_face_shifts=True`.

## Stateless design

pyvoro2 is intentionally **stateless** at the API level:

- each call to `compute(...)`, `locate(...)`, or `ghost_cells(...)` builds a Voro++
  container in C++, inserts the sites, runs the computation, and returns Python objects.

This keeps the public interface simple and results reproducible.
