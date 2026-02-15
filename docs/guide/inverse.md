# Inverse fitting (weights/radii from desired planes)

In many applications you do not start from “a tessellation”, but from a
**reference model** that tells you where the interfaces between pairs of sites
*should* be.

Examples include:

- atom-in-molecule partitions (chemistry)
- promolecular or model density partitions
- custom interface placements used as a geometric descriptor

pyvoro2 provides tools to fit a **power/Laguerre tessellation** so that the
resulting pairwise bisector planes match a set of desired locations **as well as possible**.
The output is still a mathematically standard power diagram, so it always forms a valid tessellation.

## Power bisector position along a line

In a power diagram, each site $p_i$ carries a weight $w_i$ (in pyvoro2 you normally work with
radii $r_i$ where $w_i=r_i^2$). The bisector between two sites satisfies:

$$
\lVert x-p_i \rVert^2 - w_i = \lVert x-p_j \rVert^2 - w_j.
$$

Choose a specific periodic image of $j$ (if you are in a periodic domain) and denote it by $p_j^*$.
Along the line from $p_i$ to $p_j^*$, the bisector intersects at a fractional position $t$:

$$
 t(w) = \frac{1}{2} + \frac{w_i - w_j}{2 d^2},
 \qquad d = \lVert p_j^* - p_i \rVert.
$$

- $t=0$ means “at $p_i$”,
- $t=1$ means “at $p_j^*$”,
- values outside $[0,1]$ are allowed and can occur naturally in power diagrams.

## Fitting API

Two convenience functions are provided:

- `fit_power_weights_from_plane_fractions(...)` — you provide target fractions $t_{\mathrm{target}}$
- `fit_power_weights_from_plane_positions(...)` — you provide target distances $x$ from $p_i$ along the $i\to j$ line

Constraints are given as a list of tuples:

- `(i, j, t)` or `(i, j, t, shift)`
- `(i, j, x)` or `(i, j, x, shift)`

where `shift=(na, nb, nc)` specifies which periodic image of $j$ should be used.

If you omit the shift in a periodic domain, pyvoro2 can (optionally) choose a “nearest image”.

## Restricting where the *predicted* bisector can go

Sometimes you want the *target* to be outside the segment (e.g. as a modeling choice), but you do not
want the fitted solution to place the bisector too far outside. In other cases you want to enforce
that the bisector lies strictly between the two sites.

pyvoro2 supports three regimes for the **predicted** $t(w)$:

- `t_bounds_mode='none'` — no restriction
- `t_bounds_mode='soft_quadratic'` — add a quadratic penalty for leaving an interval
- `t_bounds_mode='hard'` — enforce hard bounds (infeasible values are forbidden)

In addition, you can add a near-boundary repulsion:

- `t_near_penalty='exp'`

which discourages $t(w)$ from approaching the bounds too closely.

These options make the fit a small convex optimization problem that pyvoro2 solves in pure NumPy.

## Radii gauge and `r_min`

Power diagrams are invariant under adding a constant to all weights.
After fitting weights, pyvoro2 chooses a global shift so that the derived radii satisfy:

- `min(radii) == r_min`

This is useful if you want radii that are never exactly zero.

## “Inactive” constraints (pairs that are not a face)

A constraint between $(i,j)$ refers to a bisector plane, but in a full tessellation that plane becomes
an actual **cell face** only if $i$ and $j$ end up as neighbors.

If you pass `check_contacts=True`, pyvoro2 will compute a tessellation using the fitted radii and report
which constraints became real neighbor faces.

This is often enough for practical workflows, and it is also a stepping stone to iterative schemes
(where you refit only on active neighbor pairs).

## Typical workflow

1) Fit weights/radii from constraints
2) Compute a power tessellation with the fitted radii

```python
import pyvoro2 as pv

res = pv.fit_power_weights_from_plane_fractions(
    points,
    constraints,
    domain=cell,
    t_bounds=(0.0, 1.0),
    t_bounds_mode='hard',
    t_near_penalty='exp',
    r_min=1.0,
    check_contacts=True,
)

cells = pv.compute(
    points,
    domain=cell,
    mode='power',
    radii=res.radii,
    include_empty=True,
    return_face_shifts=True,
)
```
