# Domains (containers)

A Voronoi or power/Laguerre tessellation is always defined **inside a domain**:

- for a finite cluster you typically want an explicit boundary (a box),
- for a crystal you typically want periodic boundary conditions,
- for slabs/wires you may want periodicity only in one or two directions.

In Voro++ terminology, these are different *containers*. In pyvoro2 they are exposed as
small Python dataclasses.

## Choosing a domain

A practical rule of thumb:

- Use **`Box`** when you want a finite system and you care about boundary faces.
- Use **`OrthorhombicCell`** when your cell is axis-aligned and you want periodicity in
  **some** axes (slabs and wires are the most common cases).
- Use **`PeriodicCell`** when you have a general triclinic unit cell (fully periodic).

## `Box` (finite, non-periodic)

`Box` represents an axis-aligned rectangular domain with *no periodicity*.
Cells are clipped to the box.

```python
from pyvoro2 import Box

box = Box(((-5, 5), (-5, 5), (-5, 5)))
```

This is the most straightforward setting for:

- molecules and clusters,
- toy problems and method development,
- any workflow where “outside the box” should be treated as an explicit boundary.

## `OrthorhombicCell` (axis-aligned, optionally periodic)

`OrthorhombicCell` represents an axis-aligned cell that can be periodic in any subset
of axes. This is a natural geometry for:

- 2D periodic slabs: periodic in $(x,y)$ but not in $z$,
- 1D periodic wires: periodic in $x$ only,
- fully periodic orthorhombic crystals.

```python
from pyvoro2 import OrthorhombicCell

# Fully periodic orthorhombic unit cell
cell = OrthorhombicCell(((0, 10), (0, 10), (0, 10)), periodic=(True, True, True))

# Slab geometry: periodic in x and y only
slab = OrthorhombicCell(((0, 10), (0, 10), (0, 30)), periodic=(True, True, False))
```

Notes:

- Internally this uses the standard Voro++ rectangular container with per-axis periodic flags.
- `return_face_shifts=True` is supported whenever at least one axis is periodic.
- For non-periodic axes, query points outside the bounds are treated as out-of-domain.

Robustness note:

- pyvoro2 vendors a small Voro++ patch that makes fully periodic **power/Laguerre**
  tessellations more robust across platforms (it inflates the stored global `max_radius`
  by 1 ULP via `nextafter`, making radical pruning slightly less aggressive).
- If `return_face_shifts=True` fails with a `ValueError`, it is typically due to a
  genuine geometric degeneracy (for example, nearly co-spherical sites) rather than
  a platform-specific issue.

## `PeriodicCell` (triclinic, fully periodic)

`PeriodicCell` represents a fully periodic triclinic unit cell defined by three vectors
$a, b, c$ (in Cartesian coordinates). This is the most general setting for crystals.

```python
from pyvoro2 import PeriodicCell

cell = PeriodicCell(
    vectors=((10.0, 0.0, 0.0), (2.0, 9.0, 0.0), (1.0, 0.5, 8.0)),
    origin=(0, 0, 0),
)
```

### Constructing from Voro++ parameters

Voro++ also describes triclinic periodic cells using six lower-triangular parameters:

- `bx, bxy, by, bxz, byz, bz`

If you already have these numbers (for example from an existing Voro++ workflow), you can
construct the same cell directly:

```python
cell = PeriodicCell.from_params(bx, bxy, by, bxz, byz, bz)
```

## Coordinate wrapping helpers

Both periodic domain classes provide remapping utilities that appear in several workflows:

- `remap_cart(points, return_shifts=True|False)`

They wrap coordinates into the primary cell and optionally return the integer lattice shift.

These helpers are used internally (for example, in visualization wrapping and in periodic graph work),
but they are also useful when you want to align your own data to the primary cell.

### A note on `PeriodicCell.remap_cart` vs the geometric parallelepiped

For `OrthorhombicCell`, “the primary cell” is unambiguous.

For `PeriodicCell`, Voro++ works in a *lower-triangular internal representation*
(`bx, bxy, by, bxz, byz, bz`). The method `PeriodicCell.remap_cart(...)` is
defined to match that convention exactly.

This means that wrapping via `remap_cart` does not always coincide with wrapping
into the **geometric parallelepiped** spanned by the user vectors.

For visualization, `pyvoro2.viz3d.view_tessellation(..., wrap_cells=True)` wraps
sites into the geometric parallelepiped, because that is typically what users
expect to see.
