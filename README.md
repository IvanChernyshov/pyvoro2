# pyvoro2

[![CI](https://github.com/DeloneCommons/pyvoro2/actions/workflows/ci.yml/badge.svg)](https://github.com/DeloneCommons/pyvoro2/actions/workflows/ci.yml) [![Docs](https://github.com/DeloneCommons/pyvoro2/actions/workflows/docs.yml/badge.svg)](https://github.com/DeloneCommons/pyvoro2/actions/workflows/docs.yml) [![PyPI](https://img.shields.io/pypi/v/pyvoro2.svg)](https://pypi.org/project/pyvoro2/) [![Python Versions](https://img.shields.io/pypi/pyversions/pyvoro2.svg)](https://pypi.org/project/pyvoro2/) [![License](https://img.shields.io/pypi/l/pyvoro2.svg)](https://github.com/DeloneCommons/pyvoro2/blob/main/LICENSE)

**Documentation:** https://delonecommons.github.io/pyvoro2/


---

**pyvoro2** is a Python interface to the C++ library **Voro++** for computing
**2D and 3D Voronoi-type tessellations** around a set of points:

- **Voronoi tessellations** (standard, unweighted)
- **power / Laguerre tessellations** (weighted Voronoi, via per-site radii)
- a dedicated planar namespace, **`pyvoro2.planar`**, for 2D rectangular domains

The focus is not only on computing cells, but on making the results *usable*
in scientific and geometric settings that need **periodic boundary
conditions**, explicit **neighbor-image shifts**, reproducible
**topology/normalization** utilities, and a reusable mathematical interface to
Voronoi and power tessellations.

pyvoro2 is designed to be **honest and predictable**:

- it vendors and wraps an upstream Voro++ snapshot (with a small numeric robustness patch for power/Laguerre diagrams);
- the 3D top-level API stays separate from the 2D `pyvoro2.planar` namespace;
- the core tessellation modes are **standard Voronoi** and **power/Laguerre**.

**License note:** starting with **0.6.0**, the pyvoro2-authored code is released under **LGPLv3+**. Versions before **0.6.0** were released under **MIT**. Vendored third-party code remains under its own licenses.

## Quickstart

### 1) Standard Voronoi in a 3D bounding box

For 3D visualization, install the optional dependency: `pip install "pyvoro2[viz]"`.

```python
import numpy as np
import pyvoro2 as pv
from pyvoro2.viz3d import view_tessellation

points = np.random.default_rng(0).uniform(-1.5, 1.5, size=(10, 3))
box = pv.Box(((-2, 2), (-2, 2), (-2, 2)))
cells = pv.compute(points, domain=box, mode='standard')

view_tessellation(
    cells,
    domain=box,
    show_vertices=False,
)
```

<img src="https://raw.githubusercontent.com/DeloneCommons/pyvoro2/main/docs/assets/quickstart_box.png" width="50%" alt="Voronoi tessellation in a box" />

### 2) Planar periodic workflow

```python
import numpy as np
import pyvoro2.planar as pv2

pts2 = np.array([
    [0.2, 0.2],
    [0.8, 0.25],
    [0.4, 0.8],
], dtype=float)

cell2 = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
result2 = pv2.compute(
    pts2,
    domain=cell2,
    return_diagnostics=True,
    normalize='topology',
)

diag2 = result2.require_tessellation_diagnostics()
topo2 = result2.require_normalized_topology()
```

### 3) Power/Laguerre tessellation (weighted Voronoi)

```python
radii = np.full(len(points), 1.2)

cells = pv.compute(
    points,
    domain=box,
    mode='power',
    radii=radii,
    include_empty=True,  # power diagrams can have zero-volume cells
)
```

### 4) Periodic crystal cell with neighbor image shifts

```python
cell = pv.PeriodicCell(
    vectors=(
        (10.0, 0.0, 0.0),
        (2.0,  9.0, 0.0),
        (1.0,  0.5, 8.0),
    )
)

cells = pv.compute(points, domain=cell, return_face_shifts=True)

# Each face can include:
#   adjacent_cell  (neighbor id)
#   adjacent_shift (which periodic image produced the face)
```

## Numerical safety notes

Voro++ uses a few fixed absolute tolerances internally (most importantly a hard
near-duplicate check around ~`1e-5` in container distance units). For very small
or very large coordinate systems, this can lead to hard process termination or
loss of accuracy.

pyvoro2 does **not** silently rescale your coordinates. If you work in unusual
units, rescale explicitly before calling into the C++ layer.

As an additional safety net, you can ask pyvoro2 to run a fast Python-side
near-duplicate pre-check before entering the C++ layer:

```python
cells = pv.compute(points, domain=cell, duplicate_check='raise')
```

For stricter post-hoc checks, see:

- `pyvoro2.validate_tessellation(..., level='strict')`
- `pyvoro2.validate_normalized_topology(..., level='strict')`
- `pyvoro2.planar.validate_tessellation(..., level='strict')`
- `pyvoro2.planar.validate_normalized_topology(..., level='strict')`

Note: pyvoro2 vendors a Voro++ snapshot that includes the upstream numeric robustness fix for
*power/Laguerre* mode (radical pruning). This avoids rare cross-platform edge cases where fully
periodic power tessellations could yield a non-reciprocal face/neighbor graph under aggressive
floating-point codegen.

## Why use pyvoro2

Voro++ is fast and feature-rich, but it is a C++ library with a low-level API.
pyvoro2 aims to be a *scientific* interface that stays close to Voro++ while adding
practical pieces that are easy to get wrong:

- **triclinic periodic cells** (`PeriodicCell`) with robust coordinate mapping in 3D
- **partially periodic orthorhombic cells** (`OrthorhombicCell`) for slabs and wires
- dedicated **planar 2D support** in `pyvoro2.planar` for boxes and rectangular periodic cells
- optional **periodic image shifts** (`adjacent_shift`) on faces/edges for building periodic graphs
- **diagnostics** and **normalization utilities** for reproducible topology work
- convenience operations beyond full tessellation:
  - `locate(...)` / `pyvoro2.planar.locate(...)` (owner lookup for arbitrary query points)
  - `ghost_cells(...)` / `pyvoro2.planar.ghost_cells(...)` (probe cell at a query point without inserting it)
  - power-fitting utilities for **fitting power weights** from desired pairwise separator locations in both 2D and 3D

## Documentation overview

The documentation is written as a short scientific tutorial: it starts with the
geometric ideas, then explains domains and operations, and only then dives into
implementation-oriented details.

| Section | What it contains |
|---|---|
| [Concepts](https://delonecommons.github.io/pyvoro2/guide/concepts/) | What Voronoi and power/Laguerre tessellations are, and what you can expect from them. |
| [Domains (3D)](https://delonecommons.github.io/pyvoro2/guide/domains/) | Which spatial containers exist (`Box`, `OrthorhombicCell`, `PeriodicCell`) and how to choose between them. |
| [Planar (2D)](https://delonecommons.github.io/pyvoro2/guide/planar/) | The planar namespace, current 2D domain scope, wrapper-level diagnostics/normalization convenience, and plotting. |
| [Operations](https://delonecommons.github.io/pyvoro2/guide/operations/) | How to compute tessellations, assign query points, and compute probe (ghost) cells in the 3D and planar namespaces. |
| [Topology and graphs](https://delonecommons.github.io/pyvoro2/guide/topology/) | How to build periodic neighbor graphs and how normalization helps in both 2D and 3D. |
| [Power fitting](https://delonecommons.github.io/pyvoro2/guide/powerfit/) | Fit power weights from pairwise bisector constraints, realized-boundary matching, and self-consistent active sets in 2D or 3D. |
| [Visualization](https://delonecommons.github.io/pyvoro2/guide/visualization/) | Optional `py3Dmol` / `matplotlib` helpers for debugging and exploratory analysis. |
| [Examples (notebooks)](https://delonecommons.github.io/pyvoro2/notebooks/01_basic_compute/) | End-to-end examples, including focused power-fitting notebooks for reports and infeasibility witnesses. |
| [API reference](https://delonecommons.github.io/pyvoro2/reference/planar/) | The full reference (docstrings) for both the spatial and planar APIs. |

## Installation

Most users should install a prebuilt wheel:

```bash
pip install pyvoro2
```

Optional extras:

- `pyvoro2[viz]` for the 3D `py3Dmol` viewer (and 2D plotting too)
- `pyvoro2[viz2d]` for 2D matplotlib plotting only

To build from source (requires a C++ compiler and Python development headers):

```bash
pip install -e .
```

## Testing

pyvoro2 uses **pytest**. The default test suite is intended to be fast and deterministic:

```bash
pip install -e ".[test]"
pytest
```

Additional test groups are **opt-in**:

- **Fuzz/property tests** (randomized):

  ```bash
  pytest -m fuzz --fuzz-n 100
  ```

- **Cross-check tests vs `pyvoro`** (requires installing `pyvoro` first):

  ```bash
  pip install pyvoro
  pytest -m pyvoro --fuzz-n 100
  ```

- **Slow tests** (if any are added in the future):

  ```bash
  pytest -m slow
  ```

Tip: you can combine markers, e.g. `pytest -m "fuzz and pyvoro" --fuzz-n 100`.

## Project status

pyvoro2 is currently in **beta**.

The core tessellation modes (standard and power/Laguerre) are stable, and the
0.6.0 release now includes a first-class planar namespace.
A future 1.0 release is planned once the inverse-fitting workflow is more mature,
its disconnected-graph / coverage diagnostics are stabilized, and the project has
reassessed whether planar `PeriodicCell` support is actually needed.

## AI-assisted development

Some parts of the implementation, tests, and documentation were developed with
AI assistance (OpenAI ChatGPT). The maintainer reviews and integrates changes,
and remains responsible for the resulting code and scientific claims.

Details are documented in the [AI usage](https://delonecommons.github.io/pyvoro2/project/ai/) page.

## License

- Starting with **0.6.0**, the pyvoro2-authored code is released under the **GNU Lesser General Public License v3.0 or later (LGPLv3+)**.
- Versions **before 0.6.0** were released under the **MIT License**.
- Voro++ is vendored and redistributed under its original upstream license.

---

*This README is auto-generated from the MkDocs sources in `docs/`.*
To update it, edit the docs pages and re-run: `python tools/gen_readme.py`.

