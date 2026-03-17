# About pyvoro2

## Summary

pyvoro2 is a scientific Python package for computing **2D and 3D Voronoi-type tessellations**.
It is built on top of the established C++ library **Voro++**, and it focuses on the parts
that usually decide whether a tessellation is merely “computed” or actually **usable** in downstream analysis:

- periodic boundary conditions (including **triclinic** unit cells in 3D and rectangular periodicity in 2D),
- extraction of neighbor graphs with the correct periodic images,
- diagnostic checks and normalization utilities for reproducible topology work.

At the core, pyvoro2 exposes only two mathematically standard tessellations:

- **standard Voronoi** (unweighted), and
- **power / Laguerre** tessellations (weighted Voronoi, via per-site radii).

## What is Voro++

Voro++ is a widely used C++ library for computing Voronoi cells efficiently in 3D.
pyvoro2 also vendors the legacy upstream 2D sources for its planar namespace.
These backends are commonly used in computational physics and materials science.

pyvoro2 vendors a snapshot of upstream Voro++ and builds its Python extension against it.
The vendored snapshot includes the upstream numeric robustness fix for *power/Laguerre* (radical) pruning,
which avoids rare cross-platform edge cases in fully periodic power tessellations.

## When should you use pyvoro2?

Use pyvoro2 when you need one (or more) of the following:

- Voronoi or power/Laguerre tessellations in **2D or 3D**,
- periodic domains (especially triclinic crystal cells in 3D or rectangular periodic cells in 2D),
- a neighbor graph where the periodic image is explicit,
- “point queries” such as owner lookup (`locate`) or probe cells (`ghost_cells`).

If your task is a one-off Voronoi computation in a simple orthogonal box and you do not need
periodic graph bookkeeping, a smaller wrapper may also be sufficient.

## What does pyvoro2 add on top of Voro++?

Voro++ is a C++ library with a low-level API. pyvoro2 provides:

- a stable and test-driven Python API,
- Python-friendly outputs (dicts + NumPy arrays),
- periodic neighbor image shifts (`adjacent_shift`) for graph work,
- diagnostics (`analyze_tessellation`) and normalization helpers,
- inverse fitting tools that turn desired interface placements into a **power diagram** in both planar and spatial settings.

## Compared to `pyvoro`

`pyvoro` is an older Python wrapper around Voro++.

pyvoro2 aims to be a more modern interface with a larger emphasis on:

- periodic crystals (including triclinic),
- correctness checks and reproducible topology utilities,
- a dedicated planar namespace (`pyvoro2.planar`) rather than 2D/3D overload magic,
- additional operations beyond “compute all cells”.

## Design note: stateless API

pyvoro2 does not keep a persistent C++ container object across calls.
Each call creates a container, inserts sites, performs the operation, and returns results.

This keeps the interface small and avoids hidden state.
A persistent “index/container” object may be added in the far future for specialized workloads,
but it is not a current priority.

## Testing and validation

pyvoro2 is developed with a strong emphasis on **reproducible correctness**.
In practice, numerical geometry libraries can fail in subtle ways (degeneracies,
near-coplanar faces, roundoff on periodic images), so tests are structured in layers:

- **Deterministic unit tests** (default `pytest` run) cover the public API and
  common edge cases for all supported domains.
- **Fuzz/property tests** (`pytest -m fuzz`) generate random point sets and domains
  and assert robust invariants (finite volumes, no NaNs, reciprocal face-shift bookkeeping, etc.).
  These tests are opt-in because they are intentionally more expensive.
- **Optional cross-check tests** (`pytest -m pyvoro`) compare a subset of results against
  the older `pyvoro` wrapper (when installed). This provides an additional sanity check
  that two independent wrappers around Voro++ agree on stable quantities.

Typical commands:

```bash
pip install -e ".[test]"
pytest

# Opt-in randomized checks
pytest -m fuzz --fuzz-n 100

# Optional cross-checks (requires pyvoro)
pip install pyvoro
pytest -m pyvoro --fuzz-n 100
```

For contributor-style local validation of the whole repository, including
notebooks, docs, generated files, and distribution artifacts:

```bash
pip install -e ".[all]"
python tools/release_check.py
```

For continuous integration and local development, the recommended approach is to run
the deterministic suite frequently, and run fuzz/cross-check suites periodically.

