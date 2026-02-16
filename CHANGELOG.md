# Changelog

All notable changes to this project are documented in this file.

The format is based on *Keep a Changelog*, and this project follows *Semantic Versioning*.

## [0.4.1] - 2026-02-16

### Fixed

- Vendored Voro++: inflate the stored global `max_radius` by 1 ULP (via `nextafter`) in
  power/Laguerre mode to make radical pruning robust across platforms.
- Removed a Python-side workaround that recomputed fully periodic orthorhombic power tessellations
  via the periodic (triclinic) backend when periodic face-shift assignment failed.
- Updated documentation to reflect the patched vendored Voro++ snapshot.

## [0.4.0] - 2026-02-15

Initial public release.

pyvoro2 wraps **unmodified Voro++** and provides a Python-first interface for **3D Voronoi** and
**power/Laguerre (radical Voronoi)** tessellations, with practical utilities for periodic boundary
conditions and topology/graph workflows.

### Added

- Tessellation computation:
    - `compute(..., mode='standard')` for standard Voronoi tessellations.
    - `compute(..., mode='power')` for power/Laguerre tessellations (per-site radii).
- Domains (containers):
    - `Box`: finite axis-aligned bounding box.
    - `OrthorhombicCell`: axis-aligned cell with per-axis periodicity (useful for 1D/2D periodic
      systems such as wires and slabs).
    - `PeriodicCell`: fully periodic triclinic cell, including `PeriodicCell.from_params(...)` for
      Voro++ lower-triangular parameters.
- Periodic neighbor-image support:
    - `return_face_shifts=True` to annotate faces with `adjacent_shift=(na, nb, nc)` lattice-image
      indices (for `PeriodicCell` and periodic `OrthorhombicCell`).
- Point-query operations:
    - `locate(...)`: batched ownership queries via Voro++ `find_voronoi_cell`.
    - `ghost_cells(...)`: batched probe cells via Voro++ `compute_ghost_cell`.
- Pre-processing utilities:
    - `duplicate_check(...)` near-duplicate point detection (Python-side).
    - `compute(...)`, `locate(...)`, and `ghost_cells(...)` can optionally run the near-duplicate
      pre-check via `duplicate_check='raise'|'warn'`.
    - `PeriodicCell` performs validation of lattice vectors, including near-degeneracy
      detection (ill-conditioned bases warn; nearly degenerate cells raise).
    - Additional input validation: non-finite coordinates are rejected; user `ids` must be unique
      and non-negative; power-mode radii are required to be finite and non-negative.
- Post-processing utilities:
    - `analyze_tessellation(...)` sanity checks.
    - `validate_tessellation(...)` strict validation (raises on failure).
    - `annotate_face_properties(...)` helpers.
    - `normalize.*` helpers for reproducible periodic topology work.
    - `validate_normalized_topology(...)` strict validation for normalized topology.
- Inverse fitting utilities (`pyvoro2.inverse`) to fit power weights/radii from desired
  separating plane positions.
- Optional visualization helpers (`pyvoro2.viz3d`, extra `pyvoro2[viz]`) based on `py3Dmol`.
- Documentation site (MkDocs Material) with a narrative guide, API reference, and example notebooks.
- Test suite with deterministic unit tests, opt-in fuzz/property tests, and opt-in cross-checks
  against `pyvoro`.
