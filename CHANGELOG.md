# Changelog

All notable changes to this project are documented in this file.

The format is based on *Keep a Changelog*, and this project follows *Semantic Versioning*.

## [0.5.1] - 2026-03-15


### Added

- `tools/install_wheel_overlay.py` to support a wheel-core + repository-source
  development workflow, so the compiled extension can come from an installed
  wheel while Python imports resolve to `src/pyvoro2`.

### Fixed

- Power-fit input validation now rejects non-finite point coordinates,
  constraint values, confidence weights, and non-finite radius/weight
  conversion inputs.
- `resolve_pair_bisector_constraints(...)` now validates external `ids`
  consistently, including shape/length and uniqueness checks.
- The quadratic/analytic power-fit solver no longer crashes on zero-confidence
  constraints that would otherwise create singular gauge coupling.
- Empty resolved constraint sets now respect L2 regularization and return the
  regularization-only solution instead of silently dropping the reference.

## [0.5.0] - 2026-03-14

### Added

- New `pyvoro2.powerfit` API for inverse power fitting from generic pairwise bisector constraints.
- Power-fitting results now export plain-Python record rows for downstream reporting and diagnostics.
- Hard infeasibility reporting is simplified around explicit contradiction witnesses.
- `resolve_pair_bisector_constraints(...)` as a reusable low-level constraint-resolution primitive.
- `fit_power_weights(...)` with configurable mismatch, hard feasibility, soft penalties, and explicit infeasibility reporting.
- `match_realized_pairs(...)` for purely geometric realized-face matching with optional tessellation diagnostics.
- `solve_self_consistent_power_weights(...)` for hysteretic active-set refinement driven by realized faces.
- Rich per-constraint diagnostics, marginal-pair reporting, and optional final tessellation diagnostics.

### Changed

- The inverse-fitting surface is now math-oriented and chemistry-agnostic.
- Documentation and examples now describe the unified power-fitting workflow.
- The 0.5.x objective-model scope is explicitly documented around the current built-in convex model family.

## [0.4.2] - 2026-03-04

### Changed

- Vendored Voro++: updated the vendored snapshot to include the upstream numeric robustness fix for
  power/Laguerre (radical) pruning (fixes rare cross-platform edge cases in fully periodic power
  tessellations).
- Removed the previously vendored local `nextafter`-based `max_radius` inflation patch (no longer needed).

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
