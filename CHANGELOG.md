# Changelog

All notable changes to this project are documented in this file.

The format is based on *Keep a Changelog*, and this project follows *Semantic Versioning*.

## [0.6.1] - 2026-03-16

### Added

- Explicit realized-but-unaccounted pair diagnostics in both 3D and planar 2D power-fit realization, including public `UnaccountedRealizedPair` / `UnaccountedRealizedPairError` types and JSON/report export support.
- Structured connectivity diagnostics for low-level fits and self-consistent active-set solves, covering unconstrained points, isolated points, connected components of candidate and active graphs, and whether relative offsets are identified by the data or only by gauge policy.
- Active-set path diagnostics via `result.path_summary` and richer per-iteration `history` rows, so downstream code can distinguish final disconnectedness from transient component splits or transient candidate-absent realized pairs during optimization.
- Repo-root notebook sources plus notebook-export / notebook-check tooling, distribution-content checks, and a one-shot `tools/release_check.py` helper for local publishability validation.

### Changed

- Disconnected standalone fits no longer inherit arbitrary anchor-order gauges: each effective component is centered to mean zero by default, or aligned to the regularization-reference mean when a zero-strength reference is supplied.
- Self-consistent active-set fitting now preserves offsets per connected component of the current active effective graph by aligning each component to the previous iterate, including the final recomputed fit returned to the user.
- `weights_to_radii(...)` and the fitting APIs now support an explicit `weight_shift=` gauge, while keeping `r_min=` as a backward-compatible convenience rather than the primary convention.
- Power-fit reports now serialize connectivity diagnostics, unaccounted realized pairs, realized-diagnostics warnings, and active-set path summaries through the plain-Python report helpers.
- The example notebooks now live in a repo-root `notebooks/` directory and are exported into generated docs pages, while `README.md` and docs deployment are checked for sync in CI.
- The package metadata now includes a convenience `pyvoro2[all]` extra for contributors who want the full optional notebook/docs/release-check stack.
- The optional planar `plot_tessellation(...)` helper now accepts `domain=` and `show_sites=` to match the published guide examples.

### Fixed

- Active-set reports now nest the final low-level fit against the final active constraint subset rather than the full candidate table.
- Periodic self-image boundaries are excluded from the new unaccounted-pair diagnostics, so wrong-shift reporting does not misclassify self-adjacencies as missing candidate pairs.

## [0.6.0] - 2026-03-16

### Added

- New `pyvoro2.planar` namespace with the first 2D public surface: `Box`, `RectangularCell`, `compute`, `locate`, `ghost_cells`, duplicate checking, edge-property annotation, and optional matplotlib visualization helpers.
- Vendored legacy Voro++ 2D backend is now wired into the build as a separate `_core2d` extension target.
- New planar edge-shift reconstruction helper and pre-wheel integration tests that skip cleanly until `_core2d` wheels are available.
- New planar tessellation diagnostics and strict validation helpers: `analyze_tessellation(...)` and `validate_tessellation(...)`.
- New planar normalization helpers: `normalize_vertices(...)`, `normalize_topology(...)`, and `validate_normalized_topology(...)`.
- New `pyvoro2.planar.PlanarComputeResult` for structured wrapper-level compute results carrying raw cells, optional tessellation diagnostics, and optional normalized outputs.
- `pyvoro2.powerfit` realized-boundary matching and self-consistent active-set refinement now support planar 2D domains in addition to the original 3D path.

### Changed

- `pyvoro2.planar.compute(...)` now supports wrapper-level tessellation diagnostics (`return_diagnostics=...`, `tessellation_check=...`) and structured normalization convenience (`normalize='vertices'|'topology'`, `return_result=True`), automatically computing temporary periodic edge shifts/geometry when needed and stripping the temporary fields back out of the raw returned cells unless they were explicitly requested.
- `tools/install_wheel_overlay.py` now understands both `_core` and `_core2d`, so the editable-style wheel-overlay workflow can carry planar support once new wheels are built.
- Package metadata, release notes, and top-level documentation now describe the frozen 0.6.0 release rather than the earlier development snapshot.
- `resolve_pair_bisector_constraints(...)` now accepts both planar (2D) and spatial (3D) point sets, with dimension-aware shift validation and nearest-image resolution.
- Power-fit reports now serialize both 2D and 3D tessellation diagnostics through a shared measure-oriented schema while preserving the existing area/volume-specific fields.

### Fixed

- Periodic 2D edge reconstruction now resolves hidden periodic adjacencies that the legacy backend can surface as negative neighbor ids, so fully periodic planar tessellations expose consistent neighbor/shift data to diagnostics and normalization utilities.

## [0.5.1] - 2026-03-15


### Added

- `tools/install_wheel_overlay.py` to support a wheel-core + repository-source
  development workflow, so the compiled extension can come from an installed
  wheel while Python imports resolve to `src/pyvoro2`.
- `DEV_PLAN.md` in the repository root with the planned 0.6.x refactoring and
  2D implementation roadmap, including the current decision to ship planar 2D
  against the existing dedicated 2D backend before considering a later
  `voro-dev` migration.

### Changed

- Public API validation and block-grid resolution are now routed through shared
  internal helpers (`_inputs.py`, `_domain_geometry.py`) so 3D wrappers and the
  power-fit layer no longer duplicate the same coercion and geometry logic.
- Project status metadata is now consistently marked as **beta** across the
  package metadata and top-level documentation.

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
- `fit_power_weights(...)` and the active-set driver now return the documented
  `numerical_failure` status for linear-algebra and non-finite-iterate failures
  instead of surfacing them as uncaught exceptions or misclassified active-set
  infeasibility.
- Triclinic nearest-image resolution now warns when a chosen image touches the
  `image_search` boundary, making the search-window sensitivity explicit for
  skewed periodic cells.

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
