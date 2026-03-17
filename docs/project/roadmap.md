# Roadmap

This page lists intended future improvements. It is not a guarantee of timelines.

## Recently completed

### Planar 2D support in 0.6.0

The 0.6.0 line adds a dedicated `pyvoro2.planar` namespace built on a separate
`_core2d` backend. The supported first-release planar scope is intentionally
honest:

- `Box` and `RectangularCell` domains,
- planar compute / locate / ghost-cell operations,
- periodic edge-shift recovery for rectangular periodic domains,
- planar diagnostics, normalization, plotting, and power-fitting support.

It does **not** yet promise a planar oblique-periodic analogue of the 3D
`PeriodicCell`.

### Documentation and release hygiene in 0.6.1

The 0.6.1 line also cleans up the repository-facing documentation workflow:

- notebooks now live at the repository root and are exported into generated
  Markdown pages for the docs site;
- the package metadata now exposes a convenience `pyvoro2[all]` extra for full
  local validation;
- repository tooling now includes notebook export checks, notebook execution,
  README sync checks, distribution-content validation, and a single
  `tools/release_check.py` entry point.

### Powerfit robustness in 0.6.1

The 0.6.1 line hardens the inverse-fitting stack around underdetermined and
mis-specified candidate graphs:

- realized internal boundaries for candidate-absent pairs are now reported
  explicitly in both 3D and planar 2D workflows;
- low-level fits and self-consistent solves now expose structured connectivity
  diagnostics for candidate graphs, active graphs, unconstrained sites, and
  component identifiability;
- disconnected-component gauge handling now follows explicit component-mean or
  previous-iterate alignment policies rather than arbitrary anchor order;
- self-consistent results now distinguish final-state diagnostics from
  optimization-path diagnostics through `path_summary` and richer history rows;
- the weight-to-radius conversion path now exposes `weight_shift=` directly
  instead of relying only on the older minimum-radius convention.

## Planned / likely

The next roadmap questions are no longer about the basic powerfit surface, but
about validation depth and overall API stabilization.

## Potential / exploratory

### Planar oblique-periodic domains

A future planar `PeriodicCell` remains possible, but it is deferred rather than
promised. One possible fallback is a pseudo-3D implementation with careful
projection back to 2D, but that needs its own evaluation before it should be
part of the public contract.

### Visualization usability

The optional viewers (`pyvoro2[viz]` / `pyvoro2[viz2d]`) are intended as
lightweight debugging and exploration tools. The current direction is to keep
them simple but make the examples and notebook workflows more polished, rather
than turning visualization into a heavy core dependency.

## Release stability

pyvoro2 is currently in **beta**.

A “stable” 1.0 release is expected only after:

- the 0.6.1 robustness work is validated in downstream use,
- the current planar scope is validated in downstream use,
- the need (or non-need) for planar `PeriodicCell` is reassessed.
