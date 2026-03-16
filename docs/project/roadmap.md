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

## Planned / likely

### Powerfit robustness (next cycle)

The next high-priority design question is no longer “can the active-set loop
iterate?”, but whether the package clearly reports when the inverse model is
underspecified.

Planned work includes:

- reporting realized pair adjacencies that are present in the tessellation but
  absent from the supplied candidate set;
- graph/connectivity diagnostics for candidate and active-set graphs;
- clearer policies for disconnected components and unconstrained sites;
- revisiting the remaining chemistry-driven radius-convention legacy in the
  `weights_to_radii(...)` path.

## Potential / exploratory

### Planar oblique-periodic domains

A future planar `PeriodicCell` remains possible, but it is deferred rather than
promised. One possible fallback is a pseudo-3D implementation with careful
projection back to 2D, but that needs its own evaluation before it should be
part of the public contract.

### Visualization usability

The optional viewers (`pyvoro2[viz]` / `pyvoro2[viz2d]`) are intended as
lightweight debugging and exploration tools. Future work is expected to focus
on usability and examples, not on making visualization a heavy core
dependency.

## Release stability

pyvoro2 is currently in **beta**.

A “stable” 1.0 release is expected only after:

- the post-0.6.0 powerfit robustness work lands,
- the current planar scope is validated in downstream use,
- the need (or non-need) for planar `PeriodicCell` is reassessed.
