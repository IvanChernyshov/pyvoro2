# Roadmap

This page lists intended future improvements. It is not a guarantee of timelines.

## Planned / likely

### Native 2D support

Voro++ ships a dedicated 2D implementation. pyvoro2 plans to expose it as a **separate extension
module** (e.g. `_core2d`) so that 2D and 3D code do not collide at link time.

### Inverse-fitting iteration helpers

The inverse fitter can report constraints that do not become active faces (“inactive constraints”).
A future iteration could provide helper routines to:

1) fit weights
2) compute the diagram
3) keep only active neighbor constraints
4) refit

This would make it easier to use the inverse workflow as an iterative model-fitting loop.

## Potential

### Visualization usability

The optional `py3Dmol`-based viewer (`pyvoro2[viz]`) is intended as a lightweight debugging and
exploration tool. Future work is expected to focus on usability (better defaults, more annotations),
not on adding heavy rendering dependencies to the core.

## Release stability

pyvoro2 is currently in **beta**.

A “stable” 1.0 release is expected only after:

- the inverse-fitting workflow matures further
- native 2D support is implemented and tested
