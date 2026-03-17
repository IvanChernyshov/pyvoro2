# Development plan (post-0.6.0)

This file is the internal working plan after the 0.6.0 feature freeze. It is
more concrete than the public roadmap and may evolve during implementation.

## Release sequence

### 0.6.0 (freeze and release)

Scope of the 0.6.0 release:

- ship planar 2D support in a dedicated `pyvoro2.planar` namespace;
- keep the planar scope honest: `Box` and `RectangularCell`, but **no** planar
  oblique-periodic `PeriodicCell` yet;
- ship planar compute/locate/ghost cells, edge shifts, diagnostics,
  normalization, plotting, and planar `powerfit` support;
- finish documentation, reference pages, examples, and release cleanup.

The 0.6.0 line should **not** add major new inverse-fitting policies. Those
belong in 0.6.1.

### 0.6.1 (powerfit robustness, implemented in the current tree)

The 0.6.1 scope is now implemented in the working tree attached to this chat.

Implemented focus:

- realized pair adjacencies that exist in the tessellation but are **absent**
  from the supplied candidate set are now reported explicitly rather than being
  silently ignored or auto-added;
- low-level fits and self-consistent active-set solves now expose structured
  graph/connectivity diagnostics for unconstrained points, isolated points,
  connected components, and whether relative offsets are identified by the
  pairwise data;
- disconnected standalone fits now use an explainable component-mean gauge
  policy, while self-consistent solves preserve offsets per connected active
  component by alignment to the previous iterate;
- `weights_to_radii(...)` now supports an explicit `weight_shift=` gauge, with
  `r_min=` retained as a compatibility-oriented convenience rather than the
  preferred mathematical framing;
- plain-Python report helpers now serialize both connectivity diagnostics and
  realized-but-unaccounted pair diagnostics;
- self-consistent solves now retain optimization-path diagnostics through a
  compact `path_summary` object plus richer optional per-iteration history rows;
- notebook/documentation examples are refreshed around explicit gauge language,
  path diagnostics, and the lightweight planar plotting helper.
- repository workflow now keeps the source notebooks in a repo-root `notebooks/`
  directory, exports them to `docs/notebooks/*.md`, exposes a convenience
  `.[all]` extra, and provides single-command publishability checks through
  `tools/release_check.py`.

The current preferred default policy for disconnected components is now the
implemented behavior:

- if an explicit reference exists, align each disconnected standalone component
  to the reference mean on that component;
- otherwise, center each disconnected standalone component by its mean;
- in the self-consistent loop, preserve component offsets relative to the
  previous iterate whenever the active effective graph is disconnected.

This remains a convention, not information identified by the pairwise data, so
connectivity diagnostics continue to support `none` / `diagnose` / `warn` /
`raise` policies.

### Deferred / exploratory (candidate 0.6.2+ work)

#### Planar `PeriodicCell`

Planar oblique-periodic support is **deferred** rather than promised for a
specific release.

The options to keep in mind are:

- continue without planar `PeriodicCell` if the rectangular 2D scope proves
  sufficient in practice;
- prototype a pseudo-3D fallback (planar sites embedded in 3D, then projected
  back to 2D);
- reevaluate the backend situation later if upstream Voro++ changes become
  compelling.

The current upstream assessment still stands: `voro-dev` does not appear to add
an honest 2D analogue of pyvoro2's current 3D `PeriodicCell`, so switching
engines is not required for first-class planar support.

## 1.0 gate

Do not freeze 1.0 immediately after 0.6.0.

The intended checkpoint is:

1. release 0.6.0 with the completed planar rectangular scope;
2. implement the 0.6.1 powerfit-robustness work;
3. reassess whether planar `PeriodicCell` is actually needed;
4. only then decide whether the public API is stable enough for 1.0.

The public API should be frozen around a stable mathematical surface, not
around any particular backend snapshot.

## Notes carried forward from the backend review

- We should **not** switch pyvoro2 to `voro-dev` merely to obtain 2D support.
  The current dedicated 2D backend is already sufficient for the honest first
  planar scope.
- A later backend migration remains possible, but it should be an internal
  engineering change rather than the moment when the Python API becomes mature.
- The public Python surface should stay explicit about dimension:
  `pyvoro2` for 3D, `pyvoro2.planar` for 2D.
