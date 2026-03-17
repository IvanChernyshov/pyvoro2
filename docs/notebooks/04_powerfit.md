<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/04_powerfit.ipynb)
# Power fitting from pairwise bisector constraints

This notebook shows the new math-oriented inverse API in `pyvoro2`:

1. resolve pairwise bisector constraints,
2. fit power weights under a configurable model,
3. match realized pairs in the resulting power tessellation,
4. run the self-consistent active-set solver.
```python
import numpy as np

import pyvoro2 as pv
```
## 1) Resolve and fit a simple two-site constraint

A raw constraint tuple is `(i, j, value[, shift])`, where `value` is
interpreted in either fraction-space or absolute position-space.
```python
points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
box = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

constraints = pv.resolve_pair_bisector_constraints(
    points,
    [(0, 1, 0.25)],
    measurement='fraction',
    domain=box,
)

fit = pv.fit_power_weights(points, constraints)

print('weights:', fit.weights)
print('radii:', fit.radii)
print('predicted fraction:', fit.predicted_fraction)
print('predicted position:', fit.predicted_position)
print('status:', fit.status)
print('weight shift:', fit.weight_shift)
```
## 2) Add hard feasibility and a near-boundary penalty

The fitting model separates mismatch, hard feasibility, and soft penalties.
```python
model = pv.FitModel(
    mismatch=pv.SquaredLoss(),
    feasible=pv.Interval(0.0, 1.0),
    penalties=(
        pv.ExponentialBoundaryPenalty(
            lower=0.0,
            upper=1.0,
            margin=0.05,
            strength=1.0,
            tau=0.01,
        ),
    ),
)

fit_penalized = pv.fit_power_weights(
    points,
    [(0, 1, 1e-3)],
    measurement='fraction',
    domain=box,
    model=model,
    solver='admm',
)

print('predicted fraction with penalty:', fit_penalized.predicted_fraction[0])
```
## 3) Match realized pairs after fitting

Requested pairwise separators do not automatically become realized faces
in the full power tessellation.
```python
realized = pv.match_realized_pairs(
    points,
    domain=box,
    radii=fit.radii,
    constraints=constraints,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
)

print('realized:', realized.realized)
print('same shift:', realized.realized_same_shift)
print('boundary measure:', realized.boundary_measure)
print('tessellation ok:', realized.tessellation_diagnostics.ok)
```
## 4) Self-consistent active-set refinement

For larger candidate sets, the active-set solver repeatedly fits, tessellates,
and keeps the constraints whose requested pairs are actually realized.
```python
points3 = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    dtype=float,
)
box3 = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

result = pv.solve_self_consistent_power_weights(
    points3,
    [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
    measurement='fraction',
    domain=box3,
    options=pv.ActiveSetOptions(add_after=1, drop_after=2, relax=0.5),
    return_history=True,
    return_boundary_measure=True,
)

print('termination:', result.termination)
print('active mask:', result.active_mask)
print('constraint status:', result.diagnostics.status)
print('marginal constraints:', result.marginal_constraints)

print('path summary:', result.path_summary)
```
## Disconnected path example

The next example starts from an empty active set so the first fitted subproblem is completely disconnected, while the final active set reconnects into the expected nearest-neighbor chain. This illustrates the difference between final-state diagnostics and optimization-path diagnostics.
```python
points4 = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    dtype=float,
)
box4 = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

result_path = pv.solve_self_consistent_power_weights(
    points4,
    [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
    measurement='fraction',
    domain=box4,
    active0=np.array([False, False, False]),
    options=pv.ActiveSetOptions(add_after=1, drop_after=1, max_iter=6),
    return_history=True,
    connectivity_check='diagnose',
    unaccounted_pair_check='diagnose',
)

print('final active graph components:', result_path.connectivity.active_graph.n_components)
print('path summary:', result_path.path_summary)
print('first history row:', result_path.history[0])
```
