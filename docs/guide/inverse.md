# Power fitting from pairwise bisector constraints

`pyvoro2` can solve the inverse problem for **power / Laguerre tessellations**:
fit auxiliary power weights so that selected pairwise separators land at desired
locations along the connector between two sites.

The API is intentionally **geometry-first** and **domain-agnostic**.
Downstream code decides:

- which site pairs are candidates,
- which periodic image shift belongs to each pair,
- the target separator location for each pair,
- and any per-constraint confidence.

`pyvoro2` then provides the mathematical pieces:

- resolve and validate pair constraints,
- fit power weights under a configurable convex model,
- compute the resulting power tessellation,
- detect which constraints correspond to realized faces,
- and optionally refine an active set to self-consistency.

## Geometry of one pair

For a pair of sites `i` and `j`, choose one specific image of `j` and call it
`j*`. Let

- `d = ||p_j* - p_i||`,
- `z = w_i - w_j`,

where `w` are the fitted power weights.

Then the separator position along the connector is affine in `z`:

$$
 t(z) = \frac{1}{2} + \frac{z}{2 d^2}
$$

for normalized fraction, and

$$
 x(z) = \frac{d}{2} + \frac{z}{2 d}
$$

for absolute position measured from site `i`.

This is why `pyvoro2` exposes the measurement type explicitly: a loss in
fraction-space and a loss in position-space are **different optimization
problems**.

## Step 1: resolve pair constraints once

```python
import numpy as np
import pyvoro2 as pv

points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
box = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

constraints = pv.resolve_pair_bisector_constraints(
    points,
    [(0, 1, 0.25)],
    measurement='fraction',
    domain=box,
)
```

Each raw tuple is `(i, j, value[, shift])`, where `shift=(na, nb, nc)` is the
integer lattice image applied to site `j`.

The resolved object stores the validated pair indices, shifts, connector
geometry, and targets in both fraction and position form.

## Step 2: define the fitting model

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
```

The model separates three ideas:

- `mismatch=`: how target-vs-predicted separator locations are scored,
- `feasible=`: hard admissible sets such as an interval or fixed value,
- `penalties=`: soft penalties such as outside-interval or near-boundary
  repulsion.

Built-in pieces currently include:

- `SquaredLoss()`
- `HuberLoss(delta=...)`
- `Interval(lower, upper)`
- `FixedValue(value)`
- `SoftIntervalPenalty(lower, upper, strength=...)`
- `ExponentialBoundaryPenalty(...)`
- `ReciprocalBoundaryPenalty(...)`
- `L2Regularization(...)`

## Step 3: fit power weights

```python
fit = pv.fit_power_weights(
    points,
    constraints,
    model=model,
    r_min=1.0,
)
```

The result contains:

- fitted `weights` and shifted `radii`,
- predicted separator locations in both fraction and position form,
- residuals in the chosen measurement space,
- solver/termination metadata,
- and explicit infeasibility reporting for contradictory hard constraints.

For example, if hard interval or equality restrictions cannot all hold
simultaneously, the fit returns:

- `status == 'infeasible_hard_constraints'`
- `hard_feasible == False`
- `weights is None`

instead of pretending the issue is merely slow convergence.

## Step 4: check which pairs are actually realized

A requested pairwise separator is not automatically a realized face in the full
power tessellation. After fitting, you can ask which requested pairs became real
neighbors.

```python
realized = pv.match_realized_pairs(
    points,
    domain=box,
    radii=fit.radii,
    constraints=constraints,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
)
```

This returns purely geometric diagnostics:

- whether each pair is realized at all,
- whether it is realized with the **same** requested periodic shift,
- whether only some **other** image is realized,
- whether one of the endpoint cells is empty,
- an optional boundary measure of the matched face,
- and optional tessellation-wide diagnostics.

## Step 5: solve the self-consistent active-set problem

For sparse or noisy candidate sets, the useful high-level workflow is often:

1. fit on a current active set,
2. run the actual power tessellation,
3. keep or re-add only the constraints whose pairs are realized,
4. repeat until active and realized sets agree.

`pyvoro2` provides this as:

```python
result = pv.solve_self_consistent_power_weights(
    points,
    constraints,
    domain=box,
    model=model,
    options=pv.ActiveSetOptions(
        add_after=1,
        drop_after=2,
        relax=0.5,
        max_iter=25,
        cycle_window=8,
    ),
    return_history=True,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
)
```

The solver is generic:

- it never invents candidate pairs,
- it never silently changes the user-supplied periodic image,
- it uses realized faces rather than any domain-specific contact logic,
- it supports hysteresis, under-relaxation, cycle detection, and marginal-pair
  reporting.

## Reading the final diagnostics

`solve_self_consistent_power_weights(...)` returns both a final low-level fit and
rich per-constraint diagnostics.

Useful fields include:

- `result.constraints`: the resolved pair set used throughout the solve,
- `result.active_mask`: final active-set membership,
- `result.realized`: realized-face matching diagnostics,
- `result.diagnostics`: per-constraint targets, predictions, residuals,
  endpoint-empty flags, boundary measure, toggle counts, and generic status
  labels,
- `result.rms_residual_all` / `result.max_residual_all`: summaries over **all**
  candidate constraints,
- `result.tessellation_diagnostics`: final tessellation-wide checks,
- `result.marginal_constraints`: indices of toggling / cycle / wrong-shift pairs.

Status labels are intentionally generic, for example:

- `stable_active`
- `stable_inactive`
- `toggled_active`
- `toggled_inactive`
- `realized_other_shift`
- `active_unrealized`
- `cycle_member`

## Current scope

The current implementation is 3D because it builds on the existing Voro++-based
power tessellation core. The **API vocabulary** is already dimension-safe:
constraint fitting is phrased in terms of pairwise separators and boundary
measure rather than chemistry-specific or 3D-only semantics.
