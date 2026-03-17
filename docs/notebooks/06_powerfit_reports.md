<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/06_powerfit_reports.ipynb)
# Powerfit reports and record exports

This notebook focuses on the plain-record and nested-report helpers
around low-level fits, realized-pair matching, and the self-consistent
active-set solver.
```python
import numpy as np

import pyvoro2 as pv
```
## 1) Resolve a small candidate set

We use explicit integer ids so that exported rows already carry the labels
that downstream code wants to show.
```python
points = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ],
    dtype=float,
)
ids = np.array([100, 101, 102], dtype=int)
box = pv.Box(((-1.0, 5.0), (-2.0, 2.0), (-2.0, 2.0)))

constraints = pv.resolve_pair_bisector_constraints(
    points,
    [(0, 1, 0.35), (1, 2, 0.55), (0, 2, 0.50)],
    measurement="fraction",
    domain=box,
    ids=ids,
)
constraints.to_records(use_ids=True)
```
## 2) Fit power weights and export low-level reports
```python
model = pv.FitModel(
    mismatch=pv.SquaredLoss(),
    feasible=pv.Interval(0.0, 1.0),
    penalties=(
        pv.ExponentialBoundaryPenalty(
            lower=0.0,
            upper=1.0,
            margin=0.05,
            strength=0.2,
            tau=0.02,
        ),
    ),
)

fit = pv.fit_power_weights(
    points,
    constraints,
    model=model,
)

fit_rows = fit.to_records(constraints, use_ids=True)
fit_report = fit.to_report(constraints, use_ids=True)
fit_report["summary"]

fit_report["weight_shift"]
```
## 3) Check realized pairs against the actual power tessellation
```python
realized = pv.match_realized_pairs(
    points,
    domain=box,
    radii=fit.radii,
    constraints=constraints,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
)

realized_rows = realized.to_records(constraints, use_ids=True)
realized_report = realized.to_report(constraints, use_ids=True)
realized_report["summary"]
```
## 4) Run the self-consistent active-set solver
## Final-state vs optimization-path reports

`solve_report["connectivity"]` and `solve_report["realized"]` describe the final returned solution. `solve_report["path_summary"]` and the optional `history` rows capture transient disconnectivity or candidate-absent realized pairs that occurred during the outer iterations.
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
        max_iter=12,
        cycle_window=6,
    ),
    return_history=True,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
)

result_rows = result.to_records(use_ids=True)
solve_report = result.to_report(use_ids=True)
solve_report["summary"]

solve_report["path_summary"]
```
## 5) Serialize the report bundle
```python
text = pv.dumps_report_json(solve_report, sort_keys=True)
text[:200]
```
The numerical API stays array-oriented, while the report helpers make it
easy to hand plain Python dictionaries or rows to downstream packages.
