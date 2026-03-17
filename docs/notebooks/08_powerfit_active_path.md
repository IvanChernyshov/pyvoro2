<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/08_powerfit_active_path.ipynb)
# Active-set path diagnostics

This notebook focuses on the difference between **final-state** diagnostics and **optimization-path** diagnostics in `solve_self_consistent_power_weights(...)`. The path diagnostics are especially useful when the active graph is transiently disconnected, even though the final returned solution is connected.
```python
import numpy as np
import pyvoro2 as pv
```
## A chain example with an initially empty active set

The candidate graph is connected through the nearest-neighbor chain, but the first fitted subproblem is completely disconnected because `active0` is empty. The final active set reconnects after the first realization pass.
```python
points = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    dtype=float,
)
box = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

result = pv.solve_self_consistent_power_weights(
    points,
    [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
    measurement="fraction",
    domain=box,
    active0=np.array([False, False, False]),
    options=pv.ActiveSetOptions(add_after=1, drop_after=1, max_iter=6),
    return_history=True,
    connectivity_check="diagnose",
    unaccounted_pair_check="diagnose",
)

print("termination:", result.termination)
print("final active mask:", result.active_mask)
print("final active graph components:", result.connectivity.active_graph.n_components)
print("path summary:", result.path_summary)
```
```python
for row in result.history:
    print(row)
```
Notice the distinction between `n_active_fit` (the mask that actually generated the current iterate) and `n_active` (the post-toggle mask used for the next iterate). This lets downstream code say whether disconnectivity happened **during** optimization, not just in the final answer.
```python
solve_report = result.to_report()
solve_report["path_summary"]
```
