# Notebooks

The example notebooks are kept in the repository-root `notebooks/` directory so
that users can browse them directly on GitHub without going through the docs
site.

For the published docs, each notebook is also exported to a generated Markdown
page under `docs/notebooks/`.

## Source notebooks in the repository

The repository source notebooks are:

- `notebooks/01_basic_compute.ipynb`
- `notebooks/02_periodic_graph.ipynb`
- `notebooks/03_locate_and_ghost.ipynb`
- `notebooks/04_powerfit.ipynb`
- `notebooks/05_visualization.ipynb`
- `notebooks/06_powerfit_reports.ipynb`
- `notebooks/07_powerfit_infeasibility.ipynb`
- `notebooks/08_powerfit_active_path.ipynb`

## Published notebook pages

The generated documentation pages are:

- [01 basic compute](../notebooks/01_basic_compute.md)
- [02 periodic graph](../notebooks/02_periodic_graph.md)
- [03 locate and ghost cells](../notebooks/03_locate_and_ghost.md)
- [04 powerfit workflow](../notebooks/04_powerfit.md)
- [05 visualization](../notebooks/05_visualization.md)
- [06 powerfit reports](../notebooks/06_powerfit_reports.md)
- [07 powerfit infeasibility](../notebooks/07_powerfit_infeasibility.md)
- [08 active-set path diagnostics](../notebooks/08_powerfit_active_path.md)

## Regeneration

To refresh the generated pages after editing notebooks:

```bash
python tools/export_notebooks.py
```

To validate notebook executability against the installed `pyvoro2` package in the current environment:

```bash
python tools/check_notebooks.py
```

If you are using the wheel-overlay developer workflow and want notebook imports to resolve from `repo/src`, use:

```bash
python tools/check_notebooks.py --use-src
```
