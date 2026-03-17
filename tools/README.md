# Tooling helpers

This directory contains repository-maintenance helpers used in local
publishability checks and CI.

Main entry points:

- `python tools/export_notebooks.py` — regenerate `docs/notebooks/*.md` from
  the source notebooks in the repo-root `notebooks/` directory.
- `python tools/check_notebooks.py` — validate notebook JSON and execute
  notebook code cells against the installed `pyvoro2` package in the current
  environment. Pass `--use-src` only in a wheel-overlay developer setup where
  the compiled extensions are already available beside `src/pyvoro2/`.
- `python tools/gen_readme.py` — regenerate `README.md` from the MkDocs source.
- `python tools/release_check.py` — run the combined local release-preparation
  checks.
- `python tools/check_dist.py dist` — verify that built sdists and wheels
  contain the expected key files.

For a full local pre-release pass after installing the project with all optional
extras, run:

```bash
pip install -e ".[all]"
python tools/release_check.py
```
