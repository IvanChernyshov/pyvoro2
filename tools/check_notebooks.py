#!/usr/bin/env python3
"""Validate notebook JSON structure and execute notebook code cells."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'
NOTEBOOKS = REPO_ROOT / 'notebooks'
DEFAULT_NOTEBOOKS = (
    '01_basic_compute.ipynb',
    '02_periodic_graph.ipynb',
    '03_locate_and_ghost.ipynb',
    '04_powerfit.ipynb',
    '05_visualization.ipynb',
    '06_powerfit_reports.ipynb',
    '07_powerfit_infeasibility.ipynb',
    '08_powerfit_active_path.ipynb',
)

os.environ.setdefault('MPLBACKEND', 'Agg')


class NotebookCheckError(RuntimeError):
    """Raised when a notebook is malformed or fails to execute."""


def iter_notebooks(selected: tuple[str, ...] | None = None) -> tuple[Path, ...]:
    """Return the notebooks that should be validated and executed."""

    names = selected or DEFAULT_NOTEBOOKS
    return tuple(NOTEBOOKS / name for name in names)


def load_notebook(path: Path) -> dict[str, object]:
    """Load one notebook JSON document."""

    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise NotebookCheckError(f'{path.name}: expected top-level JSON object')
    return data


def iter_code_cells(data: dict[str, object], *, path: Path) -> tuple[str, ...]:
    """Return notebook code-cell sources in order."""

    cells = data.get('cells')
    if not isinstance(cells, list):
        raise NotebookCheckError(f'{path.name}: missing notebook cell list')

    code_cells: list[str] = []
    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            raise NotebookCheckError(f'{path.name}: cell {index} is not an object')
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', [])
        if isinstance(source, str):
            text = source
        elif isinstance(source, list) and all(isinstance(line, str) for line in source):
            text = ''.join(source)
        else:
            raise NotebookCheckError(
                f'{path.name}: cell {index} has invalid code source'
            )
        code_cells.append(text)
    if not code_cells:
        raise NotebookCheckError(f'{path.name}: contains no code cells')
    return tuple(code_cells)


def execute_notebook(path: Path) -> None:
    """Execute all code cells from one notebook in a shared namespace."""

    if not path.exists():
        raise NotebookCheckError(f'missing notebook: {path}')

    data = load_notebook(path)
    namespace = {'__name__': '__main__'}

    for index, source in enumerate(iter_code_cells(data, path=path), start=1):
        if not source.strip():
            continue
        try:
            code = compile(source, f'{path.name}::cell{index}', 'exec')
            with redirect_stdout(io.StringIO()):
                exec(code, namespace, namespace)
        except Exception as exc:  # noqa: BLE001
            raise NotebookCheckError(
                f'{path.name}: execution failed in code cell {index}: {exc}'
            ) from exc


def configure_import_path(*, use_src: bool) -> None:
    """Configure where notebook imports should resolve pyvoro2 from."""

    if use_src and str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def main() -> int:
    """Validate and execute every requested notebook."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'notebooks',
        nargs='*',
        help='optional notebook filenames under notebooks/ to validate',
    )
    parser.add_argument(
        '--use-src',
        action='store_true',
        help=(
            'prepend repo/src to sys.path before execution; use this only in '
            'a developer overlay environment where the compiled extensions are '
            'already available beside the source tree'
        ),
    )
    args = parser.parse_args()

    configure_import_path(use_src=args.use_src)

    selected = tuple(args.notebooks) if args.notebooks else None
    notebooks = iter_notebooks(selected)
    for notebook in notebooks:
        execute_notebook(notebook)
    print(f'Validated {len(notebooks)} notebook(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
