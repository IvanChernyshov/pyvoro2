from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / 'tools' / 'export_notebooks.py'
NOTEBOOKS = REPO_ROOT / 'notebooks'
EXPORTED_NOTEBOOKS = REPO_ROOT / 'docs' / 'notebooks'

EXPECTED_NOTEBOOKS = {
    '01_basic_compute.ipynb',
    '02_periodic_graph.ipynb',
    '03_locate_and_ghost.ipynb',
    '04_powerfit.ipynb',
    '05_visualization.ipynb',
    '06_powerfit_reports.ipynb',
    '07_powerfit_infeasibility.ipynb',
    '08_powerfit_active_path.ipynb',
}

EXPECTED_PAGES = {name.replace('.ipynb', '.md') for name in EXPECTED_NOTEBOOKS}


def test_notebook_files_exist() -> None:
    actual = {path.name for path in NOTEBOOKS.glob('*.ipynb')}
    assert EXPECTED_NOTEBOOKS.issubset(actual)


def test_exported_notebook_pages_are_in_sync() -> None:
    actual = {path.name for path in EXPORTED_NOTEBOOKS.glob('*.md')}
    assert EXPECTED_PAGES.issubset(actual)
    subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), '--check'],
        cwd=REPO_ROOT,
        check=True,
    )
