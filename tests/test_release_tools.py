from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_name: str) -> str:
    result = subprocess.run(
        [sys.executable, f'tools/{script_name}', '--help'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_release_check_help() -> None:
    assert 'release-preparation checks' in _run_help('release_check.py')


def test_check_notebooks_help() -> None:
    assert 'optional notebook filenames' in _run_help('check_notebooks.py')


def test_check_dist_help() -> None:
    assert 'dist_dir' in _run_help('check_dist.py')
