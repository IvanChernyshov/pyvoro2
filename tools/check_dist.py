#!/usr/bin/env python3
"""Verify that built distributions contain the project's key files."""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import zipfile


REQUIRED_WHEEL_SUFFIXES = {
    'pyvoro2/__init__.py',
    'pyvoro2/__about__.py',
    'pyvoro2/planar/__init__.py',
    'pyvoro2/powerfit/solver.py',
    'pyvoro2/viz2d.py',
    'pyvoro2/viz3d.py',
    'pyvoro2/_core',
    'pyvoro2/_core2d',
}

REQUIRED_SDIST_SUFFIXES = {
    'README.md',
    'CHANGELOG.md',
    'DEV_PLAN.md',
    'LICENSE',
    'pyproject.toml',
    'notebooks/01_basic_compute.ipynb',
    'notebooks/02_periodic_graph.ipynb',
    'notebooks/03_locate_and_ghost.ipynb',
    'notebooks/04_powerfit.ipynb',
    'notebooks/05_visualization.ipynb',
    'notebooks/06_powerfit_reports.ipynb',
    'notebooks/07_powerfit_infeasibility.ipynb',
    'notebooks/08_powerfit_active_path.ipynb',
    'docs/notebooks/01_basic_compute.md',
    'docs/notebooks/02_periodic_graph.md',
    'docs/notebooks/03_locate_and_ghost.md',
    'docs/notebooks/04_powerfit.md',
    'docs/notebooks/05_visualization.md',
    'docs/notebooks/06_powerfit_reports.md',
    'docs/notebooks/07_powerfit_infeasibility.md',
    'docs/notebooks/08_powerfit_active_path.md',
    'tools/check_dist.py',
    'tools/check_notebooks.py',
    'tools/export_notebooks.py',
    'tools/gen_readme.py',
    'tools/release_check.py',
    'tools/README.md',
}


class DistCheckError(RuntimeError):
    """Raised when a built distribution is missing required members."""


def _assert_members_present(
    actual: set[str],
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - actual)
    if missing:
        joined = ', '.join(missing)
        raise DistCheckError(f'{label} is missing required members: {joined}')


def _members_matching_suffixes(actual: set[str], suffixes: set[str]) -> set[str]:
    matched: set[str] = set()
    for suffix in suffixes:
        if suffix in {'pyvoro2/_core', 'pyvoro2/_core2d'}:
            if any(name.startswith(suffix) for name in actual):
                matched.add(suffix)
            continue
        if any(name.endswith(suffix) for name in actual):
            matched.add(suffix)
    return matched


def check_wheel(path: Path) -> None:
    """Validate the contents of one built wheel."""

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    matched = _members_matching_suffixes(names, REQUIRED_WHEEL_SUFFIXES)
    _assert_members_present(matched, REQUIRED_WHEEL_SUFFIXES, label=path.name)


def check_sdist(path: Path) -> None:
    """Validate the contents of one built source distribution."""

    with tarfile.open(path, 'r:gz') as tf:
        names = {member.name for member in tf.getmembers()}
    matched = _members_matching_suffixes(names, REQUIRED_SDIST_SUFFIXES)
    _assert_members_present(matched, REQUIRED_SDIST_SUFFIXES, label=path.name)


def main() -> None:
    """Validate wheel and sdist artifacts found in a distribution directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument('dist_dir', type=Path, nargs='?', default=Path('dist'))
    args = parser.parse_args()

    dist_dir = args.dist_dir
    wheels = sorted(dist_dir.glob('*.whl'))
    sdists = sorted(dist_dir.glob('*.tar.gz'))
    if not wheels:
        raise DistCheckError(f'no wheel files found in {dist_dir}')
    if not sdists:
        raise DistCheckError(f'no source distributions found in {dist_dir}')

    for wheel in wheels:
        check_wheel(wheel)
    for sdist in sdists:
        check_sdist(sdist)


if __name__ == '__main__':
    main()
