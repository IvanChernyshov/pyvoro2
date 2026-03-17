#!/usr/bin/env python3
"""Run the full release-preparation checks for the repository."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / 'dist'
BUILD_DIR = REPO_ROOT / 'build'


def _run(*args: str, env: dict[str, str] | None = None) -> None:
    """Run one subprocess command in the repository root."""

    print('+', ' '.join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True, env=env)


def _fresh_build_dirs() -> None:
    """Remove build artifacts from previous runs."""

    shutil.rmtree(DIST_DIR, ignore_errors=True)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)


def _smoke_test_wheel() -> None:
    """Install the built wheel into a temporary virtualenv and smoke-test it."""

    wheels = sorted(DIST_DIR.glob('*.whl'))
    if not wheels:
        raise RuntimeError('no wheel found in dist/')
    wheel = wheels[-1]

    with tempfile.TemporaryDirectory(prefix='pyvoro2-release-check-') as tmp:
        env_dir = Path(tmp) / 'venv'
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(env_dir)
        bindir = 'Scripts' if sys.platform.startswith('win') else 'bin'
        python = env_dir / bindir / 'python'
        _run(str(python), '-m', 'pip', 'install', str(wheel))
        smoke = (
            "import numpy as np; "
            "import pyvoro2 as pv; "
            "import pyvoro2.planar as pv2; "
            "pts3 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float); "
            "cells3 = pv.compute(pts3, domain=pv.Box(((-5.0, 5.0), (-5.0, 5.0), "
            "(-5.0, 5.0))), mode='standard'); "
            "assert len(cells3) == 2; "
            "pts2 = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float); "
            "cells2 = pv2.compute(pts2, domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))), "
            "return_edges=True); "
            "assert len(cells2) == 2"
        )
        _run(str(python), '-c', smoke)


def main() -> int:
    """Run lint, tests, docs, build, metadata, and wheel smoke checks."""

    parser = argparse.ArgumentParser(
        description='Run the full release-preparation checks for the repository.',
    )
    parser.add_argument(
        '--skip-docs',
        action='store_true',
        help='skip the strict MkDocs build step',
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='skip building distributions and validating dist artifacts',
    )
    parser.add_argument(
        '--skip-smoke-test',
        action='store_true',
        help='skip the temporary-virtualenv wheel smoke test',
    )
    args = parser.parse_args()

    _run('flake8', 'src', 'tests', 'tools')
    _run(sys.executable, 'tools/check_notebooks.py')
    _run(sys.executable, 'tools/export_notebooks.py', '--check')
    _run(sys.executable, 'tools/gen_readme.py', '--check')
    _run(sys.executable, '-m', 'pytest', '-q')
    if not args.skip_docs:
        _run('mkdocs', 'build', '--strict')

    if args.skip_build:
        return 0

    _fresh_build_dirs()
    _run(sys.executable, '-m', 'build')
    _run(sys.executable, '-m', 'twine', 'check', 'dist/*')
    _run(sys.executable, 'tools/check_dist.py', 'dist')
    if not args.skip_smoke_test:
        _smoke_test_wheel()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
