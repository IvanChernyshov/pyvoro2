from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'tools' / 'check_notebooks.py'

spec = importlib.util.spec_from_file_location('check_notebooks_tool', MODULE_PATH)
assert spec is not None and spec.loader is not None
check_notebooks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = check_notebooks
spec.loader.exec_module(check_notebooks)


def test_configure_import_path_default_is_noop() -> None:
    original = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != str(check_notebooks.SRC)]
        check_notebooks.configure_import_path(use_src=False)
        assert str(check_notebooks.SRC) not in sys.path
    finally:
        sys.path[:] = original


def test_configure_import_path_use_src_prepends_repo_src() -> None:
    original = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != str(check_notebooks.SRC)]
        check_notebooks.configure_import_path(use_src=True)
        assert sys.path[0] == str(check_notebooks.SRC)
    finally:
        sys.path[:] = original
