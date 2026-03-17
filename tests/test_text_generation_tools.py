from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'tools' / 'export_notebooks.py'

spec = importlib.util.spec_from_file_location('export_notebooks_tool', MODULE_PATH)
assert spec is not None and spec.loader is not None
export_notebooks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export_notebooks
spec.loader.exec_module(export_notebooks)


def test_export_notebooks_check_ignores_crlf(tmp_path: Path) -> None:
    """Notebook export checks should ignore Windows vs Unix newlines."""

    output_dir = tmp_path / 'docs'
    output_dir.mkdir()

    for notebook_path, output_path in export_notebooks.iter_notebook_pairs():
        rendered = export_notebooks.export_markdown(notebook_path)
        (output_dir / output_path.name).write_text(
            rendered.replace('\n', '\r\n'),
            encoding='utf-8',
            newline='',
        )

    assert export_notebooks.export_notebooks(output_dir, check=True) == 0
