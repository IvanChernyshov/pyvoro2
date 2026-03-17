#!/usr/bin/env python3
"""Export repository notebooks to Markdown pages for the docs site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / 'notebooks'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'docs' / 'notebooks'
HEADER = (
    '<!-- This file is generated from the matching notebook. -->\n'
    '<!-- Regenerate with: python tools/export_notebooks.py -->\n\n'
)
GITHUB_NOTEBOOK_BASE = (
    'https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks'
)


class NotebookExportError(RuntimeError):
    """Raised when notebook export fails."""


def iter_notebook_pairs() -> tuple[tuple[Path, Path], ...]:
    """Return `(notebook_path, markdown_path)` pairs in export order."""

    pairs: list[tuple[Path, Path]] = []
    for notebook in sorted(NOTEBOOKS.glob('*.ipynb')):
        pairs.append((notebook, DEFAULT_OUTPUT_DIR / f'{notebook.stem}.md'))
    return tuple(pairs)


def _load_notebook(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise NotebookExportError(f'{path.name}: expected top-level JSON object')
    return data


def _cell_source(cell: dict[str, object], *, path: Path, index: int) -> str:
    source = cell.get('source', [])
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return ''.join(source)
    raise NotebookExportError(f'{path.name}: invalid source in cell {index}')


def _output_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return ''.join(value)
    return ''


def _render_output(output: dict[str, object], *, path: Path, index: int) -> str:
    output_type = output.get('output_type')

    if output_type == 'stream':
        text = _output_text(output.get('text')).rstrip()
        if not text:
            return ''
        return f'**Output**\n\n```text\n{text}\n```\n'

    if output_type == 'error':
        traceback = _output_text(output.get('traceback')).rstrip()
        if not traceback:
            traceback = _output_text(output.get('evalue')).rstrip()
        if not traceback:
            ename = output.get('ename', 'error')
            traceback = str(ename)
        return f'**Error**\n\n```text\n{traceback}\n```\n'

    if output_type not in {'execute_result', 'display_data'}:
        raise NotebookExportError(
            f'{path.name}: unsupported output type in cell {index}: {output_type}'
        )

    data = output.get('data', {})
    if not isinstance(data, dict):
        raise NotebookExportError(f'{path.name}: invalid output data in cell {index}')

    if 'text/markdown' in data:
        text = _output_text(data['text/markdown']).strip()
        if text:
            return f'{text}\n'

    if 'text/html' in data:
        html = _output_text(data['text/html']).strip()
        if html:
            return f'{html}\n'

    if 'text/plain' in data:
        text = _output_text(data['text/plain']).rstrip()
        if text:
            return f'**Output**\n\n```text\n{text}\n```\n'

    # Some optional rich outputs (for example py3Dmol) include custom MIME
    # bundles alongside text/html. If there is no text/html or text/plain
    # fallback, skip the bundle rather than failing hard.
    return ''


def export_markdown(path: Path) -> str:
    """Render one notebook into a Markdown page without executing it."""

    data = _load_notebook(path)
    cells = data.get('cells')
    if not isinstance(cells, list):
        raise NotebookExportError(f'{path.name}: missing notebook cell list')

    parts: list[str] = [HEADER]
    parts.append(
        '[Open the original notebook on GitHub]'
        f'({GITHUB_NOTEBOOK_BASE}/{path.name})\n'
    )

    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            raise NotebookExportError(f'{path.name}: cell {index} is not an object')

        cell_type = cell.get('cell_type')
        source = _cell_source(cell, path=path, index=index)

        if cell_type == 'markdown':
            text = source.strip()
            if text:
                parts.append(f'{text}\n')
            continue

        if cell_type != 'code':
            continue

        code_text = source.rstrip()
        parts.append('```python\n')
        parts.append(f'{code_text}\n')
        parts.append('```\n')

        outputs = cell.get('outputs', [])
        if not isinstance(outputs, list):
            raise NotebookExportError(f'{path.name}: invalid outputs in cell {index}')
        for output in outputs:
            if not isinstance(output, dict):
                raise NotebookExportError(
                    f'{path.name}: output is not an object in cell {index}'
                )
            rendered = _render_output(output, path=path, index=index)
            if rendered:
                parts.append(rendered.rstrip() + '\n')

    body = '\n'.join(part.rstrip() for part in parts if part).rstrip()
    return body + '\n'


def export_notebooks(output_dir: Path, *, check: bool = False) -> int:
    """Export notebooks or verify that the exported pages are in sync."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for notebook_path, default_output in iter_notebook_pairs():
        rendered = export_markdown(notebook_path)
        output_path = output_dir / default_output.name
        if check:
            if not output_path.exists():
                print(f'missing exported page: {output_path}', file=sys.stderr)
                return 1
            current = output_path.read_text(encoding='utf-8').replace('\r\n', '\n')
            if current != rendered:
                print(
                    f'{output_path} is out of sync with {notebook_path.name}',
                    file=sys.stderr,
                )
                return 1
        else:
            output_path.write_text(rendered, encoding='utf-8', newline='\n')
    return 0


def main() -> int:
    """Export notebook Markdown pages or check that they are current."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--check',
        action='store_true',
        help='exit with status 1 when exported pages are out of sync',
    )
    args = parser.parse_args()
    return export_notebooks(args.output_dir, check=args.check)


if __name__ == '__main__':
    raise SystemExit(main())
