from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / 'README.md'
SCRIPT = REPO_ROOT / 'tools' / 'gen_readme.py'


def test_readme_is_in_sync(tmp_path: Path) -> None:
    generated = tmp_path / 'README.generated.md'
    subprocess.run(
        [sys.executable, str(SCRIPT), '--output', str(generated)],
        cwd=REPO_ROOT,
        check=True,
    )
    assert generated.read_text(encoding='utf-8') == README.read_text(encoding='utf-8')
