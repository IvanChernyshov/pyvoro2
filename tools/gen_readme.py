#!/usr/bin/env python3
"""Generate README.md from selected MkDocs markdown sources.

Why?
- Keeps the README and documentation in sync.
- Produces a single landing-page README derived from the docs sources.

How it works:
- Concatenates a curated set of docs pages in a fixed order.
- Ensures there is only one top-level H1 heading.
- Rewrites relative links:
  - `.md`/`.ipynb` links -> absolute links to the hosted docs site
  - image links under `docs/` -> absolute raw GitHub links (for PyPI rendering)

Run:
    python tools/gen_readme.py

Configuration:
- The script tries to auto-detect the GitHub `owner/repo` from `pyproject.toml`.
- You can override detection via environment variables:
  - `PYVORO2_GITHUB_REPO` (e.g. `MyOrg/pyvoro2`)
  - `PYVORO2_REPO_REF` (e.g. `main`, `v0.4.0`)
  - `PYVORO2_DOCS_SITE` (full docs base URL)
"""

from __future__ import annotations

from pathlib import Path
import os
import re
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore
from urllib.parse import quote


DEFAULT_GITHUB_REPO = 'IvanChernyshov/pyvoro2'
DEFAULT_REPO_REF = 'main'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / 'docs'
OUT = PROJECT_ROOT / 'README.md'
PYPROJECT = PROJECT_ROOT / 'pyproject.toml'
MKDOCS_YML = PROJECT_ROOT / 'mkdocs.yml'

# Curated docs pages to include in the README (in order).
#
# The README is meant to be a compact landing page. We keep full explanations
# (and especially equations) in the hosted documentation.
PAGES = [
    DOCS_ROOT / 'index.md',
]


def _load_pyproject() -> dict:
    if not PYPROJECT.exists():
        return {}
    try:
        return tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _detect_package_name(pyproject: dict) -> str:
    name = (pyproject.get('project', {}) or {}).get('name')
    if isinstance(name, str) and name.strip():
        return name.strip()
    return 'pyvoro2'


def _detect_repo_slug(pyproject: dict) -> str:
    """Return `owner/repo` slug for GitHub.

    The detection order is:
    1) env var `PYVORO2_GITHUB_REPO`
    2) parse `project.urls.Repository` / `Homepage` from pyproject
    3) fallback `DEFAULT_GITHUB_REPO`
    """

    env_slug = os.environ.get('PYVORO2_GITHUB_REPO')
    if isinstance(env_slug, str) and env_slug.strip():
        return env_slug.strip().strip('/')

    urls = (pyproject.get('project', {}) or {}).get('urls', {}) or {}
    repo_url = None
    if isinstance(urls, dict):
        repo_url = urls.get('Repository') or urls.get('Homepage')

    if isinstance(repo_url, str) and repo_url.strip():
        m = re.match(
            r'^https?://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\\.git)?/?$',
            repo_url.strip(),
        )
        if m:
            return f'{m.group("owner")}/{m.group("repo")}'

    return DEFAULT_GITHUB_REPO


def _detect_docs_site(*, owner: str, repo: str) -> str:
    """Return the hosted docs base URL (with trailing slash)."""
    env_site = os.environ.get('PYVORO2_DOCS_SITE')
    if isinstance(env_site, str) and env_site.strip():
        site = env_site.strip()
        return site if site.endswith('/') else site + '/'

    # Prefer mkdocs.yml site_url if present (keeps one source of truth).
    if MKDOCS_YML.exists():
        try:
            txt = MKDOCS_YML.read_text(encoding='utf-8')
        except Exception:
            txt = ''
        m = re.search(r'^site_url:\s*(?P<url>.+?)\s*$', txt, flags=re.MULTILINE)
        if m:
            raw = m.group('url').strip().strip('\'\"')
            if raw:
                return raw if raw.endswith('/') else raw + '/'

    return f'https://{owner}.github.io/{repo}/'


def _repo_urls(*, owner: str, repo: str, repo_ref: str) -> tuple[str, str, str]:
    repo_url = f'https://github.com/{owner}/{repo}'
    docs_site = _detect_docs_site(owner=owner, repo=repo)
    raw_base = f'https://raw.githubusercontent.com/{owner}/{repo}/{repo_ref}'
    return repo_url, docs_site, raw_base


_BADGE_TEMPLATES: list[tuple[str, str]] = [
    (
        'CI',
        '/actions/workflows/ci.yml/badge.svg',
    ),
    (
        'Docs',
        '/actions/workflows/docs.yml/badge.svg',
    ),
]


def _build_badges(*, repo_url: str, pkg_name: str, license_url: str) -> str:
    badges: list[str] = []
    for label, badge_path in _BADGE_TEMPLATES:
        workflow = badge_path.replace('/badge.svg', '')
        badges.append(f'[![{label}]({repo_url}{badge_path})]({repo_url}{workflow})')

    badges.append(
        f'[![PyPI](https://img.shields.io/pypi/v/{pkg_name}.svg)]'
        f'(https://pypi.org/project/{pkg_name}/)'
    )
    badges.append(
        f'[![Python Versions](https://img.shields.io/pypi/pyversions/{pkg_name}.svg)]'
        f'(https://pypi.org/project/{pkg_name}/)'
    )
    badges.append(
        f'[![License](https://img.shields.io/pypi/l/{pkg_name}.svg)]({license_url})'
    )

    return ' '.join(badges)


def _shift_headings(md: str, *, delta: int) -> str:
    """Increase Markdown heading levels by `delta` (outside code fences)."""
    out_lines: list[str] = []
    in_fence = False
    fence_pat = re.compile(r'^```')

    for line in md.splitlines():
        if fence_pat.match(line):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence or not line.startswith('#'):
            out_lines.append(line)
            continue

        m = re.match(r'^(#+)\s+(.*)$', line)
        if not m:
            out_lines.append(line)
            continue
        hashes, rest = m.group(1), m.group(2)
        out_lines.append('#' * (len(hashes) + delta) + ' ' + rest)
    return '\n'.join(out_lines)


def _encode_url_path(path: str) -> str:
    """URL-encode a POSIX path for safe inclusion in raw GitHub URLs."""
    # Keep forward slashes unescaped so the URL path structure remains intact.
    return quote(path, safe='/-_.~')


_LINK_PAT = re.compile(r'(?P<prefix>!?)\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)')


_IMG_SRC_PAT = re.compile(
    r'''(<img\b[^>]*?\bsrc\s*=\s*)(?P<q>["'])(?P<src>.+?)(?P=q)''',
    flags=re.IGNORECASE,
)


def _split_md_link_target(target: str) -> tuple[str, str]:
    """Split a Markdown link target into (url, trailing_title_part).

    The returned title part (if any) keeps its leading whitespace so it can be
    appended back verbatim.
    """
    t = target.strip()
    if not t:
        return '', ''
    if t.startswith('<') and '>' in t:
        url, rest = t[1:].split('>', 1)
        return url.strip(), rest
    m = re.match(r'^(\S+)(\s+.*)?$', t)
    if not m:
        return t, ''
    return m.group(1), m.group(2) or ''


def _rewrite_links(md: str, *, src_path: Path, docs_site: str, raw_base: str) -> str:
    """Rewrite relative links to absolute hosted links.

    Notes:
      - We intentionally do not rewrite inside code fences.
      - Image links under docs/ are rewritten to raw GitHub URLs so the README
        renders correctly on PyPI.
    """

    def repl(m: re.Match[str]) -> str:
        prefix = m.group('prefix')
        text = m.group('text')
        target = m.group('target')

        url_part, title_part = _split_md_link_target(target)
        url_part = url_part.strip()

        # Keep absolute and in-page links.
        if url_part.startswith(('http://', 'https://', 'mailto:', '#')):
            return m.group(0)

        # Split fragment (only from the URL part, not the optional title).
        if '#' in url_part:
            url_no_frag, frag = url_part.split('#', 1)
            frag = '#' + frag
        else:
            url_no_frag, frag = url_part, ''

        url_no_frag = url_no_frag.strip()

        # Images: if they resolve under docs/, rewrite to raw GitHub.
        if prefix == '!':
            try:
                resolved = (src_path.parent / url_no_frag).resolve()
                resolved_rel = resolved.relative_to(DOCS_ROOT)
            except Exception:
                return m.group(0)
            raw_path = _encode_url_path(f'docs/{resolved_rel.as_posix()}')
            raw = f'{raw_base}/{raw_path}'
            return f'![{text}]({raw}{frag}{title_part})'

        # Only rewrite .md and .ipynb links.
        if not (url_no_frag.endswith('.md') or url_no_frag.endswith('.ipynb')):
            return m.group(0)

        # Resolve relative link against the source file location.
        try:
            resolved = (src_path.parent / url_no_frag).resolve()
            resolved_rel = resolved.relative_to(DOCS_ROOT)
        except Exception:
            return m.group(0)

        # MkDocs directory URLs:
        #   - `guide/domains.md` -> `guide/domains/`
        #   - `guide/index.md`   -> `guide/`
        if resolved_rel.name == 'index.md':
            rel_dir = resolved_rel.parent.as_posix()
            if rel_dir == '.':
                url = docs_site
            else:
                url = docs_site + rel_dir.rstrip('/') + '/'
        else:
            url = docs_site + resolved_rel.with_suffix('').as_posix().rstrip('/') + '/'

        return f'[{text}]({url}{frag}{title_part})'

    def img_repl(m: re.Match[str]) -> str:
        prefix = m.group(1)
        q = m.group('q')
        src = m.group('src').strip()

        # Keep absolute and data URLs (and in-page fragments).
        if src.startswith(('http://', 'https://', 'data:', 'mailto:', '#')):
            return m.group(0)

        # Preserve fragments in src (rare but valid).
        if '#' in src:
            src_no_frag, frag = src.split('#', 1)
            frag = '#' + frag
        else:
            src_no_frag, frag = src, ''

        try:
            resolved = (src_path.parent / src_no_frag).resolve()
            resolved_rel = resolved.relative_to(DOCS_ROOT)
        except Exception:
            return m.group(0)

        raw_path = _encode_url_path(f'docs/{resolved_rel.as_posix()}')
        raw = f'{raw_base}/{raw_path}{frag}'
        return f'{prefix}{q}{raw}{q}'

    out_lines: list[str] = []
    in_fence = False
    fence_pat = re.compile(r'^```')
    for line in md.splitlines():
        if fence_pat.match(line):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        line2 = _LINK_PAT.sub(repl, line)
        line2 = _IMG_SRC_PAT.sub(img_repl, line2)
        out_lines.append(line2)

    return '\n'.join(out_lines)


def main() -> None:

    for p in PAGES:
        if not p.exists():
            raise FileNotFoundError(f'Missing docs page: {p}')

    pyproject = _load_pyproject()
    pkg_name = _detect_package_name(pyproject)

    slug = _detect_repo_slug(pyproject)
    owner, repo = slug.split('/', 1)

    repo_ref = os.environ.get('PYVORO2_REPO_REF', DEFAULT_REPO_REF)
    repo_url, docs_site, raw_base = _repo_urls(
        owner=owner, repo=repo, repo_ref=repo_ref
    )
    license_url = f'{repo_url}/blob/{repo_ref}/LICENSE'

    badges = _build_badges(
        repo_url=repo_url, pkg_name=pkg_name, license_url=license_url
    )

    parts: list[str] = []

    # Header + badges.
    parts.append(
        f'# {pkg_name}\n\n' + badges + '\n' + f'**Documentation:** {docs_site}\n'
    )

    # Add pages.
    for i, p in enumerate(PAGES):
        md = p.read_text(encoding='utf-8').strip() + '\n'
        md = _rewrite_links(md, src_path=p, docs_site=docs_site, raw_base=raw_base)

        if i == 0:
            # Strip the first H1 if present (we already provide the README title).
            md_lines = md.splitlines()
            if md_lines and md_lines[0].strip().lower() in (
                f'# {pkg_name}',
                f'# {pkg_name.lower()}',
            ):
                md = '\n'.join(md_lines[1:]).lstrip('\n')
        else:
            md = _shift_headings(md, delta=1)

        parts.append(md.strip())

    out = '\n\n---\n\n'.join([p for p in parts if p.strip()])

    out += '\n\n---\n\n' + (
        '*This README is auto-generated from the MkDocs sources in `docs/`.*\n'
        'To update it, edit the docs pages and re-run: `python tools/gen_readme.py`.\n'
    )

    OUT.write_text(out + '\n', encoding='utf-8')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
