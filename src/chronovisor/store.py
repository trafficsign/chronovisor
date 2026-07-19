"""Chronovisor directory management and legacy-root compatibility."""

import os
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".chronovisor"
LEGACY_ROOT = Path.home() / ".wiki"


def resolve_root() -> Path:
    """Return the one authoritative data root.

    Before migration an existing legacy root remains usable.  After migration
    ``~/.wiki`` may only coexist as a symlink to ``~/.chronovisor``.  Two
    independent trees are rejected so writers can never diverge silently.
    """

    configured = os.environ.get("CHRONOVISOR_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve(strict=False)

    new_exists = DEFAULT_ROOT.exists()
    legacy_exists = LEGACY_ROOT.exists() or LEGACY_ROOT.is_symlink()
    if new_exists and legacy_exists:
        try:
            if LEGACY_ROOT.resolve(strict=True) == DEFAULT_ROOT.resolve(strict=True):
                return DEFAULT_ROOT
        except OSError:
            pass
        raise RuntimeError(
            "split-brain data roots detected: both ~/.chronovisor and ~/.wiki "
            "exist independently"
        )
    if new_exists:
        return DEFAULT_ROOT
    if legacy_exists:
        return LEGACY_ROOT.resolve(strict=False)
    return DEFAULT_ROOT


CHRONOVISOR_ROOT = resolve_root()
RAW_DIR = CHRONOVISOR_ROOT / "raw"
PAGES_DIR = CHRONOVISOR_ROOT / "pages"
SYSTEM_DIR = CHRONOVISOR_ROOT / "system"
INDEX_FILE = CHRONOVISOR_ROOT / "index.md"
LOG_FILE = CHRONOVISOR_ROOT / "log.md"
SCHEMA_FILE = CHRONOVISOR_ROOT / "schema.md"


def all_pages() -> list[Path]:
    """Return all wiki pages (supports subdirectories)."""
    return list(PAGES_DIR.rglob("*.md"))


def find_page(page_id: str) -> Path | None:
    """Find a page by ID (filename without extension). Searches subdirectories."""
    # Direct flat path (most common case)
    flat = PAGES_DIR / f"{page_id}.md"
    if flat.exists():
        return flat
    # Search subdirectories
    matches = list(PAGES_DIR.rglob(f"{page_id}.md"))
    return matches[0] if matches else None


def page_id_from_path(path: Path) -> str:
    """Extract page ID from a path (just the stem, no folder)."""
    return path.stem


def init_chronovisor() -> None:
    """Initialize the Chronovisor directory structure."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_FILE.exists():
        INDEX_FILE.write_text(
            "---\ntitle: Index\nupdated: 1970-01-01\n---\n\n"
            "# Wiki Index\n\nNo pages yet.\n"
        )

    if not LOG_FILE.exists():
        LOG_FILE.write_text(
            "---\ntitle: Log\nupdated: 1970-01-01\n---\n\n"
            "# Change Log\n"
        )

    if not SCHEMA_FILE.exists():
        SCHEMA_FILE.write_text(SCHEMA_CONTENT)


SCHEMA_CONTENT = """\
---
title: Chronovisor Schema
updated: 2026-04-10
---

# Chronovisor Schema

## Page Granularity
1 entity = 1 page (Karpathy convention).

## Naming
- File names: kebab-case.md (English)
- Example: `jt-v10-probability-contexts.md`

## Frontmatter
Minimal, AI-first. Only two fields:
```yaml
---
title: Page Title
updated: YYYY-MM-DD
---
```

## Cross-references
Use `[[wiki-link]]` notation.
- Example: `[[jt-v10-probability-contexts]]`
- Links are bidirectional (backlinks tracked by read endpoint).

## Update Rules
LLM decides (guidelines only):
- Clear factual correction → overwrite
- State change (version upgrade, etc.) → preserve history and update
- Uncertain → append and let Lint handle it
"""
