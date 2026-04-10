"""Wiki directory management."""

from pathlib import Path

WIKI_ROOT = Path.home() / ".wiki"
RAW_DIR = WIKI_ROOT / "raw"
PAGES_DIR = WIKI_ROOT / "pages"
INDEX_FILE = WIKI_ROOT / "index.md"
LOG_FILE = WIKI_ROOT / "log.md"
SCHEMA_FILE = WIKI_ROOT / "schema.md"


def init_wiki() -> None:
    """Initialize wiki directory structure."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

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
title: Wiki Schema
updated: 2026-04-10
---

# Wiki Schema

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
