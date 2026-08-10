"""Chronovisor directory management."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".chronovisor"


def resolve_root() -> Path:
    """Return the one authoritative Chronovisor data root."""

    configured = os.environ.get("CHRONOVISOR_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve(strict=False)

    return DEFAULT_ROOT


@dataclass(frozen=True)
class RuntimeContext:
    """Immutable paths for one Chronovisor data root."""

    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def pages_dir(self) -> Path:
        return self.root / "pages"

    @property
    def system_dir(self) -> Path:
        return self.root / "system"

    @property
    def config_file(self) -> Path:
        return self.root / "config.toml"

    @property
    def model_lab_replay_file(self) -> Path:
        return self.root / "runtime" / "model-lab" / "replay.jsonl"

    @property
    def index_file(self) -> Path:
        return self.root / "index.md"

    @property
    def log_file(self) -> Path:
        return self.root / "log.md"

    @property
    def schema_file(self) -> Path:
        return self.root / "schema.md"

    @property
    def codex_state_file(self) -> Path:
        return self.root / "codex-save-state.json"

    @property
    def claude_code_state_file(self) -> Path:
        return self.root / "claude-code-save-state.json"


DEFAULT_CONTEXT = RuntimeContext(resolve_root())
CHRONOVISOR_ROOT = DEFAULT_CONTEXT.root
RAW_DIR = DEFAULT_CONTEXT.raw_dir
PAGES_DIR = DEFAULT_CONTEXT.pages_dir
SYSTEM_DIR = DEFAULT_CONTEXT.system_dir
MODEL_LAB_REPLAY_FILE = DEFAULT_CONTEXT.model_lab_replay_file
INDEX_FILE = DEFAULT_CONTEXT.index_file
LOG_FILE = DEFAULT_CONTEXT.log_file
SCHEMA_FILE = DEFAULT_CONTEXT.schema_file


def all_pages() -> list[Path]:
    """Return all wiki pages (supports subdirectories)."""
    return list(PAGES_DIR.rglob("*.md"))


def _valid_page_id(page_id: object) -> bool:
    return bool(
        isinstance(page_id, str)
        and page_id.strip()
        and page_id.strip() not in {".", ".."}
        and "/" not in page_id
        and "\\" not in page_id
        and "\x00" not in page_id
        and not any(character in page_id for character in "*?[]")
        and not Path(page_id).is_absolute()
    )


def _resolve_page_path(
    page_id: str,
    path: Path,
    *,
    allowed_roots: tuple[Path, ...],
    allow_alias: bool = False,
) -> Path | None:
    """Resolve one page path only when its location is safe to read."""
    if not _valid_page_id(page_id):
        return None

    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file() or (not allow_alias and resolved.stem != page_id):
        return None

    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            continue
        return resolved
    return None


def find_page(
    page_id: str,
    *,
    candidate: Path | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
    allow_alias: bool = False,
) -> Path | None:
    """Find a page by ID (filename without extension). Searches subdirectories."""
    if not _valid_page_id(page_id):
        return None
    roots = allowed_roots if allowed_roots is not None else (PAGES_DIR,)
    if candidate is not None:
        return _resolve_page_path(
            page_id,
            candidate,
            allowed_roots=roots,
            allow_alias=allow_alias,
        )

    # Direct flat path (most common case)
    flat = _resolve_page_path(
        page_id,
        PAGES_DIR / f"{page_id}.md",
        allowed_roots=roots,
    )
    if flat is not None:
        return flat
    # Search subdirectories
    try:
        for path in PAGES_DIR.rglob(f"{page_id}.md"):
            resolved = _resolve_page_path(
                page_id,
                path,
                allowed_roots=roots,
            )
            if resolved is not None:
                return resolved
    except (OSError, RuntimeError):
        pass
    return None


def page_id_from_path(path: Path) -> str:
    """Extract page ID from a path (just the stem, no folder)."""
    return path.stem


def init_chronovisor(context: RuntimeContext | None = None) -> None:
    """Initialize the Chronovisor directory structure."""
    if context is None:
        raw_dir, pages_dir, system_dir = RAW_DIR, PAGES_DIR, SYSTEM_DIR
        index_file, log_file, schema_file = INDEX_FILE, LOG_FILE, SCHEMA_FILE
    else:
        raw_dir, pages_dir, system_dir = (
            context.raw_dir,
            context.pages_dir,
            context.system_dir,
        )
        index_file, log_file, schema_file = (
            context.index_file,
            context.log_file,
            context.schema_file,
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)

    if not index_file.exists():
        index_file.write_text(
            "---\ntitle: Index\nupdated: 1970-01-01\n---\n\n"
            "# Wiki Index\n\nNo pages yet.\n"
        )

    if not log_file.exists():
        log_file.write_text(
            "---\ntitle: Log\nupdated: 1970-01-01\n---\n\n"
            "# Change Log\n"
        )

    if not schema_file.exists():
        schema_file.write_text(SCHEMA_CONTENT)


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
