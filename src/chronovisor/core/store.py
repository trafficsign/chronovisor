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
    def index_file(self) -> Path:
        return self.pages_dir / "index.md"

    @property
    def log_file(self) -> Path:
        return self.pages_dir / "log.md"

    @property
    def schema_file(self) -> Path:
        return self.system_dir / "schema.md"

    @property
    def activity_file(self) -> Path:
        return self.root / "runtime" / "activity.jsonl"

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
INDEX_FILE = DEFAULT_CONTEXT.index_file
LOG_FILE = DEFAULT_CONTEXT.log_file
SCHEMA_FILE = DEFAULT_CONTEXT.schema_file
ACTIVITY_FILE = DEFAULT_CONTEXT.activity_file


def all_pages() -> list[Path]:
    """Return stable, non-reserved Wiki pages (supports subdirectories)."""

    from chronovisor.core.index_store import (
        canonical_document_paths,
    )

    return canonical_document_paths(PAGES_DIR, require_stable=True)


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
    except OSError, RuntimeError:
        return None
    if not resolved.is_file() or (not allow_alias and resolved.stem != page_id):
        return None

    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve(strict=True))
        except OSError, RuntimeError, ValueError:
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
    except OSError, RuntimeError:
        pass
    return None


def page_id_from_path(path: Path) -> str:
    """Extract page ID from a path (just the stem, no folder)."""
    return path.stem


def init_chronovisor(context: RuntimeContext | None = None) -> None:
    """Create only the directories needed to capture transcripts."""
    if os.environ.get("CHRONOVISOR_READ_ONLY") == "1":
        return
    raw_dir = RAW_DIR if context is None else context.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(raw_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
