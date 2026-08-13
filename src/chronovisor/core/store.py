"""Chronovisor directory management."""

import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chronovisor.core.durable_state import atomic_write_bytes_at

if TYPE_CHECKING:
    from chronovisor.core.okf_cutover import OKFStartupDecision

DEFAULT_ROOT = Path.home() / ".chronovisor"
_BOOTSTRAP_TRANSIENT_CATEGORIES = frozenset(
    {
        "bootstrap_proof_invalid",
        "content_without_migration",
        "startup_inspection_failed",
    }
)


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

    @property
    def pi_state_file(self) -> Path:
        return self.root / "pi-save-state.json"


DEFAULT_CONTEXT = RuntimeContext(resolve_root())
CHRONOVISOR_ROOT = DEFAULT_CONTEXT.root
RAW_DIR = DEFAULT_CONTEXT.raw_dir
PAGES_DIR = DEFAULT_CONTEXT.pages_dir
SYSTEM_DIR = DEFAULT_CONTEXT.system_dir
MODEL_LAB_REPLAY_FILE = DEFAULT_CONTEXT.model_lab_replay_file
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


def okf_startup_status(root: Path) -> "OKFStartupDecision":
    """Return the read-only OKF startup decision for one data root."""
    from chronovisor.core.okf_cutover import discover_okf_startup

    return discover_okf_startup(root, root / "runtime")


def _bootstrap_discovery_needs_retry(
    root: Path, decision: "OKFStartupDecision"
) -> bool:
    if decision.category not in _BOOTSTRAP_TRANSIENT_CATEGORIES:
        return False
    try:
        return stat.S_ISREG(
            os.lstat(root / "runtime" / "bootstrap-layout.lock").st_mode
        )
    except OSError:
        return False


def _rediscover_after_bootstrap_sync(root: Path) -> "OKFStartupDecision":
    from chronovisor.core.okf_cutover import discover_okf_startup

    return discover_okf_startup(root, root / "runtime")


@contextmanager
def okf_runtime_operation(
    root: Path,
    *,
    blocking: bool = True,
    allow_bootstrap_resume: bool = False,
) -> Iterator["OKFStartupDecision"]:
    """Gate one runtime operation while holding the shared writer lease."""

    from chronovisor.core.durable_state import okf_writer_lock
    from chronovisor.core.okf_cutover import (
        OKFStartupBlocked,
        OKFStartupDecision,
        discover_okf_startup,
        require_okf_startup_allowed,
    )

    if not (root / "runtime" / "okf-writer.lock").exists():
        preflight = discover_okf_startup(root, root / "runtime")
        if allow_bootstrap_resume and _bootstrap_discovery_needs_retry(
            root, preflight
        ):
            from chronovisor.core.live_layout import bootstrap_layout_lock

            with bootstrap_layout_lock(root):
                preflight = _rediscover_after_bootstrap_sync(root)
        bootstrap_refused = (
            preflight.allowed
            and preflight.layout == "bootstrap"
            and not allow_bootstrap_resume
        )
        blocked_without_resume = not preflight.allowed and not (
            allow_bootstrap_resume
            and preflight.category == "bootstrap_in_progress"
        )
        if bootstrap_refused or blocked_without_resume:
            if bootstrap_refused:
                preflight = OKFStartupDecision(
                    False,
                    "blocked",
                    "in-progress",
                    "bootstrap_in_progress",
                )
            raise OKFStartupBlocked(preflight)
    with ExitStack() as stack:
        try:
            stack.enter_context(okf_writer_lock(root, blocking=blocking))
            decision = discover_okf_startup(root, root / "runtime")
            synchronize_bootstrap = allow_bootstrap_resume and (
                (decision.allowed and decision.layout == "bootstrap")
                or decision.category == "bootstrap_in_progress"
                or _bootstrap_discovery_needs_retry(root, decision)
            )
            if synchronize_bootstrap:
                from chronovisor.core.live_layout import bootstrap_layout_lock

                stack.enter_context(bootstrap_layout_lock(root))
                decision = _rediscover_after_bootstrap_sync(root)
            if decision.allowed and decision.layout == "bootstrap":
                if not allow_bootstrap_resume:
                    raise OKFStartupBlocked(
                        OKFStartupDecision(
                            False,
                            "blocked",
                            "in-progress",
                            "bootstrap_in_progress",
                        )
                    )
                decision = OKFStartupDecision(
                    True,
                    "bootstrap",
                    "in-progress",
                    "ok",
                )
            if not decision.allowed:
                if not (
                    allow_bootstrap_resume
                    and decision.category == "bootstrap_in_progress"
                ):
                    require_okf_startup_allowed(root, root / "runtime")
                decision = OKFStartupDecision(
                    True,
                    "bootstrap",
                    "in-progress",
                    "ok",
                )
        except OKFStartupBlocked:
            raise
        except (OSError, RuntimeError, ValueError):
            raise OKFStartupBlocked(
                OKFStartupDecision(
                    False,
                    "blocked",
                    "blocked",
                    "writer_lease_unavailable",
                )
            ) from None
        yield decision


def prepare_okf_startup(root: Path, run_id: str) -> Path:
    """Prepare one offline OKF workspace below the root's runtime directory."""
    from chronovisor.core.okf_workspace import prepare_okf_workspace

    return prepare_okf_workspace(root, root / "runtime", run_id)


def init_chronovisor(context: RuntimeContext | None = None) -> None:
    """Initialize the Chronovisor directory structure."""
    if context is None:
        raw_dir, pages_dir, system_dir = RAW_DIR, PAGES_DIR, SYSTEM_DIR
        root = raw_dir.parent
        index_file, log_file, schema_file, activity_file = (
            INDEX_FILE,
            LOG_FILE,
            SCHEMA_FILE,
            ACTIVITY_FILE,
        )
    else:
        root = context.root
        raw_dir, pages_dir, system_dir = (
            context.raw_dir,
            context.pages_dir,
            context.system_dir,
        )
        index_file, log_file, schema_file, activity_file = (
            context.index_file,
            context.log_file,
            context.schema_file,
            context.activity_file,
        )

    with okf_runtime_operation(root, allow_bootstrap_resume=True) as startup:
        from chronovisor.core.live_layout import (
            pinned_layout_directories,
            read_live_layout_proof,
            write_live_layout_proof,
        )
        from chronovisor.core.reserved_documents import (
            render_pages_index,
            render_pages_log,
        )

        if startup.layout == "legacy":
            for directory in (raw_dir, pages_dir, system_dir):
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                directory.chmod(0o700)
            return
        if startup.layout == "okf_v0_2":
            from chronovisor.core.page_mutation import chronovisor_mutation_lock

            for directory in (raw_dir, pages_dir, system_dir):
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                directory.chmod(0o700)
            with chronovisor_mutation_lock(pages_dir=pages_dir):
                pass
            return

        canonical_paths = (
            root / "raw",
            root / "pages",
            root / "system",
            root / "pages" / "index.md",
            root / "pages" / "log.md",
            root / "system" / "schema.md",
            root / "runtime" / "activity.jsonl",
        )
        configured_paths = (
            raw_dir,
            pages_dir,
            system_dir,
            index_file,
            log_file,
            schema_file,
            activity_file,
        )
        if tuple(path.absolute() for path in configured_paths) != tuple(
            path.absolute() for path in canonical_paths
        ):
            raise ValueError("fresh bootstrap paths are not canonical")

        with pinned_layout_directories(root) as (root_fd, runtime_fd):
            proof = read_live_layout_proof(root, runtime_fd=runtime_fd)
            if proof is not None and proof["state"] == "ready":
                return
            if proof is None:
                write_live_layout_proof(
                    root,
                    state="in-progress",
                    runtime_fd=runtime_fd,
                )

            directory_fds = {
                name: _bootstrap_directory(root_fd, name)
                for name in ("raw", "pages", "system")
            }
            os.fchmod(root_fd, 0o700)
            os.fchmod(runtime_fd, 0o700)
            try:
                expected = (
                    (directory_fds["pages"], "index.md", render_pages_index(())),
                    (directory_fds["pages"], "log.md", render_pages_log()),
                    (directory_fds["system"], "schema.md", SCHEMA_CONTENT.encode()),
                    (runtime_fd, "activity.jsonl", b""),
                )
                for directory_fd, name, raw in expected:
                    _ensure_bootstrap_file(directory_fd, name, raw)
                if not all(
                    _same_directory(root_fd, name, directory_fds[name])
                    for name in ("raw", "pages", "system")
                ) or not _same_directory(root_fd, "runtime", runtime_fd):
                    raise ValueError("bootstrap directory changed during publication")
                write_live_layout_proof(
                    root,
                    state="ready",
                    runtime_fd=runtime_fd,
                )
                if not all(
                    _same_directory(root_fd, name, directory_fds[name])
                    for name in ("raw", "pages", "system")
                ) or not _same_directory(root_fd, "runtime", runtime_fd):
                    write_live_layout_proof(
                        root,
                        state="in-progress",
                        runtime_fd=runtime_fd,
                    )
                    raise ValueError(
                        "bootstrap directory changed during ready publication"
                    )
            finally:
                for directory_fd in directory_fds.values():
                    os.close(directory_fd)


def _bootstrap_directory(root_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        os.mkdir(name, mode=0o700, dir_fd=root_fd)
        os.fsync(root_fd)
    except FileExistsError:
        pass
    descriptor = os.open(name, flags, dir_fd=root_fd)
    os.fchmod(descriptor, 0o700)
    return descriptor


def _ensure_bootstrap_file(directory_fd: int, name: str, raw: bytes) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        atomic_write_bytes_at(directory_fd, name, raw)
        return
    try:
        snapshot = os.fstat(descriptor)
        if (
            not stat.S_ISREG(snapshot.st_mode)
            or snapshot.st_size != len(raw)
            or os.read(descriptor, len(raw) + 1) != raw
        ):
            raise ValueError(f"unsafe partial bootstrap path: {name}")
    finally:
        os.close(descriptor)


def _same_directory(root_fd: int, name: str, directory_fd: int) -> bool:
    try:
        current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError:
        return False
    pinned = os.fstat(directory_fd)
    return (
        stat.S_ISDIR(current.st_mode)
        and current.st_dev == pinned.st_dev
        and current.st_ino == pinned.st_ino
    )


SCHEMA_CONTENT = """\
---
title: Chronovisor Schema
updated: 2026-04-10
status: stable
type: knowledge
---

# Chronovisor Schema

## Page Granularity
1 entity = 1 page (Karpathy convention).

## Naming
- File names: kebab-case.md (English)
- Example: `jt-v10-probability-contexts.md`

## Frontmatter
Minimal canonical fields:
```yaml
---
title: Page Title
updated: YYYY-MM-DD
status: stable
type: knowledge
---
```

## Cross-references
Use relative canonical Markdown links.
- Example: `[JT v10 probability contexts](<jt-v10-probability-contexts.md>)`
- Links are bidirectional (backlinks tracked by read endpoint).

## Update Rules
LLM decides (guidelines only):
- Clear factual correction → overwrite
- State change (version upgrade, etc.) → preserve history and update
- Uncertain → append and let Lint handle it
"""
