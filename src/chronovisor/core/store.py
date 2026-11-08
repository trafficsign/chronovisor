"""Canonical page lookup and document validation helpers."""

from pathlib import Path

from chronovisor.core.canonical_document import (
    CanonicalDocumentError,
    Namespace,
    validate_canonical_document,
)
from chronovisor.core.paths import PAGES_DIR

PAGE_RESERVED_FILENAMES = frozenset({"index.md", "log.md", "schema.md"})
SYSTEM_RESERVED_FILENAMES = frozenset({"index.md", "log.md"})


def contained_file(path: Path, root: Path) -> Path | None:
    """Return a regular file only when its real path stays in its namespace."""

    if root.is_symlink():
        return None
    try:
        relative = path.relative_to(root)
        root_resolved = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except OSError, RuntimeError, ValueError:
        return None
    # The final layout is direct: reject leaf and descendant symlinks, including
    # links that currently happen to resolve back inside the namespace.
    if resolved != root_resolved / relative or not resolved.is_file():
        return None
    return resolved


def _read_bytes_stable(path: Path, root: Path, retries: int = 1) -> bytes | None:
    """Read a file and verify (mtime_ns, size) didn't change mid-read.

    Guards against picking up a half-written page (ingest doesn't use
    atomic_write today). On retry exhaustion, returns the last best-effort
    read; callers should treat None as "skip this file".
    """
    last: bytes | None = None
    for _ in range(retries + 1):
        resolved = contained_file(path, root)
        if resolved is None:
            return None
        try:
            st_before = resolved.stat()
            data = resolved.read_bytes()
            st_after = resolved.stat()
        except OSError:
            return None
        if (st_before.st_mtime_ns, st_before.st_size) == (
            st_after.st_mtime_ns,
            st_after.st_size,
        ) and contained_file(path, root) == resolved:
            return data
        last = data
    return last


def canonical_document_path(
    path: Path,
    root: Path,
    *,
    namespace: Namespace,
    reserved_filenames: frozenset[str],
    require_stable: bool = False,
) -> Path | None:
    """Return one contained, non-reserved canonical document."""

    if path.name in reserved_filenames:
        return None
    resolved = contained_file(path, root)
    if resolved is None:
        return None
    data = _read_bytes_stable(resolved, root)
    if data is None:
        return None
    try:
        relative_path = resolved.relative_to(root.resolve(strict=True)).as_posix()
        validate_canonical_document(
            data,
            namespace=namespace,
            path=relative_path,
            require_stable=require_stable,
        )
    except CanonicalDocumentError, OSError, RuntimeError, ValueError:
        return None
    return resolved


def canonical_document_paths(
    pages_dir: Path,
    *,
    system_dir: Path | None = None,
    require_stable: bool = False,
    strict: bool = False,
) -> list[Path]:
    """Enumerate canonical, non-reserved documents in exact namespaces."""

    namespaces: list[tuple[Path, Namespace, frozenset[str], bool]] = [
        (pages_dir, "pages", PAGE_RESERVED_FILENAMES, True)
    ]
    if system_dir is not None:
        namespaces.append((system_dir, "system", SYSTEM_RESERVED_FILENAMES, False))
    paths: list[Path] = []
    for directory, namespace, reserved, recursive in namespaces:
        if not directory.is_dir():
            continue
        try:
            candidates = (
                directory.rglob("*.md") if recursive else directory.glob("*.md")
            )
            for path in sorted(candidates):
                if path.name in reserved:
                    continue
                resolved = canonical_document_path(
                    path,
                    directory,
                    namespace=namespace,
                    reserved_filenames=reserved,
                    require_stable=require_stable,
                )
                if resolved is not None:
                    paths.append(resolved)
                elif strict:
                    raise CanonicalDocumentError(f"invalid canonical document: {path}")
        except (OSError, RuntimeError) as exc:
            if strict:
                raise CanonicalDocumentError(
                    f"cannot enumerate canonical namespace: {directory}"
                ) from exc
            continue
    return paths


def all_pages() -> list[Path]:
    """Return stable, non-reserved Wiki pages (supports subdirectories)."""

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
