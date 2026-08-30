"""Pure renderers for the two portable OKF reserved page documents."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from chronovisor.core.canonical_document import (
    CanonicalDocument,
    extract_markdown_links,
    parse_document,
    serialize_document,
    validate_canonical_document,
)
from chronovisor.core.okf_v02 import OKF_VERSION, RESERVED_FILENAMES


@dataclass(frozen=True, slots=True)
class PageIndexEntry:
    relative_path: str
    title: str
    description: str = ""


def stable_page_index_entry(data: bytes, relative_path: str) -> PageIndexEntry | None:
    """Validate one exact page image and select stable concepts for the index."""

    document = validate_canonical_document(
        data,
        namespace="pages",
        path=relative_path,
        require_stable=False,
    )
    if document.metadata.get("status") != "stable":
        return None
    title = document.metadata.get("title")
    description = document.metadata.get("description")
    return PageIndexEntry(
        relative_path=relative_path,
        title=(
            title.strip()
            if isinstance(title, str) and title.strip()
            else PurePosixPath(relative_path).stem
        ),
        description=(
            description.strip()
            if isinstance(description, str) and description.strip()
            else ""
        ),
    )


def render_pages_index(entries: Iterable[PageIndexEntry]) -> bytes:
    """Render the deterministic portable page index, excluding reserved docs."""

    rows: dict[str, str] = {}
    for entry in sorted(entries, key=lambda item: item.relative_path):
        path = PurePosixPath(entry.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".md"
            or path.name in RESERVED_FILENAMES
        ):
            continue
        rows[entry.relative_path] = _render_page_index_row(entry)
    return _render_page_index_rows(rows)


def _render_page_index_row(entry: PageIndexEntry) -> str:
    path = PurePosixPath(entry.relative_path)
    destination = entry.relative_path
    if any(character.isspace() for character in destination):
        destination = f"<{destination}>"
    label = entry.title.strip() or path.stem
    suffix = f" - {entry.description.strip()}" if entry.description.strip() else ""
    return f"- [{_markdown_label(label)}]({destination}){suffix}"


def _render_page_index_rows(rows: dict[str, str]) -> bytes:
    body = "# Chronovisor pages\n"
    if rows:
        body += "\n" + "\n".join(rows[path] for path in sorted(rows)) + "\n"
    return serialize_document(
        CanonicalDocument(
            metadata={"okf_version": OKF_VERSION},
            body=body.encode("utf-8"),
        )
    )


def render_pages_log() -> bytes:
    """Render the activity-free portable migration history projection."""

    return b"# Derived change history\n"


def rebuild_pages_index(pages_dir: Path) -> bytes:
    """Validate the full canonical namespace and publish its stable projection."""

    from chronovisor.core.durable_state import atomic_write_bytes
    from chronovisor.core.index_store import (
        canonical_document_bytes,
        canonical_document_paths,
    )

    entries: list[PageIndexEntry] = []
    for path in canonical_document_paths(
        pages_dir,
        require_stable=False,
        strict=True,
    ):
        relative_path = path.relative_to(pages_dir.resolve(strict=True)).as_posix()
        data = canonical_document_bytes(path, pages_dir)
        if data is None:
            raise ValueError(f"canonical page changed during index rebuild: {path}")
        entry = stable_page_index_entry(data, relative_path)
        if entry is not None:
            entries.append(entry)
    rendered = render_pages_index(entries)
    atomic_write_bytes(
        pages_dir / "index.md",
        rendered,
        backup=False,
        min_free_bytes=0,
    )
    return rendered


def update_pages_index(pages_dir: Path, changed_paths: Iterable[Path]) -> bytes | None:
    """Apply an exact page receipt to the portable index without a corpus scan."""

    from chronovisor.core.durable_state import atomic_write_bytes
    from chronovisor.core.index_store import canonical_document_bytes

    root = pages_dir.resolve(strict=True)
    changes: dict[str, PageIndexEntry | None] = {}
    for supplied in dict.fromkeys(Path(path) for path in changed_paths):
        path = supplied.expanduser().resolve(strict=False)
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            continue
        relative = PurePosixPath(relative_path)
        if relative.name in RESERVED_FILENAMES or relative.suffix != ".md":
            continue
        entry: PageIndexEntry | None = None
        if path.is_file():
            data = canonical_document_bytes(path, root)
            if data is None:
                raise ValueError(f"changed page is not canonical: {path}")
            entry = stable_page_index_entry(data, relative_path)
        changes[relative_path] = entry

    if not changes:
        return None

    index_path = root / "index.md"
    try:
        current = index_path.read_bytes()
        document = parse_document(current)
        if str(document.metadata.get("okf_version")) != OKF_VERSION:
            raise ValueError("pages index OKF version mismatch")
        lines = document.body.decode("utf-8").splitlines()
        if not lines or lines[0] != "# Chronovisor pages":
            raise ValueError("pages index heading mismatch")
        rows: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            destinations = extract_markdown_links(line)
            if not line.startswith("- [") or not destinations:
                raise ValueError("pages index contains an unsupported row")
            destination = destinations[0]
            if destination in rows:
                raise ValueError(f"duplicate pages index destination: {destination}")
            rows[destination] = line
    except (FileNotFoundError, UnicodeDecodeError, ValueError):
        return rebuild_pages_index(root)

    for relative_path, entry in changes.items():
        if entry is None:
            rows.pop(relative_path, None)
        else:
            rows[relative_path] = _render_page_index_row(entry)
    rendered = _render_page_index_rows(rows)
    if rendered != current:
        atomic_write_bytes(
            index_path,
            rendered,
            backup=False,
            min_free_bytes=0,
        )
    return rendered


def _markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("]", "\\]")
