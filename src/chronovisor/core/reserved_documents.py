"""Pure renderers for the two portable OKF reserved page documents."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from chronovisor.core.canonical_document import (
    CanonicalDocument,
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

    rows: list[str] = []
    for entry in sorted(entries, key=lambda item: item.relative_path):
        path = PurePosixPath(entry.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".md"
            or path.name in RESERVED_FILENAMES
        ):
            continue
        destination = entry.relative_path
        if any(character.isspace() for character in destination):
            destination = f"<{destination}>"
        label = entry.title.strip() or path.stem
        suffix = f" - {entry.description.strip()}" if entry.description.strip() else ""
        rows.append(f"- [{_markdown_label(label)}]({destination}){suffix}")
    body = "# Chronovisor pages\n"
    if rows:
        body += "\n" + "\n".join(rows) + "\n"
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


def _markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("]", "\\]")
