"""Pure preparation of legacy pages for the one-shot OKF v0.2 migration."""

from __future__ import annotations

import hashlib
import posixpath
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from chronovisor.core.canonical_document import (
    CanonicalDocument,
    CanonicalDocumentError,
    parse_document,
    serialize_document,
)
from chronovisor.core.canonical_json import (
    canonical_json_sha256_stringifying_strict,
    canonical_json_stringifying_strict,
)
from chronovisor.core.frontmatter import parse as parse_legacy_frontmatter
from chronovisor.core.link_fix import (
    WIKI_LINK_RE,
    normalize_link_target,
    position_in_spans,
    protected_spans,
)
from chronovisor.core.okf_v02 import validate_concept

Namespace = Literal["pages", "system"]
_RESERVED_ROOTS = frozenset({"index.md", "log.md", "schema.md"})
_STATUS_MAPPING = {
    "missing": "stable",
    "active": "stable",
    "draft": "draft",
    "stable": "stable",
    "deprecated": "deprecated",
    "archived": "deprecated",
}
_LEGACY_STATUSES = frozenset(_STATUS_MAPPING) - {"missing"}
_ARCHIVE_FIELDS = ("archive_reason", "archive_provenance")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class SourceDocument:
    relative_path: str
    data: bytes
    namespace: Namespace = "pages"


@dataclass(frozen=True, slots=True)
class RawSource:
    relative_path: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    relative_path: str
    data: bytes
    uid: str


@dataclass(frozen=True, slots=True)
class DocumentManifestEntry:
    relative_path: str
    uid: str
    source_sha256: str
    output_sha256: str
    input_status: str
    output_status: str
    resolved_link_count: int
    archive_event_id: str | None


@dataclass(frozen=True, slots=True)
class StatusCohort:
    input_status: str
    output_status: str
    count: int
    uids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnresolvedLink:
    relative_path: str
    uid: str
    target: str


@dataclass(frozen=True, slots=True)
class RawManifestEntry:
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArchiveEvent:
    event_id: str
    uid: str
    relative_path: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class MigrationManifest:
    documents: tuple[DocumentManifestEntry, ...]
    status_cohorts: tuple[StatusCohort, ...]
    unresolved_links: tuple[UnresolvedLink, ...]
    raw_files: tuple[RawManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    manifest: MigrationManifest
    events: tuple[ArchiveEvent, ...]
    converted_documents: tuple[ConvertedDocument, ...]
    reserved_documents: tuple[SourceDocument, ...]
    system_documents: tuple[SourceDocument, ...]


@dataclass(frozen=True, slots=True)
class InvalidStatus:
    relative_path: str
    uid: str
    value: str


class InvalidStatusError(ValueError):
    def __init__(self, invalid_statuses: Iterable[InvalidStatus]) -> None:
        self.invalid_statuses = tuple(invalid_statuses)
        details = ", ".join(
            f"{item.uid}={item.value}" for item in self.invalid_statuses
        )
        super().__init__(f"invalid legacy status: {details}")


def prepare_okf_migration(
    documents: Iterable[SourceDocument],
    *,
    catalog: Mapping[str, str],
    raw_files: Iterable[RawSource] = (),
) -> MigrationPlan:
    """Return a deterministic write plan without reading or writing the filesystem."""

    normalized_catalog = {
        page_id: _relative_path(path) for page_id, path in sorted(catalog.items())
    }
    sources = tuple(
        sorted(
            (
                SourceDocument(
                    relative_path=_relative_path(item.relative_path),
                    data=item.data,
                    namespace=_namespace(item.namespace),
                )
                for item in documents
            ),
            key=lambda item: (item.namespace, item.relative_path),
        )
    )
    _require_unique((item.namespace, item.relative_path) for item in sources)

    reserved = tuple(
        item
        for item in sources
        if item.namespace == "pages" and item.relative_path in _RESERVED_ROOTS
    )
    system = tuple(item for item in sources if item.namespace == "system")
    concepts = tuple(
        item
        for item in sources
        if item.namespace == "pages" and item.relative_path not in _RESERVED_ROOTS
    )

    invalid_statuses: list[InvalidStatus] = []
    converted: list[ConvertedDocument] = []
    manifest_entries: list[DocumentManifestEntry] = []
    unresolved: list[UnresolvedLink] = []
    events: list[ArchiveEvent] = []
    cohorts: defaultdict[str, list[str]] = defaultdict(list)
    seen_uids: set[str] = set()

    for source in concepts:
        relative_path = source.relative_path
        document = _parse_source(source.data)
        metadata = dict(document.metadata)
        uid = _uid(metadata, relative_path)
        if uid in seen_uids:
            raise ValueError(f"duplicate concept uid: {uid}")
        seen_uids.add(uid)

        status_value = metadata.get("status", _MISSING)
        if status_value is _MISSING:
            input_status = "missing"
        elif isinstance(status_value, str) and status_value in _LEGACY_STATUSES:
            input_status = status_value
        else:
            invalid_statuses.append(
                InvalidStatus(relative_path, uid, _display_value(status_value))
            )
            continue
        output_status = _STATUS_MAPPING[input_status]
        cohorts[input_status].append(uid)

        if "type" not in metadata:
            metadata["type"] = "Concept"
        _move_summary(metadata, uid)
        metadata["status"] = output_status
        metadata.pop("chronovisor_status", None)
        event = _archive_event(metadata, uid, relative_path)
        if event is not None:
            events.append(event)

        body, resolved_count, missing = convert_wikilinks(
            document.body, relative_path, uid, normalized_catalog
        )
        unresolved.extend(missing)
        output = serialize_document(CanonicalDocument(metadata=metadata, body=body))
        reparsed = parse_document(output)
        errors = [
            issue
            for issue in validate_concept(reparsed.metadata, path=relative_path)
            if issue.severity == "error"
        ]
        if errors:
            raise ValueError(
                f"canonical concept invalid: {uid}: "
                + ", ".join(issue.code for issue in errors)
            )

        converted.append(ConvertedDocument(relative_path, output, uid))
        manifest_entries.append(
            DocumentManifestEntry(
                relative_path=relative_path,
                uid=uid,
                source_sha256=_sha256(source.data),
                output_sha256=_sha256(output),
                input_status=input_status,
                output_status=output_status,
                resolved_link_count=resolved_count,
                archive_event_id=event.event_id if event is not None else None,
            )
        )

    if invalid_statuses:
        raise InvalidStatusError(invalid_statuses)

    raw_entries = tuple(
        RawManifestEntry(
            _relative_path(raw.relative_path), len(raw.data), _sha256(raw.data)
        )
        for raw in sorted(raw_files, key=lambda item: item.relative_path)
    )
    _require_unique(item.relative_path for item in raw_entries)
    status_cohorts = tuple(
        StatusCohort(
            input_status=input_status,
            output_status=output_status,
            count=len(cohorts[input_status]),
            uids=tuple(sorted(cohorts[input_status])),
        )
        for input_status, output_status in _STATUS_MAPPING.items()
    )
    return MigrationPlan(
        manifest=MigrationManifest(
            documents=tuple(manifest_entries),
            status_cohorts=status_cohorts,
            unresolved_links=tuple(unresolved),
            raw_files=raw_entries,
        ),
        events=tuple(events),
        converted_documents=tuple(converted),
        reserved_documents=reserved,
        system_documents=system,
    )


def require_resolved_links(plan: MigrationPlan) -> None:
    """Reject a prepared plan that would leave prose wikilinks unresolved."""

    if plan.manifest.unresolved_links:
        details = ", ".join(
            f"{item.uid}={item.target!r}" for item in plan.manifest.unresolved_links
        )
        raise ValueError(f"unresolved wikilinks: {details}")


def _parse_source(data: bytes) -> CanonicalDocument:
    try:
        return parse_document(data)
    except CanonicalDocumentError as canonical_error:
        if "duplicate key" in str(canonical_error):
            raise
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise canonical_error from None
        _validate_simple_legacy_frontmatter(text, canonical_error)
        metadata, body = parse_legacy_frontmatter(text)
        if body == text:
            raise canonical_error
        return CanonicalDocument(metadata=metadata, body=body.encode("utf-8"))


def _validate_simple_legacy_frontmatter(
    text: str, canonical_error: CanonicalDocumentError
) -> None:
    if not text.startswith("---\n"):
        raise canonical_error
    closing = text.find("\n---", 4)
    if closing < 0 or text[closing + 4 : closing + 5] not in ("", "\n"):
        raise canonical_error
    keys: set[str] = set()
    for line in text[4:closing].splitlines():
        if not line.strip():
            continue
        if line[0].isspace():
            if line.lstrip().startswith("- "):
                continue
            raise canonical_error
        if ":" not in line:
            continue
        key = line.partition(":")[0].strip()
        if key in keys:
            raise CanonicalDocumentError(f"duplicate key {key!r}")
        keys.add(key)


def _move_summary(metadata: dict[str, object], uid: str) -> None:
    if "summary" not in metadata:
        return
    summary = metadata.pop("summary")
    description = metadata.get("description")
    if description is not None and description != summary:
        raise ValueError(f"summary/description conflict: {uid}")
    metadata["description"] = summary


def _archive_event(
    metadata: dict[str, object], uid: str, relative_path: str
) -> ArchiveEvent | None:
    archived = {
        field: metadata.pop(field) for field in _ARCHIVE_FIELDS if field in metadata
    }
    if not archived:
        return None
    payload = {
        "type": "okf_archive_metadata_migrated",
        "uid": uid,
        "relative_path": relative_path,
        **archived,
    }
    event_id = "okf-archive-" + canonical_json_sha256_stringifying_strict(payload)
    return ArchiveEvent(
        event_id=event_id,
        uid=uid,
        relative_path=relative_path,
        payload_json=canonical_json_stringifying_strict(payload),
    )


def convert_wikilinks(
    body: bytes,
    source_path: str,
    uid: str,
    catalog: Mapping[str, str],
) -> tuple[bytes, int, tuple[UnresolvedLink, ...]]:
    """Convert prose wikilinks using a caller-supplied destination catalog."""

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"concept body is not UTF-8: {uid}") from exc
    spans = protected_spans(text)
    replacements: list[tuple[int, int, str]] = []
    unresolved: list[UnresolvedLink] = []
    for match in WIKI_LINK_RE.finditer(text):
        if position_in_spans(match.start(), spans):
            continue
        inside = match.group(1).strip()
        target = normalize_link_target(inside)
        destination = catalog.get(target)
        if destination is None:
            unresolved.append(UnresolvedLink(source_path, uid, target))
            continue
        target_part, separator, alias = inside.partition("|")
        _, anchor_separator, anchor = target_part.partition("#")
        label = alias.strip() if separator else target_part.strip()
        relative_destination = posixpath.relpath(
            destination, PurePosixPath(source_path).parent.as_posix()
        )
        relative_destination += f"#{anchor.strip()}" if anchor_separator else ""
        rendered_destination = (
            f"<{relative_destination}>"
            if any(character.isspace() for character in relative_destination)
            else relative_destination
        )
        replacements.append(
            (match.start(), match.end(), f"[{label}]({rendered_destination})")
        )
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text.encode("utf-8"), len(replacements), tuple(unresolved)


def _uid(metadata: Mapping[str, object], relative_path: str) -> str:
    uid = metadata.get("uid")
    return uid.strip() if isinstance(uid, str) and uid.strip() else relative_path


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
        or path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or ".." in path.parts
        or path.as_posix() == "."
    ):
        raise ValueError(f"path must be bundle-relative: {value!r}")
    return path.as_posix()


def _namespace(value: str) -> Namespace:
    if value == "pages":
        return "pages"
    if value == "system":
        return "system"
    raise ValueError(f"invalid migration namespace: {value!r}")


def _require_unique(values: Iterable[object]) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate migration input: {value!r}")
        seen.add(value)


def _display_value(value: object) -> str:
    return (
        value if isinstance(value, str) else canonical_json_stringifying_strict(value)
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
