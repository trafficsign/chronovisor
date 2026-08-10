"""Repository-pinned OKF v0.2 pages-bundle conformance checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from chronovisor.core.canonical_document import (
    CanonicalDocumentError,
    parse_document,
)

OKF_VERSION = "0.2"
VALID_STATUSES = frozenset({"draft", "stable", "deprecated"})
RESERVED_FILENAMES = frozenset({"index.md", "log.md"})
RECOMMENDED_FIELDS = ("title", "description", "resource", "tags")


@dataclass(frozen=True, slots=True)
class ConformanceIssue:
    severity: Literal["error", "warning"]
    code: str
    path: str
    field: str | None = None


def scan_concept_paths(pages_root: Path) -> tuple[Path, ...]:
    """List concept Markdown files, excluding every reserved index and log."""

    return tuple(
        path
        for path in sorted(pages_root.rglob("*.md"))
        if path.name not in RESERVED_FILENAMES
    )


def validate_concept(
    metadata: Mapping[str, Any], *, path: str = ""
) -> tuple[ConformanceIssue, ...]:
    """Validate the required and recommended OKF v0.2 concept fields."""

    issues: list[ConformanceIssue] = []
    concept_type = metadata.get("type")
    if not isinstance(concept_type, str) or not concept_type.strip():
        issues.append(ConformanceIssue("error", "type_required", path, "type"))
    status = metadata.get("status")
    if "status" in metadata and (
        not isinstance(status, str) or status not in VALID_STATUSES
    ):
        issues.append(ConformanceIssue("error", "status_invalid", path, "status"))
    issues.extend(
        ConformanceIssue("warning", "recommended_field_missing", path, field)
        for field in RECOMMENDED_FIELDS
        if not _non_empty(metadata.get(field))
    )
    return tuple(issues)


def validate_production_concept(
    metadata: Mapping[str, Any], *, path: str = ""
) -> tuple[ConformanceIssue, ...]:
    """Apply Chronovisor's stricter writer contract to one OKF concept."""

    issues = list(validate_concept(metadata, path=path))
    if "status" not in metadata:
        issues.append(ConformanceIssue("error", "status_required", path, "status"))
    return tuple(issues)


def validate_pages_bundle(pages_root: Path) -> tuple[ConformanceIssue, ...]:
    """Validate only a portable pages bundle; system is intentionally separate."""

    issues: list[ConformanceIssue] = []
    root_index = pages_root / "index.md"
    if not root_index.is_file():
        issues.append(ConformanceIssue("error", "root_index_missing", "index.md"))
    else:
        try:
            metadata = parse_document(root_index.read_bytes()).metadata
        except (CanonicalDocumentError, OSError):
            issues.append(ConformanceIssue("error", "document_invalid", "index.md"))
        else:
            if metadata.get("okf_version") != OKF_VERSION:
                issues.append(
                    ConformanceIssue(
                        "error", "okf_version_invalid", "index.md", "okf_version"
                    )
                )

    for concept_path in scan_concept_paths(pages_root):
        relative_path = concept_path.relative_to(pages_root).as_posix()
        try:
            metadata = parse_document(concept_path.read_bytes()).metadata
        except (CanonicalDocumentError, OSError):
            issues.append(ConformanceIssue("error", "document_invalid", relative_path))
            continue
        issues.extend(validate_concept(metadata, path=relative_path))
    return tuple(issues)


def _non_empty(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return bool(value)
    return value is not None
