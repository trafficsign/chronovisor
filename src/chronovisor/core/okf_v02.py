"""Repository-pinned OKF v0.2 pages-bundle conformance checks."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from chronovisor.core.canonical_document import (
    CanonicalDocumentError,
    extract_markdown_links,
    parse_document,
)

OKF_VERSION = "0.2"
OKF_SPEC_REPOSITORY = "https://github.com/GoogleCloudPlatform/knowledge-catalog"
OKF_SPEC_REVISION = "374e0bc4c644310ff56cdf9c0fe81eccdec862b0"
OKF_SPEC_PATH = "okf/SPEC.md"
OKF_SPEC_SHA256 = "5a3311d270bebb16d558010e75064f5b75323f284992641732b1c8097511f948"
OKF_SPEC_URL = f"{OKF_SPEC_REPOSITORY}/blob/{OKF_SPEC_REVISION}/{OKF_SPEC_PATH}"
VALID_STATUSES = frozenset({"draft", "stable", "deprecated"})
RESERVED_FILENAMES = frozenset({"index.md", "log.md"})
RECOMMENDED_FIELDS = ("title", "description", "resource", "tags")

_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})[ \t]+(?P<title>.+?)[ \t]*$")
_LIST_RE = re.compile(r"^(?P<indent>[ \t]*)(?:[-*+])[ \t]+(?P<body>.+)$")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


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
        if path.is_file() and path.name not in RESERVED_FILENAMES
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
        issues.append(ConformanceIssue("warning", "status_invalid", path, "status"))
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

    issues = [
        issue
        for issue in validate_concept(metadata, path=path)
        if issue.code != "status_invalid"
    ]
    status = metadata.get("status")
    if "status" not in metadata:
        issues.append(ConformanceIssue("error", "status_required", path, "status"))
    elif not isinstance(status, str) or status not in VALID_STATUSES:
        issues.append(ConformanceIssue("error", "status_invalid", path, "status"))
    return tuple(issues)


def validate_pages_bundle(pages_root: Path) -> tuple[ConformanceIssue, ...]:
    """Validate only a portable pages bundle; system is intentionally separate."""

    if not pages_root.is_dir():
        return (ConformanceIssue("error", "pages_root_invalid", "."),)

    issues: list[ConformanceIssue] = []
    for reserved_path in sorted(pages_root.rglob("*.md")):
        if not reserved_path.is_file() or reserved_path.name not in RESERVED_FILENAMES:
            continue
        relative_path = reserved_path.relative_to(pages_root).as_posix()
        if reserved_path.name == "index.md":
            issues.extend(
                _validate_index(
                    reserved_path, relative_path, root=relative_path == "index.md"
                )
            )
        else:
            issues.extend(_validate_log(reserved_path, relative_path))

    for concept_path in scan_concept_paths(pages_root):
        relative_path = concept_path.relative_to(pages_root).as_posix()
        try:
            metadata = parse_document(concept_path.read_bytes()).metadata
        except (CanonicalDocumentError, OSError):
            issues.append(ConformanceIssue("error", "document_invalid", relative_path))
            continue
        issues.extend(validate_concept(metadata, path=relative_path))
    return tuple(issues)


def _validate_index(
    path: Path, relative_path: str, *, root: bool
) -> tuple[ConformanceIssue, ...]:
    issues: list[ConformanceIssue] = []
    try:
        raw = path.read_bytes()
    except OSError:
        return (ConformanceIssue("error", "document_invalid", relative_path),)
    body = raw
    if _has_frontmatter(raw):
        try:
            document = parse_document(raw)
        except CanonicalDocumentError:
            return (ConformanceIssue("error", "document_invalid", relative_path),)
        body = document.body
        if not root:
            issues.append(
                ConformanceIssue("error", "index_frontmatter_forbidden", relative_path)
            )
        else:
            for field in sorted(set(document.metadata) - {"okf_version"}):
                issues.append(
                    ConformanceIssue(
                        "warning", "index_frontmatter_key_unknown", relative_path, field
                    )
                )
            if (
                "okf_version" in document.metadata
                and document.metadata["okf_version"] != OKF_VERSION
            ):
                issues.append(
                    ConformanceIssue(
                        "warning", "okf_version_unknown", relative_path, "okf_version"
                    )
                )
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        issues.append(ConformanceIssue("error", "document_invalid", relative_path))
        return tuple(issues)

    has_heading = False
    invalid_entry = False
    relative_entry_lines: list[str] = []
    for line in text.splitlines():
        if _HEADING_RE.fullmatch(line):
            has_heading = True
        elif has_heading and _LIST_RE.fullmatch(line):
            if _has_relative_link(line):
                relative_entry_lines.append(line)
            else:
                invalid_entry = True
    if not has_heading:
        issues.append(ConformanceIssue("error", "index_heading_missing", relative_path))
    if not relative_entry_lines:
        issues.append(ConformanceIssue("error", "index_link_missing", relative_path))
    if invalid_entry:
        issues.append(ConformanceIssue("error", "index_entry_invalid", relative_path))
    if relative_entry_lines and any(
        not re.search(r"\)\s+-\s+\S", line) for line in relative_entry_lines
    ):
        issues.append(
            ConformanceIssue(
                "warning", "index_description_missing", relative_path, "description"
            )
        )
    return tuple(issues)


def _validate_log(path: Path, relative_path: str) -> tuple[ConformanceIssue, ...]:
    issues: list[ConformanceIssue] = []
    try:
        raw = path.read_bytes()
    except OSError:
        return (ConformanceIssue("error", "document_invalid", relative_path),)
    body = raw
    if _has_frontmatter(raw):
        try:
            body = parse_document(raw).body
        except CanonicalDocumentError:
            return (ConformanceIssue("error", "document_invalid", relative_path),)
    try:
        lines = body.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        issues.append(ConformanceIssue("error", "document_invalid", relative_path))
        return tuple(issues)

    nonempty = [line for line in lines if line.strip()]
    if not nonempty or not re.fullmatch(r"#[ \t]+.+", nonempty[0]):
        issues.append(ConformanceIssue("error", "log_heading_missing", relative_path))

    dates: list[date] = []
    in_date_group = False
    for line in lines:
        if not line.strip():
            continue
        heading = _HEADING_RE.fullmatch(line)
        if heading:
            level = len(heading.group("marks"))
            if level == 1:
                if line != nonempty[0]:
                    issues.append(
                        ConformanceIssue(
                            "error", "log_hierarchy_invalid", relative_path
                        )
                    )
                in_date_group = False
            elif level == 2:
                value = heading.group("title")
                try:
                    parsed = date.fromisoformat(value)
                except ValueError:
                    parsed = None
                if parsed is None or not _DATE_RE.fullmatch(value):
                    issues.append(
                        ConformanceIssue("error", "log_date_invalid", relative_path)
                    )
                    in_date_group = False
                else:
                    dates.append(parsed)
                    in_date_group = True
            else:
                issues.append(
                    ConformanceIssue("error", "log_hierarchy_invalid", relative_path)
                )
                in_date_group = False
            continue
        entry = _LIST_RE.fullmatch(line)
        if entry:
            if entry.group("indent"):
                issues.append(
                    ConformanceIssue("error", "log_hierarchy_invalid", relative_path)
                )
            if not in_date_group:
                issues.append(
                    ConformanceIssue("error", "log_entry_ungrouped", relative_path)
                )
            continue
        issues.append(ConformanceIssue("error", "log_entry_invalid", relative_path))

    if dates != sorted(dates, reverse=True):
        issues.append(
            ConformanceIssue("error", "log_dates_not_newest_first", relative_path)
        )
    return tuple(issues)


def _has_frontmatter(data: bytes) -> bool:
    return data.startswith((b"---\n", b"---\r\n"))


def _has_relative_link(line: str) -> bool:
    for target in extract_markdown_links(line):
        target = target.removeprefix("<").removesuffix(">")
        try:
            parsed = urlsplit(target)
        except ValueError:
            continue
        if (
            not parsed.scheme
            and not parsed.netloc
            and parsed.path
            and not parsed.path.startswith("/")
        ):
            return True
    return False


def _non_empty(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return bool(value)
    return value is not None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an OKF v0.2 pages bundle")
    parser.add_argument("pages_root", type=Path)
    args = parser.parse_args(argv)
    issues = sorted(
        validate_pages_bundle(args.pages_root),
        key=lambda issue: (issue.severity, issue.code, issue.path, issue.field or ""),
    )
    for issue in issues:
        print(f"{issue.severity}\t{issue.code}\t{issue.path}\t{issue.field or ''}")
    return int(any(issue.severity == "error" for issue in issues))


if __name__ == "__main__":
    raise SystemExit(main())
