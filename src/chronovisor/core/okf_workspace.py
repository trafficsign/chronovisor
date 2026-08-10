"""Offline, non-destructive staging for the one-shot OKF v0.2 migration."""

from __future__ import annotations

import hashlib
import re
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from chronovisor.core.canonical_document import (
    CanonicalDocument,
    CanonicalDocumentError,
    parse_document,
    serialize_document,
)
from chronovisor.core.canonical_json import (
    canonical_json_line_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.durable_state import (
    atomic_write_bytes,
    fsync_directory,
    okf_writer_lock,
)
from chronovisor.core.okf_prepare import (
    MigrationPlan,
    Namespace,
    RawSource,
    SourceDocument,
    convert_wikilinks,
    prepare_okf_migration,
    require_resolved_links,
)
from chronovisor.core.okf_v02 import (
    OKF_VERSION,
    ConformanceIssue,
    validate_pages_bundle,
)

SCHEMA_VERSION = 1
MANIFEST_SCHEMA = "chronovisor.okf-migration-manifest.v1"
JOURNAL_SCHEMA = "chronovisor.okf-migration-journal.v1"
SENTINEL_SCHEMA = "chronovisor.okf-restart-refusal.v1"
RESTART_REFUSAL_FILENAME = "restart-refusal.json"

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_ROOT_RESERVED = ("index.md", "log.md", "schema.md")
_MISSING = object()
_SYSTEM_STATUS_MAPPING = {
    "missing": "stable",
    "active": "stable",
    "draft": "draft",
    "stable": "stable",
    "deprecated": "deprecated",
    "archived": "deprecated",
}


@dataclass(frozen=True, slots=True)
class _ConvertedSystemDocument:
    relative_path: str
    data: bytes
    input_status: str
    output_status: str
    identity_source: str
    identity_sha256: str
    resolved_link_count: int


def prepare_okf_workspace(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
) -> Path:
    """Build and validate an offline staging workspace without changing live roots."""

    with okf_writer_lock(
        source_root,
        exclusive=True,
        allow_create=not (runtime_root / "migrations").exists(),
    ):
        return _prepare_okf_workspace_locked(source_root, runtime_root, run_id)


def _prepare_okf_workspace_locked(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
) -> Path:

    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a safe single path component")
    source_root = _existing_directory(source_root, "source root")
    runtime_root = _existing_directory(runtime_root, "runtime root")
    pages_root = _existing_directory(source_root / "pages", "pages root")
    system_root = _existing_directory(source_root / "system", "system root")
    raw_root = _existing_directory(source_root / "raw", "raw root")
    devices = {
        _device_id(path)
        for path in (source_root, pages_root, system_root, runtime_root)
    }
    if len(devices) != 1:
        raise ValueError("source and runtime roots must be on the same volume")

    reserved_sources = _reserved_sources(source_root)
    page_sources = _document_sources(pages_root, namespace="pages")
    system_sources = _document_sources(system_root, namespace="system")
    raw_sources = _raw_sources(raw_root)
    page_catalog = _catalog(page_sources)
    plan = prepare_okf_migration(
        (*reserved_sources, *page_sources, *system_sources),
        catalog=page_catalog,
        raw_files=raw_sources,
    )
    require_resolved_links(plan)
    reserved_outputs = _reserved_outputs(plan)
    system_inputs = _system_inputs(plan)
    system_outputs = _convert_system_documents(system_inputs, page_catalog)

    workspace = runtime_root / "migrations" / run_id
    journal_path = workspace / "journal.json"
    sentinel_path = workspace / RESTART_REFUSAL_FILENAME
    if journal_path.exists() or sentinel_path.exists():
        raise FileExistsError(f"migration workspace is already gated: {run_id}")
    _reject_symlinks(workspace)
    staging = workspace / "staging"
    staging_pages = staging / "pages"
    staging_system = staging / "system"
    for directory in (
        runtime_root / "migrations",
        workspace,
        staging,
        staging_pages,
        staging_system,
    ):
        _mkdir_durable(directory)

    try:
        for converted in plan.converted_documents:
            _write(staging_pages / converted.relative_path, converted.data)
        for relative_path, data in reserved_outputs.items():
            _write(staging_pages / relative_path, data)
        for system_document in system_outputs:
            _write(
                staging_system / system_document.relative_path, system_document.data
            )
        activity = _activity_jsonl(plan)
        _write(staging / "activity.jsonl", activity)
        _require_staged_inventory(
            staging_pages,
            {
                *(item.relative_path for item in plan.converted_documents),
                *reserved_outputs,
            },
            staging_system,
            {item.relative_path for item in system_outputs},
        )

        issues = validate_pages_bundle(staging_pages)
        errors = tuple(issue for issue in issues if issue.severity == "error")
        if errors:
            details = ", ".join(f"{issue.code}:{issue.path}" for issue in errors)
            raise ValueError(f"OKF conformance failed: {details}")
        _require_semantic_roundtrip(staging_pages, staging_system)

        _require_inputs_unchanged(
            source_root,
            pages_root,
            system_root,
            raw_root,
            reserved_sources,
            page_sources,
            system_sources,
            raw_sources,
        )

        manifest = _manifest(
            plan,
            issues,
            activity,
            reserved_outputs,
            system_inputs,
            system_outputs,
            run_id,
        )
        manifest_raw = canonical_json_line_bytes_strict(manifest)
        manifest_path = workspace / "dry-run-manifest.json"
        _write(manifest_path, manifest_raw)
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        _fsync_prepared_tree(runtime_root, workspace)
        _require_inputs_unchanged(
            source_root,
            pages_root,
            system_root,
            raw_root,
            reserved_sources,
            page_sources,
            system_sources,
            raw_sources,
        )
        common_gate = {
            "version": SCHEMA_VERSION,
            "run_id": run_id,
            "state": "prepared",
            "manifest_sha256": manifest_sha256,
        }
        _write(
            journal_path,
            canonical_json_line_bytes_strict(
                {"schema": JOURNAL_SCHEMA, **common_gate}
            ),
        )
        _write(
            sentinel_path,
            canonical_json_line_bytes_strict(
                {"schema": SENTINEL_SCHEMA, **common_gate}
            ),
        )
    except BaseException:
        _clear_prepared_claim(workspace, journal_path, sentinel_path)
        raise
    return workspace


def _existing_directory(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return resolved


def _device_id(path: Path) -> int:
    return path.stat().st_dev


def _reserved_sources(source_root: Path) -> tuple[SourceDocument, ...]:
    sources: list[SourceDocument] = []
    for name in _ROOT_RESERVED:
        path = source_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"reserved root document is missing or unsafe: {name}")
        sources.append(SourceDocument(name, path.read_bytes()))
    return tuple(sources)


def _document_sources(
    root: Path, *, namespace: Namespace
) -> tuple[SourceDocument, ...]:
    sources: list[SourceDocument] = []
    for path in _regular_files(root):
        if path.suffix != ".md":
            raise ValueError(f"non-Markdown canonical document: {path.name}")
        sources.append(
            SourceDocument(
                path.relative_to(root).as_posix(),
                path.read_bytes(),
                namespace=namespace,
            )
        )
    return tuple(sources)


def _raw_sources(root: Path) -> tuple[RawSource, ...]:
    return tuple(
        RawSource(path.relative_to(root).as_posix(), path.read_bytes())
        for path in _regular_files(root)
    )


def _regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in migration input: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported migration input: {path}")
        files.append(path)
    return tuple(files)


def _catalog(sources: tuple[SourceDocument, ...]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for source in sources:
        page_id = PurePosixPath(source.relative_path).stem
        if page_id in catalog:
            raise ValueError(f"duplicate page_id stem: {page_id}")
        catalog[page_id] = source.relative_path
    return catalog


def _reserved_outputs(plan: MigrationPlan) -> dict[str, bytes]:
    index_rows: list[str] = []
    for converted in plan.converted_documents:
        metadata = parse_document(converted.data).metadata
        title = metadata.get("title")
        label = (
            title.strip() if isinstance(title, str) and title.strip() else converted.uid
        )
        destination = converted.relative_path
        if any(character.isspace() for character in destination):
            destination = f"<{destination}>"
        description = metadata.get("description")
        suffix = (
            f" - {description.strip()}"
            if isinstance(description, str) and description.strip()
            else ""
        )
        index_rows.append(f"- [{_markdown_label(label)}]({destination}){suffix}")
    index_body = "# Chronovisor pages\n"
    if index_rows:
        index_body += "\n" + "\n".join(index_rows) + "\n"

    # Detailed migration history already lives in activity.jsonl and the manifest.
    log_body = "# Derived change history\n"
    return {
        "index.md": serialize_document(
            CanonicalDocument(
                metadata={"okf_version": OKF_VERSION},
                body=index_body.encode(),
            )
        ),
        "log.md": log_body.encode(),
    }


def _system_inputs(plan: MigrationPlan) -> tuple[SourceDocument, ...]:
    root_schema = next(
        item for item in plan.reserved_documents if item.relative_path == "schema.md"
    )
    inputs = (
        *plan.system_documents,
        SourceDocument("schema.md", root_schema.data, namespace="system"),
    )
    paths = [item.relative_path for item in inputs]
    if len(set(paths)) != len(paths):
        raise ValueError("root schema conflicts with system/schema.md")
    return tuple(sorted(inputs, key=lambda item: item.relative_path))


def _convert_system_documents(
    sources: tuple[SourceDocument, ...],
    page_catalog: dict[str, str],
) -> tuple[_ConvertedSystemDocument, ...]:
    system_catalog = _catalog(sources)
    overlap = sorted(set(page_catalog).intersection(system_catalog))
    if overlap:
        raise ValueError(f"page/system page_id collision: {', '.join(overlap)}")
    catalog = {
        **system_catalog,
        **{
            page_id: f"../pages/{relative_path}"
            for page_id, relative_path in page_catalog.items()
        },
    }
    converted: list[_ConvertedSystemDocument] = []
    unresolved: list[str] = []
    for source in sources:
        try:
            document = parse_document(source.data)
        except CanonicalDocumentError as exc:
            raise ValueError(
                f"system document is not full-YAML canonical: {source.relative_path}"
            ) from exc
        metadata = dict(document.metadata)
        identity_source, identity_sha256 = _system_identity(
            metadata, source.relative_path
        )
        status = metadata.get("status", _MISSING)
        if status is _MISSING:
            input_status = "missing"
        elif (
            isinstance(status, str)
            and status != "missing"
            and status in _SYSTEM_STATUS_MAPPING
        ):
            input_status = status
        else:
            raise ValueError(
                "invalid system lifecycle status: "
                f"{source.relative_path} (identity_sha256={identity_sha256})"
            )
        output_status = _SYSTEM_STATUS_MAPPING[input_status]
        metadata["status"] = output_status
        metadata.pop("chronovisor_status", None)
        body, resolved_count, missing = convert_wikilinks(
            document.body,
            source.relative_path,
            identity_sha256,
            catalog,
        )
        if missing:
            unresolved.append(
                f"{source.relative_path} (identity_sha256={identity_sha256}, "
                f"count={len(missing)})"
            )
        output = serialize_document(
            CanonicalDocument(metadata=metadata, body=body)
        )
        converted.append(
            _ConvertedSystemDocument(
                source.relative_path,
                output,
                input_status,
                output_status,
                identity_source,
                identity_sha256,
                resolved_count,
            )
        )
    if unresolved:
        raise ValueError(f"unresolved system wikilinks: {', '.join(unresolved)}")
    return tuple(converted)


def _system_identity(
    metadata: dict[str, object], relative_path: str
) -> tuple[str, str]:
    for source in ("uid", "identity"):
        value = metadata.get(source)
        if isinstance(value, str) and value.strip():
            return source, canonical_json_sha256_strict(
                {"source": source, "value": value.strip()}
            )
    return "relative_path", canonical_json_sha256_strict(
        {"source": "relative_path", "value": relative_path}
    )


def _markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("]", "\\]")


def _write(path: Path, data: bytes) -> None:
    _mkdir_durable(path.parent)
    atomic_write_bytes(path, data, backup=False, min_free_bytes=0)


def _mkdir_durable(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"migration output directory is a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"migration output parent is not a directory: {path}")
        return
    if not path.parent.exists():
        _mkdir_durable(path.parent)
    path.mkdir(mode=0o700)
    fsync_directory(path.parent)


def _reject_symlinks(workspace: Path) -> None:
    if workspace.is_symlink():
        raise ValueError(f"migration workspace is a symlink: {workspace}")
    if workspace.exists():
        for path in workspace.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"migration workspace contains a symlink: {path}")


def _require_staged_inventory(
    pages_root: Path,
    expected_pages: set[str],
    system_root: Path,
    expected_system: set[str],
) -> None:
    for root, expected in (
        (pages_root, expected_pages),
        (system_root, expected_system),
    ):
        actual = {
            path.relative_to(root).as_posix() for path in _regular_files(root)
        }
        if actual != expected:
            raise ValueError("staging document inventory does not match migration plan")


def _require_inputs_unchanged(
    source_root: Path,
    pages_root: Path,
    system_root: Path,
    raw_root: Path,
    reserved: tuple[SourceDocument, ...],
    pages: tuple[SourceDocument, ...],
    system: tuple[SourceDocument, ...],
    raw: tuple[RawSource, ...],
) -> None:
    if raw != _raw_sources(raw_root):
        raise ValueError("raw manifest changed during workspace preparation")
    if (
        reserved != _reserved_sources(source_root)
        or pages != _document_sources(pages_root, namespace="pages")
        or system != _document_sources(system_root, namespace="system")
    ):
        raise ValueError("source documents changed during workspace preparation")


def _require_semantic_roundtrip(pages_root: Path, system_root: Path) -> None:
    for root in (pages_root, system_root):
        for path in _regular_files(root):
            if root == pages_root and path.name in {"index.md", "log.md"}:
                continue
            try:
                document = parse_document(path.read_bytes())
                reparsed = parse_document(serialize_document(document))
            except CanonicalDocumentError as exc:
                relative = path.relative_to(root).as_posix()
                raise ValueError(f"semantic round-trip failed: {relative}") from exc
            if reparsed.metadata != document.metadata or reparsed.body != document.body:
                relative = path.relative_to(root).as_posix()
                raise ValueError(f"semantic round-trip changed document: {relative}")


def _activity_jsonl(plan: MigrationPlan) -> bytes:
    return (
        ("\n".join(event.payload_json for event in plan.events) + "\n").encode()
        if plan.events
        else b""
    )


def _manifest(
    plan: MigrationPlan,
    issues: tuple[ConformanceIssue, ...],
    activity: bytes,
    reserved_outputs: dict[str, bytes],
    system_inputs: tuple[SourceDocument, ...],
    system_outputs: tuple[_ConvertedSystemDocument, ...],
    run_id: str,
) -> dict[str, object]:
    system_output_by_path = {item.relative_path: item for item in system_outputs}
    return {
        "schema": MANIFEST_SCHEMA,
        "version": SCHEMA_VERSION,
        "okf_version": OKF_VERSION,
        "run_id": run_id,
        "state": "validated",
        "documents": [asdict(item) for item in plan.manifest.documents],
        "status_cohorts": [asdict(item) for item in plan.manifest.status_cohorts],
        "system_status_cohorts": [
            {
                "input_status": input_status,
                "output_status": output_status,
                "count": len(members),
                "identity_set_sha256": canonical_json_sha256_strict(
                    sorted(item.identity_sha256 for item in members)
                ),
            }
            for input_status, output_status in _SYSTEM_STATUS_MAPPING.items()
            for members in [
                tuple(
                    item
                    for item in system_outputs
                    if item.input_status == input_status
                )
            ]
        ],
        "unresolved_links": [asdict(item) for item in plan.manifest.unresolved_links],
        "raw_files": [asdict(item) for item in plan.manifest.raw_files],
        "reserved_documents": [
            {
                "source_path": item.relative_path,
                "staged_path": (
                    f"system/{item.relative_path}"
                    if item.relative_path == "schema.md"
                    else f"pages/{item.relative_path}"
                ),
                "source_sha256": hashlib.sha256(item.data).hexdigest(),
                "output_sha256": hashlib.sha256(
                    system_output_by_path[item.relative_path].data
                    if item.relative_path == "schema.md"
                    else reserved_outputs[item.relative_path]
                ).hexdigest(),
            }
            for item in plan.reserved_documents
        ],
        "system_documents": [
            {
                "relative_path": item.relative_path,
                "source_scope": (
                    "root" if item.relative_path == "schema.md" else "system"
                ),
                "input_status": system_output_by_path[item.relative_path].input_status,
                "output_status": system_output_by_path[item.relative_path].output_status,
                "identity_source": system_output_by_path[
                    item.relative_path
                ].identity_source,
                "identity_sha256": system_output_by_path[
                    item.relative_path
                ].identity_sha256,
                "resolved_link_count": system_output_by_path[
                    item.relative_path
                ].resolved_link_count,
                "source_sha256": hashlib.sha256(item.data).hexdigest(),
                "output_sha256": hashlib.sha256(
                    system_output_by_path[item.relative_path].data
                ).hexdigest(),
            }
            for item in system_inputs
        ],
        "activity": {
            "event_count": len(plan.events),
            "sha256": hashlib.sha256(activity).hexdigest(),
            "events": [
                {
                    "event_id": event.event_id,
                    "uid": event.uid,
                    "relative_path": event.relative_path,
                    "payload_sha256": hashlib.sha256(
                        event.payload_json.encode()
                    ).hexdigest(),
                }
                for event in plan.events
            ],
        },
        "conformance": [asdict(issue) for issue in issues],
    }


def _fsync_prepared_tree(runtime_root: Path, workspace: Path) -> None:
    directories = sorted(
        (path for path in workspace.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in (*directories, workspace, workspace.parent, runtime_root):
        fsync_directory(directory)


def _clear_prepared_claim(workspace: Path, *paths: Path) -> None:
    for path in paths:
        with suppress(FileNotFoundError):
            path.unlink()
    if workspace.is_dir():
        with suppress(OSError):
            fsync_directory(workspace)
