"""CAS-safe whole-file writes for generated Wiki pages.

Content-correction mutations use the stricter replacement machinery in
``page_mutation``.  Generated artifacts (hubs, reflections, and the state
register) replace their whole file, so they need a smaller primitive that
still shares the same writer lock and never rolls back bytes owned by another
writer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from chronovisor.core.canonical_document import (
    Namespace,
    patch_document_metadata,
    validate_canonical_document,
)
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.page_mutation import (
    ACTIVE_CLAIM_FRONTMATTER_FIELDS,
    chronovisor_mutation_lock,
    enforce_correction_constraints,
)
from chronovisor.core.store import PAGES_DIR, SYSTEM_DIR


@dataclass(frozen=True)
class PreparedWikiWrite:
    """A whole-file update bound to the exact observed preimage."""

    path: Path
    page_id: str
    original: bytes | None
    updated: bytes
    namespace: Namespace
    source_path: str


class WikiWriteError(RuntimeError):
    """Raised internally when a prepared write cannot be applied safely."""


def _read_optional(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def prepare_page_write(
    path: Path,
    content: str,
    *,
    page_id: str | None = None,
    namespace: Namespace | None = None,
    source_path: str | None = None,
    allowed_targets: set[tuple[Namespace, str]] | None = None,
) -> PreparedWikiWrite:
    """Capture the target's exact bytes before preparing a whole-file write."""

    original = _read_optional(path)
    location = _lexical_canonical_location(path)
    if (namespace is None) != (source_path is None):
        raise WikiWriteError("namespace and source_path must be provided together")
    if namespace is None or source_path is None:
        if location is None:
            raise WikiWriteError(
                "namespace and source_path are required outside canonical roots"
            )
        namespace, source_path = location
    _validate_write_target(path, namespace=namespace, source_path=source_path)
    updated = content.encode("utf-8")
    validate_canonical_document(
        updated,
        namespace=namespace,
        path=source_path,
        require_stable=True,
        allowed_targets=allowed_targets,
    )
    if original is not None:
        validate_canonical_document(
            original,
            namespace=namespace,
            path=source_path,
            require_stable=True,
        )
    return PreparedWikiWrite(
        path=path,
        page_id=page_id or path.stem,
        original=original,
        updated=updated,
        namespace=namespace,
        source_path=source_path,
    )


def _lexical_canonical_location(path: Path) -> tuple[Namespace, str] | None:
    target = path.expanduser().absolute()
    for namespace, root in (("pages", PAGES_DIR), ("system", SYSTEM_DIR)):
        lexical_root = root.expanduser().absolute()
        try:
            return cast(Namespace, namespace), target.relative_to(
                lexical_root
            ).as_posix()
        except ValueError:
            continue
    return None


def _validate_write_target(
    path: Path, *, namespace: Namespace, source_path: str
) -> None:
    location = _lexical_canonical_location(path)
    if location is None:
        return
    expected_namespace, expected_source_path = location
    if (namespace, source_path) != (expected_namespace, expected_source_path):
        raise WikiWriteError("canonical path does not match namespace/source_path")

    root = PAGES_DIR if expected_namespace == "pages" else SYSTEM_DIR
    lexical_root = root.expanduser().absolute()
    target = path.expanduser().absolute()
    resolved_root = lexical_root.resolve(strict=False)
    try:
        target.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise WikiWriteError("canonical target escapes through a symlink") from exc

    cursor = lexical_root
    if cursor.is_symlink():
        raise WikiWriteError("canonical root must not be a symlink")
    for part in target.relative_to(lexical_root).parts:
        cursor /= part
        if cursor.is_symlink():
            raise WikiWriteError("canonical target contains a symlink")


def _correction_safe_updated(
    item: PreparedWikiWrite,
    current: bytes | None,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Rebase generated output over approved correction state.

    The caller holds ``chronovisor_mutation_lock``.  Correction markers are copied
    before applying their durable constraints, preventing a whole-file
    generator from either restoring a stale claim or deactivating the rule
    that keeps it removed on later runs.
    """

    if current is None:
        return item.updated, []
    current_text = current.decode("utf-8")
    candidate_text = item.updated.decode("utf-8")
    current_meta = validate_canonical_document(
        current,
        namespace=item.namespace,
        path=item.source_path,
        require_stable=True,
    ).metadata
    candidate_meta = validate_canonical_document(
        item.updated,
        namespace=item.namespace,
        path=item.source_path,
        require_stable=True,
    ).metadata
    preserved_metadata = {
        key: value for key, value in current_meta.items() if key not in candidate_meta
    }
    if preserved_metadata:
        candidate_text = patch_document_metadata(
            candidate_text.encode("utf-8"), preserved_metadata
        ).decode("utf-8")
        candidate_meta = validate_canonical_document(
            candidate_text.encode("utf-8"),
            namespace=item.namespace,
            path=item.source_path,
            require_stable=True,
        ).metadata
    current_markers = current_meta.get("applied_corrections")
    if isinstance(current_markers, list) and current_markers:
        candidate_markers = candidate_meta.get("applied_corrections")
        marker_values = [
            str(value)
            for value in [
                *(candidate_markers if isinstance(candidate_markers, list) else []),
                *current_markers,
            ]
            if isinstance(value, str) and value
        ]
        updates: dict[str, Any] = {
            "applied_corrections": list(dict.fromkeys(marker_values)),
        }
        # These fields can carry active recall claims and may have been updated
        # by the approved correction alongside the body.  Preserve them while
        # allowing generated lifecycle fields (updated/type/tags) to refresh.
        for field in sorted(ACTIVE_CLAIM_FRONTMATTER_FIELDS):
            if field in current_meta:
                updates[field] = current_meta[field]
        candidate_text = patch_document_metadata(
            candidate_text.encode("utf-8"), updates
        ).decode("utf-8")

    constrained, applied = enforce_correction_constraints(
        item.page_id,
        current_text,
        candidate_text,
    )
    return constrained.encode("utf-8"), applied


def _rollback_owned_write_locked(item: PreparedWikiWrite) -> bool:
    """Restore/delete only when the path still contains our exact output."""

    try:
        _validate_write_target(
            item.path,
            namespace=item.namespace,
            source_path=item.source_path,
        )
        if _read_optional(item.path) != item.updated:
            return False
        if item.original is None:
            item.path.unlink()
            return _read_optional(item.path) is None
        atomic_write(item.path, item.original.decode("utf-8"))
        return _read_optional(item.path) == item.original
    except (OSError, UnicodeDecodeError, WikiWriteError):
        return False


def _global_page_id_conflicts(item: PreparedWikiWrite) -> list[Path]:
    """Return other Wiki files that already own ``item.page_id``.

    Page IDs are filename stems and are global across pages/ and system/.
    The check is only applied to actual Wiki targets, leaving explicit
    temporary/export paths independent from the operator's live Wiki.
    Callers hold ``chronovisor_mutation_lock`` so the check is adjacent to creation.
    """

    target = item.path.expanduser().resolve(strict=False)
    roots = (
        PAGES_DIR.expanduser().resolve(strict=False),
        SYSTEM_DIR.expanduser().resolve(strict=False),
    )
    if not any(target == root or root in target.parents for root in roots):
        return []

    conflicts: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob(f"{item.page_id}.md"):
            resolved = candidate.expanduser().resolve(strict=False)
            if resolved != target:
                conflicts.append(candidate)
    return conflicts


def apply_page_writes(items: Iterable[PreparedWikiWrite]) -> dict[str, Any]:
    """Apply a whole-file batch with lock-time CAS and owned-byte rollback."""

    plans = list(items)
    paths = [str(item.path) for item in plans]
    identities = [str(item.path.resolve(strict=False)) for item in plans]
    if len(identities) != len(set(identities)):
        return {
            "status": "retry",
            "reason": "duplicate_target_path",
            "paths": paths,
            "rolled_back": {},
        }
    page_ids = [item.page_id for item in plans]
    if len(page_ids) != len(set(page_ids)):
        return {
            "status": "retry",
            "reason": "duplicate_page_id",
            "paths": paths,
            "rolled_back": {},
        }
    if not plans:
        return {"status": "unchanged", "paths": [], "rolled_back": {}}

    written: list[PreparedWikiWrite] = []
    enforced: dict[str, list[dict[str, Any]]] = {}
    try:
        with chronovisor_mutation_lock(
            changed_paths=[item.path for item in plans],
        ):
            try:
                for item in plans:
                    # The comparison sits inside the shared lock immediately
                    # beside the replace. A stale snapshot can never overwrite
                    # a correction, ingest, or another generated artifact.
                    _validate_write_target(
                        item.path,
                        namespace=item.namespace,
                        source_path=item.source_path,
                    )
                    conflicts = _global_page_id_conflicts(item)
                    if conflicts:
                        locations = ", ".join(str(path) for path in conflicts)
                        raise WikiWriteError(
                            f"page_id {item.page_id!r} already exists at {locations}"
                        )
                    current = _read_optional(item.path)
                    if current == item.updated:
                        # A concurrent run may already have installed this
                        # exact deterministic artifact. Treat it as idempotent
                        # success, never as a reason to rewrite the file.
                        continue
                    if current != item.original:
                        raise WikiWriteError(
                            f"target changed before apply: {item.path}"
                        )
                    effective_updated, applied = _correction_safe_updated(item, current)
                    effective_item = replace(item, updated=effective_updated)
                    if applied:
                        enforced[item.page_id] = applied
                    if current == effective_item.updated:
                        continue
                    item.path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        atomic_write(item.path, effective_item.updated.decode("utf-8"))
                    except Exception:
                        # A replacement can theoretically land before a later
                        # fsync/adapter failure is raised. Include it in the
                        # rollback set only when its exact bytes are observable.
                        if _read_optional(item.path) == effective_item.updated:
                            written.append(effective_item)
                        raise
                    written.append(effective_item)
                    if _read_optional(item.path) != effective_item.updated:
                        raise WikiWriteError(
                            f"post-write verification failed: {item.path}"
                        )
            except Exception as exc:
                rolled_back = {
                    str(item.path): _rollback_owned_write_locked(item)
                    for item in reversed(written)
                }
                return {
                    "status": "retry",
                    "reason": str(exc),
                    "paths": paths,
                    "rolled_back": rolled_back,
                }
    except OSError as exc:
        return {
            "status": "retry",
            "reason": f"wiki mutation lock failed: {exc}",
            "paths": paths,
            "rolled_back": {},
        }

    return {
        "status": "applied" if written else "unchanged",
        "paths": paths,
        "rolled_back": {},
        "correction_constraints": enforced,
    }
