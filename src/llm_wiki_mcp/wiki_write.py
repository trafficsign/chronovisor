"""CAS-safe whole-file writes for generated Wiki pages.

Content-correction mutations use the stricter replacement machinery in
``page_mutation``.  Generated artifacts (hubs, reflections, and the state
register) replace their whole file, so they need a smaller primitive that
still shares the same writer lock and never rolls back bytes owned by another
writer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.frontmatter import patch as patch_frontmatter
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.page_mutation import (
    ACTIVE_CLAIM_FRONTMATTER_FIELDS,
    enforce_correction_constraints,
    wiki_mutation_lock,
)


@dataclass(frozen=True)
class PreparedWikiWrite:
    """A whole-file update bound to the exact observed preimage."""

    path: Path
    page_id: str
    original: bytes | None
    updated: bytes


class WikiWriteError(RuntimeError):
    """Raised internally when a prepared write cannot be applied safely."""


def _read_optional(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def prepare_wiki_write(
    path: Path,
    content: str,
    *,
    page_id: str | None = None,
) -> PreparedWikiWrite:
    """Capture the target's exact bytes before preparing a whole-file write."""

    original = _read_optional(path)
    if original is not None:
        # All callers write Markdown. Refuse to replace an unexpected binary or
        # corrupt file, while still retaining exact bytes for owned rollback.
        original.decode("utf-8")
    return PreparedWikiWrite(
        path=path,
        page_id=page_id or path.stem,
        original=original,
        updated=content.encode("utf-8"),
    )


def _correction_safe_updated(
    item: PreparedWikiWrite,
    current: bytes | None,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Rebase generated output over approved correction state.

    The caller holds ``wiki_mutation_lock``.  Correction markers are copied
    before applying their durable constraints, preventing a whole-file
    generator from either restoring a stale claim or deactivating the rule
    that keeps it removed on later runs.
    """

    if current is None:
        return item.updated, []
    current_text = current.decode("utf-8")
    candidate_text = item.updated.decode("utf-8")
    current_meta, _current_body = parse_frontmatter(current_text)
    candidate_meta, _candidate_body = parse_frontmatter(candidate_text)
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
        candidate_text = patch_frontmatter(candidate_text, updates)

    constrained, applied = enforce_correction_constraints(
        item.page_id,
        current_text,
        candidate_text,
    )
    return constrained.encode("utf-8"), applied


def _rollback_owned_write_locked(item: PreparedWikiWrite) -> bool:
    """Restore/delete only when the path still contains our exact output."""

    try:
        if _read_optional(item.path) != item.updated:
            return False
        if item.original is None:
            item.path.unlink()
            return _read_optional(item.path) is None
        atomic_write(item.path, item.original.decode("utf-8"))
        return _read_optional(item.path) == item.original
    except (OSError, UnicodeDecodeError):
        return False


def apply_wiki_writes(items: Iterable[PreparedWikiWrite]) -> dict[str, Any]:
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
    if not plans:
        return {"status": "unchanged", "paths": [], "rolled_back": {}}

    written: list[PreparedWikiWrite] = []
    enforced: dict[str, list[dict[str, Any]]] = {}
    try:
        with wiki_mutation_lock():
            try:
                for item in plans:
                    # The comparison sits inside the shared lock immediately
                    # beside the replace. A stale snapshot can never overwrite
                    # a correction, ingest, or another generated artifact.
                    current = _read_optional(item.path)
                    effective_updated, applied = _correction_safe_updated(item, current)
                    effective_item = replace(item, updated=effective_updated)
                    if applied:
                        enforced[item.page_id] = applied
                    if current == effective_item.updated:
                        # A concurrent run may already have installed this
                        # exact deterministic artifact. Treat it as idempotent
                        # success, never as a reason to rewrite the file.
                        continue
                    if current != item.original:
                        raise WikiWriteError(f"target changed before apply: {item.path}")
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
                        raise WikiWriteError(f"post-write verification failed: {item.path}")
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
