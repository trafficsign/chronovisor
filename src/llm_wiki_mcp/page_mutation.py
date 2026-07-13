"""Rollback-safe exact page mutations shared by autonomous content lanes."""

from __future__ import annotations

import difflib
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.frontmatter import patch as patch_frontmatter
from llm_wiki_mcp.jsonl_write import append_jsonl_durable
from llm_wiki_mcp.link_fix import atomic_write, protected_spans
from llm_wiki_mcp.wiki import PAGES_DIR, SYSTEM_DIR, WIKI_ROOT, find_page


WIKI_MUTATION_LOCK = WIKI_ROOT / "runtime" / "wiki-mutation.lock"
DECISION_AUTHORITY_LOCK = WIKI_ROOT / "runtime" / "decision-authority.lock"
CORRECTION_CONSTRAINT_SCHEMA_VERSION = 1
ACTIVE_CLAIM_FRONTMATTER_FIELDS = frozenset(
    {
        "title",
        "summary",
        "recall_questions",
        "raw_keywords",
        "entities",
    }
)
REVIEW_CONTEXT_CHARS = 1_200
REVIEW_DIFF_CHARS = 30_000
CORRECTABLE_SYSTEM_PAGE_IDS = frozenset(
    {"user-profile", "current-state", "lessons-learned"}
)


class PageMutationError(RuntimeError):
    """Raised when an exact, bounded mutation cannot be prepared safely."""


@dataclass(frozen=True)
class ExactReplacement:
    old_text: str
    new_text: str
    action: str = "replace"


@dataclass(frozen=True)
class PreparedPageMutation:
    page_id: str
    path: Path
    correction_id: str
    original: bytes
    updated: bytes
    original_sha256: str
    updated_sha256: str
    replacements: tuple[ExactReplacement, ...]
    already_applied: bool = False

    def review_payload(self, *, preview_chars: int = 12_000) -> dict[str, Any]:
        before = self.original.decode("utf-8")
        after = self.updated.decode("utf-8")
        replacement_contexts = _replacement_review_contexts(
            before,
            self.replacements,
            context_chars=REVIEW_CONTEXT_CHARS,
            already_applied=self.already_applied,
        )
        unified_diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"{self.page_id}:before:{self.original_sha256}",
                tofile=f"{self.page_id}:after:{self.updated_sha256}",
                n=4,
            )
        )
        bounded_diff = _bounded_text(unified_diff, REVIEW_DIFF_CHARS)
        return {
            "page_id": self.page_id,
            "correction_id": self.correction_id,
            "original_sha256": self.original_sha256,
            "updated_sha256": self.updated_sha256,
            "replacements": [
                {
                    "action": item.action,
                    "old_text": item.old_text,
                    "new_text": item.new_text,
                    "old_text_sha256": _sha256_text(item.old_text),
                    "new_text_sha256": _sha256_text(item.new_text),
                    **replacement_contexts[index],
                }
                for index, item in enumerate(self.replacements)
            ],
            "before_preview": _bounded_text(before, preview_chars),
            "after_preview": _bounded_text(after, preview_chars),
            "unified_diff": bounded_diff,
            "unified_diff_sha256": _sha256_text(bounded_diff),
            "full_unified_diff_sha256": _sha256_text(unified_diff),
            "unified_diff_truncated": bounded_diff != unified_diff,
        }


@contextmanager
def wiki_mutation_lock(path: Path | None = None) -> Iterator[None]:
    """Serialize cooperating Wiki writers around compare-and-replace operations."""

    lock_path = path or WIKI_MUTATION_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    locked = False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def decision_authority_lock(path: Path | None = None) -> Iterator[None]:
    """Serialize adopted-authority updates with authority-bound mutations.

    The local-evaluation artifact writer and recall auto-apply both use this
    lease.  A completed adoption artifact therefore cannot be replaced after
    its authority was revalidated but before the approved Wiki mutation is
    durably committed.
    """

    lock_path = path or DECISION_AUTHORITY_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    locked = False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def find_mutation_page(page_id: str) -> Path | None:
    """Resolve a normal page or one explicitly correctable memory page.

    Operational system files remain outside the autonomous mutation boundary.
    The allowlist intentionally contains only user-memory content that can be
    the source of a recalled factual error.
    """

    path = find_page(page_id)
    if path is not None:
        return path
    if page_id not in CORRECTABLE_SYSTEM_PAGE_IDS:
        return None
    candidate = SYSTEM_DIR / f"{page_id}.md"
    if candidate.is_symlink() or not candidate.is_file():
        return None
    return candidate


def correction_constraints_file() -> Path:
    """Return the registry paired with the currently configured Wiki lock.

    Deriving this path from ``WIKI_MUTATION_LOCK`` keeps isolated test Wikis
    and production on the same boundary without another path global to patch.
    """

    return WIKI_MUTATION_LOCK.parent / "content-correction-constraints.jsonl"


def content_correction_audit_file() -> Path:
    return WIKI_MUTATION_LOCK.parent.parent / "recall" / "content-feedback.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _constraint_row(
    mutation: PreparedPageMutation,
    replacement: ExactReplacement,
) -> dict[str, Any]:
    return {
        "schema_version": CORRECTION_CONSTRAINT_SCHEMA_VERSION,
        "kind": "content_correction_constraint",
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "correction_id": mutation.correction_id,
        "page_id": mutation.page_id,
        "action": replacement.action,
        "old_text": replacement.old_text,
        "new_text": replacement.new_text,
        "old_text_sha256": _sha256_text(replacement.old_text),
        "new_text_sha256": _sha256_text(replacement.new_text),
    }


def _constraint_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("correction_id") or ""),
        str(row.get("page_id") or ""),
        str(row.get("old_text_sha256") or _sha256_text(str(row.get("old_text") or ""))),
        str(row.get("new_text_sha256") or _sha256_text(str(row.get("new_text") or ""))),
    )


def _persist_constraints_locked(mutations: Iterable[PreparedPageMutation]) -> None:
    """Fsync constraints before a page marker can become visible.

    Registry rows written for a subsequently rolled-back page stay inert:
    readers activate them only when the page contains the matching
    ``applied_corrections`` marker. This ordering closes the crash window
    between replacing a page and writing the normal correction audit row.
    """

    path = correction_constraints_file()
    existing = {
        _constraint_identity(row)
        for row in _read_jsonl(path)
        if row.get("kind") == "content_correction_constraint"
    }
    pending: list[dict[str, Any]] = []
    for mutation in mutations:
        for replacement in mutation.replacements:
            row = _constraint_row(mutation, replacement)
            identity = _constraint_identity(row)
            if identity in existing:
                continue
            existing.add(identity)
            pending.append(row)
    if not pending:
        return
    append_jsonl_durable(path, pending, sort_keys=True)


def _audit_constraint_rows() -> list[dict[str, Any]]:
    """Project legacy/apply-audit patches into the constraint row shape."""

    rows: list[dict[str, Any]] = []
    for audit in _read_jsonl(content_correction_audit_file()):
        correction_id = str(audit.get("correction_id") or "")
        patches = audit.get("patches")
        if not correction_id or not isinstance(patches, list):
            continue
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            page_id = str(patch.get("page_id") or "")
            old_text = patch.get("old_text")
            new_text = patch.get("new_text")
            if not page_id or not isinstance(old_text, str) or not old_text:
                continue
            if not isinstance(new_text, str):
                continue
            rows.append(
                {
                    "schema_version": CORRECTION_CONSTRAINT_SCHEMA_VERSION,
                    "kind": "content_correction_constraint",
                    "correction_id": correction_id,
                    "page_id": page_id,
                    "action": str(patch.get("action") or "replace"),
                    "old_text": old_text,
                    "new_text": new_text,
                    "old_text_sha256": _sha256_text(old_text),
                    "new_text_sha256": _sha256_text(new_text),
                }
            )
    return rows


def _correction_markers(page_text: str) -> set[str]:
    meta, _body = parse_frontmatter(page_text)
    raw_markers = meta.get("applied_corrections")
    if not isinstance(raw_markers, list):
        return set()
    return {str(value) for value in raw_markers if isinstance(value, str) and value}


def _constraint_row_is_valid(row: dict[str, Any]) -> bool:
    return (
        row.get("kind") == "content_correction_constraint"
        and isinstance(row.get("correction_id"), str)
        and bool(row.get("correction_id"))
        and isinstance(row.get("page_id"), str)
        and bool(row.get("page_id"))
        and isinstance(row.get("old_text"), str)
        and bool(row.get("old_text"))
        and isinstance(row.get("new_text"), str)
        and str(row.get("action") or "replace") in {"replace", "retract", "supersede"}
    )


def active_global_correction_constraints(
    *,
    current_page_id: str = "",
    current_page_text: str = "",
) -> tuple[dict[str, Any], ...]:
    """Return every globally active exact-claim tombstone.

    Registry rows are written before page bytes, so a row becomes active only
    after either the source page carries its durable correction marker or the
    post-apply audit exists. Once active, the exact stale literal is forbidden
    in *all* generated page bodies, including a replay that chooses a different
    slug. This closes the alternate-page resurrection path while leaving an
    uncommitted/torn registry row inert.
    """

    registry_rows = [
        row
        for row in _read_jsonl(correction_constraints_file())
        if _constraint_row_is_valid(row)
    ]
    audit_rows = [
        row for row in _audit_constraint_rows() if _constraint_row_is_valid(row)
    ]
    active_ids = {str(row["correction_id"]) for row in audit_rows}
    if current_page_id and current_page_text:
        current_markers = _correction_markers(current_page_text)
        active_ids.update(
            str(row["correction_id"])
            for row in registry_rows
            if str(row["page_id"]) == current_page_id
            and str(row["correction_id"]) in current_markers
        )

    source_markers: dict[str, set[str]] = {}
    for row in registry_rows:
        correction_id = str(row["correction_id"])
        if correction_id in active_ids:
            continue
        source_page_id = str(row["page_id"])
        if source_page_id not in source_markers:
            source_path = find_mutation_page(source_page_id)
            if source_path is None:
                source_markers[source_page_id] = set()
            else:
                try:
                    source_markers[source_page_id] = _correction_markers(
                        source_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError):
                    source_markers[source_page_id] = set()
        if correction_id in source_markers[source_page_id]:
            active_ids.add(correction_id)

    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in [*registry_rows, *audit_rows]:
        if str(row["correction_id"]) not in active_ids:
            continue
        identity = _constraint_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        found.append(dict(row))
    return tuple(found)


def active_correction_constraints(
    page_id: str,
    page_text: str,
) -> tuple[dict[str, Any], ...]:
    """Return correction constraints activated by this page's markers."""

    markers = _correction_markers(page_text)
    if not markers:
        return ()
    return tuple(
        row
        for row in active_global_correction_constraints(
            current_page_id=page_id,
            current_page_text=page_text,
        )
        if str(row.get("page_id") or "") == page_id
        and str(row.get("correction_id") or "") in markers
    )


def enforce_correction_constraints(
    page_id: str,
    current_text: str,
    candidate_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Canonicalize stale generated text without undoing applied corrections."""

    constrained = candidate_text
    applied: list[dict[str, Any]] = []
    for row in active_global_correction_constraints(
        current_page_id=page_id,
        current_page_text=current_text,
    ):
        old_text = str(row["old_text"])
        count = constrained.count(old_text)
        if count == 0:
            continue
        new_text = str(row["new_text"])
        constrained = constrained.replace(old_text, new_text)
        if old_text in constrained:
            raise PageMutationError(
                f"correction constraint failed to remove stale claim: {page_id}"
            )
        applied.append(
            {
                "correction_id": str(row.get("correction_id") or ""),
                "page_id": page_id,
                "action": str(row.get("action") or "replace"),
                "replacements": count,
                "old_text_sha256": str(row.get("old_text_sha256") or ""),
                "new_text_sha256": str(row.get("new_text_sha256") or ""),
            }
        )
    if applied:
        meta, _body = parse_frontmatter(constrained)
        existing_markers = meta.get("applied_corrections")
        markers = (
            [
                str(value)
                for value in existing_markers
                if isinstance(value, str) and value
            ]
            if isinstance(existing_markers, list)
            else []
        )
        markers = list(
            dict.fromkeys([*markers, *(str(row["correction_id"]) for row in applied)])
        )
        constrained = patch_frontmatter(constrained, {"applied_corrections": markers})
    return constrained, applied


def _bounded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, (limit - 80) // 2)
    return text[:half] + "\n\n[... bounded preview ...]\n\n" + text[-half:]


def _context_window(
    text: str, start: int, end: int, *, context_chars: int
) -> dict[str, Any]:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    return {
        "body_start": start,
        "body_end": end,
        "context_start": left,
        "context_end": right,
        "context": text[left:right],
        "prefix_truncated": left > 0,
        "suffix_truncated": right < len(text),
    }


def _replacement_diff_hunk(
    replacement: ExactReplacement,
    *,
    start: int,
    max_chars: int = 8_000,
) -> str:
    old_lines = replacement.old_text.splitlines(keepends=True)
    new_lines = replacement.new_text.splitlines(keepends=True)
    header = (
        f"@@ body-char -{start},{len(replacement.old_text)} "
        f"+{start},{len(replacement.new_text)} @@\n"
    )
    diff = header
    diff += "".join(f"-{line}" for line in old_lines)
    if replacement.old_text and not replacement.old_text.endswith("\n"):
        diff += "\n"
    diff += "".join(f"+{line}" for line in new_lines)
    if replacement.new_text and not replacement.new_text.endswith("\n"):
        diff += "\n"
    return _bounded_text(diff, max_chars)


def _replacement_review_contexts(
    text: str,
    replacements: tuple[ExactReplacement, ...],
    *,
    context_chars: int,
    already_applied: bool,
) -> list[dict[str, Any]]:
    """Describe every sequential replacement even when the page preview omits it."""

    _meta, working = parse_frontmatter(text)
    contexts: list[dict[str, Any]] = []
    for replacement in replacements:
        start = working.find(replacement.old_text)
        if start < 0:
            if not already_applied:
                raise PageMutationError(
                    "prepared replacement is missing from review preimage"
                )
            new_start = (
                working.find(replacement.new_text) if replacement.new_text else 0
            )
            new_start = max(0, new_start)
            new_end = new_start + len(replacement.new_text)
            contexts.append(
                {
                    "preimage_available": False,
                    "before_context": None,
                    "after_context": _context_window(
                        working,
                        new_start,
                        new_end,
                        context_chars=context_chars,
                    ),
                    "unified_diff_hunk": _replacement_diff_hunk(
                        replacement,
                        start=new_start,
                    ),
                }
            )
            continue
        end = start + len(replacement.old_text)
        before = _context_window(working, start, end, context_chars=context_chars)
        updated = working[:start] + replacement.new_text + working[end:]
        after_end = start + len(replacement.new_text)
        after = _context_window(updated, start, after_end, context_chars=context_chars)
        contexts.append(
            {
                "preimage_available": True,
                "before_context": before,
                "after_context": after,
                "unified_diff_hunk": _replacement_diff_hunk(
                    replacement,
                    start=start,
                ),
            }
        )
        working = updated
    return contexts


def _frontmatter_value_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield item


def _active_frontmatter_occurrences(meta: dict[str, Any], text: str) -> list[str]:
    matches: list[str] = []
    for field in sorted(ACTIVE_CLAIM_FRONTMATTER_FIELDS):
        if any(text in value for value in _frontmatter_value_strings(meta.get(field))):
            matches.append(field)
    return matches


def _validate_replacement_postconditions(
    meta: dict[str, Any],
    body: str,
    replacements: Iterable[ExactReplacement],
) -> None:
    for replacement in replacements:
        if replacement.old_text in body:
            raise PageMutationError("old claim remains active in page body")
        active_fields = _active_frontmatter_occurrences(meta, replacement.old_text)
        if active_fields:
            raise PageMutationError(
                "old claim remains active in frontmatter fields: "
                + ", ".join(active_fields)
            )
        if replacement.new_text and replacement.new_text not in body:
            raise PageMutationError("new claim is missing from page body")


def _body_prefix(text: str, body: str) -> str:
    if not body:
        return text
    if not text.endswith(body):
        raise PageMutationError("frontmatter parser did not return a body suffix")
    return text[: len(text) - len(body)]


def _overlaps_protected_span(text: str, start: int, end: int) -> bool:
    return any(
        start < protected_end and end > protected_start
        for protected_start, protected_end in protected_spans(text)
    )


def _normalize_replacements(
    values: Iterable[ExactReplacement | dict[str, Any]],
) -> tuple[ExactReplacement, ...]:
    normalized: list[ExactReplacement] = []
    for value in values:
        if isinstance(value, ExactReplacement):
            item = value
        elif isinstance(value, dict):
            item = ExactReplacement(
                old_text=str(value.get("old_text") or ""),
                new_text=str(value.get("new_text") or ""),
                action=str(value.get("action") or "replace"),
            )
        else:
            raise PageMutationError("replacement must be an ExactReplacement or object")
        if item.action not in {"replace", "retract", "supersede"}:
            raise PageMutationError(f"unsupported correction action: {item.action!r}")
        if not item.old_text.strip():
            raise PageMutationError("old_text must be non-empty")
        if len(item.old_text) > 20_000 or len(item.new_text) > 20_000:
            raise PageMutationError("replacement exceeds the 20,000 character bound")
        if item.action != "retract" and not item.new_text.strip():
            raise PageMutationError(f"{item.action} requires non-empty new_text")
        if item.new_text and item.old_text in item.new_text:
            raise PageMutationError("new_text must not retain the exact old claim")
        normalized.append(item)
    if not normalized:
        raise PageMutationError("at least one replacement is required")
    if len(normalized) > 8:
        raise PageMutationError("a page correction is limited to 8 replacements")
    return tuple(normalized)


def prepare_page_mutation(
    page_id: str,
    replacements: Iterable[ExactReplacement | dict[str, Any]],
    *,
    correction_id: str,
    summary: str | None = None,
    recall_questions: list[str] | None = None,
) -> PreparedPageMutation:
    """Prepare an exact body mutation without touching the filesystem."""

    if not correction_id.strip():
        raise PageMutationError("correction_id is required")
    path = find_mutation_page(page_id)
    if path is None:
        raise PageMutationError(f"page not found: {page_id}")
    normal_page = False
    try:
        path.resolve().relative_to(PAGES_DIR.resolve())
        normal_page = True
    except ValueError:
        pass
    expected_system_path = SYSTEM_DIR / f"{page_id}.md"
    allowed_system_page = (
        page_id in CORRECTABLE_SYSTEM_PAGE_IDS
        and not path.is_symlink()
        and path.resolve() == expected_system_path.resolve()
    )
    if not normal_page and not allowed_system_page:
        raise PageMutationError("target page escapes the correctable page boundary")
    original = path.read_bytes()
    try:
        original_text = original.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PageMutationError(f"page is not UTF-8: {page_id}") from exc
    meta, body = parse_frontmatter(original_text)
    if meta.get("status") in {"deprecated", "archived"}:
        raise PageMutationError(f"target page is not active: {page_id}")

    items = _normalize_replacements(replacements)
    applied = meta.get("applied_corrections")
    applied_ids = [str(value) for value in applied] if isinstance(applied, list) else []
    if correction_id in applied_ids:
        try:
            _validate_replacement_postconditions(meta, body, items)
        except PageMutationError as exc:
            raise PageMutationError(
                "correction marker exists but postconditions do not hold"
            ) from exc
        digest = _sha256_bytes(original)
        return PreparedPageMutation(
            page_id=page_id,
            path=path,
            correction_id=correction_id,
            original=original,
            updated=original,
            original_sha256=digest,
            updated_sha256=digest,
            replacements=items,
            already_applied=True,
        )

    updated_body = body
    for item in items:
        count = updated_body.count(item.old_text)
        if count != 1:
            raise PageMutationError(
                f"old_text must occur exactly once in {page_id}; found {count}"
            )
        start = updated_body.index(item.old_text)
        end = start + len(item.old_text)
        if _overlaps_protected_span(updated_body, start, end):
            raise PageMutationError(
                "refusing to mutate text inside a protected code span"
            )
        updated_body = updated_body[:start] + item.new_text + updated_body[end:]

    updated_text = _body_prefix(original_text, body) + updated_body
    metadata_updates: dict[str, Any] = {
        "updated": date.today().isoformat(),
        # These identifiers activate durable replay constraints. Dropping an
        # older marker would eventually permit an old raw capture to restore
        # its retracted claim, so the ledger is append-only and deduplicated.
        "applied_corrections": list(dict.fromkeys([*applied_ids, correction_id])),
    }
    if summary is not None:
        clean_summary = summary.strip()
        if not clean_summary or len(clean_summary) > 1_200 or "\n" in clean_summary:
            raise PageMutationError(
                "summary must be a non-empty single line up to 1,200 chars"
            )
        metadata_updates["summary"] = clean_summary
    if recall_questions is not None:
        if not recall_questions or len(recall_questions) > 8:
            raise PageMutationError("recall_questions must contain 1 to 8 questions")
        clean_questions = [question.strip() for question in recall_questions]
        if any(
            not question
            or len(question) > 400
            or any(ch in question for ch in "[],\n\r")
            for question in clean_questions
        ):
            raise PageMutationError("recall_questions contain an unsafe value")
        metadata_updates["recall_questions"] = clean_questions
    updated_text = patch_frontmatter(updated_text, metadata_updates)
    updated_meta, verified_body = parse_frontmatter(updated_text)
    _validate_replacement_postconditions(updated_meta, verified_body, items)
    updated = updated_text.encode("utf-8")
    if updated == original:
        raise PageMutationError("prepared correction produced no change")

    return PreparedPageMutation(
        page_id=page_id,
        path=path,
        correction_id=correction_id,
        original=original,
        updated=updated,
        original_sha256=_sha256_bytes(original),
        updated_sha256=_sha256_bytes(updated),
        replacements=items,
    )


def _rollback_owned_write_locked(mutation: PreparedPageMutation) -> bool:
    """Restore an owned preimage while the caller holds ``wiki_mutation_lock``."""

    tmp: Path | None = None
    try:
        if mutation.path.read_bytes() != mutation.updated:
            return False
        tmp = mutation.path.with_name(
            f".{mutation.path.name}.{os.getpid()}.rollback.tmp"
        )
        with tmp.open("wb") as handle:
            handle.write(mutation.original)
            handle.flush()
            os.fsync(handle.fileno())
        # The owned-byte comparison is deliberately repeated immediately before
        # replace. Cooperating writers cannot enter this section while the shared
        # mutation lock is held.
        if mutation.path.read_bytes() != mutation.updated:
            return False
        os.replace(tmp, mutation.path)
        return mutation.path.read_bytes() == mutation.original
    except OSError:
        return False
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def _rollback_owned_write(mutation: PreparedPageMutation) -> bool:
    """Restore the preimage only while the page still contains our exact bytes."""

    try:
        with wiki_mutation_lock():
            return _rollback_owned_write_locked(mutation)
    except OSError:
        return False


def rollback_prepared_mutations(
    mutations: Iterable[PreparedPageMutation],
) -> dict[str, Any]:
    """Rollback only bytes still owned by a previously applied mutation."""

    items = [item for item in mutations if not item.already_applied]
    outcomes: dict[str, bool] = {}
    try:
        with wiki_mutation_lock():
            for item in reversed(items):
                try:
                    current = item.path.read_bytes()
                except OSError:
                    outcomes[item.page_id] = False
                    continue
                if current == item.original:
                    outcomes[item.page_id] = True
                    continue
                outcomes[item.page_id] = _rollback_owned_write_locked(item)
    except OSError:
        outcomes.update(
            {item.page_id: False for item in items if item.page_id not in outcomes}
        )
    return {
        "status": "rolled_back" if all(outcomes.values()) else "rollback_incomplete",
        "pages": [item.page_id for item in items],
        "outcomes": outcomes,
    }


def apply_prepared_mutations(
    mutations: Iterable[PreparedPageMutation],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """CAS-apply prepared mutations and rollback only exact owned writes."""

    items = list(mutations)
    if not items:
        return {"status": "rejected", "reason": "no_mutations", "pages": []}
    page_ids = [item.page_id for item in items]
    if len(page_ids) != len(set(page_ids)):
        return {"status": "retry", "reason": "duplicate_target_page", "pages": page_ids}
    actionable = [item for item in items if not item.already_applied]
    if dry_run:
        return {
            "status": "dry_run",
            "pages": page_ids,
            "changed": bool(actionable),
            "hashes": {item.page_id: item.updated_sha256 for item in items},
        }

    written: list[PreparedPageMutation] = []
    try:
        with wiki_mutation_lock():
            try:
                # The registry is durable before any page can expose its
                # applied_corrections marker. Rows for rolled-back pages are
                # harmless because the marker is the activation condition.
                _persist_constraints_locked(items)
                if not actionable:
                    return {"status": "already_applied", "pages": page_ids}
                for item in actionable:
                    # CAS belongs immediately next to each replace. This matters for
                    # a multi-page batch: writing page one must not leave a stale
                    # preflight result for page two.
                    if item.path.read_bytes() != item.original:
                        raise PageMutationError(
                            f"page changed before apply: {item.page_id}"
                        )
                    atomic_write(item.path, item.updated.decode("utf-8"))
                    written.append(item)
                    written_bytes = item.path.read_bytes()
                    if _sha256_bytes(written_bytes) != item.updated_sha256:
                        raise PageMutationError(
                            f"post-write hash mismatch: {item.page_id}"
                        )
                    written_meta, written_body = parse_frontmatter(
                        written_bytes.decode("utf-8")
                    )
                    try:
                        _validate_replacement_postconditions(
                            written_meta,
                            written_body,
                            item.replacements,
                        )
                    except PageMutationError as exc:
                        raise PageMutationError(f"{exc}: {item.page_id}") from exc
            except (OSError, UnicodeDecodeError, PageMutationError) as exc:
                rolled_back = {
                    item.page_id: _rollback_owned_write_locked(item)
                    for item in reversed(written)
                }
                return {
                    "status": "retry",
                    "reason": str(exc),
                    "pages": page_ids,
                    "rolled_back": rolled_back,
                }
    except OSError as exc:
        return {
            "status": "retry",
            "reason": f"wiki mutation lock failed: {exc}",
            "pages": page_ids,
            "rolled_back": {},
        }
    return {
        "status": "applied",
        "pages": page_ids,
        "hashes": {item.page_id: item.updated_sha256 for item in items},
    }
