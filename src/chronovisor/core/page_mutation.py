"""Rollback-safe exact page mutations shared by autonomous content lanes."""

from __future__ import annotations

import difflib
import fcntl
import json
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

from chronovisor.core.canonical_document import (
    CanonicalDocument,
    CanonicalDocumentError,
    Namespace,
    parse_document,
    serialize_document,
    validate_canonical_document,
)
from chronovisor.core.canonical_json import (
    canonical_json_sha256_strict as _canonical_json_sha256,
)
from chronovisor.core.hashutil import sha256_bytes as _sha256_bytes
from chronovisor.core.index_store import canonical_document_path_for_id
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.link_fix import atomic_write, protected_spans
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    PAGES_DIR,
    SYSTEM_DIR,
    okf_runtime_operation,
)

CHRONOVISOR_MUTATION_LOCK = CHRONOVISOR_ROOT / "runtime" / "chronovisor-mutation.lock"
DECISION_AUTHORITY_LOCK = CHRONOVISOR_ROOT / "runtime" / "decision-authority.lock"
CORRECTION_CONSTRAINT_SCHEMA_VERSION = 1
MUTATION_EVIDENCE_SCHEMA_VERSION = 1
MUTATION_EVIDENCE_KIND = "content_correction_mutation_evidence"
ACTIVE_CLAIM_FRONTMATTER_FIELDS = frozenset(
    {
        "title",
        "description",
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
_LOCK_STATE = threading.local()


class PageMutationError(RuntimeError):
    """Raised when an exact, bounded mutation cannot be prepared safely."""


def _canonical_location(path: Path) -> tuple[Namespace, str]:
    target = path.expanduser().resolve(strict=False)
    for namespace, root in (("pages", PAGES_DIR), ("system", SYSTEM_DIR)):
        try:
            return cast(Namespace, namespace), target.relative_to(
                root.expanduser().resolve(strict=False)
            ).as_posix()
        except ValueError:
            continue
    raise PageMutationError("target page escapes the canonical page boundary")


def _parse_canonical_text(text: str) -> tuple[dict[str, Any], str]:
    try:
        document = parse_document(text.encode("utf-8"))
        body = document.body.decode("utf-8")
    except (CanonicalDocumentError, UnicodeDecodeError) as exc:
        raise PageMutationError(f"page is not canonical Markdown: {exc}") from exc
    return document.metadata, body


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
    # These fields are deliberately outside the bounded review projection.  The
    # durable constraint receipt stores their full canonical bytes before CAS;
    # review_payload() continues to expose only bounded context/diff material.
    page_uid: str | None = None
    evidence: tuple[dict[str, Any], ...] = ()

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
def _reentrant_exclusive_lock(lock_path: Path) -> Iterator[bool]:
    """Hold one process lock, allowing only same-thread nested entry."""

    lock_path = lock_path.resolve(strict=False)
    process_id = os.getpid()
    state_pid = getattr(_LOCK_STATE, "process_id", None)
    if state_pid != process_id:
        # A fork must never inherit the parent's in-memory ownership claim.
        _LOCK_STATE.process_id = process_id
        _LOCK_STATE.depths = {}
    depths: dict[str, int] = _LOCK_STATE.depths
    identity = os.fspath(lock_path)
    depth = depths.get(identity, 0)
    if depth:
        depths[identity] = depth + 1
        try:
            yield False
        finally:
            remaining = depths[identity] - 1
            if remaining:
                depths[identity] = remaining
            else:
                depths.pop(identity, None)
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    locked = False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
        depths[identity] = 1
        yield True
    finally:
        try:
            depths.pop(identity, None)
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def chronovisor_mutation_lock(
    path: Path | None = None,
    *,
    pages_dir: Path | None = None,
    changed_paths: Iterable[Path] | None = None,
) -> Iterator[None]:
    """Serialize writers; nested same-thread mutations share the outer lease."""

    target_pages = pages_dir or CHRONOVISOR_ROOT / "pages"
    exact_changes = None if changed_paths is None else tuple(changed_paths)
    operation_root = target_pages.parent
    lock_path = path or operation_root / "runtime" / "chronovisor-mutation.lock"
    with okf_runtime_operation(operation_root) as startup:
        with _reentrant_exclusive_lock(lock_path) as outermost:
            completed = False
            try:
                yield
                completed = True
            finally:
                if outermost and startup.layout == "okf_v0_2":
                    try:
                        from chronovisor.core.reserved_documents import (
                            rebuild_pages_index,
                            update_pages_index,
                        )

                        if exact_changes is None:
                            rebuild_pages_index(target_pages)
                        else:
                            update_pages_index(target_pages, exact_changes)
                    except Exception:
                        if completed:
                            raise


@contextmanager
def decision_authority_lock(path: Path | None = None) -> Iterator[None]:
    """Serialize adopted-authority updates with authority-bound mutations.

    The local-evaluation artifact writer and recall auto-apply both use this
    lease.  A completed adoption artifact therefore cannot be replaced after
    its authority was revalidated but before the approved Wiki mutation is
    durably committed.
    """

    with _reentrant_exclusive_lock(path or DECISION_AUTHORITY_LOCK):
        yield


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _metadata_page_uid(metadata: Mapping[str, Any]) -> str | None:
    """Return an existing page UID without inventing one for legacy pages."""

    value = metadata.get("uid")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _body_byte_offset(data: bytes, body: bytes) -> int:
    """Return the byte offset of the parsed body in a canonical document."""

    # ``parse_document`` retains the body suffix byte-for-byte, so subtracting
    # its length avoids matching a repeated body phrase in YAML frontmatter.
    return len(data) - len(body)


def _byte_span(
    data: bytes,
    body: bytes,
    start: int | None,
    end: int | None,
) -> tuple[int | None, int | None]:
    if start is None or end is None:
        return None, None
    prefix = _body_byte_offset(data, body)
    text = body.decode("utf-8")
    return (
        prefix + len(text[:start].encode("utf-8")),
        prefix + len(text[:end].encode("utf-8")),
    )


def _contiguous_origin(values: list[int | None]) -> tuple[int, int] | None:
    if not values or any(value is None for value in values):
        return None
    first = cast(int, values[0])
    if values != list(range(first, first + len(values))):
        return None
    return first, first + len(values)


def _replacement_evidence_records(
    original: bytes,
    updated: bytes,
    replacements: tuple[ExactReplacement, ...],
) -> tuple[dict[str, Any], ...]:
    """Map exact replacements back to immutable pre/post body byte spans.

    The origin/token maps make a later replacement that targets text introduced
    by an earlier replacement explicitly unknown.  Such a span is never guessed
    from a fuzzy search in the final body.
    """

    try:
        original_document = parse_document(original)
        updated_document = parse_document(updated)
        original_body_bytes = original_document.body
        updated_body_bytes = updated_document.body
        original_body = original_body_bytes.decode("utf-8")
        updated_body = updated_body_bytes.decode("utf-8")
    except (CanonicalDocumentError, UnicodeDecodeError):
        return tuple(
            {
                "index": index,
                "action": replacement.action,
                "span_status": "unknown",
                "old_text_sha256": _sha256_text(replacement.old_text),
                "new_text_sha256": _sha256_text(replacement.new_text),
                "old_quote_sha256": _sha256_text(replacement.old_text),
                "new_quote_sha256": _sha256_text(replacement.new_text),
                "old_body_start": None,
                "old_body_end": None,
                "new_body_start": None,
                "new_body_end": None,
                "old_byte_start": None,
                "old_byte_end": None,
                "new_byte_start": None,
                "new_byte_end": None,
            }
            for index, replacement in enumerate(replacements)
        )

    working = original_body
    origins: list[int | None] = list(range(len(original_body)))
    tokens: list[int | None] = [None] * len(original_body)
    records: list[dict[str, Any]] = []
    for index, replacement in enumerate(replacements):
        start = working.find(replacement.old_text)
        count = working.count(replacement.old_text)
        old_end = start + len(replacement.old_text) if start >= 0 else None
        old_origin = (
            _contiguous_origin(origins[start:old_end])
            if start >= 0 and old_end is not None
            else None
        )
        # A phrase repeated in the original body is not a unique source span,
        # even when an earlier replacement happened to remove one occurrence.
        old_known = (
            count == 1
            and original_body.count(replacement.old_text) == 1
            and old_origin is not None
        )
        if not old_known or start < 0 or count != 1 or old_end is None:
            records.append(
                {
                    "index": index,
                    "action": replacement.action,
                    "span_status": "unknown",
                    "old_text_sha256": _sha256_text(replacement.old_text),
                    "new_text_sha256": _sha256_text(replacement.new_text),
                    "old_quote_sha256": _sha256_text(replacement.old_text),
                    "new_quote_sha256": _sha256_text(replacement.new_text),
                    "old_body_start": None,
                    "old_body_end": None,
                    "new_body_start": None,
                    "new_body_end": None,
                    "old_byte_start": None,
                    "old_byte_end": None,
                    "new_byte_start": None,
                    "new_byte_end": None,
                }
            )
            # The prepared mutation should already have rejected this state;
            # stop mapping rather than manufacture coordinates for later edits.
            break

        old_start, old_original_end = cast(tuple[int, int], old_origin)
        origins = origins[:start] + [None] * len(replacement.new_text) + origins[old_end:]
        tokens = tokens[:start] + [index] * len(replacement.new_text) + tokens[old_end:]
        working = working[:start] + replacement.new_text + working[old_end:]
        records.append(
            {
                "index": index,
                "action": replacement.action,
                "span_status": "verified" if old_known else "unknown",
                "old_text_sha256": _sha256_text(replacement.old_text),
                "new_text_sha256": _sha256_text(replacement.new_text),
                "old_quote_sha256": _sha256_text(replacement.old_text),
                "new_quote_sha256": _sha256_text(replacement.new_text),
                "old_body_start": old_start if old_known else None,
                "old_body_end": old_original_end if old_known else None,
                "new_body_start": None,
                "new_body_end": None,
                "old_byte_start": (
                    _byte_span(
                        original,
                        original_body_bytes,
                        old_start,
                        old_original_end,
                    )[0]
                    if old_known
                    else None
                ),
                "old_byte_end": (
                    _byte_span(
                        original,
                        original_body_bytes,
                        old_start,
                        old_original_end,
                    )[1]
                    if old_known
                    else None
                ),
                "new_byte_start": None,
                "new_byte_end": None,
            }
        )

    if working != updated_body:
        # A manually constructed/tampered PreparedPageMutation must not receive
        # apparently valid offsets from a different postimage.
        return tuple(
            {
                **record,
                "span_status": "unknown",
                "old_body_start": None,
                "old_body_end": None,
                "new_body_start": None,
                "new_body_end": None,
                "old_byte_start": None,
                "old_byte_end": None,
                "new_byte_start": None,
                "new_byte_end": None,
            }
            for record in records
        )

    for record in records:
        index = int(record["index"])
        replacement = replacements[index]
        if not replacement.new_text:
            continue
        positions = [position for position, token in enumerate(tokens) if token == index]
        contiguous = bool(positions) and positions == list(
            range(positions[0], positions[0] + len(positions))
        )
        if contiguous:
            new_start = positions[0]
            new_end = positions[-1] + 1
            record["new_body_start"] = new_start
            record["new_body_end"] = new_end
            record["new_byte_start"], record["new_byte_end"] = _byte_span(
                updated,
                updated_body_bytes,
                new_start,
                new_end,
            )
        else:
            record["span_status"] = "unknown"
            record["new_body_start"] = None
            record["new_body_end"] = None
            record["new_byte_start"] = None
            record["new_byte_end"] = None
            record["old_body_start"] = None
            record["old_body_end"] = None
            record["old_byte_start"] = None
            record["old_byte_end"] = None
    return tuple(records)


def mutation_evidence_payload(mutation: PreparedPageMutation) -> dict[str, Any]:
    """Build the pre-apply canonical mutation evidence envelope."""

    if mutation.already_applied:
        # The visible postimage is not a replacement for the original source.
        # Resolve the receipt persisted before CAS instead of minting a new
        # postimage-to-postimage evidence identity during recovery.
        matches: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(correction_constraints_file()):
            payload = row.get("mutation_evidence")
            if not isinstance(payload, Mapping) or (
                payload.get("page_id") != mutation.page_id
                or payload.get("correction_id") != mutation.correction_id
                or payload.get("postimage_sha256") != mutation.original_sha256
            ):
                continue
            digest = mutation_evidence_sha256(payload)
            if row.get("mutation_evidence_sha256") == digest and mutation_evidence_error(
                payload, expected_postimage=mutation.original
            ) is None:
                matches[digest] = dict(payload)
        if len(matches) != 1:
            raise PageMutationError("already-applied source receipt is missing or ambiguous")
        return next(iter(matches.values()))

    try:
        original_document = parse_document(mutation.original)
        updated_document = parse_document(mutation.updated)
        page_uid = _metadata_page_uid(original_document.metadata)
        updated_uid = _metadata_page_uid(updated_document.metadata)
        if page_uid != updated_uid:
            page_uid = None
    except (CanonicalDocumentError, UnicodeDecodeError):
        page_uid = mutation.page_uid
    records = mutation.evidence or _replacement_evidence_records(
        mutation.original,
        mutation.updated,
        mutation.replacements,
    )
    return {
        "schema_version": MUTATION_EVIDENCE_SCHEMA_VERSION,
        "kind": MUTATION_EVIDENCE_KIND,
        "page_id": mutation.page_id,
        "correction_id": mutation.correction_id,
        # A parsed pre/post UID mismatch is an explicit stale/tampered state;
        # do not let the dataclass fallback hide that mismatch.  The fallback
        # is only for malformed bytes, where the validator will reject the
        # envelope before it can be persisted.
        "page_uid": page_uid,
        "preimage_utf8": mutation.original.decode("utf-8", errors="strict"),
        "postimage_utf8": mutation.updated.decode("utf-8", errors="strict"),
        "preimage_sha256": mutation.original_sha256,
        "postimage_sha256": mutation.updated_sha256,
        "replacements": [dict(record) for record in records],
    }


def mutation_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(dict(evidence))


def mutation_evidence_error(
    evidence: Mapping[str, Any],
    *,
    expected_page_id: str | None = None,
    expected_correction_id: str | None = None,
    expected_preimage: bytes | None = None,
    expected_postimage: bytes | None = None,
    require_verified_spans: bool = True,
) -> str | None:
    """Validate a durable mutation receipt against bytes and recorded spans."""

    if not isinstance(evidence, Mapping):
        return "mutation evidence is missing"
    if (
        evidence.get("schema_version") != MUTATION_EVIDENCE_SCHEMA_VERSION
        or evidence.get("kind") != MUTATION_EVIDENCE_KIND
    ):
        return "mutation evidence schema is invalid"
    page_id = evidence.get("page_id")
    correction_id = evidence.get("correction_id")
    if not isinstance(page_id, str) or not page_id:
        return "mutation evidence page_id is invalid"
    if not isinstance(correction_id, str) or not correction_id:
        return "mutation evidence correction_id is invalid"
    if expected_page_id is not None and page_id != expected_page_id:
        return "mutation evidence page_id mismatch"
    if expected_correction_id is not None and correction_id != expected_correction_id:
        return "mutation evidence correction_id mismatch"
    preimage_text = evidence.get("preimage_utf8")
    postimage_text = evidence.get("postimage_utf8")
    if not isinstance(preimage_text, str) or not isinstance(postimage_text, str):
        return "mutation evidence canonical bytes are missing"
    try:
        preimage = preimage_text.encode("utf-8")
        postimage = postimage_text.encode("utf-8")
        parse_preimage = parse_document(preimage)
        parse_postimage = parse_document(postimage)
        parse_preimage.body.decode("utf-8")
        parse_postimage.body.decode("utf-8")
    except (CanonicalDocumentError, UnicodeError):
        return "mutation evidence canonical bytes are invalid"
    pre_sha = evidence.get("preimage_sha256")
    post_sha = evidence.get("postimage_sha256")
    if pre_sha != _sha256_bytes(preimage) or post_sha != _sha256_bytes(postimage):
        return "mutation evidence image hash mismatch"
    if expected_preimage is not None and preimage != expected_preimage:
        return "mutation evidence preimage changed"
    if expected_postimage is not None and postimage != expected_postimage:
        return "mutation evidence postimage changed"
    page_uid = evidence.get("page_uid")
    if page_uid is not None and (not isinstance(page_uid, str) or not page_uid.strip()):
        return "mutation evidence page_uid is invalid"
    original_uid = _metadata_page_uid(parse_preimage.metadata)
    updated_uid = _metadata_page_uid(parse_postimage.metadata)
    if page_uid != original_uid or page_uid != updated_uid:
        return "mutation evidence page_uid mismatch"
    records = evidence.get("replacements")
    if not isinstance(records, list) or not records:
        return "mutation evidence replacements are missing"
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("index") != index:
            return "mutation evidence replacement identity is invalid"
        if record.get("action") not in {"replace", "retract", "supersede"}:
            return "mutation evidence replacement action is invalid"
        status = record.get("span_status")
        if status not in {"verified", "unknown"}:
            return "mutation evidence span status is invalid"
        if require_verified_spans and status != "verified":
            return "mutation evidence span is unknown"
        old_quote = record.get("old_quote_sha256")
        new_quote = record.get("new_quote_sha256")
        if not isinstance(old_quote, str) or not isinstance(new_quote, str):
            return "mutation evidence quote hashes are missing"
        if (
            record.get("old_text_sha256") != old_quote
            or record.get("new_text_sha256") != new_quote
        ):
            return "mutation evidence text and quote hashes differ"
        old_body_start = record.get("old_body_start")
        old_body_end = record.get("old_body_end")
        new_body_start = record.get("new_body_start")
        new_body_end = record.get("new_body_end")
        old_start = record.get("old_byte_start")
        old_end = record.get("old_byte_end")
        new_start = record.get("new_byte_start")
        new_end = record.get("new_byte_end")
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in (
                old_body_start,
                old_body_end,
                new_body_start,
                new_body_end,
                old_start,
                old_end,
                new_start,
                new_end,
            )
        ):
            return "mutation evidence byte span is invalid"
        for start_value, end_value in (
            (old_body_start, old_body_end),
            (new_body_start, new_body_end),
            (old_start, old_end),
            (new_start, new_end),
        ):
            if (start_value is None) != (end_value is None):
                return "mutation evidence byte span is incomplete"
        if status == "verified" and (
            old_start is None
            or old_end is None
            or old_body_start is None
            or old_body_end is None
        ):
            return "mutation evidence old span is missing"
        # A non-empty replacement must carry a postimage span.  Retractions
        # intentionally have an empty postimage and therefore use no range;
        # a partial empty range is still tampering rather than a valid delete.
        empty_quote_sha256 = _sha256_text("")
        new_span_values = (new_body_start, new_body_end, new_start, new_end)
        if status == "verified" and new_quote != empty_quote_sha256 and any(
            value is None for value in new_span_values
        ):
            return "mutation evidence new span is missing"
        if status == "verified" and new_quote == empty_quote_sha256 and any(
            value is not None for value in new_span_values
        ):
            return "mutation evidence empty new span is invalid"
        original_body = parse_preimage.body
        updated_body = parse_postimage.body
        original_body_text = original_body.decode("utf-8")
        updated_body_text = updated_body.decode("utf-8")
        original_body_offset = _body_byte_offset(preimage, original_body)
        updated_body_offset = _body_byte_offset(postimage, updated_body)
        if old_body_start is not None:
            expected_old_start, expected_old_end = _byte_span(
                preimage,
                original_body,
                old_body_start,
                old_body_end,
            )
            if (
                old_start is None
                or old_end is None
                or old_body_end is None
                or old_body_end <= old_body_start
                or old_body_end > len(original_body_text)
                or old_start != expected_old_start
                or old_end != expected_old_end
                or old_start < original_body_offset
                or old_end > len(preimage)
            ):
                return "mutation evidence old span is out of bounds"
            old_start_in_body = len(
                original_body_text[:old_body_start].encode("utf-8")
            )
            old_end_in_body = len(
                original_body_text[:old_body_end].encode("utf-8")
            )
            old_bytes = original_body[old_start_in_body:old_end_in_body]
            if _sha256_bytes(old_bytes) != old_quote:
                return "mutation evidence old quote hash mismatch"
            if original_body.count(old_bytes) != 1:
                return "mutation evidence old quote is not unique"
        if new_body_start is not None:
            expected_new_start, expected_new_end = _byte_span(
                postimage,
                updated_body,
                new_body_start,
                new_body_end,
            )
            if (
                new_start is None
                or new_end is None
                or new_body_end is None
                or new_body_end < new_body_start
                or new_body_end > len(updated_body_text)
                or new_start != expected_new_start
                or new_end != expected_new_end
                or new_start < updated_body_offset
                or new_end > len(postimage)
            ):
                return "mutation evidence new span is out of bounds"
            new_start_in_body = len(
                updated_body_text[:new_body_start].encode("utf-8")
            )
            new_end_in_body = len(
                updated_body_text[:new_body_end].encode("utf-8")
            )
            new_bytes = updated_body[new_start_in_body:new_end_in_body]
            if _sha256_bytes(new_bytes) != new_quote:
                return "mutation evidence new quote hash mismatch"
    return None


def _mutation_evidence_ref_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project one full receipt into its compact review/audit binding."""

    return {
        "schema_version": MUTATION_EVIDENCE_SCHEMA_VERSION,
        "kind": MUTATION_EVIDENCE_KIND,
        "page_id": payload["page_id"],
        "correction_id": payload["correction_id"],
        "page_uid": payload["page_uid"],
        "preimage_sha256": payload["preimage_sha256"],
        "postimage_sha256": payload["postimage_sha256"],
        "evidence_sha256": mutation_evidence_sha256(payload),
        "replacements": [
            {
                key: record.get(key)
                for key in (
                    "index",
                    "action",
                    "span_status",
                    "old_quote_sha256",
                    "new_quote_sha256",
                    "old_byte_start",
                    "old_byte_end",
                    "new_byte_start",
                    "new_byte_end",
                )
            }
            for record in payload["replacements"]
        ],
    }


def mutation_evidence_ref(mutation: PreparedPageMutation) -> dict[str, Any]:
    """Return a compact durable binding to a pre-apply evidence receipt."""

    return _mutation_evidence_ref_payload(mutation_evidence_payload(mutation))


def find_mutation_page(page_id: str) -> Path | None:
    """Resolve a normal page or one explicitly correctable memory page.

    Operational system files remain outside the autonomous mutation boundary.
    The allowlist intentionally contains only user-memory content that can be
    the source of a recalled factual error.
    """

    path = canonical_document_path_for_id(
        page_id,
        pages_dir=PAGES_DIR,
        system_dir=SYSTEM_DIR,
    )
    if path is None:
        return None
    try:
        path.relative_to(PAGES_DIR.resolve(strict=True))
        return path
    except (OSError, RuntimeError, ValueError):
        pass
    if page_id not in CORRECTABLE_SYSTEM_PAGE_IDS:
        return None
    return path


def correction_constraints_file() -> Path:
    """Return the registry paired with the currently configured Wiki lock.

    Deriving this path from ``CHRONOVISOR_MUTATION_LOCK`` keeps isolated test stores
    and production on the same boundary without another path global to patch.
    """

    return CHRONOVISOR_MUTATION_LOCK.parent / "content-correction-constraints.jsonl"


def content_correction_audit_file() -> Path:
    return CHRONOVISOR_MUTATION_LOCK.parent.parent / "recall" / "content-feedback.jsonl"


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
    *,
    mutation_evidence: Mapping[str, Any] | None = None,
    mutation_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": CORRECTION_CONSTRAINT_SCHEMA_VERSION,
        "kind": "content_correction_constraint",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "correction_id": mutation.correction_id,
        "page_id": mutation.page_id,
        "action": replacement.action,
        "old_text": replacement.old_text,
        "new_text": replacement.new_text,
        "old_text_sha256": _sha256_text(replacement.old_text),
        "new_text_sha256": _sha256_text(replacement.new_text),
    }
    if mutation_evidence_sha256:
        row["mutation_evidence_sha256"] = mutation_evidence_sha256
    if mutation_evidence is not None:
        row["mutation_evidence"] = dict(mutation_evidence)
    return row


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
    existing_rows = [
        row
        for row in _read_jsonl(path)
        if row.get("kind") == "content_correction_constraint"
    ]
    existing = {_constraint_identity(row) for row in existing_rows}
    existing_evidence: set[str] = set()
    for row in existing_rows:
        digest = row.get("mutation_evidence_sha256")
        evidence = row.get("mutation_evidence")
        if (
            isinstance(digest, str)
            and isinstance(evidence, Mapping)
            and mutation_evidence_sha256(evidence) == digest
            and mutation_evidence_error(evidence) is None
        ):
            existing_evidence.add(digest)
    pending: list[dict[str, Any]] = []
    for mutation in mutations:
        if mutation.already_applied:
            # An already-applied mutation must reuse its earlier receipt; the
            # current postimage cannot recreate the original source bytes.
            continue
        evidence = mutation_evidence_payload(mutation)
        evidence_error = mutation_evidence_error(
            evidence,
            expected_page_id=mutation.page_id,
            expected_correction_id=mutation.correction_id,
            expected_preimage=mutation.original,
            expected_postimage=mutation.updated,
        )
        if evidence_error is not None:
            raise PageMutationError(evidence_error)
        evidence_digest = mutation_evidence_sha256(evidence)
        receipt_missing = evidence_digest not in existing_evidence
        for index, replacement in enumerate(mutation.replacements):
            row = _constraint_row(
                mutation,
                replacement,
                mutation_evidence=evidence if index == 0 else None,
                mutation_evidence_sha256=evidence_digest,
            )
            identity = _constraint_identity(row)
            if identity in existing:
                # Upgrade a legacy constraint by appending one
                # valid full receipt.  The registry is append-only, and
                # readers select the latest durable evidence by digest.
                if index == 0 and receipt_missing:
                    pending.append(row)
                    existing_evidence.add(evidence_digest)
                    receipt_missing = False
                continue
            existing.add(identity)
            pending.append(row)
            if index == 0:
                existing_evidence.add(evidence_digest)
                receipt_missing = False
    if not pending:
        return
    append_jsonl_durable(path, pending, sort_keys=True)


def read_mutation_evidence(
    evidence_sha256: str,
    *,
    page_id: str | None = None,
    correction_id: str | None = None,
) -> dict[str, Any] | None:
    """Read and revalidate one pre-apply evidence receipt from the registry."""

    if not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64:
        return None
    for row in reversed(_read_jsonl(correction_constraints_file())):
        if row.get("mutation_evidence_sha256") != evidence_sha256:
            continue
        evidence = row.get("mutation_evidence")
        if not isinstance(evidence, Mapping):
            continue
        if mutation_evidence_sha256(evidence) != evidence_sha256:
            continue
        if mutation_evidence_error(
            evidence,
            expected_page_id=page_id,
            expected_correction_id=correction_id,
        ) is not None:
            continue
        return dict(evidence)
    return None


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
    meta, _body = _parse_canonical_text(page_text)
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
        # Constraint consumers need the replacement rule, not archived bodies.
        found.append({key: value for key, value in row.items() if key != "mutation_evidence"})
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

    _parse_canonical_text(candidate_text)
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
        meta, body = _parse_canonical_text(constrained)
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
        meta["applied_corrections"] = markers
        constrained = serialize_document(
            CanonicalDocument(metadata=meta, body=body.encode("utf-8"))
        ).decode("utf-8")
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

    _meta, working = _parse_canonical_text(text)
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
    namespace, source_path = _canonical_location(path)
    try:
        document = validate_canonical_document(
            original,
            namespace=namespace,
            path=source_path,
            require_stable=True,
        )
        body = document.body.decode("utf-8")
    except (CanonicalDocumentError, UnicodeDecodeError) as exc:
        raise PageMutationError(
            f"target page is not mutable: {page_id}: {exc}"
        ) from exc
    meta = document.metadata

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
            page_uid=_metadata_page_uid(meta),
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
    updated_meta = dict(meta)
    updated_meta.update(metadata_updates)
    updated = serialize_document(
        CanonicalDocument(metadata=updated_meta, body=updated_body.encode("utf-8"))
    )
    try:
        verified = validate_canonical_document(
            updated,
            namespace=namespace,
            path=source_path,
            require_stable=True,
        )
        verified_body = verified.body.decode("utf-8")
    except (CanonicalDocumentError, UnicodeDecodeError) as exc:
        raise PageMutationError(
            f"updated page is not canonical: {page_id}: {exc}"
        ) from exc
    _validate_replacement_postconditions(updated_meta, verified_body, items)
    if updated == original:
        raise PageMutationError("prepared correction produced no change")

    evidence = _replacement_evidence_records(original, updated, items)
    if len(evidence) != len(items) or any(
        record.get("span_status") != "verified" for record in evidence
    ):
        raise PageMutationError("replacement source span is unknown")

    return PreparedPageMutation(
        page_id=page_id,
        path=path,
        correction_id=correction_id,
        original=original,
        updated=updated,
        original_sha256=_sha256_bytes(original),
        updated_sha256=_sha256_bytes(updated),
        replacements=items,
        page_uid=_metadata_page_uid(meta),
        evidence=evidence,
    )


def _rollback_owned_write_locked(mutation: PreparedPageMutation) -> bool:
    """Restore an owned preimage while the caller holds ``chronovisor_mutation_lock``."""

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
            with suppress(OSError):
                tmp.unlink()


def _rollback_owned_write(mutation: PreparedPageMutation) -> bool:
    """Restore the preimage only while the page still contains our exact bytes."""

    try:
        with chronovisor_mutation_lock():
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
        with chronovisor_mutation_lock():
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
        with chronovisor_mutation_lock():
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
                    namespace, source_path = _canonical_location(item.path)
                    written_document = validate_canonical_document(
                        written_bytes,
                        namespace=namespace,
                        path=source_path,
                        require_stable=True,
                    )
                    written_meta = written_document.metadata
                    written_body = written_document.body.decode("utf-8")
                    try:
                        _validate_replacement_postconditions(
                            written_meta,
                            written_body,
                            item.replacements,
                        )
                    except PageMutationError as exc:
                        raise PageMutationError(f"{exc}: {item.page_id}") from exc
            except (
                OSError,
                UnicodeDecodeError,
                CanonicalDocumentError,
                PageMutationError,
            ) as exc:
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
