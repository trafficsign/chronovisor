"""Append-only recall-feedback ledger helpers.

Feedback is operational evidence, so migrations must never rewrite or delete
historical JSONL rows.  A retraction instead names both the producer key and
the canonical digest of one exact ``page_ignored`` row.  Consumers use this
module to hide only that row; malformed or incomplete retractions fail closed
and leave the original feedback active.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chronovisor.core.store import find_page
from chronovisor.ingest.page_mutation import find_mutation_page
from chronovisor.recall.recall_runtime_paths import RECALL_DIR

PAGE_IGNORED_RETRACTION_KIND = "page_ignored_retracted"


def _prompt_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read valid object rows while tolerating interrupted JSONL tails."""

    try:
        lines = path.read_text(encoding="utf-8").split("\n")
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


def feedback_row_sha256(row: dict[str, Any]) -> str:
    """Return the stable identity used by an exact-row retraction."""

    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def retracted_page_ignored_targets(
    rows: Iterable[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Return fully bound ``(producer key, target digest)`` retractions."""

    targets: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("kind") != PAGE_IGNORED_RETRACTION_KIND:
            continue
        if row.get("target_kind") != "page_ignored":
            continue
        key = row.get("content_correction_key")
        digest = row.get("target_feedback_sha256")
        if (
            isinstance(key, str)
            and key
            and isinstance(digest, str)
            and len(digest) == 64
        ):
            targets.add((key, digest))
    return targets


def active_feedback_rows(path: Path) -> list[dict[str, Any]]:
    """Return semantic feedback after exact append-only retractions."""

    rows = read_jsonl_rows(path)
    retracted = retracted_page_ignored_targets(rows)
    active: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") == PAGE_IGNORED_RETRACTION_KIND:
            continue
        key = row.get("content_correction_key")
        if (
            row.get("kind") == "page_ignored"
            and isinstance(key, str)
            and (key, feedback_row_sha256(row)) in retracted
        ):
            continue
        active.append(row)
    return active


def trusted_negative_feedback_row_error(
    row: dict[str, Any],
    *,
    recall_log_file: Path | None = None,
) -> str:
    """Return why a direct negative row lacks exact frontier authority."""

    if row.get("kind") not in {"page_ignored", "contradiction"}:
        return "unsupported_kind"
    if row.get("frontier_reviewed") is not True:
        return "not_frontier_reviewed"
    if row.get("label_quality") != "strong":
        return "label_not_strong"
    producer_key = row.get("content_correction_key") or row.get("producer_key")
    if not isinstance(producer_key, str) or not producer_key:
        return "missing_producer_key"
    ref = row.get("ref")
    snapshot = row.get("snapshot")
    if not isinstance(ref, str) or not ref or not isinstance(snapshot, dict):
        return "missing_exact_ref"
    if snapshot.get("decision_id") != ref:
        return "decision_ref_mismatch"
    if snapshot.get("decision") not in {"none", "search", "read"}:
        return "decision_unknown"
    host = str(row.get("host") or "")
    if host not in {"codex", "claude-code"} or snapshot.get("host") != host:
        return "host_mismatch"
    if not isinstance(snapshot.get("session_id"), str) or not snapshot["session_id"]:
        return "missing_session_id"
    source_ref = row.get("source_turn_ref")
    if not isinstance(source_ref, dict):
        return "missing_source_turn_ref"
    if (
        source_ref.get("session_id") != snapshot.get("session_id")
        or source_ref.get("prompt_hash") != snapshot.get("prompt_hash")
        or not isinstance(row.get("prompt"), str)
        or _prompt_hash(row["prompt"]) != snapshot.get("prompt_hash")
        or not isinstance(source_ref.get("user_line"), int)
        or isinstance(source_ref.get("user_line"), bool)
        or not isinstance(source_ref.get("assistant_line"), int)
        or isinstance(source_ref.get("assistant_line"), bool)
        or source_ref["user_line"] <= 0
        or source_ref["assistant_line"] < source_ref["user_line"]
    ):
        return "source_turn_binding_mismatch"
    log_path = recall_log_file or RECALL_DIR / "recall-log.jsonl"
    matching = [
        value
        for value in read_jsonl_rows(log_path)
        if value.get("decision_id") == ref
        and value.get("session_id") == snapshot.get("session_id")
        and value.get("host") == host
        and value.get("prompt_hash") == snapshot.get("prompt_hash")
    ]
    if len(matching) != 1:
        return "decision_log_binding_missing"
    if snapshot.get("evidence_features") != matching[0].get("evidence_features"):
        return "decision_evidence_binding_mismatch"
    evidence = matching[0].get("evidence_features")
    field = evidence.get("field_shadow") if isinstance(evidence, dict) else None
    topic_epoch = field.get("topic_epoch") if isinstance(field, dict) else None
    if (
        not isinstance(topic_epoch, int)
        or isinstance(topic_epoch, bool)
        or topic_epoch < 0
        or field.get("session_hash") in {None, ""}
    ):
        return "field_topic_binding_missing"
    recalled_pages = {
        str(value)
        for field in ("pages", "injected_pages")
        for value in matching[0].get(field, [])
        if isinstance(value, str) and value
    }
    for item in matching[0].get("context_items", []):
        if isinstance(item, dict) and isinstance(item.get("page_id"), str):
            recalled_pages.add(item["page_id"])
    pages = row.get("negative_pages")
    hashes = row.get("negative_page_hashes")
    if not isinstance(pages, list) or not pages or not isinstance(hashes, dict):
        return "missing_negative_pages"
    if not set(pages).issubset(recalled_pages):
        return "negative_pages_not_recalled_subset"
    for page_id in pages:
        expected = hashes.get(page_id) if isinstance(page_id, str) else None
        if not isinstance(page_id, str) or not page_id or not isinstance(expected, str):
            return "invalid_page_binding"
        path = find_page(page_id) or find_mutation_page(page_id)
        try:
            current = hashlib.sha256(path.read_bytes()).hexdigest() if path else ""
        except OSError:
            current = ""
        if len(expected) != 64 or current != expected:
            return "page_hash_mismatch"
    return ""


def trusted_negative_feedback_rows(
    path: Path, *, recall_log_file: Path | None = None
) -> list[dict[str, Any]]:
    """Return only active exact negatives; malformed retractions change nothing."""

    return [
        row
        for row in active_feedback_rows(path)
        if not trusted_negative_feedback_row_error(
            row, recall_log_file=recall_log_file
        )
    ]
