"""Recall query hints learned from missed-candidate feedback."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core import store as chronovisor_store
from chronovisor.core.durable_state import sidecar_exclusive_lock as _hint_lock

QUERY_HINTS_FILE = chronovisor_store.CHRONOVISOR_ROOT / "recall" / "query-hints.json"
GENERIC_HINT_TOKENS = {
    "about",
    "and",
    "are",
    "as",
    "assistant",
    "available",
    "based",
    "but",
    "codex",
    "context",
    "details",
    "for",
    "from",
    "history",
    "implying",
    "in",
    "is",
    "memory",
    "not",
    "of",
    "on",
    "or",
    "past",
    "previous",
    "project",
    "prompt",
    "recall",
    "reference",
    "references",
    "response",
    "session",
    "specific",
    "stored",
    "that",
    "the",
    "this",
    "to",
    "user",
    "was",
    "which",
    "wiki",
    "with",
}


def normalize_query_text(text: str) -> str:
    text = text.casefold().replace("_", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def query_tokens(text: str) -> set[str]:
    normalized = normalize_query_text(text)
    tokens = re.findall(r"[a-z0-9][a-z0-9._-]*|[\u3040-\u30ff\u3400-\u9fff]{2,}", normalized)
    return {token for token in tokens if len(token) >= 2}


def _hint_path(path: Path | None = None) -> Path:
    return path or QUERY_HINTS_FILE


def load_query_hints(path: Path | None = None) -> list[dict[str, Any]]:
    path = _hint_path(path)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, dict):
        return []
    hints = parsed.get("hints")
    if not isinstance(hints, list):
        return []
    return [hint for hint in hints if isinstance(hint, dict)]


def _save_query_hints_unlocked(hints: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "hints": hints}
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()


def save_query_hints(hints: list[dict[str, Any]], path: Path | None = None) -> None:
    path = _hint_path(path)
    with _hint_lock(path):
        _save_query_hints_unlocked(hints, path)


def canonicalize_query_hint_targets(
    *,
    path: Path | None = None,
    aliases: dict[str, str] | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Resolve page aliases in the mutable query-hint view."""

    from chronovisor.core.alias_store import load_aliases
    from chronovisor.recall.recall_log_schema import canonicalize_page_ids

    path = _hint_path(path)
    alias_map = aliases if aliases is not None else load_aliases()

    def transform(hints: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        changed = 0
        normalized: list[dict[str, Any]] = []
        for hint in hints:
            row = dict(hint)
            page_id = row.get("page_id")
            if isinstance(page_id, str):
                resolved = canonicalize_page_ids([page_id], alias_map)
                if resolved and resolved[0] != page_id:
                    row["page_id"] = resolved[0]
                    changed += 1
            normalized.append(row)
        return normalized, changed

    if write:
        with _hint_lock(path):
            hints, changed = transform(load_query_hints(path))
            if changed:
                _save_query_hints_unlocked(hints, path)
    else:
        hints, changed = transform(load_query_hints(path))
    return {
        "status": "ok",
        "path": str(path),
        "hints": len(hints),
        "changed": changed,
        "write": write,
    }


def add_query_hint(
    *,
    page_id: str,
    query: str,
    signal: str = "",
    source: str = "recall-auto-apply",
    normalize_key: str = "",
    path: Path | None = None,
    increment_existing: bool = True,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    page_ref = page_id.strip()
    query_text = query.strip()
    if not page_ref:
        raise ValueError("query hint page_id is required")
    if not query_text:
        raise ValueError("query hint query is required")
    system_path = chronovisor_store.SYSTEM_DIR / f"{page_ref}.md"
    if chronovisor_store.find_page(page_ref) is None and not system_path.exists():
        raise ValueError(f"query hint target page does not exist: {page_ref!r}")

    path = _hint_path(path)
    with _hint_lock(path):
        hints = load_query_hints(path)
        key = normalize_query_text(query_text)
        now = datetime.now().isoformat(timespec="seconds")
        for hint in hints:
            if hint.get("page_id") == page_ref and hint.get("query_key") == key:
                if increment_existing:
                    hint["count"] = int(hint.get("count", 1) or 1) + 1
                    hint["updated_at"] = now
                    if normalize_key:
                        hint["normalize_key"] = normalize_key
                    _save_query_hints_unlocked(hints, path)
                return hint

        trusted_auto = bool(
            isinstance(provenance, dict)
            and provenance.get("schema_version") == 2
            and provenance.get("frontier_approved") is True
            and str(provenance.get("feedback_ref") or "").strip()
        )
        page_uid = ""
        try:
            from chronovisor.ingest.page_registry import PageRegistry

            resolved = PageRegistry(chronovisor_store.CHRONOVISOR_ROOT).resolve(
                page_ref
            )
            if isinstance(resolved, dict):
                page_uid = str(resolved.get("uid") or "")
        except Exception:
            pass
        record = {
            "page_id": page_ref,
            **({"page_uid": page_uid} if page_uid else {}),
            "query": query_text,
            "query_key": key,
            "tokens": sorted(query_tokens(query_text) | query_tokens(signal)),
            "signal": signal,
            "source": source,
            "normalize_key": normalize_key,
            "count": 1,
            "created_at": now,
            "updated_at": now,
            "provenance_version": 2,
            "provenance": dict(provenance or {}),
            "active": source != "recall-auto-apply" or trusted_auto,
        }
        hints.append(record)
        _save_query_hints_unlocked(hints, path)
        return record


def hint_matches_query(hint: dict[str, Any], query: str) -> bool:
    query_key = normalize_query_text(query)
    hint_key = str(hint.get("query_key") or normalize_query_text(str(hint.get("query", ""))))
    if hint_key and len(hint_key) >= 12 and len(query_key) >= 12 and (
        hint_key in query_key or query_key in hint_key
    ):
        return True
    raw_tokens = hint.get("tokens")
    hint_tokens = {str(token) for token in raw_tokens if isinstance(token, str)} if isinstance(raw_tokens, list) else query_tokens(hint_key)
    hint_tokens = meaningful_hint_tokens(hint_tokens)
    query_tokens_ = meaningful_hint_tokens(query_tokens(query))
    overlap = hint_tokens & query_tokens_
    if not hint_tokens or not query_tokens_:
        return False
    required = max(2, min(4, (min(len(hint_tokens), len(query_tokens_)) + 1) // 2))
    return len(overlap) >= required and len(overlap) / min(len(hint_tokens), len(query_tokens_)) >= 0.5


def meaningful_hint_tokens(tokens: set[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in GENERIC_HINT_TOKENS and (len(token) >= 3 or re.search(r"[\u3040-\u30ff\u3400-\u9fff]", token))
    }


def matching_hint_page_ids(queries: list[str], *, limit: int = 3, path: Path | None = None) -> list[str]:
    if not queries or limit <= 0:
        return []
    path = _hint_path(path)
    from chronovisor.core.alias_store import load_aliases
    from chronovisor.recall.recall_log_schema import canonicalize_page_ids

    aliases = load_aliases()
    out: list[str] = []
    seen: set[str] = set()
    for hint in load_query_hints(path):
        if hint.get("active") is False:
            continue
        if (
            hint.get("source") == "recall-auto-apply"
            and hint.get("provenance_version") != 2
        ):
            continue
        page_id = hint.get("page_id")
        if not isinstance(page_id, str):
            continue
        resolved = canonicalize_page_ids([page_id], aliases)
        if not resolved:
            continue
        page_id = resolved[0]
        if page_id in seen:
            continue
        if any(hint_matches_query(hint, query) for query in queries):
            if chronovisor_store.find_page(page_id) is None:
                continue
            seen.add(page_id)
            out.append(page_id)
            if len(out) >= limit:
                break
    return out


def quarantine_legacy_query_hints(
    *,
    path: Path | None = None,
    quarantine_path: Path | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Disable unversioned auto-learned hints and preserve them for audit."""
    path = _hint_path(path)
    quarantine_path = quarantine_path or path.with_name("query-hints-quarantine.json")
    hints = load_query_hints(path)
    quarantined: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    for hint in hints:
        if (
            hint.get("source") == "recall-auto-apply"
            and hint.get("provenance_version") != 2
        ):
            quarantined.append({**hint, "active": False, "quarantine_reason": "legacy_unversioned_provenance"})
        else:
            active.append(hint)
    if write:
        with _hint_lock(path):
            _save_query_hints_unlocked(active, path)
            _save_query_hints_unlocked(quarantined, quarantine_path)
    return {
        "status": "ok",
        "active": len(active),
        "quarantined": len(quarantined),
        "path": str(path),
        "quarantine_path": str(quarantine_path),
        "write": write,
    }
