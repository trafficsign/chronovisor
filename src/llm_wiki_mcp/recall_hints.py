"""Recall query hints learned from missed-candidate feedback."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp import wiki


QUERY_HINTS_FILE = wiki.WIKI_ROOT / "recall" / "query-hints.json"


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


def save_query_hints(hints: list[dict[str, Any]], path: Path | None = None) -> None:
    path = _hint_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "hints": hints}
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def add_query_hint(
    *,
    page_id: str,
    query: str,
    signal: str = "",
    source: str = "recall-auto-apply",
    normalize_key: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    page_ref = page_id.strip()
    query_text = query.strip()
    if not page_ref:
        raise ValueError("query hint page_id is required")
    if not query_text:
        raise ValueError("query hint query is required")
    if wiki.find_page(page_ref) is None:
        raise ValueError(f"query hint target page does not exist: {page_ref!r}")

    path = _hint_path(path)
    hints = load_query_hints(path)
    key = normalize_query_text(query_text)
    now = datetime.now().isoformat(timespec="seconds")
    for hint in hints:
        if hint.get("page_id") == page_ref and hint.get("query_key") == key:
            hint["count"] = int(hint.get("count", 1) or 1) + 1
            hint["updated_at"] = now
            if normalize_key:
                hint["normalize_key"] = normalize_key
            save_query_hints(hints, path)
            return hint

    record = {
        "page_id": page_ref,
        "query": query_text,
        "query_key": key,
        "tokens": sorted(query_tokens(query_text) | query_tokens(signal)),
        "signal": signal,
        "source": source,
        "normalize_key": normalize_key,
        "count": 1,
        "created_at": now,
        "updated_at": now,
    }
    hints.append(record)
    save_query_hints(hints, path)
    return record


def hint_matches_query(hint: dict[str, Any], query: str) -> bool:
    query_key = normalize_query_text(query)
    hint_key = str(hint.get("query_key") or normalize_query_text(str(hint.get("query", ""))))
    if hint_key and (hint_key in query_key or query_key in hint_key):
        return True
    raw_tokens = hint.get("tokens")
    hint_tokens = {str(token) for token in raw_tokens if isinstance(token, str)} if isinstance(raw_tokens, list) else query_tokens(hint_key)
    overlap = hint_tokens & query_tokens(query)
    return len(overlap) >= min(2, len(hint_tokens)) if hint_tokens else False


def matching_hint_page_ids(queries: list[str], *, limit: int = 3, path: Path | None = None) -> list[str]:
    if not queries or limit <= 0:
        return []
    path = _hint_path(path)
    out: list[str] = []
    seen: set[str] = set()
    for hint in load_query_hints(path):
        page_id = hint.get("page_id")
        if not isinstance(page_id, str) or page_id in seen:
            continue
        if any(hint_matches_query(hint, query) for query in queries):
            if wiki.find_page(page_id) is None:
                continue
            seen.add(page_id)
            out.append(page_id)
            if len(out) >= limit:
                break
    return out
