"""Lightweight per-session recall state.

The synchronous recall hook may read and append this state, but expensive
maintenance stays outside the hot path.  The file is intentionally small and
JSON-only so host adapters can inspect or repair it without extra dependencies.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronovisor.core.recall_runtime_paths import RECALL_DIR
from chronovisor.recall.recall_prompt import normalize_recall_prompt

SESSIONS_DIR = RECALL_DIR / "sessions"
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass
class RecallSessionState:
    session_id: str
    recent_queries: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    injected_pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    cwd: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "session_id": self.session_id,
            "recent_queries": list(self.recent_queries[-12:]),
            "recent_topics": list(self.recent_topics[-40:]),
            "injected_pages": self.injected_pages,
            "last_seen": self.last_seen,
            "cwd": self.cwd,
        }

    @classmethod
    def from_dict(cls, session_id: str, data: dict[str, Any]) -> RecallSessionState:
        recent_queries = _normalize_queries(
            _str_list(data.get("recent_queries"))
        )[-12:]
        # Rebuild topics from trusted queries instead of retaining tokens that
        # may have been extracted from an old host transport envelope.
        recent_topics = extract_topic_terms(
            " ".join(recent_queries), limit=40
        )
        raw_pages = data.get("injected_pages")
        injected_pages = raw_pages if isinstance(raw_pages, dict) else {}
        last_seen = data.get("last_seen")
        return cls(
            session_id=session_id,
            recent_queries=recent_queries,
            recent_topics=recent_topics,
            injected_pages=injected_pages,
            last_seen=float(last_seen) if isinstance(last_seen, int | float) else time.time(),
            cwd=data.get("cwd", "") if isinstance(data.get("cwd", ""), str) else "",
        )


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def session_path(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_id.strip())[:96].strip("-_.")
    return SESSIONS_DIR / f"{safe or 'anonymous'}.json"


def load_session_state(session_id: str) -> RecallSessionState | None:
    if not session_id:
        return None
    path = session_path(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RecallSessionState(session_id=session_id)
    if not isinstance(data, dict):
        return RecallSessionState(session_id=session_id)
    return RecallSessionState.from_dict(session_id, data)


def save_session_state(state: RecallSessionState) -> None:
    path = session_path(state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def update_session_after_recall(
    state: RecallSessionState | None,
    *,
    queries: list[str],
    page_ids: list[str],
    page_updated: dict[str, str] | None = None,
    cwd: str = "",
) -> None:
    if state is None:
        return
    state.last_seen = time.time()
    if state.cwd and cwd and state.cwd != cwd:
        state.recent_queries = []
        state.recent_topics = []
    state.cwd = cwd
    state.recent_queries = _normalize_queries(state.recent_queries)
    for query in _normalize_queries(queries):
        q = re.sub(r"\s+", " ", query).strip()
        # Existing session state can still contain the former product name.
        # Canonicalize it so a migration does not duplicate otherwise equal
        # recall queries forever.
        q = re.sub(r"\bllm\s+wiki\b", "Chronovisor", q, flags=re.IGNORECASE)
        if q:
            if q in state.recent_queries:
                state.recent_queries.remove(q)
            state.recent_queries.append(q)
    state.recent_queries = state.recent_queries[-12:]

    state.recent_topics = extract_topic_terms(
        " ".join(state.recent_queries), limit=40
    )

    updates = page_updated or {}
    now = time.time()
    for page_id in page_ids:
        if not page_id:
            continue
        state.injected_pages[page_id] = {
            "last_injected_at": now,
            "updated": updates.get(page_id, ""),
        }
    save_session_state(state)


def _normalize_queries(queries: list[str]) -> list[str]:
    normalized: list[str] = []
    for query in queries:
        cleaned, _reasons = normalize_recall_prompt(query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def extract_topic_terms(text: str, *, limit: int = 12) -> list[str]:
    text_lower = text.lower()
    terms: list[str] = []
    for match in re.finditer(r"[a-z0-9][a-z0-9_.-]{2,}", text_lower):
        token = match.group(0).strip("-_.")
        if token and token not in terms:
            terms.append(token)
    for run in re.findall(r"[\u3040-\u30ff\u3400-\u9fff]{2,}", text):
        if run not in terms:
            terms.append(run)
    return terms[:limit]


def session_summary(state: RecallSessionState | None, *, max_chars: int = 900) -> str:
    if state is None:
        return ""
    payload = {
        "recent_queries": state.recent_queries[-6:],
        "recent_topics": state.recent_topics[-20:],
        "injected_pages": list(state.injected_pages.keys())[-20:],
    }
    text = json.dumps(payload, ensure_ascii=False)
    return text[:max_chars]


def should_skip_page(state: RecallSessionState | None, page_id: str, updated: str) -> bool:
    if state is None:
        return False
    entry = state.injected_pages.get(page_id)
    if not isinstance(entry, dict):
        return False
    injected_updated = entry.get("updated")
    return isinstance(injected_updated, str) and injected_updated == updated


def cleanup_sessions(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> int:
    if not SESSIONS_DIR.exists():
        return 0
    cutoff = time.time() - max(60, ttl_seconds)
    removed = 0
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            last_seen = data.get("last_seen", path.stat().st_mtime)
            if float(last_seen) < cutoff:
                path.unlink()
                removed += 1
        except Exception:
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                pass
    return removed
