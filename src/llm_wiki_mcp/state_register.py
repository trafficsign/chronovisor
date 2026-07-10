"""Always-on working-memory state register."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import parse
from llm_wiki_mcp.wiki import SYSTEM_DIR
from llm_wiki_mcp.wiki_write import apply_wiki_writes, prepare_wiki_write

STATE_PAGE_ID = "current-state"
STATE_PAGE = SYSTEM_DIR / f"{STATE_PAGE_ID}.md"
STALE_AFTER_DAYS = 30
_PLACEHOLDER_IDS = {"foo", "bar", "baz", "alpha", "beta", "target", "sample", "test"}
_PLACEHOLDER_VALUES = {"body", "test", "placeholder", "sample"}



def _strip_heading_noise(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.startswith("#"):
            continue
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _looks_like_placeholder_state_candidate(meta: dict[str, Any]) -> bool:
    page_id = str(meta.get("page_id") or "").strip().casefold()
    title = str(meta.get("title") or "").strip().casefold()
    summary = str(meta.get("summary") or "").strip().casefold()
    if page_id in _PLACEHOLDER_IDS or re.fullmatch(r"p\d+", page_id):
        return True
    if title in _PLACEHOLDER_IDS and summary in _PLACEHOLDER_VALUES:
        return True
    if summary == "body" and title in _PLACEHOLDER_IDS:
        return True
    return False


def _is_state_candidate(meta: dict[str, Any]) -> bool:
    page_id = str(meta.get("page_id") or "")
    if not page_id or page_id == STATE_PAGE_ID:
        return False
    status = str(meta.get("status") or "active")
    if status != "active":
        return False
    page_type = str(meta.get("page_type") or "knowledge")
    if page_type == "reference":
        return False
    path_value = meta.get("path")
    if isinstance(path_value, str) and not Path(path_value).exists():
        return False
    folder = Path(path_value).parent.name if isinstance(path_value, str) else ""
    if folder in {"hubs", "insights"} or page_id.startswith("folder-"):
        return False
    return not _looks_like_placeholder_state_candidate(meta)


def _state_payload(path: Path = STATE_PAGE, *, max_chars: int = 1600) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {"body": "", "updated": "", "age_days": None, "stale": False}
    meta, body = parse(text)
    body = _strip_heading_noise(body)
    if not body:
        return {"body": "", "updated": "", "age_days": None, "stale": False}
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "..."
    updated = meta.get("updated") if isinstance(meta.get("updated"), str) else ""
    parsed = _parse_date(updated)
    age_days = (date.today() - parsed).days if parsed is not None else None
    stale = age_days is None or age_days > STALE_AFTER_DAYS
    return {"body": body, "updated": updated, "age_days": age_days, "stale": stale}


def load_state_register(path: Path = STATE_PAGE, *, max_chars: int = 1600) -> str:
    """Return compact current-state text, or an empty string if unavailable."""
    payload = _state_payload(path, max_chars=max_chars)
    body = str(payload.get("body") or "")
    if not body:
        return ""
    return body


def format_state_context(
    *,
    host: str,
    cwd: str = "",
    max_chars: int = 1600,
    path: Path = STATE_PAGE,
) -> str:
    """Build a small context block injected outside the recall gate."""
    payload = _state_payload(path, max_chars=max_chars)
    body = str(payload.get("body") or "")
    if not body:
        return ""
    lines = [
        "[WORKING_MEMORY]",
        "Current state from LLM Wiki. Use only when relevant; do not overfit casual chatter.",
        f"source={STATE_PAGE_ID}",
    ]
    if payload.get("updated"):
        lines.append(f"updated={payload['updated']}")
    if payload.get("age_days") is not None:
        lines.append(f"age_days={payload['age_days']}")
    if payload.get("stale"):
        lines.append("stale=true")
        lines.append("warning=This state register is stale; treat it as a dated snapshot, not current truth.")
    if host:
        lines.append(f"host={host}")
    if cwd:
        lines.append(f"cwd={cwd}")
    lines.append("content:")
    lines.append(body)
    lines.append("[/WORKING_MEMORY]")
    return "\n".join(lines)


def should_inject_state(host: str) -> bool:
    return host in {"codex", "claude-code", "generic"}


def refresh_state_register(
    page_ids: list[str] | None = None,
    *,
    source_raw: str = "",
    limit: int = 12,
    write: bool = True,
    path: Path = STATE_PAGE,
) -> dict[str, Any]:
    """Refresh the working-memory state from recently changed wiki pages."""
    from llm_wiki_mcp.index_store import get_store

    store = get_store()
    store.refresh()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates: list[dict[str, Any]]
    if page_ids:
        candidates = []
        for page_id in page_ids:
            if page_id in seen:
                continue
            seen.add(page_id)
            meta = store.meta(page_id)
            if meta is not None:
                candidates.append(meta)
    else:
        candidates = store.all_pages_meta(include_system=False)

    for meta in candidates:
        page_id = str(meta.get("page_id") or "")
        full = store.meta(page_id)
        if isinstance(full, dict):
            meta = {**meta, **full}
        if not _is_state_candidate(meta):
            continue
        page_type = str(meta.get("page_type") or "knowledge")
        title = str(meta.get("title") or page_id)
        summary = str(meta.get("summary") or "").strip()
        selected.append(
            {
                "page_id": page_id,
                "title": title,
                "summary": summary,
                "updated": str(meta.get("updated") or "unknown"),
                "page_type": page_type,
            }
        )
        if len(selected) >= max(1, limit):
            break

    today = date.today().isoformat()
    lines = [
        "---",
        "title: Current State",
        f"updated: {today}",
        "type: state",
        "tags: [d/llm-wiki, t/state, s/current]",
        "summary: Auto-maintained working-memory snapshot from recent LLM Wiki updates.",
        "---",
        "",
        "# Current State",
        "",
        "This file is auto-maintained by LLM Wiki ingest/sleep. Treat items as working-memory hints, not final facts.",
        "",
        "## Recent Memory Updates",
    ]
    if selected:
        for row in selected:
            suffix = f" — {row['summary']}" if row["summary"] else ""
            lines.append(f"- [[{row['page_id']}]] — {row['title']} ({row['updated']}){suffix}")
    else:
        lines.append("- No recent non-reference updates found.")
    if source_raw:
        lines.extend(["", "## Source", f"- Latest raw: `{Path(source_raw).name}`"])
    text = "\n".join(lines).rstrip() + "\n"
    mutation: dict[str, Any] | None = None
    if write:
        mutation = apply_wiki_writes([prepare_wiki_write(path, text)])
    return {
        "status": (
            "ok"
            if mutation is None or mutation["status"] in {"applied", "unchanged"}
            else "retry"
        ),
        "path": str(path),
        "updated": today,
        "pages": [row["page_id"] for row in selected],
        "write": write,
        "mutation": mutation,
    }
