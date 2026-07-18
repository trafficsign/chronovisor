"""Read-only tools available to the bounded research orchestrator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.raw_store import RawStore
from llm_wiki_mcp.research_config import ResearchConfig
from llm_wiki_mcp.research_store import ResearchStore
from llm_wiki_mcp.research_types import Action, ActionType
from llm_wiki_mcp.search import search as run_search
from llm_wiki_mcp.wiki import RAW_DIR, WIKI_ROOT, find_page


@dataclass
class ToolContext:
    config: ResearchConfig
    store: ResearchStore
    web_provider: Any = None


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(minimum, min(maximum, value))
    return default


def wiki_search(arguments: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    limit = _bounded_int(arguments.get("limit"), default=8, minimum=1, maximum=20)
    semantic = arguments.get("semantic") is not False
    results, mode = run_search(query=query, top_n=limit, semantic=semantic)
    return {
        "query": query,
        "search_mode": mode,
        "results": [
            {
                "page_id": row.page_id,
                "title": row.title,
                "updated": row.updated,
                "score": round(float(row.score), 6),
                "page_type": row.page_type,
            }
            for row in results
        ],
    }


def wiki_read(arguments: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    page_id = str(arguments.get("page_id") or "").strip()
    if not page_id:
        raise ValueError("page_id is required")
    path = find_page(page_id)
    if path is None:
        raise FileNotFoundError(page_id)
    content = path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(content)
    max_chars = _bounded_int(arguments.get("max_chars"), default=12_000, minimum=500, maximum=50_000)
    index = get_store()
    return {
        "page_id": page_id,
        "title": str(metadata.get("title") or page_id),
        "updated": str(metadata.get("updated") or "unknown"),
        "body": body[:max_chars],
        "truncated": len(body) > max_chars,
        "outlinks": index.outlinks(page_id)[:50],
        "backlinks": index.backlinks(page_id)[:50],
        "citation": f"wiki:{page_id}",
    }


def wiki_neighbors(arguments: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    page_id = str(arguments.get("page_id") or "").strip()
    if not page_id:
        raise ValueError("page_id is required")
    index = get_store()
    return {
        "page_id": page_id,
        "outlinks": index.outlinks(page_id)[:50],
        "backlinks": index.backlinks(page_id)[:50],
    }


def verified_claims(arguments: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip().casefold()
    if not query:
        raise ValueError("query is required")
    tokens = [token for token in re.findall(r"[a-z0-9]{3,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", query)][:12]
    rows: list[dict[str, Any]] = []
    path = WIKI_ROOT / "claims" / "claims-index.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in reversed(lines[-10_000:]):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        haystack = json.dumps(row, ensure_ascii=False).casefold()
        if tokens and not any(token in haystack for token in tokens):
            continue
        rows.append(row)
        if len(rows) >= 20:
            break
    return {"query": query, "claims": rows, "citation": "claims:claims-index"}


def raw_search(arguments: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    limit = _bounded_int(arguments.get("limit"), default=5, minimum=1, maximum=10)
    scan_limit = _bounded_int(arguments.get("scan_limit"), default=1000, minimum=10, maximum=5000)
    tokens = [token.casefold() for token in re.findall(r"[a-z0-9]{3,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", query)][:12]
    store = RawStore(RAW_DIR)
    units = list(store.iter_units())[-scan_limit:]
    hits: list[dict[str, Any]] = []
    for unit in reversed(units):
        try:
            text = store.read_text(unit)
        except (OSError, UnicodeError, ValueError):
            continue
        folded = text.casefold()
        positions = [folded.find(token) for token in tokens if token and token in folded]
        if not positions:
            continue
        offset = min(positions)
        start = max(0, offset - 500)
        excerpt = text[start : start + 2500]
        hits.append(
            {
                "raw_id": unit.raw_id,
                "captured_date": unit.captured_date,
                "excerpt": excerpt,
                "offset": start,
                "citation": f"raw:{unit.raw_id}#offset={start}",
            }
        )
        if len(hits) >= limit:
            break
    return {"query": query, "scanned": len(units), "hits": hits}


def web_search(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    from llm_wiki_mcp.web_provider import search_web

    query = str(arguments.get("query") or "")
    limit = _bounded_int(arguments.get("limit"), default=5, minimum=1, maximum=10)
    return search_web(
        query,
        config=context.config.web,
        provider=context.web_provider,
        limit=limit,
    ).to_dict()


def web_fetch(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    from llm_wiki_mcp.web_fetch import fetch_web

    url = str(arguments.get("url") or "").strip()
    if not url:
        raise ValueError("url is required")
    return fetch_web(url, config=context.config.web, store=context.store).to_dict()


Tool = Callable[[dict[str, Any], ToolContext], dict[str, Any]]

TOOL_REGISTRY: dict[ActionType, Tool] = {
    ActionType.WIKI_SEARCH: wiki_search,
    ActionType.WIKI_READ: wiki_read,
    ActionType.WIKI_NEIGHBORS: wiki_neighbors,
    ActionType.VERIFIED_CLAIMS: verified_claims,
    ActionType.RAW_SEARCH: raw_search,
    ActionType.WEB_SEARCH: web_search,
    ActionType.WEB_FETCH: web_fetch,
}

TOOL_PERMISSIONS: dict[ActionType, str] = {
    ActionType.WIKI_SEARCH: "wiki_read",
    ActionType.WIKI_READ: "wiki_read",
    ActionType.WIKI_NEIGHBORS: "wiki_read",
    ActionType.VERIFIED_CLAIMS: "claims_read",
    ActionType.RAW_SEARCH: "raw_read",
    ActionType.WEB_SEARCH: "web_search_egress",
    ActionType.WEB_FETCH: "web_fetch_egress",
    ActionType.FINISH: "none",
}


def execute_tool(action: Action, context: ToolContext) -> dict[str, Any]:
    if action.type == ActionType.FINISH:
        return {"status": "finished", "answer": str(action.arguments.get("answer") or "")}
    tool = TOOL_REGISTRY.get(action.type)
    if tool is None:
        raise ValueError(f"tool not registered: {action.type.value}")
    return tool(action.arguments, context)
