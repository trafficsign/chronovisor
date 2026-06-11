"""Agentic search/read/link/requery retrieval for MCP deep dives."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from typing import Any

from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.jobs import JobStatus, job_store
from llm_wiki_mcp.search import ScoredPage, search as run_search
from llm_wiki_mcp.wiki import find_page


def _compact(text: str, *, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _page_record(page_id: str, *, max_chars: int = 1200) -> dict[str, Any] | None:
    store = get_store()
    meta = store.meta(page_id)
    path = find_page(page_id)
    if meta is None and path is None:
        return None
    title = page_id
    updated = "unknown"
    body = ""
    if path is not None:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = ""
        parsed_meta, parsed_body = parse_frontmatter(content)
        body = parsed_body
        title = str(parsed_meta.get("title") or title)
        updated = str(parsed_meta.get("updated") or updated)
    if meta is not None:
        title = str(meta.get("title") or title)
        updated = str(meta.get("updated") or updated)
    return {
        "page_id": page_id,
        "title": title,
        "updated": updated,
        "snippet": _compact(body, limit=max_chars),
        "outlinks": store.outlinks(page_id),
        "backlinks": store.backlinks(page_id),
    }


def _search_rows(results: list[ScoredPage]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page.page_id,
            "title": page.title,
            "score": round(float(page.score), 6),
            "updated": page.updated,
        }
        for page in results
    ]


def _linked_page_ids(page_ids: list[str], *, limit: int) -> list[str]:
    store = get_store()
    linked: list[str] = []
    seen = set(page_ids)
    for page_id in page_ids:
        for candidate in store.outlinks(page_id) + store.backlinks(page_id):
            if candidate in seen:
                continue
            if store.meta(candidate) is None and find_page(candidate) is None:
                continue
            seen.add(candidate)
            linked.append(candidate)
            if len(linked) >= limit:
                return linked
    return linked


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _fallback_requeries(original_query: str, pages: list[dict[str, Any]], *, limit: int) -> list[str]:
    titles = [str(page.get("title") or page.get("page_id") or "") for page in pages[:4]]
    title_text = " ".join(title for title in titles if title)
    query = _compact(f"{original_query} {title_text}", limit=180)
    return [query] if query else []


def _llm_requeries(
    original_query: str,
    current_query: str,
    pages: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    from llm_wiki_mcp.ollama import generate, is_available

    if not is_available():
        return []
    page_lines = "\n".join(
        f"- {page['page_id']}: {page.get('title', '')} :: {page.get('snippet', '')[:220]}"
        for page in pages[:6]
    )
    prompt = (
        "You are improving a local wiki retrieval query.\n"
        "Return compact JSON only: {\"queries\":[\"...\"]}.\n"
        "Write 1-2 follow-up search queries that would find missing or adjacent pages.\n\n"
        f"Original query: {original_query}\n"
        f"Current query: {current_query}\n"
        f"Read pages:\n{page_lines}\n"
    )
    try:
        raw = generate(prompt, system="Return JSON only. No markdown.")
    except Exception:
        return []
    payload = _extract_json_object(raw) or {}
    queries = payload.get("queries")
    if not isinstance(queries, list):
        return []
    out = []
    for item in queries:
        if isinstance(item, str) and item.strip():
            out.append(_compact(item, limit=180))
        if len(out) >= limit:
            break
    return out


def run_deep_dive(
    query: str,
    *,
    max_iterations: int = 3,
    fanout: int = 5,
    semantic: bool = True,
    use_llm: bool = True,
) -> dict[str, Any]:
    store = get_store()
    store.refresh()

    max_iterations = max(1, min(5, int(max_iterations)))
    fanout = max(1, min(10, int(fanout)))
    queued_queries = [_compact(query, limit=220)]
    used_queries: set[str] = set()
    collected_pages: dict[str, dict[str, Any]] = {}
    iterations: list[dict[str, Any]] = []

    for idx in range(max_iterations):
        current_query = next((q for q in queued_queries if q and q not in used_queries), "")
        if not current_query:
            break
        used_queries.add(current_query)
        results, mode = run_search(query=current_query, top_n=fanout, semantic=semantic)
        direct_ids = [page.page_id for page in results]
        linked_ids = _linked_page_ids(direct_ids, limit=fanout)
        read_ids = direct_ids + [pid for pid in linked_ids if pid not in direct_ids]
        read_pages = []
        for page_id in read_ids:
            record = _page_record(page_id)
            if record is None:
                continue
            collected_pages.setdefault(page_id, record)
            read_pages.append(record)

        next_queries: list[str] = []
        if idx < max_iterations - 1:
            if use_llm:
                next_queries = _llm_requeries(query, current_query, read_pages, limit=2)
            if not next_queries:
                next_queries = _fallback_requeries(query, read_pages, limit=2)
            for next_query in next_queries:
                if next_query and next_query not in used_queries and next_query not in queued_queries:
                    queued_queries.append(next_query)

        iterations.append(
            {
                "iteration": idx + 1,
                "query": current_query,
                "search_mode": mode,
                "direct_hits": _search_rows(results),
                "linked_page_ids": linked_ids,
                "read_page_ids": [page["page_id"] for page in read_pages],
                "next_queries": next_queries,
            }
        )

    return {
        "status": "completed",
        "query": query,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "iterations": iterations,
        "pages": list(collected_pages.values()),
    }


def start_deep_dive(
    query: str,
    *,
    max_iterations: int = 3,
    fanout: int = 5,
    semantic: bool = True,
    use_llm: bool = True,
) -> str:
    job = job_store.create(processor="deep-retrieval")
    job_store.update(
        job.job_id,
        status=JobStatus.RUNNING,
        stage="search-read-requery",
        total_ops=max(1, min(5, int(max_iterations))),
        completed_ops=0,
    )

    def worker() -> None:
        try:
            result = run_deep_dive(
                query,
                max_iterations=max_iterations,
                fanout=fanout,
                semantic=semantic,
                use_llm=use_llm,
            )
            job_store.update(
                job.job_id,
                status=JobStatus.COMPLETED,
                completed_at=datetime.now().isoformat(),
                completed_ops=len(result.get("iterations", [])),
                result=result,
            )
        except Exception as exc:
            job_store.update(
                job.job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now().isoformat(),
                error=str(exc),
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return job.job_id
