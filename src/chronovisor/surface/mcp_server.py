"""MCP page search and explicit raw memory capture."""

import contextlib
import json
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP

from chronovisor.capture import capture_status
from chronovisor.capture.background_jobs import snapshot
from chronovisor.common.paths import CHRONOVISOR_ROOT
from chronovisor.index.index_store import (
    canonical_document_path_for_id,
    get_store,
    stable_indexed_document_path,
)
from chronovisor.index.page_registry import PageRegistry, PageRegistryError
from chronovisor.pages.frontmatter import parse as _frontmatter_parse
from chronovisor.pages.markdown_sections import markdown_sections
from chronovisor.pages.store import PAGE_RESERVED_FILENAMES, canonical_document_path
from chronovisor.raw.record_raw import record_raw

mcp = FastMCP(
    "chronovisor",
    instructions=(
        "Chronovisor searches existing knowledge pages. Use chronovisor_init at session "
        "start, chronovisor_search during conversation, and chronovisor_read for full "
        "pages with backlinks. chronovisor_status reports indexes, capture status and capture jobs. "
        "Use chronovisor_record for explicit 'remember this' requests; it saves raw "
        "content without generating pages."
    ),
)
READ_MAX_CHARS = 40_000
_READ_SECTION_INDEX_LIMIT = 80
_EXPANDED_PER_HIT = 3
_EXPANDED_TOTAL = 20


def _find_page_with_alias(
    page_id: str,
    *,
    require_stable: bool = True,
) -> Path | None:
    """Resolve a canonical page or a durable legacy page-id alias."""

    registry = PageRegistry(CHRONOVISOR_ROOT)
    if registry.path.exists():
        state = registry.load()
        normalized_key = str(page_id or "").strip().removesuffix(".md").casefold()
        known_key = (
            normalized_key in state["keys"] or normalized_key in state["ambiguous_keys"]
        )
        if not known_key:
            known_key = any(
                normalized_key
                in {
                    str(uid).casefold(),
                    str(row.get("uid") or "").casefold(),
                    str(row.get("page_id") or "")
                    .strip()
                    .removesuffix(".md")
                    .casefold(),
                }
                for uid, row in state["pages"].items()
                if isinstance(row, dict)
            ) or normalized_key in {str(uid).casefold() for uid in state["redirects"]}
        if known_key:
            try:
                registry.resolve_from_state(state, page_id)
            except PageRegistryError as exc:
                if str(exc).startswith("ambiguous page key "):
                    return None
                raise
            return registry.path_for(page_id, require_stable=require_stable)
    path = canonical_document_path_for_id(
        page_id,
        pages_dir=CHRONOVISOR_ROOT / "pages",
        system_dir=CHRONOVISOR_ROOT / "system",
        require_stable=require_stable,
    )
    if path is not None:
        return path
    try:
        from chronovisor.index.alias_store import resolve_alias_path

        alias_path = resolve_alias_path(page_id)
    except Exception:
        # Read paths remain fail-closed to a normal not-found result when the
        # optional alias ledger is absent or malformed.
        return None
    if alias_path is None:
        return None
    return canonical_document_path(
        alias_path,
        CHRONOVISOR_ROOT / "pages",
        namespace="pages",
        reserved_filenames=PAGE_RESERVED_FILENAMES,
        require_stable=require_stable,
    )


def _content_window(content: str, offset: int, max_chars: int) -> dict[str, Any]:
    total = len(content)
    offset = min(max(0, offset), total)
    max_chars = max(1, min(max_chars, READ_MAX_CHARS))
    end = min(total, offset + max_chars)
    window: dict[str, Any] = {"content": content[offset:end]}
    if offset or end < total:
        sections = []
        position = 0
        for section in markdown_sections(content):
            if section.heading:
                sections.append({"heading": section.heading, "offset": position})
            position += len(section.content)
        window.update(
            {
                "truncated": True,
                "content_chars": total,
                "offset": offset,
                "next_offset": end if end < total else None,
                "sections": sections[:_READ_SECTION_INDEX_LIMIT],
                "section_count": len(sections),
            }
        )
    return window


def _filter_search_results(
    results: list[Any],
    *,
    store: Any,
    tags: list[str] | None,
    tag_match: str,
) -> tuple[list[Any], list[str]]:
    """Apply exact metadata filters after ranked retrieval."""

    tag_filter = [tag for tag in (tags or []) if isinstance(tag, str) and tag]
    if tag_filter:
        target = set(tag_filter)
        match_all = tag_match != "any"
        results = [
            result
            for result in results
            if (
                target.issubset(set(store.tags(result.page_id)))
                if match_all
                else bool(target & set(store.tags(result.page_id)))
            )
        ]

    return results, tag_filter


def _direct_search_hits(
    results: list[Any],
    *,
    query: str,
    store: Any,
) -> list[dict[str, Any]]:
    """Project ranked results into the stable MCP direct-hit shape."""

    query_terms = query.lower().split()
    direct_hits = []
    for result in results:
        meta = store.meta(result.page_id)
        path = stable_indexed_document_path(
            meta,
            pages_dir=CHRONOVISOR_ROOT / "pages",
            system_dir=CHRONOVISOR_ROOT / "system",
        )
        if path is None:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError, UnicodeDecodeError:
            continue
        snippet = _extract_snippet(content, query_terms)
        identity = meta or {}
        direct_hits.append(
            {
                "page_id": result.page_id,
                **({"uid": identity["uid"]} if identity.get("uid") else {}),
                "title": result.title,
                "updated": result.updated,
                "score": round(result.score, 4),
                "snippets": [snippet] if snippet else [],
                "tags": store.tags(result.page_id),
            }
        )
    return direct_hits


def _expanded_search_hits(
    direct_hits: list[dict[str, Any]],
    *,
    depth: int,
    store: Any,
    tag_filter: list[str],
    tag_match: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand direct hits through one-hop links while preserving filters."""

    expanded_hits: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    if depth <= 0 or not direct_hits:
        return expanded_hits, edges
    seen = {hit["page_id"] for hit in direct_hits}
    target_tags = set(tag_filter)
    for hit in direct_hits:
        taken = 0
        for link in store.outlinks(hit["page_id"]):
            if taken >= _EXPANDED_PER_HIT or len(expanded_hits) >= _EXPANDED_TOTAL:
                break
            if link in seen:
                continue
            meta = store.meta(link)
            if meta is None:
                continue
            if tag_filter:
                link_tags = set(store.tags(link))
                matches = (
                    bool(target_tags & link_tags)
                    if tag_match == "any"
                    else target_tags.issubset(link_tags)
                )
                if not matches:
                    continue
            seen.add(link)
            taken += 1
            identity = meta
            expanded_hits.append(
                {
                    "page_id": link,
                    **({"uid": identity["uid"]} if identity.get("uid") else {}),
                    "title": meta["title"],
                    "updated": meta["updated"],
                    "distance": 1,
                    "via": [hit["page_id"]],
                    "score": round(hit["score"] * 0.5, 4),
                    "reason": "linked from direct hit",
                    "tags": store.tags(link),
                }
            )
            edges.append(
                {
                    "from": hit["page_id"],
                    "to": link,
                    "from_uid": hit.get("uid"),
                    "to_uid": identity.get("uid"),
                    "type": "wikilink",
                }
            )
    return expanded_hits, edges


def _search_filters_applied(
    *,
    folder: str | None,
    updated_after: str | None,
    updated_before: str | None,
    tag_filter: list[str],
    tag_match: str,
) -> dict[str, Any]:
    """Render only filters that materially constrained this search."""

    filters: dict[str, Any] = {}
    if folder:
        filters["folder"] = folder
    if updated_after:
        filters["updated_after"] = updated_after
    if updated_before:
        filters["updated_before"] = updated_before
    if tag_filter:
        filters["tags"] = tag_filter
        filters["tag_match"] = tag_match if tag_match in {"all", "any"} else "all"
    return filters


def _extract_snippet(content: str, terms: list[str], max_len: int = 150) -> str | None:
    """Extract a relevant snippet from content."""
    _meta, body = _frontmatter_parse(content)
    body = body.strip()

    body_lower = body.lower()
    for term in terms:
        idx = body_lower.find(term)
        if idx != -1:
            start = max(0, idx - 50)
            end = min(len(body), idx + max_len)
            snippet = body[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(body):
                snippet = snippet + "..."
            return snippet
    return None


@mcp.tool()
def chronovisor_read(
    page: str,
    offset: int = 0,
    max_chars: int = READ_MAX_CHARS,
) -> str:
    """Read a wiki page with outlinks and backlinks.

    Searches pages/ first, then system/ for system files. Large pages are
    returned in windows: when ``truncated`` is true, call again with
    ``offset=next_offset`` or jump to a heading offset from ``sections``.

    Args:
        page: Page ID (filename without .md extension)
        offset: Character offset into the page body to start from.
        max_chars: Maximum characters of content to return (capped at 40000).
    """
    store = get_store()
    store.refresh_if_stale()

    try:
        path = _find_page_with_alias(page, require_stable=False)
    except PageRegistryError, ValueError:
        return json.dumps({"error": "Page registry unavailable"})
    if path is None:
        return json.dumps({"error": f"Page '{page}' not found"})

    canonical_page_id = path.stem
    resolved_identity = None
    try:
        resolved_identity = PageRegistry(CHRONOVISOR_ROOT).resolve(page)
        if resolved_identity is None:
            resolved_identity = PageRegistry(CHRONOVISOR_ROOT).resolve(
                canonical_page_id
            )
    except PageRegistryError:
        resolved_identity = None
    indexed_meta = (
        store.meta(canonical_page_id)
        if callable(getattr(store, "meta", None))
        else None
    )
    canonical_uid = (
        str(resolved_identity.get("uid"))
        if isinstance(resolved_identity, dict) and resolved_identity.get("uid")
        else str(indexed_meta.get("uid") or "")
        if isinstance(indexed_meta, dict)
        else ""
    )
    content = path.read_text()
    outlinks = store.outlinks(canonical_page_id)
    backlinks = store.backlinks(canonical_page_id)
    return json.dumps(
        {
            "page_id": canonical_page_id,
            **({"uid": canonical_uid} if canonical_uid else {}),
            **(
                {"alias": {"requested": page, "target": canonical_page_id}}
                if canonical_page_id != page
                else {}
            ),
            **_content_window(content, offset, max_chars),
            "outlinks": outlinks,
            "backlinks": backlinks,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_search(
    query: str,
    depth: int = 1,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    tags: list[str] | None = None,
    tag_match: str = "all",
) -> str:
    """Search wiki pages with BM25 and link expansion.
    Returns direct_hits (pages matching query) and expanded_hits (linked pages).
    Args:
        query: Search query string
        depth: How many link-hops to follow (0=direct only, 1=one hop, default 1)
        folder: Filter by folder prefix (e.g. "career", "project")
        updated_after: Filter pages updated after this date (ISO format)
        updated_before: Filter pages updated before this date (ISO format)
        sort_by: Sort order - "relevance" (default), "date", or "title"
        tags: Filter results to pages whose frontmatter ``tags:`` field
            includes these values. Tags must be the full prefixed form
            (e.g. ``d/ai-industry``). Empty list / None disables the filter.
        tag_match: ``"all"`` (default) requires every tag in ``tags`` to be
            present; ``"any"`` matches if at least one tag overlaps. Ignored
            when ``tags`` is empty / None.
    """
    from chronovisor.search.search import last_search_trace
    from chronovisor.search.search import search as run_search

    store = get_store()
    store.refresh_if_stale()

    results, search_mode = run_search(
        query=query,
        top_n=10,
        folder=folder,
        updated_after=updated_after,
        updated_before=updated_before,
        sort_by=sort_by,
    )
    retrieval_trace = last_search_trace()
    results, tag_filter = _filter_search_results(
        results,
        store=store,
        tags=tags,
        tag_match=tag_match,
    )

    direct_hits = _direct_search_hits(
        results,
        query=query,
        store=store,
    )

    expanded_hits, edges = _expanded_search_hits(
        direct_hits,
        depth=depth,
        store=store,
        tag_filter=tag_filter,
        tag_match=tag_match,
    )
    filters_applied = _search_filters_applied(
        folder=folder,
        updated_after=updated_after,
        updated_before=updated_before,
        tag_filter=tag_filter,
        tag_match=tag_match,
    )
    return json.dumps(
        {
            "query": query,
            "depth": depth,
            "search_mode": search_mode,
            "filters_applied": filters_applied,
            "retrieval": retrieval_trace,
            "direct_hits": direct_hits,
            "expanded_hits": expanded_hits,
            "edges": edges,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_init() -> str:
    """Return user-profile, lessons-learned, page count and queued capture jobs."""
    store = get_store()
    store.refresh_if_stale()
    jobs = snapshot()
    return json.dumps(
        {
            "status": {
                "page_count": store.page_count(include_system=False),
                "queued_capture_jobs": jobs["by_status"].get("queued", 0),
            },
            "system_pages": {
                page: json.loads(chronovisor_read(page))
                for page in ("user-profile", "lessons-learned")
            },
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_status() -> str:
    """Return page count, index freshness and capture status."""
    store = get_store()
    store.refresh_if_stale()
    indexes = {}
    for name in ("pages.json", "lexical.sqlite"):
        path = CHRONOVISOR_ROOT / ".index" / name
        indexes[name] = {
            "exists": path.is_file(),
            "mtime": path.stat().st_mtime if path.is_file() else None,
        }
    return json.dumps(
        {
            "page_count": store.page_count(include_system=False),
            "indexes": indexes,
            "capture_jobs": snapshot(),
            "capture_status": capture_status.snapshot(
                capture_status.status_path(CHRONOVISOR_ROOT)
            ),
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_record(
    content: str,
    session_id: str | None = None,
    keywords: list[str] | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Save an explicit memory to raw/ and return only publication metadata.

    Keywords are stored as raw_keywords in frontmatter. Empty keywords and
    keywords containing inline-list delimiters or control characters are
    excluded. Repeating the same content and idempotency key reuses the first
    atomically published raw.
    """
    result = record_raw(
        content,
        session_id=session_id,
        keywords=keywords,
        idempotency_key=idempotency_key,
    )
    metadata = {"saved": result["saved"]}
    try:
        metadata["path"] = (
            Path(result["path"])
            .resolve(strict=False)
            .relative_to(CHRONOVISOR_ROOT.resolve())
            .as_posix()
        )
    except OSError, ValueError:
        pass
    if "deduplicated" in result:
        metadata["deduplicated"] = result["deduplicated"]
    return json.dumps(metadata, ensure_ascii=False)


def main() -> None:
    with contextlib.suppress(Exception):
        get_store().refresh()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
