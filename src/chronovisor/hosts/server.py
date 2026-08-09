"""Chronovisor MCP Server."""

import contextlib
import hashlib
import json
import os
import re
import secrets
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import Context, FastMCP
except ModuleNotFoundError:
    from mcp.server.mcpserver import Context
    from mcp.server.mcpserver import MCPServer as FastMCP
from chronovisor.core.durable_state import fsync_directory as _fsync_directory
from chronovisor.core.frontmatter import parse as _frontmatter_parse
from chronovisor.core.frontmatter import patch as _frontmatter_patch
from chronovisor.core.link_fix import extract_targets as _extract_targets
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    LOG_FILE,
    RAW_DIR,
    SYSTEM_DIR,
    find_page,
    init_chronovisor,
)
from chronovisor.ingest.page_registry import PageRegistry, PageRegistryError
from chronovisor.raw.save_transaction import parse_save_transaction_receipt
from chronovisor.search.index_store import get_store

mcp = FastMCP(
    "chronovisor",
    instructions=(
        "Chronovisor is your structured knowledge base. "
        "Use chronovisor_init at session start, chronovisor_search during "
        "conversation, and chronovisor_read for full pages with backlinks. "
        "Before finishing an answer, call chronovisor_recall_used with the "
        "forwarded decision and session IDs for every recalled page that "
        "materially affected the answer; never mark exposure-only pages used."
    ),
)


def _mcp_client_host(ctx: Context | None) -> str:
    """Map an MCP connection's client identity onto a Chronovisor host."""

    if ctx is None:
        return ""
    candidates: list[str] = []
    with contextlib.suppress(Exception):
        candidates.append(str(ctx.client_id or ""))
    with contextlib.suppress(Exception):
        client_params = ctx.session.client_params
        client_info = getattr(client_params, "clientInfo", None) or getattr(
            client_params,
            "client_info",
            None,
        )
        candidates.append(str(getattr(client_info, "name", "") or ""))
    folded = " ".join(candidates).casefold()
    if "claude" in folded:
        return "claude-code"
    if "codex" in folded or "openai" in folded:
        return "codex"
    return ""


def _record_mcp_field_activity(
    *,
    ctx: Context | None,
    session_id: str | None,
    page_ids: list[str],
    activity_kind: str,
) -> dict[str, Any]:
    """Best-effort bridge from actual MCP use to the live Recall Field."""

    if ctx is None:
        return {}
    try:
        from chronovisor.recall.recall_field import record_mcp_activity

        return record_mcp_activity(
            host=_mcp_client_host(ctx),
            session_id=session_id or "",
            page_ids=page_ids,
            activity_kind=activity_kind,
        )
    except Exception:
        return {"status": "error"}


def _parse_frontmatter(text: str) -> dict:
    """Extract frontmatter from markdown.

    Thin wrapper around :func:`frontmatter.parse` that returns just the
    metadata dict (no body). Behaves as a strict superset of the legacy
    scalar-only parser: scalar values still come back as ``str``, while
    inline/block lists are decoded as ``list[str]``.
    """
    meta, _ = _frontmatter_parse(text)
    return meta


def _extract_wiki_links(text: str) -> list[str]:
    """Extract [[wiki-link]] references from text, normalized to page_id.

    Code fence / frontmatter / inline code 内のリンクは除外される。
    ``[[foo|label]]`` / ``[[foo#sec]]`` は ``foo`` に normalize される。
    """
    return _extract_targets(text, strip=True)


def _find_backlinks(page_id: str) -> list[str]:
    """Find pages that link to the given page.

    Backed by the persistent IndexStore. Callers expect scan-order
    (`pages/` then `system/`, rglob within each), source-deduped, and
    edges recorded for non-existent targets too — all preserved by
    `IndexStore._rebuild_backlinks`.

    Tool handlers must call ``get_store().refresh()`` once at entry;
    this helper does *not* refresh on its own to avoid redundant scans
    when called multiple times per tool invocation.
    """
    return get_store().backlinks(page_id)


def _page_metadata(path: Path) -> dict:
    """Extract metadata from a page file."""
    content = path.read_text()
    fm = _parse_frontmatter(content)
    return {
        "page_id": path.stem,
        "title": fm.get("title", path.stem),
        "updated": fm.get("updated", "unknown"),
    }


def _find_page_with_alias(page_id: str) -> Path | None:
    """Resolve a canonical page or a durable legacy page-id alias."""

    registry = PageRegistry(CHRONOVISOR_ROOT)
    try:
        registry_path = registry.path_for(page_id)
        if registry_path is not None:
            return registry_path
    except PageRegistryError:
        # Once the registry exists, ambiguous/corrupt identity must fail closed.
        # Falling back to stem lookup could select an arbitrary duplicate.
        if registry.path.exists():
            return None
    path = find_page(page_id)
    if path is not None:
        return path
    try:
        from chronovisor.core.alias_store import resolve_alias_path

        return resolve_alias_path(page_id)
    except Exception:
        # Read paths remain fail-closed to a normal not-found result when the
        # optional alias ledger is absent or malformed.
        return None


@mcp.tool()
def chronovisor_read(
    page: str,
    session_id: str | None = None,
    decision_id: str | None = None,
    ctx: Context = None,
) -> str:
    """Read a wiki page with outlinks and backlinks.

    Searches pages/ first, then system/ for system files.

    Args:
        page: Page ID (filename without .md extension)
        session_id: Optional session id for recall pull feedback.
        decision_id: Optional automatic-Recall decision id for turn tracing.
    """
    store = get_store()
    store.refresh()

    path = _find_page_with_alias(page)
    if not path:
        # Check system/ directory
        path = SYSTEM_DIR / f"{page}.md"
    if not path or not path.exists():
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
    outlinks = store.outlinks(canonical_page_id) or _extract_wiki_links(content)
    backlinks = store.backlinks(canonical_page_id)
    field_activity = _record_mcp_field_activity(
        ctx=ctx,
        session_id=session_id,
        page_ids=[canonical_page_id],
        activity_kind="read",
    )
    _append_pull_log(
        {
            "type": "read",
            "stage": "read",
            "session_id": session_id or "",
            "decision_id": decision_id or "",
            "page_id": canonical_page_id,
            **({"host": field_activity["host"]} if field_activity.get("host") else {}),
            **(
                {"field_session_hash": field_activity["session_hash"]}
                if field_activity.get("session_hash")
                else {}
            ),
            **({"page_uid": canonical_uid} if canonical_uid else {}),
            **({"requested_page_id": page} if canonical_page_id != page else {}),
        }
    )

    return json.dumps(
        {
            "page_id": canonical_page_id,
            **({"uid": canonical_uid} if canonical_uid else {}),
            **(
                {"alias": {"requested": page, "target": canonical_page_id}}
                if canonical_page_id != page
                else {}
            ),
            "content": content,
            "outlinks": outlinks,
            "backlinks": backlinks,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_index(limit: int = 50, cursor: int = 0) -> str:
    """Return structured catalog of wiki pages with pagination.

    Args:
        limit: Max number of entries to return (default 50)
        cursor: Offset for pagination (default 0)
    """
    store = get_store()
    store.refresh()
    entries = store.all_pages_meta(include_system=False)
    try:
        registry_state = PageRegistry(CHRONOVISOR_ROOT).load()
    except PageRegistryError:
        registry_state = PageRegistry.empty()
    for entry in entries:
        if entry.get("uid"):
            continue
        uid = registry_state.get("keys", {}).get(
            str(entry.get("page_id") or "").casefold()
        )
        if uid:
            entry["uid"] = uid
    total = len(entries)
    sliced = entries[cursor : cursor + limit]

    return json.dumps(
        {
            "total": total,
            "cursor": cursor,
            "limit": limit,
            "has_more": cursor + limit < total,
            "pages": sliced,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_log(limit: int = 20) -> str:
    """Return recent change history.

    Args:
        limit: Number of recent log entries to return
    """
    if not LOG_FILE.exists():
        return json.dumps({"entries": []})

    lines = LOG_FILE.read_text().splitlines()
    # Skip frontmatter and header
    log_lines = [line for line in lines if line.startswith("- ")]
    recent = log_lines[-limit:] if len(log_lines) > limit else log_lines

    return json.dumps({"entries": recent}, ensure_ascii=False)


def _raw_defer_counts() -> tuple[int, int, int]:
    """Return raw total plus semantic and operational queue-hold counts."""
    from chronovisor.raw.failure_supervisor import (
        SEMANTIC_NO_QUORUM_DEFER_REASON,
        operational_deferred_raw_files,
    )
    from chronovisor.raw.raw_store import RawStore

    raw_store = RawStore(RAW_DIR)
    reference_dir = RAW_DIR.parent / "runtime" / "raw-projections" / "parents"
    raw_paths = sorted(
        unit.path
        if unit.storage == "legacy_file"
        else raw_store.materialize_ingest(unit, reference_dir)
        for unit in raw_store.iter_units()
    )
    artifact_dir = RAW_DIR.parent / "runtime" / "raw-projections" / "artifacts"
    if artifact_dir.exists():
        raw_paths.extend(sorted(artifact_dir.glob("*.md")))
        raw_paths = sorted(dict.fromkeys(raw_paths), key=lambda path: path.name)
    deferred = operational_deferred_raw_files(raw_paths)
    semantic_deferred = sum(
        reason == SEMANTIC_NO_QUORUM_DEFER_REASON for reason in deferred.values()
    )
    operational_deferred = len(deferred) - semantic_deferred
    return len(raw_paths), semantic_deferred, operational_deferred


@mcp.tool()
def chronovisor_status() -> str:
    """Return wiki health, Ollama status, and basic statistics."""
    from chronovisor.ingest.orchestrator import get_pending_raw_files

    store = get_store()
    store.refresh()

    page_count = store.page_count(include_system=False)
    raw_total, semantic_deferred, operational_deferred = _raw_defer_counts()
    raw_pending = len(get_pending_raw_files())

    # Check Ollama via the shared httpx client used by every other Ollama
    # call in the process. The status string preserves the original
    # vocabulary ("running" / "error" / "stopped") so callers see no
    # behaviour change.
    ollama_status = "unknown"
    try:
        from chronovisor.core.ollama import client

        resp = client().get("/api/tags", timeout=3)
        ollama_status = "running" if resp.status_code == 200 else "error"
    except Exception:
        ollama_status = "stopped"

    # Find oldest/newest page from the index (mtime-sorted).
    sorted_meta = store.all_pages_meta(include_system=False)
    oldest = sorted_meta[-1] if sorted_meta else None
    newest = sorted_meta[0] if sorted_meta else None

    # Orphan count: pages with no inbound backlinks.
    orphan_count = len(store.orphans(include_system=False))
    page_types: dict[str, int] = {}
    for meta in store.all_pages_meta(include_system=False):
        page_type = str(meta.get("page_type") or "knowledge")
        page_types[page_type] = page_types.get(page_type, 0) + 1
    try:
        from chronovisor.ops.health import health_snapshot

        health = health_snapshot()
    except Exception:
        health = {}
    try:
        from chronovisor.librarian.librarian_status import build_librarian_status

        librarian = build_librarian_status(CHRONOVISOR_ROOT)
    except Exception:
        librarian = {
            "state": "BLOCKED",
            "detail": "librarian status unavailable",
        }

    return json.dumps(
        {
            "page_count": page_count,
            "raw_total": raw_total,
            "raw_pending": raw_pending,
            "semantic_deferred": semantic_deferred,
            "operational_deferred": operational_deferred,
            "raw_outstanding": raw_pending + semantic_deferred + operational_deferred,
            "orphan_count": orphan_count,
            "page_types": page_types,
            "health": health,
            "librarian": librarian,
            "ollama_status": ollama_status,
            "oldest_page": oldest,
            "newest_page": newest,
            "chronovisor_root": str(CHRONOVISOR_ROOT),
        },
        ensure_ascii=False,
    )


def _append_log(message: str) -> None:
    """Append an entry to log.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n- [{timestamp}] {message}")

    # Update log frontmatter
    content = LOG_FILE.read_text()
    today = date.today().isoformat()
    content = re.sub(r"updated: .+", f"updated: {today}", content, count=1)
    LOG_FILE.write_text(content)


def _append_pull_log(record: dict) -> bool:
    """Append pull telemetry and return a durable-write receipt."""

    try:
        from chronovisor.recall.recall_runtime import RECALL_PULL_LOG_FILE, append_jsonl

        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            **record,
        }
        append_jsonl(RECALL_PULL_LOG_FILE, record)
        return True
    except Exception:
        return False


def _validate_used_recall_decision(
    decision_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Resolve a used receipt to one durable Recall decision before accepting it."""

    from chronovisor.core.jsonl import read_jsonl
    from chronovisor.core.recall_log_schema import page_ids_from_record
    from chronovisor.recall.recall_runtime import RECALL_LOG_FILE, RECALL_PULL_LOG_FILE

    matches = [
        row
        for row in read_jsonl(RECALL_LOG_FILE, limit=10_000)
        if str(row.get("decision_id") or "") == decision_id
    ]
    if not matches:
        return {"status": "error", "error": "unknown recall decision"}
    if len(matches) != 1:
        return {"status": "error", "error": "ambiguous recall decision"}
    recall = matches[0]
    recall_session = str(recall.get("session_id") or "")
    if recall_session and session_id and session_id != recall_session:
        return {"status": "error", "error": "recall session mismatch"}
    canonical_session = session_id or recall_session
    observable_pages = page_ids_from_record(recall)
    for pull in read_jsonl(RECALL_PULL_LOG_FILE, limit=10_000):
        if str(pull.get("decision_id") or "") != decision_id:
            continue
        pull_session = str(pull.get("session_id") or "")
        if canonical_session and pull_session and pull_session != canonical_session:
            continue
        if pull.get("type") == "read":
            page_id = pull.get("page_id")
            if isinstance(page_id, str) and page_id:
                observable_pages.append(page_id)
        elif pull.get("type") == "search":
            returned = pull.get("returned_pages")
            if isinstance(returned, list):
                observable_pages.extend(
                    value for value in returned if isinstance(value, str) and value
                )
    features = recall.get("evidence_features")
    shadow = features.get("processor_shadow") if isinstance(features, dict) else None
    shadow_pages = (
        [
            value
            for value in shadow.get("committed_page_ids", [])
            if isinstance(value, str) and value
        ]
        if isinstance(shadow, dict)
        else []
    )
    return {
        "status": "ok",
        "session_id": canonical_session,
        "observable_page_ids": list(dict.fromkeys(observable_pages)),
        "processor_shadow_page_ids": list(dict.fromkeys(shadow_pages)),
    }


def _existing_used_receipt(decision_id: str, session_id: str) -> dict[str, Any] | None:
    """Return the aggregate immutable used receipts for a decision, if any."""

    from chronovisor.core.jsonl import read_jsonl
    from chronovisor.recall.recall_runtime import RECALL_PULL_LOG_FILE

    matches: list[dict[str, Any]] = []
    pages: list[str] = []
    for row in read_jsonl(RECALL_PULL_LOG_FILE, limit=10_000):
        if row.get("type") != "used":
            continue
        if str(row.get("decision_id") or "") != decision_id:
            continue
        row_session = str(row.get("session_id") or "")
        if row_session and session_id and row_session != session_id:
            continue
        matches.append(row)
        values = row.get("page_ids")
        if isinstance(values, list):
            pages.extend(value for value in values if isinstance(value, str) and value)
    if not matches:
        return None
    return {
        "event_id": str(matches[0].get("event_id") or ""),
        "event_count": len(matches),
        "page_ids": list(dict.fromkeys(pages)),
    }


def _record_search_pull(
    *,
    ctx: Context | None,
    session_id: str | None,
    decision_id: str | None,
    query: str,
    direct_hits: list[dict],
    expanded_hits: list[dict],
    provisional_hits: list[dict],
    retrieval_trace: dict,
) -> None:
    """Record search telemetry and project returned pages into the live Field."""

    direct_page_ids = [hit["page_id"] for hit in direct_hits]
    all_hits = [*direct_hits, *expanded_hits]
    field_activity = _record_mcp_field_activity(
        ctx=ctx,
        session_id=session_id,
        page_ids=direct_page_ids,
        activity_kind="search",
    )
    _append_pull_log(
        {
            "type": "search",
            "stage": "returned",
            "session_id": session_id or "",
            "decision_id": decision_id or "",
            "query": query,
            "direct_pages": direct_page_ids,
            "expanded_pages": [hit["page_id"] for hit in expanded_hits],
            "returned_pages": [hit["page_id"] for hit in all_hits],
            "direct_uids": [hit["uid"] for hit in direct_hits if hit.get("uid")],
            "expanded_uids": [hit["uid"] for hit in expanded_hits if hit.get("uid")],
            "returned_uids": [hit["uid"] for hit in all_hits if hit.get("uid")],
            "provisional_ids": [hit["provisional_id"] for hit in provisional_hits],
            "retrieval": retrieval_trace,
            **({"host": field_activity["host"]} if field_activity.get("host") else {}),
            **(
                {"field_session_hash": field_activity["session_hash"]}
                if field_activity.get("session_hash")
                else {}
            ),
        }
    )


@mcp.tool()
def chronovisor_init() -> str:
    """Initialize session: returns system pages + status in a single call.

    Replaces the need for separate chronovisor_status + 3x chronovisor_read at session start.
    Returns user-profile, current-state, lessons-learned, and basic wiki stats.
    """
    from concurrent.futures import ThreadPoolExecutor

    from chronovisor.core.ollama import is_available
    from chronovisor.ingest.orchestrator import get_pending_raw_files

    store = get_store()

    # Run the index refresh, raw-dir count, and Ollama health probe in
    # parallel. The Ollama probe is the heaviest (network) so doing it
    # alongside the on-disk scans hides its latency.
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_refresh = ex.submit(store.refresh)
        f_raw = ex.submit(_raw_defer_counts)
        f_pending = ex.submit(lambda: len(get_pending_raw_files()))
        f_ollama = ex.submit(is_available)
        f_refresh.result()
        raw_total, semantic_deferred, operational_deferred = f_raw.result()
        raw_pending = f_pending.result()
        ollama_status = "running" if f_ollama.result() else "stopped"

    page_count = store.page_count(include_system=False)
    try:
        from chronovisor.librarian.librarian_status import build_librarian_status

        librarian = build_librarian_status(CHRONOVISOR_ROOT)
    except Exception:
        librarian = {
            "state": "BLOCKED",
            "detail": "librarian status unavailable",
        }
    system_pages = {}
    for page_id in ("user-profile", "current-state", "lessons-learned"):
        path = SYSTEM_DIR / f"{page_id}.md"
        if not path.exists():
            system_pages[page_id] = {"error": f"Page '{page_id}' not found"}
            continue
        content = path.read_text()
        system_pages[page_id] = {
            "page_id": page_id,
            "content": content,
            "outlinks": store.outlinks(page_id) or _extract_wiki_links(content),
            "backlinks": store.backlinks(page_id),
        }

    return json.dumps(
        {
            "status": {
                "page_count": page_count,
                "raw_total": raw_total,
                "raw_pending": raw_pending,
                "semantic_deferred": semantic_deferred,
                "operational_deferred": operational_deferred,
                "raw_outstanding": (
                    raw_pending + semantic_deferred + operational_deferred
                ),
                "ollama_status": ollama_status,
                "chronovisor_root": str(CHRONOVISOR_ROOT),
                "librarian": librarian,
            },
            "system_pages": system_pages,
        },
        ensure_ascii=False,
    )


def _filter_search_results(
    results: list[Any],
    *,
    store: Any,
    registry_row: Callable[[str], dict[str, Any]],
    tags: list[str] | None,
    tag_match: str,
    classification_notation: str | None,
    classification_status: str | None,
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

    if classification_notation or classification_status:
        filtered = []
        for result in results:
            row = registry_row(result.page_id)
            classification = row.get("classification")
            classification = classification if isinstance(classification, dict) else {}
            primary = classification.get("primary")
            primary = primary if isinstance(primary, dict) else {}
            if (
                classification_notation
                and str(primary.get("notation") or "") != classification_notation
            ):
                continue
            if (
                classification_status
                and str(row.get("classification_status") or "") != classification_status
            ):
                continue
            filtered.append(result)
        results = filtered
    return results, tag_filter


def _direct_search_hits(
    results: list[Any],
    *,
    query: str,
    store: Any,
    registry_row: Callable[[str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project ranked results into the stable MCP direct-hit shape."""

    query_terms = query.lower().split()
    direct_hits = []
    for result in results:
        path = find_page(result.page_id)
        content = path.read_text() if path else None
        snippet = (
            _extract_snippet(content, query_terms) if content is not None else None
        )
        identity = registry_row(result.page_id)
        classification = identity.get("classification")
        direct_hits.append(
            {
                "page_id": result.page_id,
                **({"uid": identity["uid"]} if identity.get("uid") else {}),
                "title": result.title,
                "updated": result.updated,
                "score": round(result.score, 4),
                "snippets": [snippet] if snippet else [],
                "tags": store.tags(result.page_id),
                **(
                    {"classification": classification}
                    if isinstance(classification, dict)
                    else {}
                ),
            }
        )
    return direct_hits


def _expanded_search_hits(
    direct_hits: list[dict[str, Any]],
    *,
    depth: int,
    store: Any,
    registry_row: Callable[[str], dict[str, Any]],
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
        for link in store.outlinks(hit["page_id"]):
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
            identity = registry_row(link)
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
    classification_notation: str | None,
    classification_status: str | None,
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
    if classification_notation:
        filters["classification_notation"] = classification_notation
    if classification_status:
        filters["classification_status"] = classification_status
    return filters


@mcp.tool()
def chronovisor_search(
    query: str,
    depth: int = 1,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    semantic: bool = True,
    tags: list[str] | None = None,
    tag_match: str = "all",
    classification_notation: str | None = None,
    classification_status: str | None = None,
    session_id: str | None = None,
    decision_id: str | None = None,
    ctx: Context = None,
) -> str:
    """Search wiki pages with BM25 + semantic search and link expansion.
    Returns direct_hits (pages matching query) and expanded_hits (linked pages).
    Args:
        query: Search query string
        depth: How many link-hops to follow (0=direct only, 1=one hop, default 1)
        folder: Filter by folder prefix (e.g. "career", "project")
        updated_after: Filter pages updated after this date (ISO format)
        updated_before: Filter pages updated before this date (ISO format)
        sort_by: Sort order - "relevance" (default), "date", or "title"
        semantic: Use semantic search when Ollama is available (default True)
        tags: Filter results to pages whose frontmatter ``tags:`` field
            includes these values. Tags must be the full prefixed form
            (e.g. ``d/ai-industry``). Empty list / None disables the filter.
        tag_match: ``"all"`` (default) requires every tag in ``tags`` to be
            present; ``"any"`` matches if at least one tag overlaps. Ignored
            when ``tags`` is empty / None.
        classification_notation: Optional exact UDC notation. While the
            classification authority is inactive this filters shadow proposals
            and the response explicitly reports that status.
        classification_status: Optional exact classification disposition.
        session_id: Optional session id for recall pull feedback.
        decision_id: Optional automatic-Recall decision id for turn tracing.
    """
    from chronovisor.core.runtime_config import load_reranker_config
    from chronovisor.search.pipeline import apply_rerank_stage
    from chronovisor.search.reranker import rerank_results
    from chronovisor.search.search import last_search_trace
    from chronovisor.search.search import search as run_search

    store = get_store()
    store.refresh()
    reranker_cfg = load_reranker_config()
    rerank_allowed = reranker_cfg.enabled and sort_by == "relevance"
    search_top_n = max(10, reranker_cfg.top_n) if rerank_allowed else 10

    results, search_mode = run_search(
        query=query,
        top_n=search_top_n,
        folder=folder,
        updated_after=updated_after,
        updated_before=updated_before,
        sort_by=sort_by,
        semantic=semantic,
    )
    retrieval_trace = last_search_trace()
    registry = PageRegistry(CHRONOVISOR_ROOT)
    try:
        registry_state = registry.load()
    except PageRegistryError:
        registry_state = PageRegistry.empty()
    try:
        from chronovisor.classification.classification import (
            classification_authority_status,
        )

        classification_authority = classification_authority_status(CHRONOVISOR_ROOT)
    except Exception:
        classification_authority = {
            "active": False,
            "reason": "classification_authority_unavailable",
        }

    def registry_row(page_id: str) -> dict:
        uid = registry_state.get("keys", {}).get(page_id.casefold())
        row = registry_state.get("pages", {}).get(uid) if uid else None
        return dict(row) if isinstance(row, dict) else {}

    results, tag_filter = _filter_search_results(
        results,
        store=store,
        registry_row=registry_row,
        tags=tags,
        tag_match=tag_match,
        classification_notation=classification_notation,
        classification_status=classification_status,
    )

    rerank_stage = apply_rerank_stage(
        query,
        results,
        reranker_config=reranker_cfg,
        rerank_results=rerank_results,
        sort_by=sort_by,
    )
    results = rerank_stage.results
    reranker_meta = rerank_stage.metadata
    if rerank_stage.applied:
        search_mode = f"{search_mode}+rerank"
    results = results[:10]

    direct_hits = _direct_search_hits(
        results,
        query=query,
        store=store,
        registry_row=registry_row,
    )

    # Fail-closed semantic holds remain visible only through a separate,
    # projection-only namespace. They never compete as normal wiki pages and
    # can never be used as mutation authority.
    provisional_hits: list[dict] = []
    if (
        sort_by == "relevance"
        and not folder
        and not updated_after
        and not updated_before
        and not tag_filter
    ):
        try:
            from chronovisor.recall.provisional_recall import search_provisional

            provisional_hits = search_provisional(
                query, chronovisor_root=CHRONOVISOR_ROOT
            )
        except Exception:
            provisional_hits = []

    expanded_hits, edges = _expanded_search_hits(
        direct_hits,
        depth=depth,
        store=store,
        registry_row=registry_row,
        tag_filter=tag_filter,
        tag_match=tag_match,
    )
    filters_applied = _search_filters_applied(
        folder=folder,
        updated_after=updated_after,
        updated_before=updated_before,
        tag_filter=tag_filter,
        tag_match=tag_match,
        classification_notation=classification_notation,
        classification_status=classification_status,
    )
    _record_search_pull(
        ctx=ctx,
        session_id=session_id,
        decision_id=decision_id,
        query=query,
        direct_hits=direct_hits,
        expanded_hits=expanded_hits,
        provisional_hits=provisional_hits,
        retrieval_trace=retrieval_trace,
    )

    return json.dumps(
        {
            "query": query,
            "depth": depth,
            "search_mode": search_mode,
            "filters_applied": filters_applied,
            "classification_authority": {
                **classification_authority,
                "mode": (
                    "active" if classification_authority.get("active") else "shadow"
                ),
                "note": (
                    "classification authority adopted"
                    if classification_authority.get("active")
                    else (
                        "classification filters use non-authoritative shadow "
                        "proposals until calibration and activation"
                    )
                ),
            },
            "reranker": reranker_meta,
            "retrieval": retrieval_trace,
            "direct_hits": direct_hits,
            "provisional_hits": provisional_hits,
            "expanded_hits": expanded_hits,
            "edges": edges,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_recall_used(
    decision_id: str,
    page_ids: list[str],
    session_id: str | None = None,
    note: str = "",
) -> str:
    """Record which recalled pages materially affected the answer.

    This is the only pull-trace event that is positive learning evidence.
    Search results and page reads remain exploration telemetry. The decision
    and session are validated synchronously so a successful response is
    guaranteed to be joinable by the growth controller.
    """

    decision_id = decision_id.strip()
    if not decision_id:
        return json.dumps({"status": "error", "error": "decision_id is required"})
    pages = list(
        dict.fromkeys(
            page.strip() for page in page_ids if isinstance(page, str) and page.strip()
        )
    )[:20]
    if not pages:
        return json.dumps({"status": "error", "error": "page_ids is required"})
    validation = _validate_used_recall_decision(
        decision_id,
        str(session_id or "").strip(),
    )
    if validation.get("status") != "ok":
        return json.dumps(
            {
                "status": "error",
                "error": str(validation.get("error") or "invalid recall decision"),
                "decision_id": decision_id,
            },
            ensure_ascii=False,
        )
    canonical_session = str(validation.get("session_id") or "")
    observable_pages = {
        value
        for value in validation.get("observable_page_ids", [])
        if isinstance(value, str) and value
    }
    unobserved_pages = [page for page in pages if page not in observable_pages]
    if unobserved_pages:
        return json.dumps(
            {
                "status": "error",
                "error": "used pages were not returned, injected, or read",
                "decision_id": decision_id,
                "page_ids": unobserved_pages,
            },
            ensure_ascii=False,
        )
    existing = _existing_used_receipt(decision_id, canonical_session)
    existing_page_list = [
        value
        for value in (existing or {}).get("page_ids", [])
        if isinstance(value, str) and value
    ]
    existing_pages = set(existing_page_list)
    new_pages = [page for page in pages if page not in existing_pages]
    if existing is not None and not new_pages:
        return json.dumps(
            {
                "status": "already_recorded",
                "event_id": str(existing.get("event_id") or ""),
                "decision_id": decision_id,
                "page_ids": existing.get("page_ids") or [],
                "learning_join": "ready",
            },
            ensure_ascii=False,
        )
    event_identity = json.dumps(
        [decision_id, canonical_session, sorted(new_pages)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    event_id = hashlib.sha256(event_identity.encode()).hexdigest()[:32]
    recorded = _append_pull_log(
        {
            "type": "used",
            "stage": "used",
            "event_id": event_id,
            "session_id": canonical_session,
            "decision_id": decision_id,
            "page_ids": new_pages,
            "note": note[:500],
        }
    )
    if recorded is not True:
        return json.dumps(
            {
                "status": "error",
                "error": "used receipt was not durably recorded",
                "decision_id": decision_id,
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "status": "recorded",
            "event_id": event_id,
            "decision_id": decision_id,
            "page_ids": [*existing_page_list, *new_pages],
            "new_page_ids": new_pages,
            "learning_join": "ready",
            "processor_shadow_covered_page_ids": sorted(
                set(pages) & set(validation.get("processor_shadow_page_ids") or [])
            ),
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_reindex() -> str:
    """Rebuild search embeddings for all wiki pages.

    Call this after bulk changes or to initialize semantic search.
    """
    from chronovisor.core.runtime_config import load_search_embedding_config
    from chronovisor.search.search import update_embeddings

    config = load_search_embedding_config()
    count = update_embeddings()
    if config.enabled and config.backend == "nemotron_service":
        return json.dumps(
            {
                "status": "queued",
                "backend": config.backend,
                "message": "immutable semantic generation rebuild queued",
            }
        )
    if count == 0:
        return json.dumps(
            {
                "status": "skipped",
                "message": "Ollama not available or no pages to update",
            }
        )
    return json.dumps({"status": "ok", "pages_updated": count})


def _extract_snippet(content: str, terms: list[str], max_len: int = 150) -> str | None:
    """Extract a relevant snippet from content."""
    # Skip frontmatter
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3 :].strip()

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


_RAW_PREFIX_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_RAW_SLUG_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_RAW_UUIDISH_RE = re.compile(r"^[0-9a-f]{8,}(?:-[0-9a-f]{4,})*$", re.IGNORECASE)
_RAW_TOPIC_STOPWORDS = {
    "and",
    "code",
    "codex",
    "claude",
    "memory",
    "save",
    "session",
    "the",
}
_RAW_SOURCE_PREFIXES = ("claude-code", "codex", "ingest")


def _sanitize_raw_prefix(prefix: str) -> str:
    """Sanitize a caller-supplied prefix so it can't break out of RAW_DIR.

    Slashes, ``..``, control chars, etc. would let a malicious or
    malformed ``session_id`` (passed straight from an MCP client) place
    the raw file outside ``RAW_DIR``. Strip everything but ASCII
    alphanumerics, dash, and underscore; clamp length to 64 chars.
    """
    if not prefix:
        return ""
    cleaned = _RAW_PREFIX_RE.sub("-", prefix.strip())[:64]
    cleaned = cleaned.strip("-_")
    return f"-{cleaned}" if cleaned else ""


def _sanitize_raw_component(value: str, *, max_len: int = 64) -> str:
    cleaned = _RAW_PREFIX_RE.sub("-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-_")[:max_len].strip("-_")


def _raw_source_label(session_id: str | None) -> str:
    if not session_id:
        return ""
    cleaned = _sanitize_raw_component(session_id, max_len=64)
    for source in _RAW_SOURCE_PREFIXES:
        if cleaned == source or cleaned.startswith(f"{source}-"):
            return source
    if _RAW_UUIDISH_RE.match(cleaned):
        return "session"
    return cleaned[:28].strip("-_")


def _raw_topic_slug(
    content: str, keywords: list[str] | None = None, *, max_len: int = 56
) -> str:
    """Create a readable raw filename slug while keeping ASCII-only safety."""

    parts: list[str] = []
    if keywords:
        for keyword in keywords:
            for match in _RAW_SLUG_TOKEN_RE.finditer(keyword.lower()):
                token = match.group(0)
                if (
                    len(token) < 2
                    or token in _RAW_TOPIC_STOPWORDS
                    or _RAW_UUIDISH_RE.match(token)
                ):
                    continue
                parts.append(token)
                slug = "-".join(parts)
                if len(slug) >= max_len:
                    return slug[:max_len].strip("-")
        if parts:
            return "-".join(parts)[:max_len].strip("-")

    candidates: list[str] = []
    for line in content.splitlines():
        stripped = line.strip(" #-\t")
        if not stripped:
            continue
        lower = stripped.lower()
        if lower in {
            "codex session memory save",
            "claude code session memory save",
            "memory",
            "writer reason",
            "rejected keywords",
        }:
            continue
        if lower.startswith(
            (
                "source:",
                "session id:",
                "cwd:",
                "session file:",
                "lines:",
                "memory writer model:",
                "generated at:",
                "raw_keywords:",
            )
        ):
            continue
        candidates.append(stripped)
        break

    for candidate in candidates:
        for match in _RAW_SLUG_TOKEN_RE.finditer(candidate.lower()):
            token = match.group(0)
            if (
                len(token) < 2
                or token in _RAW_TOPIC_STOPWORDS
                or _RAW_UUIDISH_RE.match(token)
            ):
                continue
            parts.append(token)
            slug = "-".join(parts)
            if len(slug) >= max_len:
                return slug[:max_len].strip("-")
    return "-".join(parts)[:max_len].strip("-")


_RAW_ALLOC_MAX_RETRIES = 32
_RAW_IDEMPOTENCY_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$")


def _raw_readable_component(prefix: str, topic_slug: str) -> str:
    source = _raw_source_label(prefix)
    topic = _sanitize_raw_component(topic_slug, max_len=56)
    name_parts = [part for part in (source, topic) if part]
    return f"-{'-'.join(name_parts)}" if name_parts else _sanitize_raw_prefix(prefix)


def _raw_candidate_path(prefix: str = "", topic_slug: str = "") -> Path:
    from chronovisor.raw.raw_segment import capture_date
    from chronovisor.raw.raw_store import raw_layout_mode

    readable = _raw_readable_component(prefix, topic_slug)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(4)  # 8 hex chars / 32 bits
    if raw_layout_mode(chronovisor_root=RAW_DIR.parent) == "v2":
        day_dir = RAW_DIR / capture_date()
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir / f"manual-{ts}{readable}-{suffix}.md"
    return RAW_DIR / f"{ts}{readable}-{suffix}.md"


def _allocate_raw_path(prefix: str = "", topic_slug: str = "") -> Path:
    """Reserve a unique, non-ingestable staging path for a raw publish.

    The old allocator reserved the *final* ``raw/*.md`` filename by creating
    a zero-byte file.  Concurrent ingest could discover that file before the
    caller filled it.  Staging files are now dot-prefixed ``*.tmp`` entries,
    so every raw glob sees either no entry or a complete final file.
    """
    readable = _raw_readable_component(prefix, topic_slug)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for _ in range(_RAW_ALLOC_MAX_RETRIES):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = secrets.token_hex(4)  # 8 hex chars / 32 bits
        path = RAW_DIR / f".{ts}{readable}-{suffix}.tmp"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return path
        except FileExistsError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"could not allocate unique raw path after "
        f"{_RAW_ALLOC_MAX_RETRIES} retries: {last_err}"
    )


def _link_raw_no_replace(staging: Path, target: Path) -> None:
    """Atomically publish ``staging`` at ``target`` without replacement."""
    os.link(staging, target)


def _publish_raw(content: str, *, prefix: str = "", topic_slug: str = "") -> Path:
    """Write a complete raw entry, then atomically expose its final name.

    A hard link within ``RAW_DIR`` is the portable no-replace primitive we
    need here: the target appears atomically and ``EEXIST`` never overwrites a
    prior raw.  The hidden staging inode is unlinked after publication.
    """
    staging = _allocate_raw_path(prefix=prefix, topic_slug=topic_slug)
    published: Path | None = None
    try:
        with staging.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        last_err: Exception | None = None
        for _ in range(_RAW_ALLOC_MAX_RETRIES):
            target = _raw_candidate_path(prefix=prefix, topic_slug=topic_slug)
            try:
                _link_raw_no_replace(staging, target)
                _fsync_directory(target.parent)
                published = target
                break
            except FileExistsError as exc:
                last_err = exc
        if published is None:
            raise RuntimeError(
                "could not publish unique raw path after "
                f"{_RAW_ALLOC_MAX_RETRIES} retries: {last_err}"
            )
        return published
    finally:
        with contextlib.suppress(FileNotFoundError):
            staging.unlink()


def _publish_raw_idempotent(
    content: str,
    *,
    idempotency_key: str,
    prefix: str = "",
    topic_slug: str = "",
) -> tuple[Path, bool]:
    """Atomically publish one complete raw per idempotency key.

    The first file wins. Exact-byte retries are reused; nondeterministic saver
    retries are reused only when both self-verifying transaction receipts
    identify the same source interval. Collisions and corrupt receipts fail.
    """
    if not _RAW_IDEMPOTENCY_RE.fullmatch(idempotency_key):
        raise ValueError(
            "idempotency_key must contain only ASCII letters, digits, dash, "
            "or underscore and be at most 160 characters"
        )
    staging = _allocate_raw_path(prefix=prefix, topic_slug=topic_slug)
    target = RAW_DIR / f"save-{idempotency_key}.md"
    try:
        with staging.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _link_raw_no_replace(staging, target)
        except FileExistsError as exc:
            try:
                existing = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise RuntimeError(
                    "idempotent raw target exists but cannot be verified"
                ) from exc
            if existing != content:
                incoming_receipt = parse_save_transaction_receipt(content)
                existing_receipt = parse_save_transaction_receipt(existing)
                if (
                    incoming_receipt is None
                    or existing_receipt is None
                    or incoming_receipt.transaction != existing_receipt.transaction
                    or incoming_receipt.transaction.idempotency_key != idempotency_key
                ):
                    raise RuntimeError(
                        "idempotency key collision with different or corrupt raw content"
                    ) from exc
            return target, True
        _fsync_directory(RAW_DIR)
        return target, False
    finally:
        with contextlib.suppress(FileNotFoundError):
            staging.unlink()


@mcp.tool()
def chronovisor_ingest(content: str, force: bool = True) -> str:
    """Persist raw content and trigger ingest.

    The previous implementation called ``start_ingest`` directly, bypassing
    the orchestrator lock and never persisting the content. We now write
    to ``raw/`` first (so a partial failure can still be retried) and then
    invoke the orchestrator.

    Args:
        content: Raw session data to structure into wiki pages.
        force: If True (default, matching the historical contract of
            "ingest now"), trigger ingest even if pending count is below
            ``INGEST_THRESHOLD``. Pass ``force=False`` to defer to the
            normal threshold check.
    """
    from chronovisor.ingest.orchestrator import run_pending_ingest

    path = _publish_raw(
        content,
        prefix="ingest",
        topic_slug=_raw_topic_slug(content),
    )

    result = run_pending_ingest(force=force)
    payload = {
        "saved": path.name,
        "ingest": result,
    }
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def chronovisor_check() -> str:
    """Run lint checks on the wiki. Returns list of detected issues.

    Issues include: broken links, stale pages, orphan pages, duplicates.
    Does NOT auto-fix anything.
    """
    from chronovisor.ops.lint import check, issue_lane, summarize_issues

    issues = check()
    issue_limit = 40

    def compact(issue: dict) -> dict:
        detail = str(issue.get("detail") or "")
        if len(detail) > 180:
            detail = detail[:177].rstrip() + "..."
        return {
            "type": issue.get("type"),
            "severity": issue.get("severity"),
            "lane": issue_lane(issue),
            "page": issue.get("page"),
            "detail": detail,
            "auto_fixable": bool(issue.get("auto_fixable")),
        }

    return json.dumps(
        {
            "total_issues": len(issues),
            "summary": summarize_issues(issues),
            "issues": [compact(issue) for issue in issues[:issue_limit]],
            "omitted_issues": max(0, len(issues) - issue_limit),
            "output_budget": {"issue_limit": issue_limit, "detail_chars": 180},
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_apply(dry_run: bool = False, fuzzy: bool = True) -> str:
    """Apply safe auto-fixes to the wiki.

    broken_link は fuzzy match で近い page_id に置換し、見つからなければ
    plaintext 化する。system/ 配下に実在する target は false positive と見なして
    書き換えない。Contradictions / orphans / stale などは flag のまま残す。
    Run chronovisor_check first to see what will be fixed.

    Args:
        dry_run: True なら実際には書き込まず actions のプレビューだけ返す。
        fuzzy: False にすると broken_link の自動書き換えを無効化する (より保守的)。
    """
    from chronovisor.ops.lint import (
        apply_safe_fixes,
        check,
        issue_lane,
        summarize_issues,
        write_repair_queue,
    )

    issues = check()
    snapshot = {"status": "skipped", "reason": "dry_run"} if dry_run else None
    if snapshot is None:
        try:
            from chronovisor.ops.snapshot import snapshot_chronovisor

            snapshot = snapshot_chronovisor("before chronovisor_apply")
        except Exception as exc:
            snapshot = {"status": "error", "error": str(exc)}
    actions = apply_safe_fixes(issues, dry_run=dry_run, fuzzy=fuzzy)
    remaining = [i for i in issues if not i.get("auto_fixable")]
    try:
        repair_queue = str(write_repair_queue(remaining))
    except Exception:
        repair_queue = None
    issue_limit = 40

    def compact(issue: dict) -> dict:
        detail = str(issue.get("detail") or "")
        if len(detail) > 180:
            detail = detail[:177].rstrip() + "..."
        return {
            "type": issue.get("type"),
            "severity": issue.get("severity"),
            "lane": issue_lane(issue),
            "page": issue.get("page"),
            "detail": detail,
            "auto_fixable": bool(issue.get("auto_fixable")),
        }

    return json.dumps(
        {
            "actions_taken": actions,
            "snapshot": snapshot,
            "summary": summarize_issues(remaining),
            "remaining_issues": [compact(issue) for issue in remaining[:issue_limit]],
            "omitted_remaining_issues": max(0, len(remaining) - issue_limit),
            "output_budget": {"issue_limit": issue_limit, "detail_chars": 180},
            "repair_queue": repair_queue,
            "dry_run": dry_run,
            "fuzzy": fuzzy,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_jobs(job_id: str | None = None) -> str:
    """Check job progress.

    Args:
        job_id: Specific job ID to check. If None, returns recent jobs.
    """
    from chronovisor.core.jobs import job_store

    if job_id:
        job = job_store.get(job_id)
        if not job:
            from chronovisor.core.background_jobs import get_job

            durable = get_job(job_id)
            if durable is None:
                return json.dumps({"error": f"Job '{job_id}' not found"})
            return json.dumps(
                {
                    "job_id": durable.get("job_id"),
                    "status": durable.get("status"),
                    "processor": durable.get("name"),
                    "stage": "durable-worker",
                    "created_at": durable.get("created_at"),
                    "updated_at": durable.get("updated_at"),
                    "attempts": durable.get("attempts"),
                    "output_tail": durable.get("output_tail"),
                    "error": (
                        durable.get("output_tail")
                        if durable.get("status") in {"failed", "quarantined"}
                        else None
                    ),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "processor": job.processor,
                "stage": job.stage,
                "total_ops": job.total_ops,
                "completed_ops": job.completed_ops,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "pages_created": job.pages_created,
                "pages_updated": job.pages_updated,
                "result": job.result,
                "error": job.error,
            },
            ensure_ascii=False,
        )

    jobs = job_store.recent()
    return json.dumps(
        {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status.value,
                    "processor": j.processor,
                    "created_at": j.created_at,
                }
                for j in jobs
            ]
        },
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_deep_dive(
    query: str,
    max_iterations: int = 3,
    fanout: int = 5,
    semantic: bool = True,
    use_llm: bool = True,
    background: bool = True,
    engine: str = "v2",
) -> str:
    """Run agentic search -> read -> wikilink -> requery retrieval.

    Args:
        query: Initial research query.
        max_iterations: Search/read/requery loops to run, capped at 5.
        fanout: Direct and linked pages to inspect per loop, capped at 10.
        semantic: Use hybrid semantic search for each loop.
        use_llm: Ask the local heavy model to propose follow-up queries.
            When unavailable, the tool falls back to deterministic requery.
        background: When True, return a job_id immediately and inspect with
            chronovisor_jobs(job_id). When False, run synchronously and return result.
    """
    from chronovisor.research.deep_retrieval import (
        run_deep_dive,
        run_deep_dive_v2,
        start_deep_dive,
    )

    if background:
        job_id = start_deep_dive(
            query,
            max_iterations=max_iterations,
            fanout=fanout,
            semantic=semantic,
            use_llm=use_llm,
            engine=engine,
        )
        return json.dumps(
            {
                "status": "started",
                "job_id": job_id,
                "processor": "deep-retrieval",
                "query": query,
            },
            ensure_ascii=False,
        )

    runner = run_deep_dive_v2 if engine == "v2" else run_deep_dive
    result = runner(
        query,
        max_iterations=max_iterations,
        fanout=fanout,
        semantic=semantic,
        use_llm=use_llm,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def chronovisor_research(
    query: str,
    claims: list[str] | None = None,
    challenge: bool = True,
    background: bool = True,
) -> str:
    """Run bounded source-backed research with citations and local challenge.

    The authority ladder is Wiki -> verified claims -> Raw -> Web. Web is
    unavailable unless its independent kill switches are enabled. This tool is
    asynchronous by default so the host prompt path never waits on a 35B model.

    Args:
        query: Research goal.
        claims: Optional concrete claims to classify and verify.
        challenge: Ask the configured local challenger and conditional tie-breaker.
        background: Return a durable job immediately when True.
    """
    from chronovisor.research.research_service import (
        enqueue_evidence_research,
        run_evidence_research,
    )

    if background:
        job = enqueue_evidence_research(query, claims=claims, challenge=challenge)
        return json.dumps(
            {
                "status": "started"
                if job.get("status") in {"queued", "running"}
                else job.get("status"),
                "job_id": job.get("job_id"),
                "processor": "research",
                "query": query,
                "coalesced": job.get("coalesced", False),
            },
            ensure_ascii=False,
        )
    return json.dumps(
        run_evidence_research(query, claims=claims, challenge=challenge),
        ensure_ascii=False,
    )


@mcp.tool()
def chronovisor_provenance(page: str) -> str:
    """Trace the provenance of a wiki page back to raw session data.

    Args:
        page: Page ID to trace
    """
    page_path = _find_page_with_alias(page)
    if not page_path:
        return json.dumps({"error": f"Page '{page}' not found"})

    page_mtime = datetime.fromtimestamp(page_path.stat().st_mtime)

    # Read target page metadata once (was reread inside loop and again at return)
    page_content = page_path.read_text()
    page_fm = _parse_frontmatter(page_content)
    page_title_lower = page_fm.get("title", page).lower()
    page_dehyphen = page.replace("-", " ")
    page_updated = page_fm.get("updated", "unknown")
    threshold = page_mtime + timedelta(minutes=30)

    # Find raw files that might be the source.
    # raw files are walked in mtime-descending order; once a file exceeds the
    # threshold, every subsequent file also exceeds it, so we can break.
    from chronovisor.raw.raw_store import RawStore

    raw_candidates = []
    raw_store = RawStore(RAW_DIR)
    candidates = []
    for unit in raw_store.iter_units():
        raw_mtime = (
            datetime.fromisoformat(unit.captured_at).replace(tzinfo=None)
            if unit.captured_at
            else datetime.fromtimestamp(unit.path.stat().st_mtime)
        )
        candidates.append((raw_mtime, unit))
    for raw_mtime, unit in sorted(candidates, key=lambda row: row[0], reverse=True):
        if raw_mtime > threshold:
            break
        try:
            raw_content = raw_store.read_text(unit)[:500]
        except (OSError, UnicodeError):
            continue
        raw_lower = raw_content.lower()
        if page_dehyphen in raw_lower or page_title_lower in raw_lower:
            raw_candidates.append(
                {
                    "raw_file": unit.raw_id,
                    "created": raw_mtime.isoformat(),
                    "preview": raw_content[:200].strip(),
                }
            )

    # Check log for ingest records
    log_entries = []
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            if page in line and "ingest" in line:
                log_entries.append(line.strip())

    return json.dumps(
        {
            "page_id": page,
            "page_updated": page_updated,
            "page_mtime": page_mtime.isoformat(),
            "raw_sources": raw_candidates[:5],
            "log_entries": log_entries,
        },
        ensure_ascii=False,
    )


# Characters that would break the inline-list serialization or carry
# semantic meaning in YAML-flavored frontmatter. Keywords containing any
# of these are rejected at the writer boundary; consumers never see them.
_RAW_KEYWORD_FORBIDDEN_CHARS = frozenset(",[]:#{}\n\r")


def _validate_raw_keyword(kw: object) -> bool:
    """Return True iff ``kw`` is safe to serialize as an inline-list item."""
    if not isinstance(kw, str):
        return False
    if not kw or not kw.strip():
        return False
    for ch in kw:
        if ch in _RAW_KEYWORD_FORBIDDEN_CHARS:
            return False
        if ord(ch) < 0x20:  # control characters
            return False
    return True


@mcp.tool()
def chronovisor_record(
    content: str,
    session_id: str | None = None,
    keywords: list[str] | None = None,
    trigger_ingest: bool = True,
    idempotency_key: str | None = None,
    ctx: Context = None,
) -> str:
    """Save raw session data to raw/ for later ingest.

    This is the entry point for hooks to dump session logs.

    Args:
        content: Raw session content to save
        session_id: Optional session identifier. Auto-generated if not provided.
        keywords: Optional list of keywords carried into the ingest context.
            Items are written to the raw frontmatter as ``raw_keywords: [...]``.
            Keywords containing characters that would break inline-list
            serialization (``,[]:#{}\\n\\r`` or control chars) or that are
            empty/whitespace-only are rejected; rejected items are returned
            in the ``rejected_keywords`` field of the result.
        trigger_ingest: When True, preserve the historical behavior of
            running pending ingest immediately if the raw threshold is met.
            Set False for hooks that must not block on local LLM ingestion.
        idempotency_key: Optional stable transaction identity. Repeated calls
            with the same key reuse the first atomically published raw.
    """
    accepted: list[str] = []
    rejected: list[str] = []
    if keywords:
        for kw in keywords:
            if _validate_raw_keyword(kw):
                accepted.append(kw)
            else:
                rejected.append(kw if isinstance(kw, str) else repr(kw))

    if accepted:
        body = _frontmatter_patch(content, {"raw_keywords": accepted})
    else:
        body = content

    raw_slug = _raw_topic_slug(body, accepted)
    if idempotency_key:
        path, deduplicated = _publish_raw_idempotent(
            body,
            idempotency_key=idempotency_key,
            prefix=session_id or "",
            topic_slug=raw_slug,
        )
    else:
        # session_id is advisory; ordinary callers still get a unique path.
        path = _publish_raw(body, prefix=session_id or "", topic_slug=raw_slug)
        deduplicated = False
    filename = path.name

    # Check if orchestrator should trigger ingest
    from chronovisor.ingest.orchestrator import run_pending_ingest, should_ingest

    should, reason = should_ingest()

    result: dict = {
        "saved": filename,
        "path": str(path),
        "raw_slug": raw_slug,
        "ingest_pending": should,
        "ingest_reason": reason,
    }
    if idempotency_key:
        result["deduplicated"] = deduplicated
    if rejected:
        result["rejected_keywords"] = rejected

    if should and trigger_ingest:
        ingest_result = run_pending_ingest()
        result["ingest_triggered"] = ingest_result
    elif should:
        result["ingest_deferred"] = True

    if ctx is not None:
        try:
            from chronovisor.recall.recall_field import record_mcp_content_activity

            result["field_activity"] = record_mcp_content_activity(
                host=_mcp_client_host(ctx),
                session_id=session_id or "",
                content="\n".join([*accepted, content]),
            )
        except Exception:
            result["field_activity"] = {"status": "error"}

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def chronovisor_tick() -> str:
    """Run orchestration tick. Checks and triggers ingest/lint if needed.

    Call this periodically or at session boundaries.
    """
    from chronovisor.ingest.orchestrator import tick

    result = tick()
    return json.dumps(result, ensure_ascii=False, default=str)


def main():
    """Run the ``chronovisor-mcp`` command-line entry point."""
    init_chronovisor()
    # Importing torch, resolving the Hugging Face snapshot, and compiling the
    # first MPS inference used to add ~15 s to the first interactive search.
    # Warm it in parallel with the existing index startup work. reranker.py
    # serializes a truly immediate search against the same model instance.
    reranker_warmup = None
    try:
        from chronovisor.search.reranker import start_reranker_warmup

        reranker_warmup = start_reranker_warmup()
    except Exception:
        pass
    # job_store is in-memory: any current_job_id persisted from a previous
    # process is, by definition, stale. Clear it so a crash mid-ingest
    # doesn't permanently lock out run_pending_ingest.
    try:
        from chronovisor.ingest.orchestrator import reset_stale_lock

        reset_stale_lock()
    except Exception:
        pass
    # Warm both the page index and the BM25 cache on startup so the first
    # tool call doesn't pay the full-scan cost. Failures are non-fatal —
    # lazy refresh inside each tool will catch up on the next call.
    with contextlib.suppress(Exception):
        get_store().refresh()
    try:
        from chronovisor.search.search import get_bm25

        get_bm25().build()
    except Exception:
        pass
    # Cached production models complete in a few seconds. Bound the wait so a
    # first-ever model download can never prevent MCP from advertising tools.
    if reranker_warmup is not None:
        reranker_warmup.join(timeout=8.0)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
