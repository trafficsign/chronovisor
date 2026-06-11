"""LLM Wiki MCP Server."""

import json
import re
from datetime import datetime, date, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from llm_wiki_mcp.wiki import (
    WIKI_ROOT, RAW_DIR, PAGES_DIR, SYSTEM_DIR, INDEX_FILE, LOG_FILE, SCHEMA_FILE,
    init_wiki, all_pages, find_page, page_id_from_path,
)
from llm_wiki_mcp.link_fix import extract_targets as _extract_targets
from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.frontmatter import parse as _frontmatter_parse, patch as _frontmatter_patch

mcp = FastMCP(
    "llm-wiki",
    instructions=(
        "LLM Wiki is your structured knowledge base. "
        "Use wiki.index() at session start, wiki.search() during conversation, "
        "and wiki.read() to get full page content with backlinks."
    ),
)


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


@mcp.tool()
def wiki_read(page: str, session_id: str | None = None) -> str:
    """Read a wiki page with outlinks and backlinks.

    Searches pages/ first, then system/ for system files.

    Args:
        page: Page ID (filename without .md extension)
        session_id: Optional session id for recall pull feedback.
    """
    store = get_store()
    store.refresh()

    path = find_page(page)
    if not path:
        # Check system/ directory
        path = SYSTEM_DIR / f"{page}.md"
    if not path or not path.exists():
        return json.dumps({"error": f"Page '{page}' not found"})

    content = path.read_text()
    outlinks = store.outlinks(page) or _extract_wiki_links(content)
    backlinks = store.backlinks(page)
    _append_pull_log({"type": "read", "session_id": session_id or "", "page_id": page})

    return json.dumps({
        "page_id": page,
        "content": content,
        "outlinks": outlinks,
        "backlinks": backlinks,
    }, ensure_ascii=False)


@mcp.tool()
def wiki_index(limit: int = 50, cursor: int = 0) -> str:
    """Return structured catalog of wiki pages with pagination.

    Args:
        limit: Max number of entries to return (default 50)
        cursor: Offset for pagination (default 0)
    """
    store = get_store()
    store.refresh()
    entries = store.all_pages_meta(include_system=False)
    total = len(entries)
    sliced = entries[cursor:cursor + limit]

    return json.dumps({
        "total": total,
        "cursor": cursor,
        "limit": limit,
        "has_more": cursor + limit < total,
        "pages": sliced,
    }, ensure_ascii=False)


@mcp.tool()
def wiki_log(limit: int = 20) -> str:
    """Return recent change history.

    Args:
        limit: Number of recent log entries to return
    """
    if not LOG_FILE.exists():
        return json.dumps({"entries": []})

    lines = LOG_FILE.read_text().splitlines()
    # Skip frontmatter and header
    log_lines = [l for l in lines if l.startswith("- ")]
    recent = log_lines[-limit:] if len(log_lines) > limit else log_lines

    return json.dumps({"entries": recent}, ensure_ascii=False)


@mcp.tool()
def wiki_status() -> str:
    """Return wiki health, Ollama status, and basic statistics."""
    from llm_wiki_mcp.orchestrator import get_pending_raw_files

    store = get_store()
    store.refresh()

    page_count = store.page_count(include_system=False)
    raw_total = len(list(RAW_DIR.glob("*.md")))
    raw_pending = len(get_pending_raw_files())

    # Check Ollama via the shared httpx client used by every other Ollama
    # call in the process. The status string preserves the original
    # vocabulary ("running" / "error" / "stopped") so callers see no
    # behaviour change.
    ollama_status = "unknown"
    try:
        from llm_wiki_mcp.ollama import _client
        resp = _client().get("/api/tags", timeout=3)
        if resp.status_code == 200:
            ollama_status = "running"
        else:
            ollama_status = "error"
    except Exception:
        ollama_status = "stopped"

    # Find oldest/newest page from the index (mtime-sorted).
    sorted_meta = store.all_pages_meta(include_system=False)
    oldest = sorted_meta[-1] if sorted_meta else None
    newest = sorted_meta[0] if sorted_meta else None

    # Orphan count: pages with no inbound backlinks.
    orphan_count = len(store.orphans(include_system=False))

    return json.dumps({
        "page_count": page_count,
        "raw_total": raw_total,
        "raw_pending": raw_pending,
        "orphan_count": orphan_count,
        "ollama_status": ollama_status,
        "oldest_page": oldest,
        "newest_page": newest,
        "wiki_root": str(WIKI_ROOT),
    }, ensure_ascii=False)


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


def _append_pull_log(record: dict) -> None:
    try:
        from llm_wiki_mcp.recall_runtime import RECALL_PULL_LOG_FILE, append_jsonl

        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            **record,
        }
        append_jsonl(RECALL_PULL_LOG_FILE, record)
    except Exception:
        pass


@mcp.tool()
def wiki_init() -> str:
    """Initialize session: returns system pages + status in a single call.

    Replaces the need for separate wiki_status + 3x wiki_read at session start.
    Returns user-profile, current-state, lessons-learned, and basic wiki stats.
    """
    from concurrent.futures import ThreadPoolExecutor
    from llm_wiki_mcp.ollama import is_available
    from llm_wiki_mcp.orchestrator import get_pending_raw_files

    store = get_store()

    # Run the index refresh, raw-dir count, and Ollama health probe in
    # parallel. The Ollama probe is the heaviest (network) so doing it
    # alongside the on-disk scans hides its latency.
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_refresh = ex.submit(store.refresh)
        f_raw = ex.submit(lambda: len(list(RAW_DIR.glob("*.md"))))
        f_pending = ex.submit(lambda: len(get_pending_raw_files()))
        f_ollama = ex.submit(is_available)
        f_refresh.result()
        raw_total = f_raw.result()
        raw_pending = f_pending.result()
        ollama_status = "running" if f_ollama.result() else "stopped"

    page_count = store.page_count(include_system=False)
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

    return json.dumps({
        "status": {
            "page_count": page_count,
            "raw_total": raw_total,
            "raw_pending": raw_pending,
            "ollama_status": ollama_status,
            "wiki_root": str(WIKI_ROOT),
        },
        "system_pages": system_pages,
    }, ensure_ascii=False)


@mcp.tool()
def wiki_search(
    query: str,
    depth: int = 1,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    semantic: bool = True,
    tags: list[str] | None = None,
    tag_match: str = "all",
    session_id: str | None = None,
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
        session_id: Optional session id for recall pull feedback.
    """
    from llm_wiki_mcp.search import search as run_search
    from llm_wiki_mcp.reranker import rerank_results
    from llm_wiki_mcp.runtime_config import load_reranker_config

    store = get_store()
    store.refresh()
    reranker_cfg = load_reranker_config()
    rerank_allowed = reranker_cfg.enabled and sort_by == "relevance"
    search_top_n = max(10, reranker_cfg.top_n) if rerank_allowed else 10

    results, search_mode = run_search(
        query=query, top_n=search_top_n,
        folder=folder, updated_after=updated_after,
        updated_before=updated_before, sort_by=sort_by,
        semantic=semantic,
    )

    # Tag filter: post-process search results so the tag axis composes
    # with relevance / date / folder cleanly. Done in Python rather than
    # pushed into BM25 because tag membership is exact-match, not scored.
    tag_filter = [t for t in (tags or []) if isinstance(t, str) and t]
    if tag_filter:
        match_mode = "any" if tag_match == "any" else "all"
        target = set(tag_filter)
        kept: list = []
        for r in results:
            page_tags = set(store.tags(r.page_id))
            if match_mode == "all":
                if target.issubset(page_tags):
                    kept.append(r)
            else:  # any
                if target & page_tags:
                    kept.append(r)
        results = kept

    reranker_meta = {
        "status": "disabled" if not reranker_cfg.enabled else "skipped",
        "reason": "config_disabled" if not reranker_cfg.enabled else "sort_by_not_relevance",
        "model": reranker_cfg.model,
        "backend": reranker_cfg.backend,
    }
    if rerank_allowed and results:
        rerank_outcome = rerank_results(query, results, config=reranker_cfg)
        results = rerank_outcome.results
        reranker_meta = rerank_outcome.metadata
        if reranker_meta.get("status") == "applied":
            search_mode = f"{search_mode}+rerank"
    results = results[:10]

    query_terms = query.lower().split()
    direct_hits = []
    for r in results:
        path = find_page(r.page_id)
        content = path.read_text() if path else None
        snippet = _extract_snippet(content, query_terms) if content is not None else None
        direct_hits.append({
            "page_id": r.page_id,
            "title": r.title,
            "updated": r.updated,
            "score": round(r.score, 4),
            "snippets": [snippet] if snippet else [],
            "tags": store.tags(r.page_id),
        })

    # Expand via links — outlinks and link metadata both come from the
    # IndexStore, so no extra disk reads are needed in this pass.
    # When a tag filter is active, expanded hits inherit the same tag
    # constraint so a casual link from a matching page to a wildly
    # off-tag page doesn't sneak past the filter.
    expanded_hits = []
    edges = []
    if depth > 0 and direct_hits:
        seen = {h["page_id"] for h in direct_hits}
        for hit in direct_hits:
            outlinks = store.outlinks(hit["page_id"])
            for link in outlinks:
                if link in seen:
                    continue
                meta = store.meta(link)
                if meta is None:
                    # Link points to a non-existent page; skip (matches
                    # legacy behaviour, which used find_page() == None).
                    continue
                if tag_filter:
                    link_tags = set(store.tags(link))
                    if tag_match == "any":
                        if not (set(tag_filter) & link_tags):
                            continue
                    else:
                        if not set(tag_filter).issubset(link_tags):
                            continue
                seen.add(link)
                expanded_hits.append({
                    "page_id": link,
                    "title": meta["title"],
                    "updated": meta["updated"],
                    "distance": 1,
                    "via": [hit["page_id"]],
                    "score": round(hit["score"] * 0.5, 4),
                    "reason": "linked from direct hit",
                    "tags": store.tags(link),
                })
                edges.append({
                    "from": hit["page_id"],
                    "to": link,
                    "type": "wikilink",
                })

    filters_applied = {}
    if folder:
        filters_applied["folder"] = folder
    if updated_after:
        filters_applied["updated_after"] = updated_after
    if updated_before:
        filters_applied["updated_before"] = updated_before
    if tag_filter:
        filters_applied["tags"] = tag_filter
        filters_applied["tag_match"] = tag_match if tag_match in ("all", "any") else "all"
    _append_pull_log(
        {
            "type": "search",
            "session_id": session_id or "",
            "query": query,
            "direct_pages": [hit["page_id"] for hit in direct_hits],
            "expanded_pages": [hit["page_id"] for hit in expanded_hits],
        }
    )

    return json.dumps({
        "query": query,
        "depth": depth,
        "search_mode": search_mode,
        "filters_applied": filters_applied,
        "reranker": reranker_meta,
        "direct_hits": direct_hits,
        "expanded_hits": expanded_hits,
        "edges": edges,
    }, ensure_ascii=False)


@mcp.tool()
def wiki_reindex() -> str:
    """Rebuild search embeddings for all wiki pages.

    Call this after bulk changes or to initialize semantic search.
    """
    from llm_wiki_mcp.search import update_embeddings
    count = update_embeddings()
    if count == 0:
        return json.dumps({"status": "skipped", "message": "Ollama not available or no pages to update"})
    return json.dumps({"status": "ok", "pages_updated": count})


def _extract_snippet(content: str, terms: list[str], max_len: int = 150) -> str | None:
    """Extract a relevant snippet from content."""
    # Skip frontmatter
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:].strip()

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


def _raw_topic_slug(content: str, keywords: list[str] | None = None, *, max_len: int = 56) -> str:
    """Create a readable raw filename slug while keeping ASCII-only safety."""

    parts: list[str] = []
    if keywords:
        for keyword in keywords:
            for match in _RAW_SLUG_TOKEN_RE.finditer(keyword.lower()):
                token = match.group(0)
                if len(token) < 2 or token in _RAW_TOPIC_STOPWORDS or _RAW_UUIDISH_RE.match(token):
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
        if lower.startswith((
            "source:",
            "session id:",
            "cwd:",
            "session file:",
            "lines:",
            "memory writer model:",
            "generated at:",
            "raw_keywords:",
        )):
            continue
        candidates.append(stripped)
        break

    for candidate in candidates:
        for match in _RAW_SLUG_TOKEN_RE.finditer(candidate.lower()):
            token = match.group(0)
            if len(token) < 2 or token in _RAW_TOPIC_STOPWORDS or _RAW_UUIDISH_RE.match(token):
                continue
            parts.append(token)
            slug = "-".join(parts)
            if len(slug) >= max_len:
                return slug[:max_len].strip("-")
    return "-".join(parts)[:max_len].strip("-")


_RAW_ALLOC_MAX_RETRIES = 32


def _allocate_raw_path(prefix: str = "", topic_slug: str = "") -> Path:
    """Reserve a unique raw/*.md path even under sub-millisecond contention.

    Filename = ``YYYYMMDD-HHMMSS-{source}-{topic}-{8hex}.md`` when source or
    topic can be derived, falling back to the old timestamp/hash shape. The
    8-hex suffix is from :func:`secrets.token_hex` (32 bits of OS entropy,
    far less collision-prone than ``time.time_ns() & 0xFFFF``). Combined
    with ``O_CREAT|O_EXCL`` exclusive create this makes accidental overwrite
    from concurrent callers effectively impossible. We bound retries so a
    misconfigured filesystem doesn't spin forever.
    """
    import os
    import secrets
    from datetime import datetime

    source = _raw_source_label(prefix)
    topic = _sanitize_raw_component(topic_slug, max_len=56)
    name_parts = [part for part in (source, topic) if part]
    readable = f"-{'-'.join(name_parts)}" if name_parts else _sanitize_raw_prefix(prefix)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for _ in range(_RAW_ALLOC_MAX_RETRIES):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = secrets.token_hex(4)  # 8 hex chars / 32 bits
        path = RAW_DIR / f"{ts}{readable}-{suffix}.md"
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


@mcp.tool()
def wiki_ingest(content: str, force: bool = True) -> str:
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
    from llm_wiki_mcp.orchestrator import run_pending_ingest

    path = _allocate_raw_path(prefix="ingest", topic_slug=_raw_topic_slug(content))
    path.write_text(content)

    result = run_pending_ingest(force=force)
    payload = {
        "saved": path.name,
        "ingest": result,
    }
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def wiki_check() -> str:
    """Run lint checks on the wiki. Returns list of detected issues.

    Issues include: broken links, stale pages, orphan pages, duplicates.
    Does NOT auto-fix anything.
    """
    from llm_wiki_mcp.lint import check
    issues = check()
    return json.dumps({
        "total_issues": len(issues),
        "issues": issues,
    }, ensure_ascii=False)


@mcp.tool()
def wiki_apply(dry_run: bool = False, fuzzy: bool = True) -> str:
    """Apply safe auto-fixes to the wiki.

    broken_link は fuzzy match で近い page_id に置換し、見つからなければ
    plaintext 化する。system/ 配下に実在する target は false positive と見なして
    書き換えない。Contradictions / orphans / stale などは flag のまま残す。
    Run wiki.check() first to see what will be fixed.

    Args:
        dry_run: True なら実際には書き込まず actions のプレビューだけ返す。
        fuzzy: False にすると broken_link の自動書き換えを無効化する (より保守的)。
    """
    from llm_wiki_mcp.lint import check, apply_safe_fixes
    issues = check()
    actions = apply_safe_fixes(issues, dry_run=dry_run, fuzzy=fuzzy)
    remaining = [i for i in issues if not i.get("auto_fixable")]
    return json.dumps({
        "actions_taken": actions,
        "remaining_issues": remaining,
        "dry_run": dry_run,
        "fuzzy": fuzzy,
    }, ensure_ascii=False)


@mcp.tool()
def wiki_jobs(job_id: str | None = None) -> str:
    """Check job progress.

    Args:
        job_id: Specific job ID to check. If None, returns recent jobs.
    """
    from llm_wiki_mcp.jobs import job_store

    if job_id:
        job = job_store.get(job_id)
        if not job:
            return json.dumps({"error": f"Job '{job_id}' not found"})
        return json.dumps({
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
            "error": job.error,
        }, ensure_ascii=False)

    jobs = job_store.recent()
    return json.dumps({
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "processor": j.processor,
                "created_at": j.created_at,
            }
            for j in jobs
        ]
    }, ensure_ascii=False)


@mcp.tool()
def wiki_provenance(page: str) -> str:
    """Trace the provenance of a wiki page back to raw session data.

    Args:
        page: Page ID to trace
    """
    page_path = find_page(page)
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
    raw_candidates = []
    for raw_path in sorted(RAW_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        raw_mtime = datetime.fromtimestamp(raw_path.stat().st_mtime)
        if raw_mtime > threshold:
            break
        raw_content = raw_path.read_text()[:500]
        raw_lower = raw_content.lower()
        if page_dehyphen in raw_lower or page_title_lower in raw_lower:
            raw_candidates.append({
                "raw_file": raw_path.name,
                "created": raw_mtime.isoformat(),
                "preview": raw_content[:200].strip(),
            })

    # Check log for ingest records
    log_entries = []
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            if page in line and "ingest" in line:
                log_entries.append(line.strip())

    return json.dumps({
        "page_id": page,
        "page_updated": page_updated,
        "page_mtime": page_mtime.isoformat(),
        "raw_sources": raw_candidates[:5],
        "log_entries": log_entries,
    }, ensure_ascii=False)


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
def wiki_save_raw(
    content: str,
    session_id: str | None = None,
    keywords: list[str] | None = None,
    trigger_ingest: bool = True,
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

    # session_id is advisory; always allocate a unique path so two callers
    # in the same second don't silently overwrite each other.
    raw_slug = _raw_topic_slug(body, accepted)
    path = _allocate_raw_path(prefix=session_id or "", topic_slug=raw_slug)
    filename = path.name
    path.write_text(body)

    # Check if orchestrator should trigger ingest
    from llm_wiki_mcp.orchestrator import should_ingest, run_pending_ingest
    should, reason = should_ingest()

    result: dict = {
        "saved": filename,
        "path": str(path),
        "raw_slug": raw_slug,
        "ingest_pending": should,
        "ingest_reason": reason,
    }
    if rejected:
        result["rejected_keywords"] = rejected

    if should and trigger_ingest:
        ingest_result = run_pending_ingest()
        result["ingest_triggered"] = ingest_result
    elif should:
        result["ingest_deferred"] = True

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def wiki_tick() -> str:
    """Run orchestration tick. Checks and triggers ingest/lint if needed.

    Call this periodically or at session boundaries.
    """
    from llm_wiki_mcp.orchestrator import tick
    result = tick()
    return json.dumps(result, ensure_ascii=False, default=str)


def main():
    init_wiki()
    # job_store is in-memory: any current_job_id persisted from a previous
    # process is, by definition, stale. Clear it so a crash mid-ingest
    # doesn't permanently lock out run_pending_ingest.
    try:
        from llm_wiki_mcp.orchestrator import reset_stale_lock
        reset_stale_lock()
    except Exception:
        pass
    # Warm both the page index and the BM25 cache on startup so the first
    # tool call doesn't pay the full-scan cost. Failures are non-fatal —
    # lazy refresh inside each tool will catch up on the next call.
    try:
        get_store().refresh()
    except Exception:
        pass
    try:
        from llm_wiki_mcp.search import get_bm25
        get_bm25().build()
    except Exception:
        pass
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
