"""LLM Wiki MCP Server."""

import json
import re
from datetime import datetime, date, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from llm_wiki_mcp.wiki import (
    WIKI_ROOT, RAW_DIR, PAGES_DIR, SYSTEM_DIR, INDEX_FILE, LOG_FILE, SCHEMA_FILE,
    init_wiki,
)

mcp = FastMCP(
    "llm-wiki",
    instructions=(
        "LLM Wiki is your structured knowledge base. "
        "Use wiki.index() at session start, wiki.search() during conversation, "
        "and wiki.read() to get full page content with backlinks."
    ),
)


def _parse_frontmatter(text: str) -> dict:
    """Extract frontmatter from markdown."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
    return fm


def _extract_wiki_links(text: str) -> list[str]:
    """Extract [[wiki-link]] references from text."""
    return re.findall(r"\[\[([^\]]+)\]\]", text)


def _find_backlinks(page_id: str) -> list[str]:
    """Find pages that link to the given page."""
    backlinks = []
    for p in PAGES_DIR.glob("*.md"):
        content = p.read_text()
        if f"[[{page_id}]]" in content:
            backlinks.append(p.stem)
    return backlinks


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
def wiki_read(page: str) -> str:
    """Read a wiki page with outlinks and backlinks.

    Searches pages/ first, then system/ for system files.

    Args:
        page: Page ID (filename without .md extension)
    """
    path = PAGES_DIR / f"{page}.md"
    if not path.exists():
        # Check system/ directory
        path = SYSTEM_DIR / f"{page}.md"
    if not path.exists():
        return json.dumps({"error": f"Page '{page}' not found"})

    content = path.read_text()
    outlinks = _extract_wiki_links(content)
    backlinks = _find_backlinks(page)

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
    pages = sorted(PAGES_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(pages)
    sliced = pages[cursor:cursor + limit]

    entries = [_page_metadata(p) for p in sliced]

    return json.dumps({
        "total": total,
        "cursor": cursor,
        "limit": limit,
        "has_more": cursor + limit < total,
        "pages": entries,
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
    page_count = len(list(PAGES_DIR.glob("*.md")))
    raw_count = len(list(RAW_DIR.glob("*.md")))

    # Check Ollama
    ollama_status = "unknown"
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            ollama_status = "running"
        else:
            ollama_status = "error"
    except Exception:
        ollama_status = "stopped"

    # Find oldest/newest page
    pages = list(PAGES_DIR.glob("*.md"))
    oldest = None
    newest = None
    if pages:
        by_mtime = sorted(pages, key=lambda p: p.stat().st_mtime)
        oldest = _page_metadata(by_mtime[0])
        newest = _page_metadata(by_mtime[-1])

    # Count orphan pages (no backlinks)
    orphan_count = 0
    for p in pages:
        if not _find_backlinks(p.stem):
            orphan_count += 1

    return json.dumps({
        "page_count": page_count,
        "raw_pending": raw_count,
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


@mcp.tool()
def wiki_search(query: str, depth: int = 1) -> str:
    """Search wiki pages with chain expansion via [[wiki-links]].

    Returns direct_hits (pages matching query) and expanded_hits (linked pages).

    Args:
        query: Search query string
        depth: How many link-hops to follow (0=direct only, 1=one hop, default 1)
    """
    query_lower = query.lower()
    query_terms = query_lower.split()

    # Score all pages
    scored = []
    for path in PAGES_DIR.glob("*.md"):
        content = path.read_text()
        content_lower = content.lower()
        fm = _parse_frontmatter(content)
        title = fm.get("title", path.stem)
        title_lower = title.lower()

        score = 0.0
        match_reasons = []

        # Title match (high weight)
        for term in query_terms:
            if term in title_lower:
                score += 0.5
                if "title" not in match_reasons:
                    match_reasons.append("title")

        # Filename match
        for term in query_terms:
            if term in path.stem.lower().replace("-", " "):
                score += 0.3
                if "filename" not in match_reasons:
                    match_reasons.append("filename")

        # Content match
        for term in query_terms:
            count = content_lower.count(term)
            if count > 0:
                score += min(0.1 * count, 0.4)
                if "content" not in match_reasons:
                    match_reasons.append("content")

        if score > 0:
            # Extract snippet
            snippet = _extract_snippet(content, query_terms)
            scored.append({
                "page_id": path.stem,
                "title": title,
                "updated": fm.get("updated", "unknown"),
                "score": round(score, 2),
                "match_reasons": match_reasons,
                "snippets": [snippet] if snippet else [],
            })

    # Sort by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    direct_hits = scored[:10]

    # Expand via links
    expanded_hits = []
    edges = []
    if depth > 0 and direct_hits:
        seen = {h["page_id"] for h in direct_hits}
        for hit in direct_hits:
            path = PAGES_DIR / f"{hit['page_id']}.md"
            if not path.exists():
                continue
            outlinks = _extract_wiki_links(path.read_text())
            for link in outlinks:
                if link in seen:
                    continue
                link_path = PAGES_DIR / f"{link}.md"
                if not link_path.exists():
                    continue
                seen.add(link)
                fm = _parse_frontmatter(link_path.read_text())
                expanded_hits.append({
                    "page_id": link,
                    "title": fm.get("title", link),
                    "updated": fm.get("updated", "unknown"),
                    "distance": 1,
                    "via": [hit["page_id"]],
                    "score": round(hit["score"] * 0.5, 2),
                    "reason": "linked from direct hit",
                })
                edges.append({
                    "from": hit["page_id"],
                    "to": link,
                    "type": "wikilink",
                })

    return json.dumps({
        "query": query,
        "depth": depth,
        "direct_hits": direct_hits,
        "expanded_hits": expanded_hits,
        "edges": edges,
        "truncated": len(scored) > 10,
    }, ensure_ascii=False)


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


@mcp.tool()
def wiki_ingest(content: str) -> str:
    """Ingest raw content into the wiki asynchronously.

    Returns a job_id for tracking progress.

    Args:
        content: Raw session data to structure into wiki pages
    """
    from llm_wiki_mcp.ingest import start_ingest
    from llm_wiki_mcp.ollama import is_available

    job_id = start_ingest(content)
    processor = "ollama" if is_available() else "sonnet"

    return json.dumps({
        "job_id": job_id,
        "accepted": True,
        "processor": processor,
    }, ensure_ascii=False)


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
def wiki_apply() -> str:
    """Apply safe auto-fixes to the wiki.

    Only fixes safe issues (broken links, formatting).
    Contradictions and ambiguous issues are left as flags.
    Run wiki.check() first to see what will be fixed.
    """
    from llm_wiki_mcp.lint import check, apply_safe_fixes
    issues = check()
    actions = apply_safe_fixes(issues)
    remaining = [i for i in issues if not i.get("auto_fixable")]
    return json.dumps({
        "actions_taken": actions,
        "remaining_issues": remaining,
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
    page_path = PAGES_DIR / f"{page}.md"
    if not page_path.exists():
        return json.dumps({"error": f"Page '{page}' not found"})

    page_mtime = datetime.fromtimestamp(page_path.stat().st_mtime)

    # Find raw files that might be the source
    # Match by checking raw files created before or around the page creation time
    raw_candidates = []
    for raw_path in sorted(RAW_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        raw_mtime = datetime.fromtimestamp(raw_path.stat().st_mtime)
        # Raw file should be older than or close to page creation
        if raw_mtime <= page_mtime + timedelta(minutes=30):
            # Check if raw content mentions anything related to the page
            raw_content = raw_path.read_text()[:500]
            page_content = page_path.read_text()
            fm = _parse_frontmatter(page_content)
            title = fm.get("title", page)

            # Simple relevance check
            if (page.replace("-", " ") in raw_content.lower()
                    or title.lower() in raw_content.lower()):
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
        "page_updated": _parse_frontmatter(page_path.read_text()).get("updated", "unknown"),
        "page_mtime": page_mtime.isoformat(),
        "raw_sources": raw_candidates[:5],
        "log_entries": log_entries,
    }, ensure_ascii=False)


@mcp.tool()
def wiki_save_raw(content: str, session_id: str | None = None) -> str:
    """Save raw session data to raw/ for later ingest.

    This is the entry point for hooks to dump session logs.

    Args:
        content: Raw session content to save
        session_id: Optional session identifier. Auto-generated if not provided.
    """
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    filename = f"{session_id}.md"
    path = RAW_DIR / filename
    path.write_text(content)

    # Check if orchestrator should trigger ingest
    from llm_wiki_mcp.orchestrator import should_ingest, run_pending_ingest
    should, reason = should_ingest()

    result = {
        "saved": filename,
        "path": str(path),
        "ingest_pending": should,
        "ingest_reason": reason,
    }

    if should:
        ingest_result = run_pending_ingest()
        result["ingest_triggered"] = ingest_result

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
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
