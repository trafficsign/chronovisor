"""Ingest engine - structures raw data into wiki pages (two-stage pipeline)."""

import json
import re
import threading
from datetime import date
from pathlib import Path

from llm_wiki_mcp.wiki import PAGES_DIR, INDEX_FILE, LOG_FILE, all_pages, find_page, page_id_from_path
from llm_wiki_mcp.jobs import job_store, JobStatus
from llm_wiki_mcp.ollama import (
    generate, is_available,
    INGEST_SYSTEM_PROMPT, TRIAGE_SYSTEM_PROMPT, GENERATE_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Stage 1: Triage — analyze raw content and produce a structured plan
# ---------------------------------------------------------------------------

def _triage(content: str) -> list[dict]:
    """Stage 1: Analyze raw content and return a list of page operations as JSON."""
    # Build lightweight context: existing folders + page catalog (no full text)
    existing_folders = sorted({p.parent.name for p in all_pages() if p.parent != PAGES_DIR})
    catalog_lines = [f"Existing folders: {', '.join(f'{f}/' for f in existing_folders)}", ""]
    catalog_lines.append("Existing wiki pages (page_id — title):")
    for path in all_pages():
        content_text = path.read_text()
        fm_match = re.search(r"title:\s*(.+)", content_text)
        title = fm_match.group(1).strip() if fm_match else path.stem
        catalog_lines.append(f"  [[{page_id_from_path(path)}]] — {title}")

    catalog = "\n".join(catalog_lines)

    prompt = f"""{catalog}

---
Raw session data to triage:
---
{content}
---

Analyze the raw data above. Output a JSON array of page operations (create/update)."""

    output = generate(prompt, system=TRIAGE_SYSTEM_PROMPT)

    # Parse JSON from output (strip markdown fences if present)
    output = output.strip()
    if output.startswith("```"):
        output = re.sub(r"^```\w*\n?", "", output)
        output = re.sub(r"\n?```$", "", output)
    output = output.strip()

    try:
        plan = json.loads(output)
        if not isinstance(plan, list):
            _append_log(f"ingest | triage returned non-list: {type(plan)}")
            return []
        return plan
    except json.JSONDecodeError as e:
        _append_log(f"ingest | triage JSON parse failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Stage 2: Generate — produce each page with focused context
# ---------------------------------------------------------------------------

def _build_focused_context(op: dict, raw_content: str) -> str:
    """Build context focused on a single page operation."""
    lines = []

    # For updates, include the current page content
    if op.get("type") == "update":
        filename = op.get("filename", "")
        page_id = filename.replace(".md", "").split("/")[-1]
        existing_path = find_page(page_id)
        if existing_path:
            lines.append(f"--- Current content of [[{page_id}]] ---")
            lines.append(existing_path.read_text())
            lines.append("--- End current content ---\n")

    # Search for related pages using keywords from the triage plan
    keywords = op.get("keywords", [])
    if keywords:
        related = _search_related_pages(keywords, top_n=5)
        if related:
            lines.append("Related existing pages for cross-referencing:")
            for path in related:
                content = path.read_text()
                lines.append(f"\n--- [[{page_id_from_path(path)}]] ---")
                lines.append(content)
    elif op.get("type") == "create":
        # For creates without keywords, use title words as fallback
        title = op.get("title", "")
        if title:
            title_keywords = [w for w in title.split() if len(w) >= 2]
            related = _search_related_pages(title_keywords, top_n=3)
            if related:
                lines.append("Related existing pages:")
                for path in related:
                    lines.append(f"\n--- [[{page_id_from_path(path)}]] ---")
                    lines.append(path.read_text())

    return "\n".join(lines)


def _search_related_pages(keywords: list[str], min_score: float = 0.5, top_n: int = 8) -> list[Path]:
    """Search for related pages using keywords. Uses BM25 if available, falls back to simple matching."""
    try:
        from llm_wiki_mcp.search import get_bm25
        bm25 = get_bm25()
        bm25.build()
        results = bm25.query(" ".join(keywords), top_n=top_n)
        return [find_page(r.page_id) for r in results if find_page(r.page_id)]
    except Exception:
        pass

    # Fallback: simple keyword matching
    query_terms = [k.lower() for k in keywords]
    scored = []
    for path in all_pages():
        content = path.read_text()
        content_lower = content.lower()
        fm_match = re.search(r"title:\s*(.+)", content)
        title = fm_match.group(1) if fm_match else path.stem
        title_lower = title.lower()

        score = 0.0
        for term in query_terms:
            if term in title_lower:
                score += 0.5
            if term in path.stem.lower().replace("-", " "):
                score += 0.3
            count = content_lower.count(term)
            if count > 0:
                score += min(0.1 * count, 0.4)

        if score >= min_score:
            scored.append((score, path))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in scored[:top_n]]


def _generate_one(op: dict, raw_content: str) -> str | None:
    """Stage 2: Generate a single page using focused context."""
    context = _build_focused_context(op, raw_content)

    op_type = op.get("type", "create").upper()
    filename = op.get("filename", "unknown.md")
    summary = op.get("summary", "")
    title = op.get("title", "")

    prompt = f"""{context}

---
Raw session data (source material):
---
{raw_content}
---

Task: {op_type} page "{filename}"
Title: {title}
Summary: {summary}

Generate the page content based on the raw data and context above."""

    try:
        output = generate(prompt, system=GENERATE_SYSTEM_PROMPT)
        return output
    except Exception as e:
        _append_log(f"ingest | generate failed for {filename}: {e}")
        return None


# ---------------------------------------------------------------------------
# Parse & Apply (unchanged from original)
# ---------------------------------------------------------------------------

def _parse_output(output: str) -> list[dict]:
    """Parse LLM output into page operations."""
    operations = []

    for match in re.finditer(
        r"=== NEW PAGE:\s*(\S+)\s*===\n(.*?)\n=== END PAGE ===",
        output,
        re.DOTALL,
    ):
        filename = match.group(1)
        content = match.group(2).strip()
        operations.append({
            "type": "create",
            "filename": filename,
            "content": content,
        })

    for match in re.finditer(
        r"=== UPDATE PAGE:\s*(\S+)\s*===\n(.*?)\n=== END PAGE ===",
        output,
        re.DOTALL,
    ):
        filename = match.group(1)
        content = match.group(2).strip()
        operations.append({
            "type": "update",
            "filename": filename,
            "content": content,
        })

    return operations


def _apply_operations(operations: list[dict]) -> tuple[list[str], list[str]]:
    """Apply page operations and return (created, updated) lists."""
    created = []
    updated = []

    for op in operations:
        filename = op["filename"]
        if not filename.endswith(".md"):
            filename += ".md"
        path = PAGES_DIR / filename
        page_id = path.stem

        if op["type"] == "create":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(op["content"] + "\n")
            created.append(page_id)
            _append_log(f"ingest | created {page_id}")

        elif op["type"] == "update":
            existing_path = find_page(page_id)
            if existing_path:
                path = existing_path
            if path.exists():
                existing = path.read_text()
                today = date.today().isoformat()
                existing = re.sub(
                    r"updated:\s*.+",
                    f"updated: {today}",
                    existing,
                    count=1,
                )
                path.write_text(existing.rstrip() + "\n\n" + op["content"] + "\n")
                updated.append(page_id)
                _append_log(f"ingest | updated {page_id}")

    return created, updated


def _rebuild_index() -> None:
    """Rebuild index.md from all pages."""
    pages = sorted(all_pages())
    lines = [
        "---",
        f"title: Index",
        f"updated: {date.today().isoformat()}",
        "---",
        "",
        "# Wiki Index",
        "",
    ]
    for p in pages:
        content = p.read_text()
        title_match = re.search(r"title:\s*(.+)", content)
        title = title_match.group(1) if title_match else p.stem
        lines.append(f"- [[{p.stem}]] — {title}")

    INDEX_FILE.write_text("\n".join(lines) + "\n")


def _append_log(message: str) -> None:
    """Append to log.md."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n- [{timestamp}] {message}")


# ---------------------------------------------------------------------------
# Main entry point — two-stage pipeline
# ---------------------------------------------------------------------------

def run_ingest(content: str, job_id: str, on_complete: "callable | None" = None) -> None:
    """Run two-stage ingest in background thread."""
    job_store.update(job_id, status=JobStatus.RUNNING)

    try:
        processor = "ollama" if is_available() else "sonnet"
        if processor == "sonnet":
            raise NotImplementedError("Sonnet fallback not yet implemented")
        job_store.update(job_id, processor=processor)

        # Stage 1: Triage
        job_store.update(job_id, stage="triage")
        _append_log("ingest | stage 1: triage started")
        plan = _triage(content)

        if not plan:
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=_now(),
                result={"message": "No wiki-worthy content extracted"},
                pages_created=[],
                pages_updated=[],
            )
            _append_log("ingest | triage: no operations planned")
            return

        _append_log(f"ingest | triage: {len(plan)} operations planned")
        job_store.update(job_id, stage="generate", total_ops=len(plan), completed_ops=0)

        # Stage 2: Generate each page
        all_operations = []
        for i, op in enumerate(plan):
            _append_log(f"ingest | generating {i+1}/{len(plan)}: {op.get('filename', '?')}")
            output = _generate_one(op, content)
            if output:
                operations = _parse_output(output)
                all_operations.extend(operations)
            job_store.update(job_id, completed_ops=i + 1)

        # Apply all operations
        if all_operations:
            created, updated = _apply_operations(all_operations)
            _rebuild_index()

            # Refresh the IndexStore so subsequent reads see fresh
            # backlinks/outlinks. Do this *before* embedding updates so
            # that any code path that consults the index during embedding
            # gets the new state.
            try:
                from llm_wiki_mcp.index_store import get_store
                get_store().refresh()
            except Exception as e:
                _append_log(f"ingest | index_store refresh failed: {e}")

            # Update search embeddings
            changed_pages = created + updated
            if changed_pages:
                try:
                    from llm_wiki_mcp.search import update_embeddings
                    update_embeddings(page_ids=changed_pages)
                except Exception:
                    pass
        else:
            created, updated = [], []

        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=_now(),
            pages_created=created,
            pages_updated=updated,
        )
        _append_log(f"ingest | completed: {len(created)} created, {len(updated)} updated")

        if on_complete:
            try:
                on_complete()
            except Exception as cb_err:
                _append_log(f"ingest | on_complete callback failed: {cb_err}")

    except Exception as e:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            completed_at=_now(),
            error=str(e),
        )
        _append_log(f"ingest | failed: {e}")


def start_ingest(content: str, on_complete: "callable | None" = None) -> str:
    """Start an async ingest job. Returns job_id."""
    processor = "ollama" if is_available() else "sonnet"
    job = job_store.create(processor=processor)

    thread = threading.Thread(
        target=run_ingest,
        args=(content, job.job_id, on_complete),
        daemon=True,
    )
    thread.start()

    return job.job_id


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
