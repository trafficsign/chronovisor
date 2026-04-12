"""Ingest engine - structures raw data into wiki pages."""

import re
import threading
from datetime import date
from pathlib import Path

from llm_wiki_mcp.wiki import PAGES_DIR, INDEX_FILE, LOG_FILE, all_pages, find_page, page_id_from_path
from llm_wiki_mcp.jobs import job_store, JobStatus
from llm_wiki_mcp.ollama import generate, is_available, INGEST_SYSTEM_PROMPT


def _extract_keywords_from_raw(content: str) -> list[str]:
    """Extract keywords from raw content frontmatter."""
    keywords = []
    fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if fm_match:
        kw_match = re.search(r"keywords:\s*\[([^\]]*)\]", fm_match.group(1))
        if kw_match:
            keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]
    return keywords


def _search_related_pages(keywords: list[str], min_score: float = 0.5) -> list[Path]:
    """Search for related pages using keywords."""
    if not keywords:
        return []

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
    return [path for _, path in scored]


def _build_context(raw_content: str) -> str:
    """Build context by searching for related pages using keywords from raw content."""
    keywords = _extract_keywords_from_raw(raw_content)
    related_pages = _search_related_pages(keywords)

    if not related_pages:
        # Fallback: most recently updated pages
        all_pages = sorted(all_pages(), key=lambda p: p.stat().st_mtime, reverse=True)
        related_pages = all_pages[:50]

    if not related_pages:
        return "No existing pages in wiki."

    # Show existing folder structure
    existing_folders = sorted({p.parent.name for p in all_pages() if p.parent != PAGES_DIR})
    lines = [f"Existing folders: {', '.join(f'{f}/' for f in existing_folders)}"]
    lines.append("")
    lines.append("Existing related wiki pages (use [[page-id]] to cross-reference):")
    for p in related_pages:
        content = p.read_text()
        folder = p.parent.name if p.parent != PAGES_DIR else "(root)"
        lines.append(f"\n--- [[{p.stem}]] (in {folder}/) ---")
        lines.append(content)
    return "\n".join(lines)


def _parse_output(output: str) -> list[dict]:
    """Parse LLM output into page operations."""
    operations = []

    # Match NEW PAGE blocks
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

    # Match UPDATE PAGE blocks
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
            # Create subdirectory if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(op["content"] + "\n")
            created.append(page_id)
            _append_log(f"ingest | created {page_id}")

        elif op["type"] == "update":
            # Find existing page (may be in a subdirectory)
            existing_path = find_page(page_id)
            if existing_path:
                path = existing_path
            if path.exists():
                existing = path.read_text()
                # Update the 'updated' field in frontmatter
                today = date.today().isoformat()
                existing = re.sub(
                    r"updated:\s*.+",
                    f"updated: {today}",
                    existing,
                    count=1,
                )
                # Append new content
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


def run_ingest(content: str, job_id: str) -> None:
    """Run ingest in background thread."""
    job_store.update(job_id, status=JobStatus.RUNNING)

    try:
        context = _build_context(content)

        prompt = f"""{context}

---
Raw session data to ingest:
---
{content}
---

Extract wiki-worthy knowledge from the above and produce structured pages."""

        # Choose processor with retry
        processor = "ollama" if is_available() else "sonnet"
        output = None

        if processor == "ollama":
            for attempt in range(2):
                try:
                    output = generate(prompt, system=INGEST_SYSTEM_PROMPT)
                    break
                except Exception as e:
                    if attempt == 0:
                        _append_log(f"ingest | ollama attempt 1 failed: {e}, retrying...")
                        import time
                        time.sleep(5)
                    else:
                        _append_log(f"ingest | ollama failed after 2 attempts, falling back to sonnet")
                        processor = "sonnet"

        if processor == "sonnet" or output is None:
            processor = "sonnet"
            # TODO: Sonnet fallback implementation
            raise NotImplementedError("Sonnet fallback not yet implemented")

        job_store.update(job_id, processor=processor)

        # Parse and apply
        operations = _parse_output(output)
        if not operations:
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=_now(),
                result={"message": "No wiki-worthy content extracted"},
                pages_created=[],
                pages_updated=[],
            )
            return

        created, updated = _apply_operations(operations)
        _rebuild_index()

        # Update search embeddings for new/updated pages
        changed_pages = created + updated
        if changed_pages:
            try:
                from llm_wiki_mcp.search import update_embeddings
                update_embeddings(page_ids=changed_pages)
            except Exception:
                pass  # Non-critical: embeddings will be built on next reindex

        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=_now(),
            pages_created=created,
            pages_updated=updated,
        )

    except Exception as e:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            completed_at=_now(),
            error=str(e),
        )


def start_ingest(content: str) -> str:
    """Start an async ingest job. Returns job_id."""
    processor = "ollama" if is_available() else "sonnet"
    job = job_store.create(processor=processor)

    thread = threading.Thread(
        target=run_ingest,
        args=(content, job.job_id),
        daemon=True,
    )
    thread.start()

    return job.job_id


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
