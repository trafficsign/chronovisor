"""Ingest engine - structures raw data into wiki pages."""

import re
import threading
from datetime import date
from pathlib import Path

from llm_wiki_mcp.wiki import PAGES_DIR, INDEX_FILE, LOG_FILE
from llm_wiki_mcp.jobs import job_store, JobStatus
from llm_wiki_mcp.ollama import generate, is_available, INGEST_SYSTEM_PROMPT


def _build_context(existing_pages: list[Path]) -> str:
    """Build context of existing pages for the LLM."""
    if not existing_pages:
        return "No existing pages in wiki."

    lines = ["Existing wiki pages:"]
    for p in existing_pages[:50]:  # Limit to avoid context overflow
        content = p.read_text()
        title_match = re.search(r"title:\s*(.+)", content)
        title = title_match.group(1) if title_match else p.stem
        lines.append(f"- [[{p.stem}]]: {title}")
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
            path.write_text(op["content"] + "\n")
            created.append(page_id)
            _append_log(f"ingest | created {page_id}")

        elif op["type"] == "update":
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
    pages = sorted(PAGES_DIR.glob("*.md"))
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
        existing_pages = list(PAGES_DIR.glob("*.md"))
        context = _build_context(existing_pages)

        prompt = f"""{context}

---
Raw session data to ingest:
---
{content}
---

Extract wiki-worthy knowledge from the above and produce structured pages."""

        # Choose processor
        if is_available():
            processor = "ollama"
            output = generate(prompt, system=INGEST_SYSTEM_PROMPT)
        else:
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
