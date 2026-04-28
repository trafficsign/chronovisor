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
    INGEST_SYSTEM_PROMPT, TRIAGE_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT, UPDATE_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Stage 1: Triage — analyze raw content and produce a structured plan
# ---------------------------------------------------------------------------

def _extract_json_array(output: str) -> list[dict] | None:
    """Best-effort extraction of a JSON array from an LLM response.

    Uses ``json.JSONDecoder.raw_decode`` to try every ``[`` position as the
    start of a valid JSON value, taking the longest array that parses. This
    is robust against:

    * preamble fluff (``---\\n[...]``, ``Here is the plan: [...]``).
    * postamble prose containing brackets (``[...]\\nNote: [done]``).
    * markdown code fences.
    * literal ``]`` inside summary fields (the parser doesn't care).

    Returns ``None`` on parse failure so the caller can distinguish failure
    from a legitimate empty plan (``[]``).
    """
    if not output:
        return None

    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    decoder = json.JSONDecoder()
    best: list | None = None
    best_len = -1
    pos = 0
    while True:
        idx = text.find("[", pos)
        if idx == -1:
            break
        try:
            value, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(value, list) and end > best_len:
            best = value
            best_len = end
        pos = idx + max(end, 1)

    return best


def _triage(content: str) -> list[dict] | None:
    """Stage 1: Analyze raw content and return a plan, or None on parse failure.

    Distinguishing ``None`` (parser/model failure) from ``[]`` (model said
    "nothing wiki-worthy") matters for the caller: failures should leave
    raw files un-marked so the next tick retries them, while a legitimate
    empty plan should mark the raws processed to avoid forever-retry.
    """
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
    plan = _extract_json_array(output)
    if plan is None:
        _append_log(
            f"ingest | triage parse failed (output preview: {output[:120]!r})"
        )
        return None
    return plan


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


_FRONTMATTER_BLOCK_RE = re.compile(
    r"^---\n.*?\n---(?:\n|$)", re.MULTILINE | re.DOTALL
)


def _has_frontmatter(text: str) -> bool:
    """True if ``text`` starts with a ``---\\n...\\n---\\n`` block containing ``title:``."""
    if not text.startswith("---\n"):
        return False
    m = _FRONTMATTER_BLOCK_RE.match(text)
    if not m:
        return False
    return bool(re.search(r"^title:\s*\S", m.group(0), re.MULTILINE))


def _strip_all_frontmatter(text: str) -> str:
    """Remove every ``---\\n...\\n---\\n`` block from ``text``.

    Used as a defensive scrub on update bodies: even if the model ignored
    UPDATE_SYSTEM_PROMPT and wrote frontmatter, we drop it before append.
    """
    return _FRONTMATTER_BLOCK_RE.sub("", text)


def _extract_page_body(output: str, op_type: str = "create") -> str | None:
    """Pull a page body out of generate-stage LLM output.

    Op-type rules:
    - ``create``: body MUST contain a frontmatter block with ``title:``. We accept
      strict (`=== NEW PAGE: ... ===`), lenient (`=== anything ===`), or
      no-wrapper formats — but reject anything without frontmatter so we never
      persist refusals or malformed pages.
    - ``update``: body MUST NOT contain a frontmatter block. Stray FM is
      stripped here as a belt-and-braces against UPDATE_SYSTEM_PROMPT
      drift. Empty body after stripping → reject.
    """
    if not output:
        return None

    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    body: str | None = None

    # 1. Strict: === NEW/UPDATE PAGE: filename === ... === END PAGE ===
    m = re.search(
        r"===\s*(?:NEW|UPDATE)\s+PAGE:\s*\S+\s*===\n(.*?)\n===\s*END\s+PAGE\s*===",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        body = m.group(1).strip() or None

    # 2. Lenient: === <anything> === ... === END PAGE === (NEW PAGE: keyword dropped)
    if body is None:
        m = re.search(
            r"===[^\n]*===\n(.*?)\n===\s*END\s+PAGE\s*===",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            body = m.group(1).strip() or None

    # 3. Fallback: bare frontmatter-led page with no wrapper.
    if body is None and _has_frontmatter(text):
        body = re.sub(
            r"\n=+\s*END[^\n]*\s*=+\s*$", "", text, flags=re.IGNORECASE
        ).strip() or None

    if body is None:
        return None

    # Op-type sanity: enforce frontmatter contract.
    op_type = (op_type or "create").lower()
    if op_type == "create":
        if not _has_frontmatter(body):
            return None
        return body

    # update: drop any stray FM blocks, reject if nothing meaningful is left.
    cleaned = _strip_all_frontmatter(body).strip()
    return cleaned or None


def _generate_one(op: dict, raw_content: str) -> dict | None:
    """Stage 2: generate one page; return an operation dict ready for apply."""
    context = _build_focused_context(op, raw_content)

    op_type = op.get("type", "create").lower()
    if op_type not in ("create", "update"):
        op_type = "create"
    filename = op.get("filename", "unknown.md")
    summary = op.get("summary", "")
    title = op.get("title", "")

    prompt = f"""{context}

---
Raw session data (source material):
---
{raw_content}
---

Task: {op_type.upper()} page "{filename}"
Title: {title}
Summary: {summary}

Generate the page content based on the raw data and context above."""

    system_prompt = UPDATE_SYSTEM_PROMPT if op_type == "update" else GENERATE_SYSTEM_PROMPT

    try:
        output = generate(prompt, system=system_prompt)
    except Exception as e:
        _append_log(f"ingest | generate failed for {filename}: {e}")
        return None

    body = _extract_page_body(output, op_type=op_type)
    if not body:
        _append_log(
            f"ingest | generate parse failed for {filename} ({op_type}, preview: {output[:120]!r})"
        )
        return None

    return {
        "type": op_type,
        "filename": filename,
        "content": body,
    }


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


def _reconcile_links(content: str, allowed_ids: set[str]) -> tuple[str, dict]:
    """Repair or unwrap [[wiki-links]] in prose, leaving code & frontmatter alone.

    For each ``[[target|alias#anchor]]`` outside of frontmatter / fenced code /
    inline code:

      * target resolves → leave intact (anchor + alias preserved).
      * target has ``folder/`` or ``.md`` clutter that strips to a known id
        → rewrite to the canonical form.
      * target unresolvable → unwrap to plain text (alias if given, else target),
        so the body keeps the entity name without polluting the link graph.

    Code / frontmatter regions are detected via ``link_fix`` (the existing
    canonical implementation used by lint/server) so we never break
    ``x = data[[1]]`` or fenced examples.

    Returns ``(rewritten_content, stats)``.
    """
    from llm_wiki_mcp.link_fix import (
        WIKI_LINK_RE,
        _FRONTMATTER_RE,
        _FENCED_CODE_RE,
        _INLINE_CODE_RE,
    )

    stats = {"resolved": 0, "rewritten": 0, "unwrapped": 0}

    # Pre-compute byte spans we must NOT touch.
    skip_ranges: list[tuple[int, int]] = []
    for pattern in (_FRONTMATTER_RE, _FENCED_CODE_RE, _INLINE_CODE_RE):
        for sm in pattern.finditer(content):
            skip_ranges.append(sm.span())
    skip_ranges.sort()

    def in_skip(pos: int) -> bool:
        # Linear scan is fine — N is small per page.
        for start, end in skip_ranges:
            if pos < start:
                return False
            if pos < end:
                return True
        return False

    def replace(m: re.Match) -> str:
        if in_skip(m.start()):
            return m.group(0)  # inside code/frontmatter — never rewrite

        inside = m.group(1)
        target_part, _, alias_raw = inside.partition("|")
        alias = alias_raw if "|" in inside else None
        if "#" in target_part:
            target, anchor_body = target_part.split("#", 1)
            anchor = "#" + anchor_body
        else:
            target, anchor = target_part, ""
        target = target.strip()

        if target in allowed_ids:
            stats["resolved"] += 1
            return m.group(0)

        # Try canonicalizing: strip a single leading folder and a trailing .md.
        candidate = target
        if "/" in candidate:
            candidate = candidate.rsplit("/", 1)[-1]
        if candidate.endswith(".md"):
            candidate = candidate[:-3]
        candidate = candidate.strip()
        if candidate and candidate != target and candidate in allowed_ids:
            stats["rewritten"] += 1
            tail = anchor + (f"|{alias}" if alias is not None else "")
            return f"[[{candidate}{tail}]]"

        # Unresolvable → unwrap to plain text. Keep alias as display, else target.
        stats["unwrapped"] += 1
        return alias if alias is not None else target

    new_content = WIKI_LINK_RE.sub(replace, content)
    return new_content, stats


class IngestApplyError(Exception):
    """Raised when an operation cannot be safely applied (fail-closed)."""


def _apply_operations(operations: list[dict]) -> tuple[list[str], list[str]]:
    """Apply page operations and return (created, updated) lists.

    Fail-closed: any unrecoverable problem (index_store unavailable, missing
    update target, write failure) raises ``IngestApplyError``. The caller
    surfaces that as ``JobStatus.FAILED`` *without* invoking ``on_complete``,
    so the source raws stay pending for retry.
    """
    from llm_wiki_mcp.link_fix import atomic_write

    created: list[str] = []
    updated: list[str] = []

    # Build the universe of valid link targets: every existing page plus every
    # page about to be created in this batch (so siblings can cross-reference).
    # We fail closed here — a stale or missing index would otherwise cause
    # _reconcile_links to silently unwrap every link in the corpus.
    try:
        from llm_wiki_mcp.index_store import get_store
        store = get_store()
        store.refresh()
        allowed_ids: set[str] = {
            m["page_id"] for m in store.all_pages_meta(include_system=True)
        }
    except Exception as e:
        raise IngestApplyError(f"index_store unavailable: {e}") from e

    for op in operations:
        fn = op["filename"]
        if not fn.endswith(".md"):
            fn += ".md"
        allowed_ids.add(Path(fn).stem)

    totals = {"resolved": 0, "rewritten": 0, "unwrapped": 0}

    for op in operations:
        filename = op["filename"]
        if not filename.endswith(".md"):
            filename += ".md"
        path = PAGES_DIR / filename
        page_id = path.stem

        body, stats = _reconcile_links(op["content"], allowed_ids)
        for k in totals:
            totals[k] += stats[k]

        if op["type"] == "create":
            if find_page(page_id) is not None:
                # Stem collision: a page with the same id already exists in a
                # different folder. Refuse silently overwriting it.
                raise IngestApplyError(
                    f"create op would overwrite existing page_id {page_id!r} "
                    f"(target path: {filename})"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(path, body + "\n")
            created.append(page_id)
            _append_log(f"ingest | created {page_id}")

        elif op["type"] == "update":
            existing_path = find_page(page_id)
            if existing_path is None or not existing_path.exists():
                raise IngestApplyError(
                    f"update target not found for page_id {page_id!r} "
                    f"(filename: {filename})"
                )
            existing = existing_path.read_text()
            today = date.today().isoformat()
            existing = re.sub(
                r"updated:\s*.+",
                f"updated: {today}",
                existing,
                count=1,
            )
            atomic_write(existing_path, existing.rstrip() + "\n\n" + body + "\n")
            updated.append(page_id)
            _append_log(f"ingest | updated {page_id}")

        else:
            raise IngestApplyError(f"unknown op type: {op.get('type')!r}")

    if any(totals.values()):
        _append_log(
            f"ingest | link reconcile: resolved={totals['resolved']} "
            f"rewritten={totals['rewritten']} unwrapped={totals['unwrapped']}"
        )

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

def run_ingest(
    content: str,
    job_id: str,
    on_complete: "callable | None" = None,
    on_finally: "callable | None" = None,
) -> None:
    """Run two-stage ingest in background thread.

    ``on_complete`` fires only on full success (so the orchestrator can mark
    raws processed). ``on_finally`` fires after every terminal state — success,
    partial, parse failure, or apply failure — so the orchestrator can release
    its in-flight lock and (on failure) bump retry counters.
    """
    job_store.update(job_id, status=JobStatus.RUNNING)
    failed = True  # flipped to False on full-success path

    try:
        processor = "ollama" if is_available() else "sonnet"
        if processor == "sonnet":
            raise NotImplementedError("Sonnet fallback not yet implemented")
        job_store.update(job_id, processor=processor)

        # Stage 1: Triage
        job_store.update(job_id, stage="triage")
        _append_log("ingest | stage 1: triage started")
        plan = _triage(content)

        # None = parser/model failure → leave raws un-marked so the next
        # tick can retry. [] = model legitimately said "nothing here" →
        # mark the raws processed (via on_complete) so we don't loop on
        # them forever.
        if plan is None:
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                completed_at=_now(),
                error="triage parse failed",
            )
            _append_log("ingest | triage: parse failed (raws left pending for retry)")
            return

        if not plan:
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=_now(),
                result={"message": "No wiki-worthy content extracted"},
                pages_created=[],
                pages_updated=[],
            )
            _append_log("ingest | triage: no operations planned (raws marked processed)")
            failed = False
            if on_complete:
                try:
                    on_complete()
                except Exception as cb_err:
                    _append_log(f"ingest | on_complete callback failed: {cb_err}")
            return

        _append_log(f"ingest | triage: {len(plan)} operations planned")
        job_store.update(job_id, stage="generate", total_ops=len(plan), completed_ops=0)

        # Stage 2: Generate each page
        all_operations: list[dict] = []
        failed_ops: list[str] = []
        for i, op in enumerate(plan):
            fname = op.get("filename", "?")
            _append_log(f"ingest | generating {i+1}/{len(plan)}: {fname}")
            generated = _generate_one(op, content)
            if generated:
                all_operations.append(generated)
            else:
                failed_ops.append(fname)
            job_store.update(job_id, completed_ops=i + 1)

        # Apply whatever succeeded. Fail-closed: a hard error during apply
        # raises out of this try block and is caught by the outer except,
        # which marks the job FAILED *without* invoking on_complete.
        if all_operations:
            created, updated = _apply_operations(all_operations)
            _rebuild_index()

            try:
                from llm_wiki_mcp.index_store import get_store
                get_store().refresh()
            except Exception as e:
                _append_log(f"ingest | index_store refresh failed: {e}")

            changed_pages = created + updated
            if changed_pages:
                try:
                    from llm_wiki_mcp.search import update_embeddings
                    update_embeddings(page_ids=changed_pages)
                except Exception:
                    pass
        else:
            created, updated = [], []

        # Distinguish full success / partial / total failure. Only full success
        # marks raws processed — anything less keeps them pending for retry so
        # we never lose the source content because of a flaky generate call.
        if failed_ops:
            status = JobStatus.FAILED if not all_operations else JobStatus.COMPLETED
            err = (
                f"generate failed for {len(failed_ops)}/{len(plan)} ops: "
                f"{', '.join(failed_ops[:5])}"
                + ("..." if len(failed_ops) > 5 else "")
            )
            job_store.update(
                job_id,
                status=status,
                completed_at=_now(),
                pages_created=created,
                pages_updated=updated,
                error=err,
            )
            _append_log(
                f"ingest | partial: applied {len(created)} created / "
                f"{len(updated)} updated, but {len(failed_ops)} ops failed "
                f"— raws left pending for retry"
            )
            return

        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=_now(),
            pages_created=created,
            pages_updated=updated,
        )
        _append_log(f"ingest | completed: {len(created)} created, {len(updated)} updated")
        failed = False

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
    finally:
        if on_finally:
            try:
                on_finally(failed=failed)
            except Exception as cb_err:
                _append_log(f"ingest | on_finally callback failed: {cb_err}")


def start_ingest(
    content: str,
    on_complete: "callable | None" = None,
    on_finally: "callable | None" = None,
) -> str:
    """Start an async ingest job. Returns job_id."""
    processor = "ollama" if is_available() else "sonnet"
    job = job_store.create(processor=processor)

    thread = threading.Thread(
        target=run_ingest,
        args=(content, job.job_id, on_complete, on_finally),
        daemon=True,
    )
    thread.start()

    return job.job_id


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
