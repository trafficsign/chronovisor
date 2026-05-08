"""Ingest engine - structures raw data into wiki pages (two-stage pipeline)."""

import json
import re
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from llm_wiki_mcp.wiki import PAGES_DIR, INDEX_FILE, LOG_FILE, all_pages, find_page, page_id_from_path
from llm_wiki_mcp.jobs import job_store, JobStatus
from llm_wiki_mcp.ollama import (
    generate, is_available,
    TRIAGE_SYSTEM_PROMPT,
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
    n = len(text)
    first_idx_in_text = text.find("[")
    if first_idx_in_text == -1:
        return None

    candidates: list[tuple[int, int, list]] = []  # (idx, consumed, value)
    pos = 0
    while pos < n:
        idx = text.find("[", pos)
        if idx == -1:
            break
        try:
            # Pass the full text + an offset so we don't re-allocate a slice
            # for every candidate position (was O(N²) on bracket-heavy input).
            value, end_offset = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(value, list):
            candidates.append((idx, end_offset - idx, value))
        pos = max(end_offset, idx + 1)

    if not candidates:
        return None

    # If the outermost `[` parsed cleanly, trust the LLM's intent and return
    # the longest array we found (preserves the historical "preamble [done]
    # then real plan" behavior).
    if candidates[0][0] == first_idx_in_text:
        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates[0][2]

    # The outer array did not parse — usually because the model truncated
    # mid-stream. raw_decode then picks up inner ``"keywords": []`` or
    # ``"keywords": ["x", "y"]`` lists as the "best" valid array, which
    # silently routes truncated triage as either "nothing wiki-worthy"
    # (empty plan → raws marked processed) or "schema invalid" (counted
    # toward dead-letter quarantine). Both are wrong: the LLM had more to
    # say. Only accept inner matches that fit the contract — non-empty
    # arrays of dicts; otherwise return ``None`` so the caller treats this
    # as a parse failure and the raws stay pending for retry.
    dict_arrays = [
        (consumed, value)
        for _, consumed, value in candidates
        if value and all(isinstance(e, dict) for e in value)
    ]
    if dict_arrays:
        dict_arrays.sort(reverse=True)
        return dict_arrays[0][1]
    return None


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
    raw_plan = _extract_json_array(output)
    if raw_plan is None:
        _safe_log(
            f"ingest | triage parse failed (output preview: {output[:120]!r})"
        )
        return None
    validated = _validate_triage_plan(raw_plan)
    if validated is None:
        _safe_log(
            f"ingest | triage schema invalid (preview: {str(raw_plan)[:120]!r})"
        )
        return None
    return validated


# Filename hardening: kebab-case ASCII, optional single folder segment,
# .md suffix, capped length. Anything else is treated as a triage failure
# so it accrues toward dead-letter instead of crashing later in apply.
_FILENAME_PATTERN = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?/)?"  # optional folder/
    r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"          # stem
    r"(?:\.md)?$"                               # optional .md
)
_MAX_FILENAME_LEN = 200


def _validate_triage_plan(plan: list) -> list[dict] | None:
    """Reject any plan that doesn't match the documented operation schema.

    Validation is **op-type aware**:

    * ``create``: filename must be ASCII kebab-case (``[a-z0-9-]``,
      ≤200 chars, optional single folder segment, optional ``.md``).
      We're choosing the canonical id for a brand-new page, so strict
      hygiene is appropriate.
    * ``update``: filename must resolve to an existing page in the
      corpus. Legacy pages predating the kebab rule (e.g. ``Foo.md``,
      ``snake_case.md``, non-ASCII titles) must remain updatable; the
      strict regex would block them forever. We only reject control
      characters and length blowups here.

    Anything else (string entries, nonsense types, missing filenames,
    control chars) returns ``None`` so the caller treats it as a triage
    failure and counts it toward dead-letter quarantine.

    Empty plan ([]) is valid and means "nothing wiki-worthy".
    """
    if not isinstance(plan, list):
        return None
    cleaned: list[dict] = []
    for entry in plan:
        if not isinstance(entry, dict):
            return None
        op_type = entry.get("type")
        if op_type not in ("create", "update"):
            return None
        filename = entry.get("filename")
        if not isinstance(filename, str):
            return None
        fn = filename.strip()
        if not fn or len(fn) > _MAX_FILENAME_LEN:
            return None
        # Reject any control char (NUL, newline, tab, etc.) for any op type.
        if any(ord(c) < 0x20 or c == "\x7f" for c in fn):
            return None
        # Reject path-traversal markers up front for both op types — the
        # apply layer would catch these but we want them to count as a
        # triage failure (LLM produced garbage), not an apply failure.
        if ".." in fn.split("/"):
            return None
        if op_type == "create":
            if not _FILENAME_PATTERN.fullmatch(fn):
                return None
        # For update we don't enforce kebab — apply will look the page up
        # via find_page() (case-insensitive on macOS APFS) and reject if
        # the target doesn't exist. That way legacy corpus stays updatable.
        cleaned.append(entry)
    return cleaned


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

    # 4. Truncation fallback: the wrapper opened (``=== ... ===``) but the
    #    model ran out of tokens before emitting ``=== END PAGE ===``. Patterns
    #    1 and 2 require the close, and pattern 3 requires the body to start
    #    with ``---\n`` — none of them salvage this very common Ollama failure
    #    mode. Peel the leading wrapper line and trust the remainder; the
    #    op-type contract check at the bottom (frontmatter required for
    #    create, no FM for update) still gates output.
    if body is None:
        m = re.match(r"===[^\n]*===\n(.*)$", text, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            # Strip any partial trailing close fence: anything from the last
            # ``\n=`` to EOF. This covers full ``=== END PAGE ===``, partials
            # like ``=== EN`` or ``===`` alone, and any other ``=``-prefixed
            # truncation tail, without touching mid-body lines.
            candidate = re.sub(r"\n=[^\n]*$", "", candidate).strip()
            body = candidate or None

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
    if not cleaned:
        return None
    # A partial frontmatter (opening `---` with no closing) cannot be safely
    # stripped; refuse rather than appending it raw.
    has_open_fence = bool(re.match(r"^---\s*$", cleaned, re.MULTILINE))
    has_closed_block = bool(_FRONTMATTER_BLOCK_RE.search(cleaned))
    if has_open_fence and not has_closed_block:
        return None
    return cleaned


def _generate_one(
    op: dict,
    raw_content: str,
    *,
    raw_keywords: list[str] | None = None,
) -> dict | None:
    """Stage 2: generate one page; return an operation dict ready for apply.

    ``raw_keywords`` is a side-channel list lifted from the source raw's
    frontmatter (not from triage). It rides on the returned operation dict
    so the apply layer can patch it onto the page frontmatter without an
    extra LLM round-trip. ``None`` means "no metadata propagation" — the
    field is omitted from the output, distinguishing it from an explicit
    empty list which would survive as ``[]``.
    """
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
        _safe_log(f"ingest | generate failed for {filename}: {e}")
        return None

    body = _extract_page_body(output, op_type=op_type)
    if not body:
        _safe_log(
            f"ingest | generate parse failed for {filename} ({op_type}, preview: {output[:120]!r})"
        )
        return None

    result: dict = {
        "type": op_type,
        "filename": filename,
        "content": body,
    }
    if raw_keywords is not None:
        result["raw_keywords"] = list(raw_keywords)
    return result


# ---------------------------------------------------------------------------
# Apply (link reconciliation, write phase, rollback)
# ---------------------------------------------------------------------------


def _fenced_code_spans(text: str) -> list[tuple[int, int]]:
    """Return spans of fenced code blocks, including unclosed ones at EOF.

    `_FENCED_CODE_RE` from ``link_fix`` requires a closing ```````,
    so a model that emits an opener without a closer (truncation, formatting
    error) leaves the rest of the body unprotected. Walk the text once,
    toggling on every ```````: if we end inside a fence, treat
    the trailing region as a fence too.
    """
    from llm_wiki_mcp.link_fix import fenced_code_spans
    return fenced_code_spans(text)


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
    ``x = data[[1]]`` or fenced examples. Unclosed fences (truncated LLM
    output) are also covered — we treat everything after the dangling
    opener as code.

    Returns ``(rewritten_content, stats)``.
    """
    from llm_wiki_mcp.link_fix import (
        WIKI_LINK_RE,
        position_in_spans,
        protected_spans,
    )

    stats = {"resolved": 0, "rewritten": 0, "unwrapped": 0}

    # Pre-compute byte spans we must NOT touch.
    skip_ranges = protected_spans(content)

    def replace(m: re.Match) -> str:
        if position_in_spans(m.start(), skip_ranges):
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


def _safe_resolve_page_path(filename: str) -> Path:
    """Resolve ``filename`` to a path strictly under ``PAGES_DIR``.

    The triage stage is LLM-controlled and can be steered by adversarial
    raw content, so we treat its filenames as untrusted input. Reject:

      * absolute paths (``/etc/passwd``)
      * parent traversal (``../../etc/passwd``)
      * symlink-escape after resolution
      * empty / whitespace-only / dotfile-only filenames

    Returns the resolved path; raises :class:`IngestApplyError` otherwise.
    """
    if not filename or not filename.strip():
        raise IngestApplyError("empty filename")

    # Normalize the .md suffix so callers don't need to do it themselves.
    fn = filename.strip()
    if not fn.endswith(".md"):
        fn = fn + ".md"

    candidate = Path(fn)
    if candidate.is_absolute():
        raise IngestApplyError(f"absolute filename refused: {filename!r}")
    if any(part in ("..", "") for part in candidate.parts):
        raise IngestApplyError(f"parent-traversal filename refused: {filename!r}")
    # Disallow filenames that resolve outside PAGES_DIR (e.g. via symlink).
    pages_root = PAGES_DIR.resolve()
    full = (PAGES_DIR / candidate).resolve()
    try:
        full.relative_to(pages_root)
    except ValueError as e:
        raise IngestApplyError(
            f"filename escapes PAGES_DIR: {filename!r}"
        ) from e

    if full.name in (".md", ""):
        raise IngestApplyError(f"degenerate filename: {filename!r}")

    return full


def _normalize_for_collision(name: str) -> str:
    """Canonical key for case- and Unicode-insensitive collision detection.

    macOS's default APFS is case-insensitive AND can ship the same logical
    name in two byte representations (NFC vs NFD): ``café.md`` (NFC,
    one ``é``) and ``café.md`` (NFD, ``e`` + combining acute) resolve to
    the same file but compare as different strings. NFC-normalize first,
    then casefold.
    """
    import unicodedata
    return unicodedata.normalize("NFC", name).casefold()


def _find_page_casefold(page_id: str) -> Path | None:
    """find_page with macOS-case-insensitive + NFC-normalized semantics."""
    direct = find_page(page_id)
    if direct is not None:
        return direct
    target = _normalize_for_collision(page_id)
    for p in PAGES_DIR.rglob("*.md"):
        if _normalize_for_collision(p.stem) == target:
            return p
    return None


def _process_tags_in_body(
    body: str, existing_tags: list[str], parse, patch
) -> str:
    """For ``create`` bodies: validate, dedupe, record tags from frontmatter.

    Soft-fail throughout — a malformed tag drops itself rather than
    aborting the page. Strict enforcement is wiki_check's job.

    Steps for each tag in the LLM output:
      1. ``validate_tag`` — drop on form-rule failure
      2. ``dedupe_with_existing`` — if cosine similarity to an existing
         same-axis tag is ``>= 0.80``, replace the new tag with the
         existing one (prevents proliferation of near-synonyms)
      3. ``record_new_tag`` — append truly-new tags to the changelog
    """
    from llm_wiki_mcp.tags import (
        dedupe_with_existing,
        record_new_tag,
        validate_tag,
    )

    meta, _ = parse(body)
    tags_raw = meta.get("tags")
    if not isinstance(tags_raw, list) or not tags_raw:
        return body

    existing_set = set(existing_tags)
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags_raw:
        if not isinstance(raw_tag, str):
            continue
        ok, _reason = validate_tag(raw_tag)
        if not ok:
            continue
        # Dedup against the corpus first; the LLM may have invented a
        # synonym for something we already have.
        canonical = dedupe_with_existing(raw_tag, existing_tags, threshold=0.80)
        if canonical in seen:
            continue
        seen.add(canonical)
        cleaned.append(canonical)
        # Audit only tags that survived as truly new (not redirected by
        # dedupe and not present in the corpus).
        if canonical == raw_tag and canonical not in existing_set:
            record_new_tag(canonical, reason="ingest auto-gen")

    if cleaned == tags_raw:
        return body
    return patch(body, {"tags": cleaned})


def _apply_operations(operations: list[dict]) -> tuple[list[str], list[str]]:
    """Apply page operations and return (created, updated) lists.

    Two-phase contract: a **prepare phase** validates every op and stages
    the final on-disk content; only after every op passes does the **write
    phase** mutate disk. If a write fails midway, we attempt a best-effort
    rollback (delete created files, restore updated bodies) so the wiki
    never stays in a partially-applied state.

    Fail-closed: any unrecoverable problem raises :class:`IngestApplyError`.
    The caller marks the job FAILED without invoking ``on_complete``.

    Phase 4 propagation: any op that carries a non-empty ``raw_keywords``
    list (from the source raw's frontmatter, riding on metadata since
    Phase 3) gets that list patched onto the page frontmatter inside the
    prepare phase — never inside the write phase, so a partial-write
    rollback restores either the pre-batch text or nothing at all, never
    a half-patched frontmatter.

    Plan-4 tag processing: ``create`` op bodies whose generated frontmatter
    already includes a ``tags:`` list (per ``GENERATE_SYSTEM_PROMPT``) get
    each tag form-validated, dedup'd against the existing corpus's tag
    pool (cosine similarity >= 0.80 → reuse), and audited via
    ``tag-changelog.md``. ``update`` ops never touch ``tags`` because
    ``UPDATE_SYSTEM_PROMPT`` forbids the LLM from emitting frontmatter.
    """
    from llm_wiki_mcp.link_fix import atomic_write
    from llm_wiki_mcp.frontmatter import (
        parse as _frontmatter_parse,
        patch as _frontmatter_patch,
    )

    # Build the universe of valid link targets: every existing page plus every
    # page about to be created in this batch (so siblings can cross-reference).
    # Fail closed — stale or missing index would silently unwrap every link.
    try:
        from llm_wiki_mcp.index_store import get_store
        store = get_store()
        store.refresh()
        allowed_ids: set[str] = {
            m["page_id"] for m in store.all_pages_meta(include_system=True)
        }
        # Snapshot the tag pool once for the whole batch so dedupe doesn't
        # re-walk the index on every op. Same-batch siblings can't see
        # each other's newly-coined tags here, but that's fine: dedup is
        # only meaningful against the *committed* corpus, and within-batch
        # divergence will be reconciled the next time wiki_check runs.
        existing_tags_snapshot = store.all_tags(include_system=False)
    except Exception as e:
        raise IngestApplyError(f"index_store unavailable: {e}") from e

    # ---- Prepare phase -----------------------------------------------------
    # Resolve every filename, validate every op, build the final write plan.
    # Nothing here touches disk except for read-only stat/read calls.

    @dataclass
    class _Plan:
        op_type: str  # "create" | "update"
        path: Path
        page_id: str
        new_body: str  # bytes-equivalent; what we'll write
        previous_text: str | None  # for rollback on update; None for create

    planned: list[_Plan] = []
    seen_norm_ids: set[str] = set()
    seen_paths: set[Path] = set()

    for op in operations:
        op_type = op.get("type")
        if op_type not in ("create", "update"):
            raise IngestApplyError(f"unknown op type: {op_type!r}")

        full_path = _safe_resolve_page_path(op["filename"])
        page_id = full_path.stem

        # Detect intra-batch dups using the same case/Unicode-insensitive key
        # we use against the existing corpus, so two ops whose ids differ
        # only in case or NFC/NFD form are caught before any write.
        norm_key = _normalize_for_collision(page_id)
        if norm_key in seen_norm_ids:
            raise IngestApplyError(
                f"duplicate page_id within batch (case/Unicode-insensitive): "
                f"{page_id!r}"
            )
        if full_path in seen_paths:
            raise IngestApplyError(
                f"duplicate target path within batch: {full_path}"
            )
        seen_norm_ids.add(norm_key)
        seen_paths.add(full_path)

        allowed_ids.add(page_id)

    totals = {"resolved": 0, "rewritten": 0, "unwrapped": 0}

    for op in operations:
        op_type = op["type"]
        full_path = _safe_resolve_page_path(op["filename"])
        page_id = full_path.stem

        body, stats = _reconcile_links(op["content"], allowed_ids)
        for k in totals:
            totals[k] += stats[k]

        # Phase 4: lift the raw_keywords side channel off the op. Empty
        # lists are treated as "no propagation" — writing ``raw_keywords:
        # []`` to a page would create a zero-information diff against the
        # existing frontmatter. The propagate flag distinguishes "list[str]
        # with content" from anything else.
        op_raw_keywords = op.get("raw_keywords")
        propagate_raw_keywords = (
            isinstance(op_raw_keywords, list)
            and all(isinstance(v, str) for v in op_raw_keywords)
            and len(op_raw_keywords) > 0
        )

        if op_type == "create":
            existing = _find_page_casefold(page_id)
            if existing is not None:
                raise IngestApplyError(
                    f"create op would overwrite existing page_id {page_id!r} "
                    f"(existing: {existing}, target: {full_path})"
                )
            # Tag processing happens BEFORE raw_keywords patch so the
            # final frontmatter goes through one consistent serialization
            # path. Soft-fail: a missing or malformed ``tags`` list just
            # passes the body through unchanged — wiki_check lint will
            # surface the absent-tags case for human attention.
            body = _process_tags_in_body(
                body,
                existing_tags_snapshot,
                _frontmatter_parse,
                _frontmatter_patch,
            )
            if propagate_raw_keywords:
                # generate output already carries a frontmatter block
                # (enforced by ``_extract_page_body`` for create), so
                # ``patch`` will splice raw_keywords into it without
                # synthesizing a new block.
                body = _frontmatter_patch(body, {"raw_keywords": op_raw_keywords})
            planned.append(
                _Plan(
                    op_type="create",
                    path=full_path,
                    page_id=page_id,
                    new_body=body.rstrip() + "\n",
                    previous_text=None,
                )
            )

        else:  # update
            existing_path = find_page(page_id)
            if existing_path is None or not existing_path.exists():
                raise IngestApplyError(
                    f"update target not found for page_id {page_id!r}"
                )
            previous = existing_path.read_text()
            # Preserve the on-disk text for rollback BEFORE we mutate
            # ``previous`` with a frontmatter patch — the rollback path
            # restores the file as it was before this batch ran, not as
            # it was after the patch.
            previous_text_for_rollback = previous

            # raw_keywords union with the existing page's value, preserving
            # insertion order so the diff stays deterministic. If the
            # existing field is missing or malformed (legacy data, manual
            # edit), treat it as empty rather than raising — the apply
            # phase shouldn't reject otherwise-valid updates because of
            # frontmatter rot somewhere upstream.
            if propagate_raw_keywords:
                existing_meta, _existing_body = _frontmatter_parse(previous)
                existing_kw_raw = existing_meta.get("raw_keywords")
                if isinstance(existing_kw_raw, list) and all(
                    isinstance(v, str) for v in existing_kw_raw
                ):
                    existing_kw = existing_kw_raw
                else:
                    existing_kw = []
                union_kw = list(dict.fromkeys(existing_kw + op_raw_keywords))
                previous = _frontmatter_patch(previous, {"raw_keywords": union_kw})

            today = date.today().isoformat()
            stamped = re.sub(
                r"updated:\s*.+",
                f"updated: {today}",
                previous,
                count=1,
            )
            new_body = stamped.rstrip() + "\n\n" + body + "\n"
            planned.append(
                _Plan(
                    op_type="update",
                    path=existing_path,
                    page_id=page_id,
                    new_body=new_body,
                    previous_text=previous_text_for_rollback,
                )
            )

    # ---- Write phase -------------------------------------------------------
    # Each atomic_write is itself crash-safe (tempfile + os.replace + fsync).
    # If any write raises (disk full, permissions, etc.) we best-effort
    # rollback every page we already touched in this batch.

    written: list[_Plan] = []
    created: list[str] = []
    updated: list[str] = []

    try:
        for entry in planned:
            if entry.op_type == "create":
                entry.path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(entry.path, entry.new_body)
                # Append BEFORE logging so a log failure could never drop
                # an entry from the rollback set. _safe_log additionally
                # ensures a logging exception (which atomic_write success
                # already proves is irrelevant to data) never triggers
                # rollback of a write that succeeded.
                written.append(entry)
                created.append(entry.page_id)
                _safe_log(f"ingest | created {entry.page_id}")
            else:
                atomic_write(entry.path, entry.new_body)
                written.append(entry)
                updated.append(entry.page_id)
                _safe_log(f"ingest | updated {entry.page_id}")
    except Exception as write_err:
        # Best-effort rollback. Each revert is gated by a CAS check: only
        # restore if the file still contains exactly what we wrote. If
        # another writer (e.g. wiki_apply) has modified it since, we leave
        # their change intact and surface that fact in the raised error.
        rollback_errors: list[str] = []
        for entry in reversed(written):
            try:
                if entry.op_type == "create":
                    if entry.path.exists() and entry.path.read_text() == entry.new_body:
                        entry.path.unlink()
                    elif entry.path.exists():
                        rollback_errors.append(
                            f"{entry.page_id}: skipped (modified by another writer)"
                        )
                else:
                    if entry.path.exists() and entry.path.read_text() == entry.new_body:
                        atomic_write(entry.path, entry.previous_text or "")
                    elif entry.path.exists():
                        rollback_errors.append(
                            f"{entry.page_id}: skipped (modified by another writer)"
                        )
            except Exception as rb_err:
                rollback_errors.append(f"{entry.page_id}: {rb_err}")
        if rollback_errors:
            partial_summary = "; ".join(rollback_errors)
            _safe_log(
                "ingest | rollback partial (other writers or IO failures): "
                + partial_summary
            )
            raise IngestApplyError(
                f"apply write failed: {write_err}; partial rollback: "
                f"{partial_summary}"
            ) from write_err
        _safe_log(
            f"ingest | rolled back {len(written)} writes after error: {write_err}"
        )
        raise IngestApplyError(f"apply write failed: {write_err}") from write_err

    if any(totals.values()):
        _safe_log(
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
    """Append to log.md. Failures are intentionally swallowed.

    A dropped log line is recoverable; an exception escaping into the
    ingest pipeline is not. Letting an IO error here propagate would,
    for example, override a freshly-set ``COMPLETED`` job status with
    ``FAILED`` from the outer except block and skip ``on_complete`` —
    leaving disk pages persisted but raws marked pending, so the next
    tick collides on every page we just created.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"\n- [{timestamp}] {message}")
    except Exception:
        pass


def _safe_log(message: str) -> None:
    """Defense-in-depth wrapper used by atomicity-critical call sites.

    ``_append_log`` is already internally crash-safe, but a test (or a
    monkeypatch in some future caller) can replace it with something
    that raises. The rollback path and the post-apply success path
    cannot afford to propagate such exceptions: doing so would either
    break atomicity (rollback aborts mid-loop) or override a
    successfully-set ``COMPLETED`` status with ``FAILED`` and skip
    ``on_complete``. So we wrap, swallow, and move on.
    """
    try:
        _append_log(message)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main entry point — two-stage pipeline
# ---------------------------------------------------------------------------

def run_ingest(
    content: str,
    job_id: str,
    on_complete: "callable | None" = None,
    on_finally: "callable | None" = None,
    *,
    metadata: dict | None = None,
) -> None:
    """Run two-stage ingest in background thread.

    ``on_complete`` fires whenever any pages were applied — full success or
    partial-with-some-ops-failed. The orchestrator uses it to mark raws
    processed; partial apply still counts because the next tick mustn't
    re-triage the same content (that's how the prior contract caused
    duplicate page creation). ``on_finally`` fires after every terminal
    state — success, partial, parse failure, or apply failure — and is
    called with two flags:

      * ``failed`` — True for any non-success terminal state.
      * ``triage_failed`` — True only when triage (stage 1) could not produce
        a parseable plan. The orchestrator counts these specifically: a
        triage failure means the raw content itself is unprocessable, so
        repeated occurrences justify quarantine. Other failures (Ollama
        unavailable, generate parse, apply error) are transient or per-op
        and must not feed the dead-letter counter.

    ``metadata`` is a keyword-only optional dict carrying side-channel data
    that should be attached to the resulting operations. Currently supports
    ``raw_keywords: list[str]`` — keywords lifted from the raw frontmatter
    that need to land on the page frontmatter without re-running an LLM.
    Unknown keys are ignored so future extensions don't break callers.
    """
    job_store.update(job_id, status=JobStatus.RUNNING)
    failed = True  # flipped to False on full-success path
    triage_failed = False

    # Extract the raw_keywords side channel from metadata once, up front.
    # Every operation generated from this raw shares the same propagated
    # value: a single raw can produce N operations (e.g. 1 create + 1
    # update from the same session), and the source-of-truth keywords
    # belong to all of them. Anything that isn't a list[str] is treated
    # as "no metadata" so we don't fabricate values.
    raw_keywords_for_ops: list[str] | None = None
    if metadata is not None:
        candidate = metadata.get("raw_keywords")
        if isinstance(candidate, list) and all(isinstance(v, str) for v in candidate):
            raw_keywords_for_ops = list(candidate)

    try:
        processor = "ollama" if is_available() else "sonnet"
        if processor == "sonnet":
            raise NotImplementedError("Sonnet fallback not yet implemented")
        job_store.update(job_id, processor=processor)

        # Stage 1: Triage. Every log call here is _safe_log so a wedged
        # log file can't promote a successful triage into a FAILED job.
        job_store.update(job_id, stage="triage")
        _safe_log("ingest | stage 1: triage started")
        plan = _triage(content)

        # None = parser/model failure → leave raws un-marked so the next
        # tick can retry. [] = model legitimately said "nothing here" →
        # mark the raws processed (via on_complete) so we don't loop on
        # them forever.
        if plan is None:
            triage_failed = True
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                completed_at=_now(),
                error="triage parse failed",
            )
            _safe_log("ingest | triage: parse failed (raws left pending for retry)")
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
            _safe_log("ingest | triage: no operations planned (raws marked processed)")
            failed = False
            if on_complete:
                try:
                    on_complete()
                except Exception as cb_err:
                    _safe_log(f"ingest | on_complete callback failed: {cb_err}")
            return

        _safe_log(f"ingest | triage: {len(plan)} operations planned")
        job_store.update(job_id, stage="generate", total_ops=len(plan), completed_ops=0)

        # Stage 2: Generate each page. Failed ops are retried once before
        # being dead-lettered — most generate failures are transient
        # (truncation, malformed wrapper) and a second sample from the
        # model usually succeeds.
        all_operations: list[dict] = []
        failed_op_specs: list[dict] = []  # full op dicts for the dead-letter record
        for i, op in enumerate(plan):
            fname = op.get("filename", "?")
            _safe_log(f"ingest | generating {i+1}/{len(plan)}: {fname}")
            generated = _generate_one(op, content, raw_keywords=raw_keywords_for_ops)
            if generated is None:
                _safe_log(f"ingest | retry generate for {fname}")
                generated = _generate_one(op, content, raw_keywords=raw_keywords_for_ops)
            if generated:
                all_operations.append(generated)
            else:
                failed_op_specs.append({
                    "filename": fname,
                    "type": op.get("type", "?"),
                    "title": op.get("title", ""),
                    "summary": op.get("summary", ""),
                })
            job_store.update(job_id, completed_ops=i + 1)

        failed_ops = [spec["filename"] for spec in failed_op_specs]

        # Apply policy:
        # - All ops failed → discard, leave raws pending so the next tick
        #   can re-triage with fresh model output. This avoids creating an
        #   empty "completed" job that silently consumes the raws.
        # - Some ops succeeded, some failed → apply the successful ops AND
        #   mark raws processed (via on_complete). The previous "discard
        #   everything" contract caused two problems: (1) successful ops
        #   were thrown away, and (2) raws stayed pending so the next tick
        #   re-triaged the same content, often producing a different plan
        #   that would either redo work or generate divergent page_ids.
        #   Marking raws processed after a partial apply prevents both.
        # - All ops succeeded → standard happy path.
        if failed_ops and not all_operations:
            err = (
                f"generate failed for all {len(failed_ops)}/{len(plan)} ops: "
                f"{', '.join(failed_ops[:5])}"
                + ("..." if len(failed_ops) > 5 else "")
            )
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                completed_at=_now(),
                pages_created=[],
                pages_updated=[],
                error=err,
            )
            _safe_log(
                f"ingest | all {len(failed_ops)} generate ops failed "
                f"— discarded, raws left pending for retry"
            )
            return

        # All ops generated successfully (or partial — apply what we have).
        # _apply_operations raises IngestApplyError on any unrecoverable
        # problem; the outer except marks the job FAILED *without* invoking
        # on_complete, so raws stay pending for retry.
        created, updated = _apply_operations(all_operations)

        # Side effects (rebuild_index, IndexStore refresh, embeddings) are
        # derived artifacts. Pages are already on disk; failures here must
        # NOT undo the apply or block on_complete — that would leave raws
        # pending forever and re-create the same pages on retry. Use
        # _safe_log so a logging error in a side-effect handler doesn't
        # promote a derived-artifact failure into a hard ingest failure.
        try:
            _rebuild_index()
        except Exception as e:
            _safe_log(f"ingest | index.md rebuild failed (non-fatal): {e}")

        try:
            from llm_wiki_mcp.index_store import get_store
            get_store().refresh()
        except Exception as e:
            _safe_log(f"ingest | index_store refresh failed: {e}")

        changed_pages = created + updated
        if changed_pages:
            try:
                from llm_wiki_mcp.search import update_embeddings
                update_embeddings(page_ids=changed_pages)
            except Exception:
                pass

        # Build job result. For partial runs, surface the failed op specs
        # so a human can `wiki_jobs <id>` to see what was dropped.
        job_result: dict | None = None
        if failed_op_specs:
            job_result = {
                "partial": True,
                "failed_ops": failed_op_specs,
            }

        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=_now(),
            pages_created=created,
            pages_updated=updated,
            result=job_result,
        )
        # _safe_log so a log failure here can't fall through to the outer
        # except, override COMPLETED with FAILED, and skip on_complete.
        # That was the R5-Critical regression path.
        if failed_op_specs:
            _safe_log(
                f"ingest | partial: {len(created)} created, {len(updated)} updated, "
                f"{len(failed_op_specs)} dead-lettered "
                f"({', '.join(failed_ops[:3])}"
                + ("..." if len(failed_ops) > 3 else "")
                + ")"
            )
        else:
            _safe_log(
                f"ingest | completed: {len(created)} created, {len(updated)} updated"
            )
        failed = False

        if on_complete:
            try:
                on_complete()
            except Exception as cb_err:
                _safe_log(f"ingest | on_complete callback failed: {cb_err}")

    except Exception as e:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            completed_at=_now(),
            error=str(e),
        )
        _safe_log(f"ingest | failed: {e}")
    finally:
        if on_finally:
            try:
                on_finally(failed=failed, triage_failed=triage_failed)
            except Exception as cb_err:
                _safe_log(f"ingest | on_finally callback failed: {cb_err}")


def start_ingest(
    content: str,
    on_complete: "callable | None" = None,
    on_finally: "callable | None" = None,
    *,
    metadata: dict | None = None,
) -> str:
    """Start an async ingest job. Returns job_id.

    ``metadata`` is forwarded to :func:`run_ingest` (keyword-only) so callers
    that want raw-side context (e.g. ``raw_keywords``) propagated to the
    resulting operations can pass it through without changing positional
    argument order.
    """
    processor = "ollama" if is_available() else "sonnet"
    job = job_store.create(processor=processor)

    thread = threading.Thread(
        target=run_ingest,
        args=(content, job.job_id, on_complete, on_finally),
        kwargs={"metadata": metadata},
        daemon=True,
    )
    thread.start()

    return job.job_id


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
