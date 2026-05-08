"""Orphan link suggestion (plan-2).

Generates a *dry-run* report listing, for every orphan page, the existing
``source`` pages most likely to benefit from gaining an inbound
``[[orphan]]`` link. Direction matters: this module proposes
``source_page → orphan_page`` edges, NOT the reverse — adding outbound
links from the orphan would not change ``orphan_count``.

The pipeline is:
    1. Pull orphans from ``IndexStore.orphans()``.
    2. For each orphan, build a query (title + body head) and use the
       existing semantic search to surface candidate sources.
    3. Filter candidates to ``pages/`` only (no system, no the orphan
       itself, no duplicates), and rank by well-connectedness so the
       LLM scores the most-cited candidates first.
    4. Ask the LLM, per (source, orphan) pair, whether adding the link
       makes contextual sense — emit a structured JSON answer with
       ``confidence`` / ``reason`` / ``suggested_anchor`` / ``suggested_section``.
       Page IDs never appear in the LLM output (they're held in the
       caller's context, fabrication is impossible).
    5. Drop any candidate below ``confidence_threshold``.
    6. Write a human-reviewable Markdown report. Pages on disk are NOT
       modified.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from llm_wiki_mcp.wiki import find_page


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Suggestion:
    source_page_id: str
    confidence: float
    reason: str
    suggested_anchor: str
    suggested_section: str


@dataclass
class OrphanReport:
    orphan_page_id: str
    orphan_title: str
    candidates_considered: int
    suggestions: list[Suggestion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM contract
# ---------------------------------------------------------------------------


SUGGESTION_SYSTEM_PROMPT = """\
You are evaluating whether adding a wiki link from a SOURCE page to a
TARGET page is contextually appropriate.

You will be given:
  - SOURCE page: title + body head
  - TARGET page: title + body head

Your job: judge whether the SOURCE page's content would naturally
warrant a [[wiki-link]] to TARGET, and where to place it.

Output a SINGLE JSON object with exactly these fields, no extra text:

{
  "confidence": <float 0.0 to 1.0>,
  "reason": "<one short sentence in Japanese explaining the call>",
  "suggested_anchor": "<short phrase from the SOURCE body where the link belongs, or empty string>",
  "suggested_section": "<existing or recommended section heading, e.g. '関連' / '参考' / 'See also'>"
}

Rules:
- Output JSON ONLY, no markdown fences, no preamble.
- Do NOT output page IDs — those are tracked by the caller.
- confidence < 0.5 means the link would feel forced; emit it honestly,
  the caller will drop it.
- Prefer anchors that already exist in the SOURCE body verbatim.
- If the link belongs in a brand-new section, name a conventional one
  (関連, 参考, See also).
"""


_ALLOWED_FIELDS = {"confidence", "reason", "suggested_anchor", "suggested_section"}


def parse_llm_response(raw: str) -> dict | None:
    """Validate the LLM response against the contract.

    Returns the cleaned dict, or ``None`` if anything's off (extra
    fields, missing required fields, wrong types, out-of-range
    confidence). Failing closed: a malformed response means we drop the
    candidate, not invent values.
    """
    if not raw:
        return None
    text = raw.strip()
    # Tolerate a single set of code fences.
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    # Find the first balanced top-level JSON object.
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) - _ALLOWED_FIELDS:
        return None
    if not _ALLOWED_FIELDS.issubset(obj.keys()):
        return None
    if not isinstance(obj["confidence"], (int, float)):
        return None
    confidence = float(obj["confidence"])
    if not (0.0 <= confidence <= 1.0):
        return None
    if not all(
        isinstance(obj[k], str)
        for k in ("reason", "suggested_anchor", "suggested_section")
    ):
        return None
    return {
        "confidence": confidence,
        "reason": obj["reason"].strip(),
        "suggested_anchor": obj["suggested_anchor"].strip(),
        "suggested_section": obj["suggested_section"].strip(),
    }


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def _page_head(page_id: str, max_chars: int = 500) -> str:
    """Return ``title + first N body chars`` for prompt construction."""
    path = find_page(page_id)
    if path is None:
        return ""
    try:
        text = path.read_text()
    except OSError:
        return ""
    # Strip frontmatter.
    body = re.sub(r"^---\n.*?\n---\n?", "", text, count=1, flags=re.DOTALL).lstrip()
    return body[:max_chars]


def _build_query(orphan_id: str, store) -> str:
    """Search query string from the orphan's title + body head."""
    meta = store.meta(orphan_id)
    title = meta["title"] if meta else orphan_id
    head = _page_head(orphan_id, max_chars=500)
    return f"{title}\n\n{head}"


def gather_candidates(
    orphan_id: str,
    store,
    max_candidates: int = 5,
    semantic_top_n: int = 20,
    semantic_search_fn: Callable[[str, int], list] | None = None,
) -> list[str]:
    """Return up to ``max_candidates`` source page IDs for this orphan.

    Filters: pages-only (no system), no orphan self-link, exists in
    IndexStore. Ranking: well-connectedness (``len(backlinks)`` desc),
    semantic score breaks ties.

    ``semantic_search_fn`` is injectable so tests can avoid the real
    embedding store. The default falls back to ``search.semantic_search``.
    """
    if semantic_search_fn is None:
        from llm_wiki_mcp.search import semantic_search
        semantic_search_fn = semantic_search

    query = _build_query(orphan_id, store)
    if not query.strip():
        return []
    results = semantic_search_fn(query, semantic_top_n)

    candidates: list[tuple[str, float, int]] = []  # (id, sem_score, backlink_count)
    seen: set[str] = set()
    for r in results:
        pid = getattr(r, "page_id", None)
        if not pid or pid == orphan_id or pid in seen:
            continue
        meta = store.meta(pid)
        if meta is None or meta.get("is_system"):
            continue
        seen.add(pid)
        backlink_count = len(store.backlinks(pid))
        candidates.append((pid, float(getattr(r, "score", 0.0)), backlink_count))

    # Well-connected first, semantic score as tiebreaker.
    candidates.sort(key=lambda t: (-t[2], -t[1]))
    return [c[0] for c in candidates[:max_candidates]]


# ---------------------------------------------------------------------------
# LLM scoring
# ---------------------------------------------------------------------------


def _build_prompt(source_id: str, orphan_id: str, store) -> str:
    src_meta = store.meta(source_id) or {}
    orph_meta = store.meta(orphan_id) or {}
    return f"""\
SOURCE page:
  title: {src_meta.get('title', source_id)}
  body head:
{_page_head(source_id, max_chars=500)}

TARGET page (orphan, currently has zero inbound links):
  title: {orph_meta.get('title', orphan_id)}
  body head:
{_page_head(orphan_id, max_chars=500)}

Question: should the SOURCE page gain an inbound link to TARGET? Output
one JSON object per the rules.
"""


def score_candidate(
    source_id: str,
    orphan_id: str,
    store,
    generate_fn: Callable[..., str],
) -> dict | None:
    """Ask the LLM to score one (source, orphan) pair. None on any failure
    (LLM down, malformed response, schema violation)."""
    prompt = _build_prompt(source_id, orphan_id, store)
    try:
        raw = generate_fn(prompt, system=SUGGESTION_SYSTEM_PROMPT)
    except Exception:
        return None
    return parse_llm_response(raw)


# ---------------------------------------------------------------------------
# Top-level dry-run runner
# ---------------------------------------------------------------------------


def run_dry_run(
    output_path: Path,
    *,
    max_candidates: int = 5,
    confidence_threshold: float = 0.5,
    semantic_top_n: int = 20,
    store=None,
    generate_fn: Callable[..., str] | None = None,
    semantic_search_fn: Callable[[str, int], list] | None = None,
    orphan_limit: int | None = None,
) -> dict:
    """Run the pipeline end-to-end and write a Markdown report.

    Returns a stats dict. Pages on disk are not modified — only
    ``output_path`` is written.

    ``orphan_limit`` truncates the orphan list (handy for sample runs
    before committing to the full ~335-orphan sweep).
    """
    if store is None:
        from llm_wiki_mcp.index_store import get_store
        store = get_store()
        store.refresh()
    if generate_fn is None:
        from llm_wiki_mcp.ollama import generate
        generate_fn = generate

    orphans = store.orphans(include_system=False)
    if orphan_limit is not None:
        orphans = orphans[:orphan_limit]

    reports: list[OrphanReport] = []
    started = datetime.now()

    for orphan_id in orphans:
        meta = store.meta(orphan_id) or {}
        report = OrphanReport(
            orphan_page_id=orphan_id,
            orphan_title=meta.get("title", orphan_id),
            candidates_considered=0,
        )
        candidates = gather_candidates(
            orphan_id,
            store,
            max_candidates=max_candidates,
            semantic_top_n=semantic_top_n,
            semantic_search_fn=semantic_search_fn,
        )
        report.candidates_considered = len(candidates)
        for source_id in candidates:
            scored = score_candidate(source_id, orphan_id, store, generate_fn)
            if scored is None:
                continue
            if scored["confidence"] < confidence_threshold:
                continue
            report.suggestions.append(
                Suggestion(
                    source_page_id=source_id,
                    confidence=scored["confidence"],
                    reason=scored["reason"],
                    suggested_anchor=scored["suggested_anchor"],
                    suggested_section=scored["suggested_section"],
                )
            )
        # Highest confidence first per orphan.
        report.suggestions.sort(key=lambda s: -s.confidence)
        reports.append(report)

    elapsed = (datetime.now() - started).total_seconds()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_report(reports, elapsed=elapsed, store=store))

    with_suggestion = sum(1 for r in reports if r.suggestions)
    total_suggestions = sum(len(r.suggestions) for r in reports)
    return {
        "orphans_total": len(orphans),
        "with_suggestion": with_suggestion,
        "without_suggestion": len(orphans) - with_suggestion,
        "total_suggestions": total_suggestions,
        "elapsed_seconds": round(elapsed, 1),
        "output_path": str(output_path),
    }


def format_report(
    reports: list[OrphanReport],
    *,
    elapsed: float = 0.0,
    store=None,
) -> str:
    """Render the dry-run report as Markdown."""
    today = date.today().isoformat()
    total_pages = "?" if store is None else str(len(store.all_page_ids(include_system=False)))
    with_sug = sum(1 for r in reports if r.suggestions)
    total_sug = sum(len(r.suggestions) for r in reports)

    confidences = [s.confidence for r in reports for s in r.suggestions]
    if confidences:
        conf_min = min(confidences)
        conf_max = max(confidences)
        conf_avg = sum(confidences) / len(confidences)
    else:
        conf_min = conf_max = conf_avg = 0.0

    lines: list[str] = []
    lines.append("---")
    lines.append("title: Orphan Link Suggestions (dry-run)")
    lines.append(f"updated: {today}")
    lines.append("---")
    lines.append("")
    lines.append("# Orphan Link Suggestions (dry-run)")
    lines.append("")
    lines.append("Direction: each suggestion proposes `source_page → orphan_page`.")
    lines.append("Pages on disk are NOT modified by this report — review and apply by hand.")
    lines.append("")
    lines.append("## Run statistics")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- elapsed_seconds: {elapsed:.1f}")
    lines.append(f"- total_pages: {total_pages}")
    lines.append(f"- orphans_evaluated: {len(reports)}")
    lines.append(f"- orphans_with_suggestion: {with_sug}")
    lines.append(f"- orphans_without_suggestion: {len(reports) - with_sug}")
    lines.append(f"- total_suggestions: {total_sug}")
    lines.append(
        f"- confidence: min={conf_min:.2f} avg={conf_avg:.2f} max={conf_max:.2f}"
    )
    lines.append("")
    lines.append("## Suggestions")
    lines.append("")

    for r in reports:
        lines.append(f"### orphan: `{r.orphan_page_id}`")
        lines.append(f"- title: {r.orphan_title}")
        lines.append(f"- candidates_considered: {r.candidates_considered}")
        if not r.suggestions:
            lines.append("- _no suggestion above threshold_")
            lines.append("")
            continue
        lines.append("- suggested incoming links (sources to add link FROM):")
        for s in r.suggestions:
            lines.append(
                f"  - [ ] **`{s.source_page_id}`** "
                f"confidence={s.confidence:.2f}, "
                f"anchor=\"{s.suggested_anchor}\", "
                f"section=\"{s.suggested_section}\""
            )
            lines.append(f"        reason: {s.reason}")
        lines.append("")

    return "\n".join(lines) + "\n"
