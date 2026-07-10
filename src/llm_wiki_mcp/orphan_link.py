"""Orphan link proposal and autonomous convergence.

The legacy report API generates a *dry-run* listing for every orphan page. The
nightly API uses the same candidate/scoring path, then applies only a
frontier-approved suggestion with a content compare-and-swap.

The report lists the existing
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
    6. Either write a diagnostic Markdown report without page mutations, or
       pass the best proposal through bounded frontier review and CAS apply.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from llm_wiki_mcp.wiki import find_page
from llm_wiki_mcp.wiki import WIKI_ROOT
from llm_wiki_mcp.link_fix import atomic_write, protected_spans
from llm_wiki_mcp.page_mutation import wiki_mutation_lock


DECISIONS_FILE = WIKI_ROOT / "autonomy" / "orphan-link-decisions.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOLVER_VERSION = "orphan-link-v1"
DEFAULT_FRONTIER_CONFIDENCE_THRESHOLD = 0.8

ORPHAN_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {"type": "string", "enum": ["approved", "rejected", "needs_retry"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}


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
    if isinstance(obj["confidence"], bool) or not isinstance(
        obj["confidence"], (int, float)
    ):
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
    outcome = _score_candidate_outcome(source_id, orphan_id, store, generate_fn)
    score = outcome.get("score")
    return score if isinstance(score, dict) else None


def _score_candidate_outcome(
    source_id: str,
    orphan_id: str,
    store,
    generate_fn: Callable[..., str],
) -> dict[str, Any]:
    """Keep transient model failures distinct from a valid low score."""
    prompt = _build_prompt(source_id, orphan_id, store)
    try:
        raw = generate_fn(prompt, system=SUGGESTION_SYSTEM_PROMPT)
    except Exception as exc:
        return {
            "status": "call_error",
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    parsed = parse_llm_response(raw)
    if parsed is None:
        return {
            "status": "schema_error",
            "error": "local suggestion did not match the required schema",
        }
    return {"status": "ok", "score": parsed}


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


def _content_hash(page_id: str) -> str:
    path = find_page(page_id)
    if path is None:
        return "missing"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def _wiki_link_spans(text: str) -> list[tuple[int, int]]:
    """Return spans for complete and dangling wiki links.

    Anchors inside an existing ``[[...]]`` must never be wrapped again: doing
    so creates nested markup that the wiki-link parser cannot recover from.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    while True:
        start = text.find("[[", offset)
        if start < 0:
            break
        end = text.find("]]", start + 2)
        if end < 0:
            line_end = text.find("\n", start + 2)
            spans.append((start, len(text) if line_end < 0 else line_end))
            offset = len(text) if line_end < 0 else line_end + 1
            continue
        spans.append((start, end + 2))
        offset = end + 2
    return spans


def _sanitize_section_heading(value: object) -> str:
    """Reduce an untrusted model heading to one plain single-line label."""
    first_line = str(value or "").splitlines()[0].strip() if str(value or "") else ""
    first_line = first_line.lstrip("#").strip()
    cleaned = "".join(
        char
        for char in first_line
        if char.isalnum() or char in {" ", "-", "_", "/", "&", "・", "／"}
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()[:80]
    return cleaned or "Related"


def apply_suggestion(
    orphan_id: str,
    suggestion: Suggestion,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply one frontier-approved inbound link with a content CAS."""
    source_path = find_page(suggestion.source_page_id)
    target_path = find_page(orphan_id)
    if source_path is None or target_path is None:
        return {"status": "error", "reason": "source_or_target_missing"}
    try:
        original = source_path.read_text(encoding="utf-8")
        target_preimage = target_path.read_bytes()
    except OSError as exc:
        return {"status": "error", "reason": f"read_error:{exc}"}
    link = f"[[{orphan_id}]]"
    if re.search(r"\[\[" + re.escape(orphan_id) + r"(?:[#|\]])", original):
        return {"status": "already_applied", "source": suggestion.source_page_id, "target": orphan_id}

    updated = original
    anchor = suggestion.suggested_anchor.strip()
    if (
        "\n" in anchor
        or "\r" in anchor
        or "[[" in anchor
        or "]]" in anchor
        or len(anchor) > 200
    ):
        anchor = ""
    if anchor:
        spans = sorted([*protected_spans(original), *_wiki_link_spans(original)])
        positions = [match.start() for match in re.finditer(re.escape(anchor), original)]
        position = next(
            (
                pos
                for pos in positions
                if not any(
                    pos < protected_end and pos + len(anchor) > protected_start
                    for protected_start, protected_end in spans
                )
            ),
            None,
        )
        if position is not None:
            replacement = f"[[{orphan_id}|{anchor}]]"
            updated = original[:position] + replacement + original[position + len(anchor):]
    if updated == original:
        section = _sanitize_section_heading(suggestion.suggested_section)
        suffix = "" if original.endswith("\n") else "\n"
        updated = f"{original}{suffix}\n## {section}\n\n- {link}\n"
    if dry_run:
        return {
            "status": "dry_run",
            "source": suggestion.source_page_id,
            "target": orphan_id,
            "changed": updated != original,
        }
    wrote_source = False
    try:
        with wiki_mutation_lock():
            try:
                if source_path.read_text(encoding="utf-8") != original:
                    return {"status": "retry", "reason": "source_changed_before_apply"}
                if target_path.read_bytes() != target_preimage:
                    return {"status": "retry", "reason": "target_changed_before_apply"}
                atomic_write(source_path, updated)
                wrote_source = True
                written = source_path.read_text(encoding="utf-8")
                target_after = target_path.read_bytes()
            except OSError as exc:
                if wrote_source:
                    try:
                        if source_path.read_text(encoding="utf-8") == updated:
                            atomic_write(source_path, original)
                    except OSError:
                        pass
                return {"status": "error", "reason": f"write_error:{exc}"}
            if f"[[{orphan_id}" not in written or target_after != target_preimage:
                # Roll back under the same lock, and only while the page still
                # contains our exact bytes.  A foreign post-write edit wins.
                try:
                    if source_path.read_text(encoding="utf-8") == updated:
                        atomic_write(source_path, original)
                except OSError:
                    pass
                return {"status": "error", "reason": "post_write_verification_failed"}
    except OSError as exc:
        return {"status": "error", "reason": f"write_error:{exc}"}
    return {"status": "applied", "source": suggestion.source_page_id, "target": orphan_id}


def _frontier_failure_class(review: dict[str, Any]) -> str | None:
    failure = review.get("frontier_failure")
    if isinstance(failure, dict) and isinstance(failure.get("failure_class"), str):
        return failure["failure_class"]
    return None


def _normalize_frontier_review(value: object) -> dict[str, Any]:
    """Fail closed for custom reviewers that bypass Codex output-schema."""
    if not isinstance(value, Mapping):
        return {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": "frontier result is not an object",
            "valid": False,
        }
    review = dict(value)
    decision = review.get("decision")
    summary = review.get("summary")
    confidence = review.get("confidence")
    errors: list[str] = []
    if decision not in {"approved", "rejected", "needs_retry"}:
        errors.append("invalid decision")
        decision = "needs_retry"
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary is required")
        summary = str(summary or "frontier result is missing a summary")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append("confidence must be numeric")
        numeric_confidence = 0.0
    else:
        numeric_confidence = float(confidence)
        if not 0.0 <= numeric_confidence <= 1.0:
            errors.append("confidence is outside [0, 1]")
            numeric_confidence = 0.0
    return {
        **review,
        "decision": decision if not errors else "needs_retry",
        "confidence": numeric_confidence,
        "summary": summary.strip(),
        "valid": not errors,
        "validation_errors": errors,
    }


def _review_orphan_proposal(
    orphan_id: str,
    proposal: dict[str, Any],
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    kind = str(proposal.get("kind") or "")
    candidate: dict[str, Any] = {
        "proposal_kind": kind,
        "orphan_page_id": orphan_id,
        "target_excerpt": _page_head(orphan_id, max_chars=1200),
        "proposal": proposal,
    }
    raw_suggestion = proposal.get("suggestion")
    if kind == "link" and isinstance(raw_suggestion, dict):
        suggestion = Suggestion(**raw_suggestion)
        candidate.update(
            {
                "source_page_id": suggestion.source_page_id,
                "confidence": suggestion.confidence,
                "reason": suggestion.reason,
                "suggested_anchor": suggestion.suggested_anchor,
                "suggested_section": suggestion.suggested_section,
                "source_excerpt": _page_head(
                    suggestion.source_page_id, max_chars=1200
                ),
            }
        )
    if reviewer is not None:
        return reviewer(candidate)
    from llm_wiki_mcp.frontier_review import run_structured_review

    prompt = f"""\
You are the final autonomous reviewer for an LLM Wiki orphan-link disposition.
For proposal_kind=link, approve only if SOURCE naturally benefits from linking
to TARGET. For proposal_kind=no_link, approve only if the supplied candidates
support the conclusion that no safe link should be created. For
proposal_kind=retry, approve only if evidence is genuinely unavailable or
transiently broken and another autonomous attempt is required. Reject an
unsupported disposition. Do not edit files or ask a human. Return JSON matching
the schema.

Candidate:
{json.dumps(candidate, ensure_ascii=False, indent=2)}
"""
    return run_structured_review(prompt, ORPHAN_FRONTIER_SCHEMA, repo_root=PROJECT_ROOT)


def run_autonomous(
    *,
    orphan_limit: int = 3,
    max_candidates: int = 3,
    confidence_threshold: float = 0.75,
    store=None,
    generate_fn: Callable[..., str] | None = None,
    semantic_search_fn: Callable[[str, int], list] | None = None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    convergence_store=None,
    budget=None,
    frontier_confidence_threshold: float = DEFAULT_FRONTIER_CONFIDENCE_THRESHOLD,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Boundedly drain orphan proposals through local + frontier review."""
    from llm_wiki_mcp.convergence import (
        ConvergenceStore,
        CycleBudget,
        TERMINAL_STATUSES,
        stable_item_key,
    )

    if store is None:
        from llm_wiki_mcp.index_store import get_store

        store = get_store()
        store.refresh()
    if generate_fn is None:
        from llm_wiki_mcp.ollama import generate

        generate_fn = generate
    state = convergence_store or ConvergenceStore()
    cycle_budget = budget or CycleBudget(max_local_calls=max(1, orphan_limit * max_candidates), max_frontier_calls=2, max_mutations=2)
    orphans = store.orphans(include_system=False)
    retired_absent = state.retire_absent_sources(
        lane="orphan_link",
        active_source_ids={f"orphan:{page_id}" for page_id in orphans},
        reason="page_is_no_longer_orphaned",
        dry_run=dry_run,
    )
    retired_stale = state.retire_stale(
        lane="orphan_link",
        reason="orphan_candidate_expired",
        dry_run=dry_run,
    )
    work_limit = max(0, int(orphan_limit))
    frontier_confidence_threshold = max(0.0, min(1.0, float(frontier_confidence_threshold)))
    results: list[dict[str, Any]] = []
    work_items = 0
    scanned = 0

    for orphan_id in orphans:
        scanned += 1
        discovery_error: str | None = None
        try:
            candidates = gather_candidates(
                orphan_id,
                store,
                max_candidates=max_candidates,
                semantic_search_fn=semantic_search_fn,
            )
        except Exception as exc:
            candidates = []
            discovery_error = f"{exc.__class__.__name__}: {exc}"
        # The production semantic path returns [] both for an empty result and
        # for unavailable/missing embeddings. Treat that ambiguous condition as
        # retryable. Injected search functions in tests/tools can explicitly
        # establish a deterministic empty result.
        if not candidates and semantic_search_fn is None and discovery_error is None:
            discovery_error = "production semantic search returned no candidates"
        source_id = candidates[0] if candidates else ""
        input_data = {
            "orphan": orphan_id,
            "orphan_hash": _content_hash(orphan_id),
            "candidates": [
                {"source": candidate_id, "source_hash": _content_hash(candidate_id)}
                for candidate_id in candidates
            ],
        }
        key = stable_item_key(
            "orphan_link",
            f"orphan:{orphan_id}",
            input_data,
            resolver_version=RESOLVER_VERSION,
        )
        existing = state.get(key)
        if existing is not None and existing.get("status") in TERMINAL_STATUSES:
            results.append(
                {
                    "orphan": orphan_id,
                    "source": source_id,
                    "status": existing.get("status"),
                    "cached": True,
                    "key": key,
                }
            )
            continue

        # Candidate discovery with no result is resolved/retried durably below,
        # but it performs no model review and therefore must not consume the
        # bounded model-work allowance or starve later actionable orphans.
        if candidates and work_items >= work_limit:
            break
        if existing is None:
            merged = state.merge_item(
                lane="orphan_link",
                source_id=f"orphan:{orphan_id}",
                input_data=input_data,
                resolver_version=RESOLVER_VERSION,
                metadata={
                    "orphan": orphan_id,
                    "source": source_id,
                    "candidate_discovery_error": discovery_error,
                },
                dry_run=dry_run,
            )
            item = merged["item"]
        else:
            # In particular, retain the locally-approved suggestion while a
            # frontier retry/backoff is pending.  Re-merging observational
            # metadata here would otherwise replace that durable hand-off.
            item = existing
        key = item["key"]
        # A concurrent worker may have completed the item after the preflight
        # read but before the merge lock was acquired.
        if item.get("status") in TERMINAL_STATUSES:
            results.append(
                {
                    "orphan": orphan_id,
                    "source": source_id,
                    "status": item.get("status"),
                    "cached": True,
                    "key": key,
                }
            )
            continue
        if dry_run:
            if candidates:
                work_items += 1
            status = (
                "would_retry_candidate_discovery"
                if discovery_error
                else "would_reject_no_candidate"
                if not candidates
                else "would_process"
            )
            results.append({"orphan": orphan_id, "source": source_id, "status": status, "key": key})
            continue
        if not candidates and item.get("status") in {"pending_local", "local_retry"}:
            claim = state.claim_attempt(key, "local")
            if not claim["claimed"]:
                results.append({"orphan": orphan_id, "status": claim["reason"], "key": key})
                continue
            frontier_proposal = {
                "kind": "retry" if discovery_error else "no_link",
                "reason": discovery_error or "no_semantic_candidate",
                "candidates": [],
                "failure_class": (
                    "candidate_discovery_error" if discovery_error else None
                ),
            }
            state.merge_item(
                lane="orphan_link",
                source_id=f"orphan:{orphan_id}",
                input_data=input_data,
                resolver_version=RESOLVER_VERSION,
                metadata={
                    "orphan": orphan_id,
                    "source": source_id,
                    "candidate_discovery_error": discovery_error,
                    "frontier_proposal": frontier_proposal,
                },
            )
            state.escalate(
                key,
                reason="deterministic orphan disposition requires frontier final review",
                owner=claim["owner"],
            )
            item = state.get(key) or item

        item_counted = False
        if item["status"] == "pending_local" or item["status"] == "local_retry":
            claim = state.claim_attempt(key, "local", budget=cycle_budget)
            if not claim["claimed"]:
                results.append({"orphan": orphan_id, "status": claim["reason"], "key": key})
                continue
            work_items += 1
            item_counted = True
            local_suggestion: Suggestion | None = None
            valid_scores: list[dict[str, Any]] = []
            local_errors: list[dict[str, Any]] = []
            for index, candidate_id in enumerate(candidates):
                if index:
                    allowed, budget_reason = cycle_budget.consume("local")
                    if not allowed:
                        local_errors.append({"status": "budget_deferred", "error": budget_reason})
                        break
                outcome = _score_candidate_outcome(candidate_id, orphan_id, store, generate_fn)
                scored = outcome.get("score")
                if isinstance(scored, dict):
                    valid_scores.append(scored)
                else:
                    local_errors.append(
                        {"source": candidate_id, "status": outcome.get("status"), "error": outcome.get("error")}
                    )
                    continue
                if scored["confidence"] >= confidence_threshold:
                    local_suggestion = Suggestion(source_page_id=candidate_id, **scored)
                    break
            if local_suggestion is None:
                if valid_scores and not local_errors:
                    frontier_proposal = {
                        "kind": "no_link",
                        "reason": "all_local_candidates_below_confidence_threshold",
                        "max_confidence": max(
                            float(score["confidence"]) for score in valid_scores
                        ),
                        "scores": valid_scores,
                    }
                else:
                    frontier_proposal = {
                        "kind": "retry",
                        "reason": (
                            "; ".join(
                                str(error.get("error") or error.get("status"))
                                for error in local_errors
                            )
                            or "all local candidate reviews failed"
                        ),
                        "failure_class": "local_model_or_schema_error",
                        "scores": valid_scores,
                        "local_errors": local_errors,
                    }
                suggestion_payload = None
            else:
                suggestion_payload = local_suggestion.__dict__
                frontier_proposal = {
                    "kind": "link",
                    "suggestion": suggestion_payload,
                }
            state.merge_item(
                lane="orphan_link",
                source_id=f"orphan:{orphan_id}",
                input_data=input_data,
                resolver_version=RESOLVER_VERSION,
                metadata={
                    "orphan": orphan_id,
                    "source": source_id,
                    "candidate_discovery_error": None,
                    "suggestion": suggestion_payload,
                    "frontier_proposal": frontier_proposal,
                },
            )
            state.escalate(key, reason="local suggestion requires frontier final review", owner=claim["owner"])
            item = state.get(key) or item

        if item.get("status") not in {"pending_frontier", "frontier_retry"}:
            results.append({"orphan": orphan_id, "status": item.get("status"), "key": key})
            continue
        claim = state.claim_attempt(key, "frontier", budget=cycle_budget)
        if not claim["claimed"]:
            results.append({"orphan": orphan_id, "status": claim["reason"], "key": key})
            continue
        metadata = (state.get(key) or {}).get("metadata") or {}
        frontier_proposal = (
            metadata.get("frontier_proposal") if isinstance(metadata, dict) else None
        )
        raw_suggestion = metadata.get("suggestion") if isinstance(metadata, dict) else None
        if not isinstance(frontier_proposal, dict) and isinstance(raw_suggestion, dict):
            frontier_proposal = {"kind": "link", "suggestion": raw_suggestion}
        if not isinstance(frontier_proposal, dict) or frontier_proposal.get("kind") not in {
            "link",
            "no_link",
            "retry",
        }:
            state.fail_attempt(
                key,
                "frontier",
                error="durable frontier proposal missing",
                owner=claim["owner"],
            )
            results.append({"orphan": orphan_id, "status": "frontier_retry", "key": key})
            continue
        proposal_kind = str(frontier_proposal["kind"])
        if not item_counted and not (
            proposal_kind == "no_link" and not candidates
        ):
            work_items += 1
        suggestion: Suggestion | None = None
        if proposal_kind == "link":
            proposal_suggestion = frontier_proposal.get("suggestion")
            if not isinstance(proposal_suggestion, dict):
                proposal_suggestion = raw_suggestion
            try:
                suggestion = Suggestion(**proposal_suggestion)
            except (TypeError, ValueError) as exc:
                failed = state.fail_attempt(
                    key,
                    "frontier",
                    error=f"invalid persisted suggestion: {exc}",
                    owner=claim["owner"],
                )
                results.append(
                    {"orphan": orphan_id, "status": failed["item"]["status"], "key": key}
                )
                continue
        try:
            raw_review = _review_orphan_proposal(
                orphan_id,
                frontier_proposal,
                reviewer=reviewer,
            )
        except Exception as exc:
            raw_review = {
                "decision": "needs_retry",
                "confidence": 0.0,
                "summary": f"{exc.__class__.__name__}: {exc}",
            }
        review = _normalize_frontier_review(raw_review)
        decision = review.get("decision")
        if decision == "rejected":
            if proposal_kind in {"link", "retry"}:
                state.complete(
                    key,
                    "rejected",
                    result={"frontier": review, "proposal": frontier_proposal},
                    owner=claim["owner"],
                )
                results.append({"orphan": orphan_id, "status": "rejected", "key": key})
            else:
                failed = state.fail_attempt(
                    key,
                    "frontier",
                    error=str(review.get("summary") or "frontier rejected no-link disposition"),
                    owner=claim["owner"],
                )
                results.append(
                    {"orphan": orphan_id, "status": failed["item"]["status"], "key": key}
                )
            continue
        if decision != "approved":
            failed = state.fail_attempt(
                key,
                "frontier",
                error=str(review.get("summary") or "frontier needs retry"),
                failure_class=_frontier_failure_class(review),
                owner=claim["owner"],
            )
            results.append({"orphan": orphan_id, "status": failed["item"]["status"], "key": key})
            continue
        if float(review.get("confidence") or 0.0) < frontier_confidence_threshold:
            failed = state.fail_attempt(
                key,
                "frontier",
                error="frontier_confidence_below_threshold",
                owner=claim["owner"],
            )
            results.append(
                {"orphan": orphan_id, "status": failed["item"]["status"], "key": key}
            )
            continue
        if proposal_kind == "no_link":
            completed = state.complete(
                key,
                "rejected",
                result={"frontier": review, "proposal": frontier_proposal},
                owner=claim["owner"],
            )
            results.append(
                {"orphan": orphan_id, "status": completed["item"]["status"], "key": key}
            )
            continue
        if proposal_kind == "retry":
            failed = state.fail_attempt(
                key,
                "frontier",
                error=str(frontier_proposal.get("reason") or "autonomous retry required"),
                failure_class=str(frontier_proposal.get("failure_class") or "") or None,
                owner=claim["owner"],
            )
            results.append(
                {"orphan": orphan_id, "status": failed["item"]["status"], "key": key}
            )
            continue
        assert suggestion is not None
        allowed, reason = cycle_budget.consume("mutation")
        if not allowed:
            state.fail_attempt(key, "frontier", error=reason, owner=claim["owner"])
            results.append({"orphan": orphan_id, "status": reason, "key": key})
            continue
        applied = apply_suggestion(orphan_id, suggestion)
        if applied["status"] in {"applied", "already_applied"}:
            state.complete(key, "applied", result={"frontier": review, "apply": applied}, owner=claim["owner"])
        else:
            state.fail_attempt(key, "frontier", error=str(applied.get("reason") or applied["status"]), owner=claim["owner"])
        results.append({"orphan": orphan_id, "status": (state.get(key) or {}).get("status"), "key": key})

    return {
        "status": "ok",
        "orphans_seen": scanned,
        "orphans_total": len(orphans),
        "work_items": work_items,
        "results": results,
        "retired": sorted(
            set(retired_absent.get("retired", []))
            | set(retired_stale.get("retired", []))
        ),
        "budget": cycle_budget.snapshot(),
        "dry_run": dry_run,
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
    lines.append("Pages are not modified by this diagnostic report; nightly convergence reviews and applies bounded proposals.")
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
