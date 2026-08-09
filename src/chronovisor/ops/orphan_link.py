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

import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.hashutil import sha256_bytes as _sha256_bytes
from chronovisor.core.link_fix import atomic_write, protected_spans
from chronovisor.core.page_mutation import (
    chronovisor_mutation_lock,
    decision_authority_lock,
)
from chronovisor.core.runtime_config import (
    load_decision_router_config,
    runtime_repo_root,
)
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.decision.decision_authority import (
    compare_semantic_authority,
    current_semantic_authority,
    seal_semantic_artifact,
    semantic_verdict_authority_error,
    semantic_verdict_authority_provenance_error,
)
from chronovisor.decision.decision_schema_manifest import ORPHAN_FRONTIER_SCHEMA
from chronovisor.decision.local_structured import ChatRequest, LocalStructuredSession
from chronovisor.decision.semantic_hold import is_local_semantic_no_quorum

DECISIONS_FILE = CHRONOVISOR_ROOT / "autonomy" / "orphan-link-decisions.jsonl"
PROJECT_ROOT = runtime_repo_root()
RESOLVER_VERSION = "orphan-link-v1"
DECISION_LANE = "orphan_link"

ORPHAN_SUGGESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["confidence", "reason", "suggested_anchor", "suggested_section"],
    "properties": {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "suggested_anchor": {"type": "string"},
        "suggested_section": {"type": "string"},
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
        from chronovisor.search.search import semantic_search

        # Orphan resolution must distinguish a healthy empty result from the
        # search layer's normal fail-open ``[]``. Strict mode preserves that
        # distinction so legitimate no-candidate pages terminate as no-link,
        # while an unavailable semantic service remains retryable.
        def semantic_search_fn(query, top_n):
            return semantic_search(
                    query,
                    top_n,
                    strict=True,
                )

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
  title: {src_meta.get("title", source_id)}
  body head:
{_page_head(source_id, max_chars=500)}

TARGET page (orphan, currently has zero inbound links):
  title: {orph_meta.get("title", orphan_id)}
  body head:
{_page_head(orphan_id, max_chars=500)}

Question: should the SOURCE page gain an inbound link to TARGET? Output
one JSON object per the rules.
"""


def score_candidate(
    source_id: str,
    orphan_id: str,
    store,
    generate_fn: Callable[..., str] | None = None,
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
    generate_fn: Callable[..., str] | None,
) -> dict[str, Any]:
    """Keep transient model failures distinct from a valid low score."""
    prompt = _build_prompt(source_id, orphan_id, store)
    config = load_decision_router_config()
    last_output: list[str] = []
    transport = None
    if generate_fn is not None:

        def transport(request: ChatRequest) -> str:
            system = request.messages[0]["content"] if request.messages else ""
            transcript = "\n\n".join(
                f"<{message['role'].upper()}>\n{message['content']}"
                for message in request.messages[1:]
            )
            output = generate_fn(transcript, system=system)
            last_output[:] = [output]
            return output

    session_kwargs = {
        "model": config.primary_model,
        "transport": transport,
        "role": "orphan_candidate",
        "num_ctx": config.num_ctx,
        "num_predict": min(config.num_predict, 1_024),
        "keep_alive": config.primary_keep_alive,
        "read_timeout_ms": config.read_timeout_ms,
        "max_input_chars": min(config.max_input_chars, 16_000),
        "max_output_chars": min(config.max_output_chars, 4_000),
        "max_feedback_chars": config.max_feedback_chars,
    }
    if generate_fn is None:
        result = LocalStructuredSession(**session_kwargs).run(
            prompt,
            ORPHAN_SUGGESTION_SCHEMA,
            system=SUGGESTION_SYSTEM_PROMPT,
        )
    else:
        # Injected generators are a compatibility/test seam, not a production
        # decision source. Keep their audit artifacts isolated and ephemeral.
        with tempfile.TemporaryDirectory(prefix="chronovisor-orphan-structured-") as root:
            result = LocalStructuredSession(
                **session_kwargs,
                audit_root=Path(root),
            ).run(
                prompt,
                ORPHAN_SUGGESTION_SCHEMA,
                system=SUGGESTION_SYSTEM_PROMPT,
            )
    if not result.ok:
        return {
            "status": (
                "call_error"
                if result.failure_class in {"transport_error", "transport_timeout"}
                else "schema_error"
            ),
            "error": result.failure_reason
            or result.failure_class
            or "structured review failed",
            "last_output": last_output[-1] if last_output else "",
        }
    parsed = result.value if isinstance(result.value, dict) else None
    if parsed is not None:
        parsed = parse_llm_response(json.dumps(parsed, ensure_ascii=False))
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
        from chronovisor.search.index_store import get_store

        store = get_store()
        store.refresh()
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




def _render_suggestion_postimage(
    original: str,
    orphan_id: str,
    suggestion: Suggestion,
) -> str:
    """Render the exact source-page postimage without mutating the Wiki."""
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
        positions = [
            match.start() for match in re.finditer(re.escape(anchor), original)
        ]
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
            updated = (
                original[:position] + replacement + original[position + len(anchor) :]
            )
    if updated == original:
        section = _sanitize_section_heading(suggestion.suggested_section)
        suffix = "" if original.endswith("\n") else "\n"
        updated = f"{original}{suffix}\n## {section}\n\n- [[{orphan_id}]]\n"
    return updated


def _prepare_suggestion_effect(
    orphan_id: str,
    suggestion: Suggestion,
    *,
    convergence_key: str,
    proposal_fingerprint: str,
    semantic_artifact: Mapping[str, Any],
    claim_owner: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build a durable, exact-CAS page effect from current page bytes."""
    source_path = find_page(suggestion.source_page_id)
    target_path = find_page(orphan_id)
    if source_path is None or target_path is None:
        return None, {"status": "error", "reason": "source_or_target_missing"}
    try:
        source_preimage = source_path.read_bytes()
        target_preimage = target_path.read_bytes()
        original = source_preimage.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, {"status": "error", "reason": f"read_error:{exc}"}
    if re.search(r"\[\[" + re.escape(orphan_id) + r"(?:[#|\]])", original):
        return None, {
            "status": "already_applied",
            "source": suggestion.source_page_id,
            "target": orphan_id,
            "recovery_only": False,
            "semantic_effect": False,
        }
    postimage = _render_suggestion_postimage(original, orphan_id, suggestion)
    postimage_bytes = postimage.encode("utf-8")
    return (
        {
            "schema_version": 2,
            "kind": "orphan_link_page_effect",
            "convergence_key": convergence_key,
            "proposal_fingerprint": proposal_fingerprint,
            "claim_owner": claim_owner,
            "source_page_id": suggestion.source_page_id,
            "target_page_id": orphan_id,
            "source_preimage_sha256": _sha256_bytes(source_preimage),
            "source_preimage_size": len(source_preimage),
            "target_preimage_sha256": _sha256_bytes(target_preimage),
            "target_preimage_size": len(target_preimage),
            "source_postimage_sha256": _sha256_bytes(postimage_bytes),
            "source_postimage_size": len(postimage_bytes),
            "source_postimage": postimage,
            "semantic_artifact": dict(semantic_artifact),
        },
        None,
    )


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _orphan_semantic_epoch(
    item: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a semantic hold to the exact orphan evidence and local proposal."""

    return {
        "resolver_version": RESOLVER_VERSION,
        "input_hash": str(item.get("input_hash") or ""),
        "proposal_sha256": _canonical_payload_sha256(proposal),
    }


def _effect_artifact_path(state: Any, key: str) -> Path:
    state_file = getattr(state, "state_file", DECISIONS_FILE)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(state_file).parent / "orphan-link-effects" / f"{digest}.json"


def _persist_effect_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically fsync an effect intent before its page CAS can run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
    return _canonical_payload_sha256(payload)


def _load_effect_artifact(
    metadata: object,
    *,
    expected_key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(metadata, Mapping):
        return None, "durable orphan effect metadata is missing"
    path_value = metadata.get("effect_artifact_path")
    expected_hash = metadata.get("effect_artifact_sha256")
    expected_proposal = metadata.get("effect_proposal_fingerprint")
    if not isinstance(path_value, str) or not path_value:
        return None, "durable orphan effect artifact path is missing"
    try:
        payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read durable orphan effect artifact: {exc}"
    if not isinstance(payload, dict):
        return None, "durable orphan effect artifact is not an object"
    if _canonical_payload_sha256(payload) != expected_hash:
        return None, "durable orphan effect artifact hash mismatch"
    if (
        payload.get("schema_version") != 2
        or payload.get("kind") != "orphan_link_page_effect"
    ):
        return None, "durable orphan effect artifact schema mismatch"
    if payload.get("convergence_key") != expected_key:
        return None, "durable orphan effect convergence key mismatch"
    if (
        not isinstance(expected_proposal, str)
        or payload.get("proposal_fingerprint") != expected_proposal
    ):
        return None, "durable orphan effect proposal fingerprint mismatch"
    semantic = payload.get("semantic_artifact")
    if not isinstance(semantic, Mapping):
        return None, "durable orphan semantic artifact is missing"
    proposal = semantic.get("proposal")
    if not isinstance(proposal, Mapping):
        return None, "durable orphan proposal is missing"
    if _canonical_payload_sha256(proposal) != expected_proposal:
        return None, "durable orphan proposal identity mismatch"
    authority = semantic.get("authority")
    verdict_error = semantic_verdict_authority_error(
        semantic.get("frontier"),
        authority,
        lane=DECISION_LANE,
    )
    if verdict_error is not None:
        return None, verdict_error
    return payload, None


def _apply_prepared_effect(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only the exact postimage authorized by a durable effect artifact."""
    source_id = str(payload.get("source_page_id") or "")
    target_id = str(payload.get("target_page_id") or "")
    source_path = find_page(source_id)
    target_path = find_page(target_id)
    if source_path is None or target_path is None:
        return {"status": "error", "reason": "source_or_target_missing"}
    postimage = payload.get("source_postimage")
    if not isinstance(postimage, str):
        return {"status": "error", "reason": "source_postimage_missing"}
    postimage_bytes = postimage.encode("utf-8")
    if _sha256_bytes(postimage_bytes) != payload.get("source_postimage_sha256"):
        return {"status": "error", "reason": "source_postimage_hash_mismatch"}
    try:
        with chronovisor_mutation_lock():
            source_before = source_path.read_bytes()
            target_before = target_path.read_bytes()
            source_hash = _sha256_bytes(source_before)
            target_hash = _sha256_bytes(target_before)
            if source_hash == payload.get(
                "source_postimage_sha256"
            ) and target_hash == payload.get("target_preimage_sha256"):
                return {
                    "status": "already_applied",
                    "source": source_id,
                    "target": target_id,
                    "recovery_only": True,
                    "semantic_effect": False,
                }
            if source_hash != payload.get("source_preimage_sha256"):
                return {"status": "retry", "reason": "source_changed_before_apply"}
            if target_hash != payload.get("target_preimage_sha256"):
                return {"status": "retry", "reason": "target_changed_before_apply"}
            atomic_write(source_path, postimage)
            source_after = source_path.read_bytes()
            target_after = target_path.read_bytes()
            if _sha256_bytes(source_after) != payload.get(
                "source_postimage_sha256"
            ) or _sha256_bytes(target_after) != payload.get("target_preimage_sha256"):
                if _sha256_bytes(source_after) == payload.get(
                    "source_postimage_sha256"
                ):
                    atomic_write(source_path, source_before.decode("utf-8"))
                return {"status": "error", "reason": "post_write_verification_failed"}
    except (OSError, UnicodeDecodeError) as exc:
        return {"status": "error", "reason": f"write_error:{exc}"}
    return {
        "status": "applied",
        "source": source_id,
        "target": target_id,
        "recovery_only": False,
        "semantic_effect": True,
    }


def _recover_pending_effects(
    state: Any,
    *,
    eligible_keys: set[str] | None = None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Finalize only a byte-exact postimage left by a crashed worker.

    Recovery intentionally does not re-run the semantic effect.  A pending
    artifact whose source is anything other than its exact recorded postimage
    is left pending (and may later be retired), never replayed here.
    """
    results: list[dict[str, Any]] = []
    for item in state.list_items(lane=DECISION_LANE):
        if item.get("status") in {
            "applied",
            "rejected",
            "quarantined",
            "human_required",
        }:
            continue
        key = str(item.get("key") or "")
        if eligible_keys is not None and key not in eligible_keys:
            continue
        payload, artifact_error = _load_effect_artifact(
            item.get("metadata"),
            expected_key=key,
        )
        if payload is None:
            if isinstance(item.get("metadata"), Mapping) and item["metadata"].get(
                "effect_artifact_path"
            ):
                results.append(
                    {
                        "key": key,
                        "orphan": str(item.get("source_id") or "").removeprefix(
                            "orphan:"
                        ),
                        "status": "recovery_artifact_invalid",
                        "reason": artifact_error,
                    }
                )
            continue
        source_path = find_page(str(payload.get("source_page_id") or ""))
        target_path = find_page(str(payload.get("target_page_id") or ""))
        try:
            source_hash = (
                _sha256_bytes(source_path.read_bytes())
                if source_path is not None
                else None
            )
            target_hash = (
                _sha256_bytes(target_path.read_bytes())
                if target_path is not None
                else None
            )
        except OSError:
            continue
        if source_hash != payload.get(
            "source_postimage_sha256"
        ) or target_hash != payload.get("target_preimage_sha256"):
            continue
        orphan_id = str(payload.get("target_page_id") or "")
        if dry_run:
            results.append(
                {
                    "key": key,
                    "orphan": orphan_id,
                    "status": "would_recover_exact_postimage",
                    "recovery_only": True,
                    "semantic_effect": False,
                }
            )
            continue
        recovery = {
            "status": "already_applied",
            "source": str(payload.get("source_page_id") or ""),
            "target": orphan_id,
            "recovery_only": True,
            "semantic_effect": False,
        }
        semantic = payload["semantic_artifact"]
        completed = state.complete(
            key,
            "applied",
            result={
                **semantic,
                "apply": recovery,
                "recovery_only": True,
                "semantic_effect": False,
            },
            owner=str(payload.get("claim_owner") or "") or None,
        )
        results.append(
            {
                "key": key,
                "orphan": orphan_id,
                "status": completed["item"]["status"],
                "recovery_only": True,
                "semantic_effect": False,
            }
        )
    return results


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
    if re.search(r"\[\[" + re.escape(orphan_id) + r"(?:[#|\]])", original):
        return {
            "status": "already_applied",
            "source": suggestion.source_page_id,
            "target": orphan_id,
        }

    updated = _render_suggestion_postimage(original, orphan_id, suggestion)
    if dry_run:
        return {
            "status": "dry_run",
            "source": suggestion.source_page_id,
            "target": orphan_id,
            "changed": updated != original,
        }
    wrote_source = False
    try:
        with chronovisor_mutation_lock():
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
    return {
        "status": "applied",
        "source": suggestion.source_page_id,
        "target": orphan_id,
    }


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
                "source_excerpt": _page_head(suggestion.source_page_id, max_chars=1200),
            }
        )
    if reviewer is not None:
        return reviewer(candidate)
    from chronovisor.decision.decision_lane_prompts import (
        build_orphan_link_review_prompt,
    )
    from chronovisor.decision.routine_review import run_structured_review

    prompt = build_orphan_link_review_prompt(candidate)
    return run_structured_review(
        prompt,
        ORPHAN_FRONTIER_SCHEMA,
        repo_root=PROJECT_ROOT,
        decision_lane="orphan_link",
    )


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
    frontier_confidence_threshold: float | None = None,
    eligible_keys: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Boundedly drain orphan proposals through local + frontier review."""
    from chronovisor.ingest.convergence import (
        TERMINAL_STATUSES,
        ConvergenceStore,
        CycleBudget,
        stable_item_key,
    )

    state = convergence_store or ConvergenceStore()
    recovered = _recover_pending_effects(
        state,
        eligible_keys=eligible_keys,
        dry_run=dry_run,
    )
    recovered_keys = {str(row.get("key") or "") for row in recovered}
    if store is None:
        from chronovisor.search.index_store import get_store

        store = get_store()
        store.refresh()
    cycle_budget = budget or CycleBudget(
        max_local_calls=max(1, orphan_limit * max_candidates),
        max_frontier_calls=2,
        max_mutations=2,
    )
    orphans = store.orphans(include_system=False)
    active_statuses = {
        "pending_local",
        "local_retry",
        "pending_frontier",
        "frontier_retry",
        "local_running",
        "frontier_running",
    }
    eligible = (
        {str(key) for key in eligible_keys} if eligible_keys is not None else None
    )
    active_items = [
        item
        for item in state.list_items(lane=DECISION_LANE, statuses=active_statuses)
        if eligible is None or str(item.get("key") or "") in eligible
    ]
    active_sources: dict[str, str] = {}
    active_status_by_source: dict[str, str] = {}
    for item in sorted(
        active_items,
        key=lambda row: (
            str(row.get("created_at") or ""),
            str(row.get("key") or ""),
        ),
    ):
        source = str(item.get("source_id") or "").removeprefix("orphan:")
        if source:
            active_sources.setdefault(source, str(item.get("created_at") or ""))
            active_status_by_source.setdefault(source, str(item.get("status") or ""))
    if eligible_keys is not None:
        eligible_sources = {
            str(item.get("source_id") or "").removeprefix("orphan:")
            for key in eligible_keys
            if (item := state.get(key)) is not None
            and item.get("lane") == DECISION_LANE
        }
        orphans = [page_id for page_id in orphans if page_id in eligible_sources]
    # Drain durable work oldest-first.  Previously the producer's page order
    # forced every pass to re-embed and skip a growing prefix of terminal
    # orphans before reaching pending items, making throughput collapse as the
    # queue converged.
    original_order = {page_id: index for index, page_id in enumerate(orphans)}
    orphans.sort(
        key=lambda page_id: (
            0 if page_id in active_sources else 1,
            active_sources.get(page_id, ""),
            original_order[page_id],
        )
    )
    if eligible_keys is None:
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
    else:
        retired_absent = {
            "status": "skipped",
            "reason": "targeted_allowlist",
            "retired": [],
        }
        retired_stale = dict(retired_absent)
    work_limit = max(0, int(orphan_limit))
    # Deprecated compatibility input. Consensus confidence is diagnostic only.
    del frontier_confidence_threshold
    results: list[dict[str, Any]] = list(recovered)
    work_items = 0
    scanned = 0
    stop_reason: str | None = None

    for orphan_id in orphans:
        if cycle_budget.remaining_elapsed_seconds <= 0:
            stop_reason = "elapsed_budget_exhausted"
            break
        active_status = active_status_by_source.get(orphan_id)
        remaining = cycle_budget.snapshot()["remaining"]
        if (
            active_status in {"pending_local", "local_retry", "local_running"}
            and int(remaining["local"]) <= 0
        ):
            stop_reason = "local_lane_budget_exhausted"
            break
        if (
            active_status in {"pending_frontier", "frontier_retry", "frontier_running"}
            and int(remaining["frontier"]) <= 0
        ):
            stop_reason = "frontier_lane_budget_exhausted"
            break
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
        source_id = candidates[0] if candidates else ""
        decision_authority, authority_error = current_semantic_authority(
            DECISION_LANE,
            injected_reviewer=reviewer is not None,
        )
        if decision_authority is None or authority_error is not None:
            results.append(
                {
                    "orphan": orphan_id,
                    "source": source_id,
                    "status": "decision_authority_unavailable",
                    "reason": authority_error or "decision authority is unavailable",
                }
            )
            continue
        input_data = {
            "orphan": orphan_id,
            "orphan_hash": _content_hash(orphan_id),
            # Make pending and terminal convergence keys authority-epoch
            # specific.  A new contract, adoption artifact/model triplet, or
            # lane-mode change can never reuse an old semantic disposition.
            "decision_authority": decision_authority,
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
        if eligible_keys is not None and key not in eligible_keys:
            results.append(
                {
                    "orphan": orphan_id,
                    "source": source_id,
                    "status": "out_of_scope",
                    "key": key,
                }
            )
            continue
        existing = state.get(key)
        if existing is not None and existing.get("status") in TERMINAL_STATUSES:
            if key in recovered_keys:
                continue
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
                supersede_eligible_keys=eligible_keys,
            )
            item = merged["item"]
            if item is None:
                results.append(
                    {
                        "orphan": orphan_id,
                        "source": source_id,
                        "status": "out_of_scope_source_changed",
                        "key": key,
                        "blocked_by_out_of_scope": merged.get(
                            "blocked_by_out_of_scope", []
                        ),
                    }
                )
                continue
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
            results.append(
                {"orphan": orphan_id, "source": source_id, "status": status, "key": key}
            )
            continue
        if not candidates and item.get("status") in {"pending_local", "local_retry"}:
            claim = state.claim_attempt(key, "local")
            if not claim["claimed"]:
                results.append(
                    {"orphan": orphan_id, "status": claim["reason"], "key": key}
                )
                continue
            frontier_proposal = {
                "kind": "retry" if discovery_error else "no_link",
                "reason": discovery_error or "no_semantic_candidate",
                "candidates": [],
                "failure_class": (
                    "candidate_discovery_error" if discovery_error else None
                ),
            }
            proposal_merge = state.merge_item(
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
                supersede_eligible_keys=eligible_keys,
            )
            if proposal_merge["item"] is None:
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": "out_of_scope_source_changed",
                        "key": key,
                        "blocked_by_out_of_scope": proposal_merge.get(
                            "blocked_by_out_of_scope", []
                        ),
                    }
                )
                continue
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
                results.append(
                    {"orphan": orphan_id, "status": claim["reason"], "key": key}
                )
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
                        local_errors.append(
                            {"status": "budget_deferred", "error": budget_reason}
                        )
                        break
                outcome = _score_candidate_outcome(
                    candidate_id, orphan_id, store, generate_fn
                )
                scored = outcome.get("score")
                if isinstance(scored, dict):
                    valid_scores.append(scored)
                else:
                    local_errors.append(
                        {
                            "source": candidate_id,
                            "status": outcome.get("status"),
                            "error": outcome.get("error"),
                        }
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
            proposal_merge = state.merge_item(
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
                supersede_eligible_keys=eligible_keys,
            )
            if proposal_merge["item"] is None:
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": "out_of_scope_source_changed",
                        "key": key,
                        "blocked_by_out_of_scope": proposal_merge.get(
                            "blocked_by_out_of_scope", []
                        ),
                    }
                )
                continue
            state.escalate(
                key,
                reason="local suggestion requires frontier final review",
                owner=claim["owner"],
            )
            item = state.get(key) or item

        if item.get("status") not in {"pending_frontier", "frontier_retry"}:
            results.append(
                {"orphan": orphan_id, "status": item.get("status"), "key": key}
            )
            continue
        claim = state.claim_attempt(key, "frontier", budget=cycle_budget)
        if not claim["claimed"]:
            results.append({"orphan": orphan_id, "status": claim["reason"], "key": key})
            continue
        metadata = (state.get(key) or {}).get("metadata") or {}
        frontier_proposal = (
            metadata.get("frontier_proposal") if isinstance(metadata, dict) else None
        )
        raw_suggestion = (
            metadata.get("suggestion") if isinstance(metadata, dict) else None
        )
        if not isinstance(frontier_proposal, dict) and isinstance(raw_suggestion, dict):
            frontier_proposal = {"kind": "link", "suggestion": raw_suggestion}
        if not isinstance(frontier_proposal, dict) or frontier_proposal.get(
            "kind"
        ) not in {
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
            results.append(
                {"orphan": orphan_id, "status": "frontier_retry", "key": key}
            )
            continue
        proposal_kind = str(frontier_proposal["kind"])
        if not item_counted and not (proposal_kind == "no_link" and not candidates):
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
                    {
                        "orphan": orphan_id,
                        "status": failed["item"]["status"],
                        "key": key,
                    }
                )
                continue
        review_authority, review_authority_error = current_semantic_authority(
            DECISION_LANE,
            injected_reviewer=reviewer is not None,
        )
        authority_changed = review_authority_error
        if review_authority is not None and authority_changed is None:
            authority_changed = compare_semantic_authority(
                decision_authority,
                review_authority,
                lane=DECISION_LANE,
            )
        if review_authority is None or authority_changed is not None:
            failed = state.fail_attempt(
                key,
                "frontier",
                error=authority_changed or "decision authority is unavailable",
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
        semantic_no_quorum = is_local_semantic_no_quorum(review)
        verdict_error = (
            semantic_verdict_authority_provenance_error(
                review,
                review_authority,
                lane=DECISION_LANE,
            )
            if semantic_no_quorum
            else semantic_verdict_authority_error(
                review,
                review_authority,
                lane=DECISION_LANE,
            )
        )
        if verdict_error is not None:
            failed = state.fail_attempt(
                key,
                "frontier",
                error=verdict_error,
                owner=claim["owner"],
            )
            results.append(
                {"orphan": orphan_id, "status": failed["item"]["status"], "key": key}
            )
            continue
        semantic_result = seal_semantic_artifact(
            {
                "schema_version": 2,
                "frontier": review,
                "proposal": frontier_proposal,
            },
            authority=review_authority,
            lane=DECISION_LANE,
        )

        # Every durable semantic disposition and page effect is committed in
        # the same authority epoch that produced the review.  This prevents a
        # concurrent lane disable, contract update, or model re-adoption from
        # installing a now-stale verdict.
        with decision_authority_lock():
            current_authority, current_authority_error = current_semantic_authority(
                DECISION_LANE,
                injected_reviewer=reviewer is not None,
            )
            effect_authority_error = current_authority_error
            if current_authority is not None and effect_authority_error is None:
                effect_authority_error = compare_semantic_authority(
                    review_authority,
                    current_authority,
                    lane=DECISION_LANE,
                )
            if effect_authority_error is None:
                effect_authority_error = (
                    semantic_verdict_authority_provenance_error(
                        review,
                        review_authority,
                        lane=DECISION_LANE,
                    )
                    if semantic_no_quorum
                    else semantic_verdict_authority_error(
                        review,
                        review_authority,
                        lane=DECISION_LANE,
                    )
                )
            if current_authority is None or effect_authority_error is not None:
                failed = state.fail_attempt(
                    key,
                    "frontier",
                    error=(
                        effect_authority_error
                        or "decision authority is unavailable before effect"
                    ),
                    owner=claim["owner"],
                )
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": failed["item"]["status"],
                        "key": key,
                    }
                )
                continue

            decision = review.get("decision")
            if decision == "rejected":
                if proposal_kind in {"link", "retry"}:
                    state.complete(
                        key,
                        "rejected",
                        result=semantic_result,
                        owner=claim["owner"],
                    )
                    results.append(
                        {"orphan": orphan_id, "status": "rejected", "key": key}
                    )
                else:
                    failed = state.fail_attempt(
                        key,
                        "frontier",
                        error=str(
                            review.get("summary")
                            or "frontier rejected no-link disposition"
                        ),
                        owner=claim["owner"],
                    )
                    results.append(
                        {
                            "orphan": orphan_id,
                            "status": failed["item"]["status"],
                            "key": key,
                        }
                    )
                continue
            if decision != "approved":
                if semantic_no_quorum:
                    try:
                        current_item = state.get(key) or item
                        failed = state.hold_semantic_no_quorum(
                            key,
                            lane=DECISION_LANE,
                            stage="frontier",
                            review=review,
                            epoch=_orphan_semantic_epoch(
                                current_item,
                                frontier_proposal,
                            ),
                            authority=current_authority,
                            owner=claim["owner"],
                            error=str(
                                review.get("summary") or "local semantic no quorum"
                            ),
                        )
                    except (TypeError, ValueError) as exc:
                        failed = state.fail_attempt(
                            key,
                            "frontier",
                            error=f"semantic hold rejected: {exc}",
                            failure_class="review_artifact_invalid",
                            owner=claim["owner"],
                        )
                else:
                    failed = state.fail_attempt(
                        key,
                        "frontier",
                        error=str(review.get("summary") or "frontier needs retry"),
                        failure_class=_frontier_failure_class(review),
                        owner=claim["owner"],
                    )
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": failed["item"]["status"],
                        "key": key,
                    }
                )
                continue
            if proposal_kind == "no_link":
                completed = state.complete(
                    key,
                    "rejected",
                    result=semantic_result,
                    owner=claim["owner"],
                )
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": completed["item"]["status"],
                        "key": key,
                    }
                )
                continue
            if proposal_kind == "retry":
                failed = state.fail_attempt(
                    key,
                    "frontier",
                    error=str(
                        frontier_proposal.get("reason") or "autonomous retry required"
                    ),
                    failure_class=(
                        str(frontier_proposal.get("failure_class") or "") or None
                    ),
                    owner=claim["owner"],
                )
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": failed["item"]["status"],
                        "key": key,
                    }
                )
                continue
            assert suggestion is not None
            allowed, reason = cycle_budget.consume("mutation")
            if not allowed:
                state.fail_attempt(key, "frontier", error=reason, owner=claim["owner"])
                results.append({"orphan": orphan_id, "status": reason, "key": key})
                continue
            proposal_fingerprint = _canonical_payload_sha256(frontier_proposal)
            effect_payload, preparation_error = _prepare_suggestion_effect(
                orphan_id,
                suggestion,
                convergence_key=key,
                proposal_fingerprint=proposal_fingerprint,
                semantic_artifact=semantic_result,
                claim_owner=str(claim["owner"]),
            )
            if preparation_error is not None:
                if preparation_error.get("status") == "already_applied":
                    state.complete(
                        key,
                        "applied",
                        result={
                            **semantic_result,
                            "apply": preparation_error,
                            "recovery_only": bool(
                                preparation_error.get("recovery_only")
                            ),
                            "semantic_effect": False,
                        },
                        owner=claim["owner"],
                    )
                else:
                    state.fail_attempt(
                        key,
                        "frontier",
                        error=str(
                            preparation_error.get("reason")
                            or preparation_error.get("status")
                        ),
                        owner=claim["owner"],
                    )
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": (state.get(key) or {}).get("status"),
                        "key": key,
                    }
                )
                continue
            assert effect_payload is not None
            artifact_path = _effect_artifact_path(state, key)
            artifact_hash = _persist_effect_artifact(artifact_path, effect_payload)
            durable_metadata = {
                **metadata,
                "effect_artifact_path": str(artifact_path),
                "effect_artifact_sha256": artifact_hash,
                "effect_proposal_fingerprint": proposal_fingerprint,
            }
            effect_merge = state.merge_item(
                lane=DECISION_LANE,
                source_id=f"orphan:{orphan_id}",
                input_data=input_data,
                resolver_version=RESOLVER_VERSION,
                metadata=durable_metadata,
                supersede_eligible_keys=eligible_keys,
            )
            if effect_merge["item"] is None:
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": "out_of_scope_source_changed",
                        "key": key,
                        "blocked_by_out_of_scope": effect_merge.get(
                            "blocked_by_out_of_scope", []
                        ),
                    }
                )
                continue
            persisted = state.get(key) or {}
            effect_payload, artifact_error = _load_effect_artifact(
                persisted.get("metadata"),
                expected_key=key,
            )
            if effect_payload is None:
                state.fail_attempt(
                    key,
                    "frontier",
                    error=artifact_error or "durable orphan effect artifact is invalid",
                    owner=claim["owner"],
                )
                results.append(
                    {
                        "orphan": orphan_id,
                        "status": (state.get(key) or {}).get("status"),
                        "key": key,
                    }
                )
                continue
            applied = _apply_prepared_effect(effect_payload)
            if applied["status"] in {"applied", "already_applied"}:
                state.complete(
                    key,
                    "applied",
                    result={
                        **semantic_result,
                        "apply": applied,
                        "recovery_only": bool(applied.get("recovery_only")),
                        "semantic_effect": bool(applied.get("semantic_effect")),
                    },
                    owner=claim["owner"],
                )
            else:
                state.fail_attempt(
                    key,
                    "frontier",
                    error=str(applied.get("reason") or applied["status"]),
                    owner=claim["owner"],
                )
        results.append(
            {
                "orphan": orphan_id,
                "status": (state.get(key) or {}).get("status"),
                "key": key,
            }
        )

    return {
        "status": "ok",
        "orphans_seen": scanned,
        "orphans_total": len(orphans),
        "work_items": work_items,
        "stop_reason": stop_reason,
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
    total_pages = (
        "?" if store is None else str(len(store.all_page_ids(include_system=False)))
    )
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
    lines.append(
        "Pages are not modified by this diagnostic report; nightly convergence reviews and applies bounded proposals."
    )
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
                f'anchor="{s.suggested_anchor}", '
                f'section="{s.suggested_section}"'
            )
            lines.append(f"        reason: {s.reason}")
        lines.append("")

    return "\n".join(lines) + "\n"
