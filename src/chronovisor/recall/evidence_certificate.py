"""Page-level evidence certificates for precision-first Recall injection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.frontmatter import parse
from chronovisor.core.search_types import tokenize
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page

CERTIFICATE_LEDGER = CHRONOVISOR_ROOT / "recall" / "evidence-certificate-ledger.jsonl"
CERTIFICATE_LEDGER_LOCK = CERTIFICATE_LEDGER.parent / "evidence-certificate-ledger.jsonl.lock"


@dataclass(frozen=True)
class EvidenceCertificate:
    certificate_id: str
    page_id: str
    outcome: str
    confidence: float
    label_quality: str
    supporting_span: str
    source_line: int
    query_sha256: str
    content_sha256: str
    policy_sha256: str
    model_revision: str
    features: dict[str, Any]
    reasons: tuple[str, ...]
    created_at: str

    def public_summary(self) -> dict[str, Any]:
        """Return diagnostics safe for Recall logs and dashboard aggregation."""

        return {
            "certificate_id": self.certificate_id,
            "page_id": self.page_id,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "label_quality": self.label_quality,
            "source_line": self.source_line,
            "content_sha256": self.content_sha256,
            "policy_sha256": self.policy_sha256,
            "model_revision": self.model_revision,
            "components": {
                key: self.features[key]
                for key in (
                    "anti_index",
                    "hub_penalty",
                    "lexical_coverage",
                    "reranker_raw",
                    "reranker_rank",
                    "reranker_margin",
                )
                if isinstance(self.features.get(key), int | float)
            },
            "reasons": list(self.reasons),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _query_tokens(query: str) -> set[str]:
    generic = {
        "about",
        "chronovisor",
        "from",
        "have",
        "memory",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return {token for token in tokenize(query) if token not in generic}


def _span_score(query_tokens: set[str], text: str) -> tuple[int, float]:
    if not text.strip():
        return 0, 0.0
    span_tokens = set(tokenize(text))
    overlap = len(query_tokens & span_tokens)
    coverage = overlap / max(1, min(len(query_tokens), 12))
    return overlap, coverage


def supporting_span(
    page_id: str, query: str, snippet: str = ""
) -> tuple[str, int, float, str]:
    """Choose one exact, bounded span and return its content digest."""

    path = find_page(page_id)
    if path is None:
        digest = _sha256_bytes(page_id.encode("utf-8"))
        overlap, coverage = _span_score(_query_tokens(query), snippet)
        return snippet[:320].strip(), 0, coverage if overlap else 0.0, digest
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        content = ""
    digest = _sha256_bytes(content.encode("utf-8"))
    try:
        meta, body = parse(content)
    except Exception:
        meta, body = {}, content
    query_tokens = _query_tokens(query)
    candidates: list[tuple[str, int]] = []
    if snippet.strip():
        candidates.append((snippet.strip(), 0))
    summary = meta.get("summary")
    if isinstance(summary, str) and summary.strip():
        candidates.append((summary.strip(), 0))
    try:
        from chronovisor.search.claims import page_claims

        for claim in page_claims(page_id):
            claim_span = claim.get("evidence_span")
            claim_value = claim.get("value")
            text = (
                claim_span
                if isinstance(claim_span, str) and claim_span.strip()
                else claim_value
                if isinstance(claim_value, str)
                else ""
            )
            source_line = claim.get("source_line")
            if text.strip():
                candidates.append(
                    (
                        text.strip(),
                        source_line
                        if isinstance(source_line, int) and source_line > 0
                        else 0,
                    )
                )
    except Exception:
        pass
    for line_no, raw_line in enumerate(body.splitlines(), start=1):
        line = raw_line.strip(" #-*\t")
        if line:
            candidates.append((line, line_no))
    if not candidates:
        return "", 0, 0.0, digest
    ranked = sorted(
        (
            (
                *_span_score(query_tokens, text),
                -line_no,
                text,
                line_no,
            )
            for text, line_no in candidates
        ),
        reverse=True,
    )
    _overlap, coverage, _line_order, text, line_no = ranked[0]
    return text[:320].strip(), line_no, coverage, digest


def _reranker_probability(raw_score: float) -> float:
    if raw_score <= -40:
        return 0.0
    if raw_score >= 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-raw_score))


def certify_candidate(
    query: str,
    candidate: Any,
    *,
    policy: dict[str, Any],
    reranker_score: dict[str, Any] | None = None,
    ranking_components: dict[str, Any] | None = None,
) -> EvidenceCertificate:
    """Build a deterministic, auditable page certificate."""

    span, source_line, lexical_coverage, content_sha = supporting_span(
        str(candidate.page_id),
        query,
        str(getattr(candidate, "snippet", "") or ""),
    )
    fused_raw = max(0.0, float(getattr(candidate, "score", 0.0) or 0.0))
    fused_calibrated = min(1.0, fused_raw / 0.08)
    reranker_raw: float | None = None
    reranker_rank = 0
    reranker_count = 0
    reranker_margin = 0.0
    model_revision = ""
    if isinstance(reranker_score, dict):
        try:
            reranker_raw = float(reranker_score.get("raw_score"))
            reranker_margin = max(
                0.0, float(reranker_score.get("margin_to_next") or 0.0)
            )
            reranker_rank = max(0, int(reranker_score.get("rerank_rank") or 0))
            reranker_count = max(
                reranker_rank,
                int(reranker_score.get("candidate_count") or 0),
            )
            model_revision = str(reranker_score.get("model_revision") or "")
        except (TypeError, ValueError):
            reranker_raw = None
    if reranker_raw is None:
        confidence = (0.72 * lexical_coverage) + (0.28 * fused_calibrated)
    else:
        rank_probability = (
            1.0 - ((reranker_rank - 1) / max(1, reranker_count))
            if reranker_rank > 0
            else _reranker_probability(reranker_raw)
        )
        confidence = (
            (0.32 * lexical_coverage)
            + (0.50 * max(_reranker_probability(reranker_raw), rank_probability))
            + (0.18 * fused_calibrated)
        )
    confidence = max(0.0, min(1.0, confidence))
    independent_retrieval = fused_calibrated >= 0.12
    lexical_support = lexical_coverage >= 0.08
    # BGE v2-m3 logits are not zero-calibrated: a relevant conversational
    # paraphrase can be rank 1 with a negative raw logit. Relative rank is the
    # cross-query stable signal; the low raw floor only rejects a collapsed
    # tail score.
    reranker_support = bool(
        reranker_raw is not None
        and reranker_rank in range(1, min(5, max(1, reranker_count)) + 1)
        and reranker_raw >= -12.0
    )
    passed = bool(
        span
        and independent_retrieval
        and (
            (lexical_support and confidence >= 0.30)
            or (reranker_support and confidence >= 0.52)
        )
    )
    reasons: list[str] = []
    if not span:
        reasons.append("no_supporting_span")
    if not independent_retrieval:
        reasons.append("weak_retrieval")
    if not lexical_support:
        reasons.append("weak_lexical_support")
    if reranker_raw is not None and not reranker_support:
        reasons.append("reranker_reject")
    if not reasons and passed:
        reasons.append("independent_signals_agree")
    elif not passed and confidence < 0.30:
        reasons.append("low_confidence")
    policy_sha = _sha256_bytes(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    query_sha = _sha256_bytes(query.encode("utf-8"))
    certificate_id = _sha256_bytes(
        ":".join(
            [
                query_sha,
                str(candidate.page_id),
                content_sha,
                policy_sha,
                model_revision,
            ]
        ).encode("utf-8")
    )[:24]
    quality = (
        "strong"
        if passed and reranker_support and lexical_support and source_line > 0
        else "silver"
    )
    contextual = ranking_components or {}
    return EvidenceCertificate(
        certificate_id=certificate_id,
        page_id=str(candidate.page_id),
        outcome="pass" if passed else "reject",
        confidence=round(confidence, 6),
        label_quality=quality,
        supporting_span=span,
        source_line=source_line,
        query_sha256=query_sha,
        content_sha256=content_sha,
        policy_sha256=policy_sha,
        model_revision=model_revision,
        features={
            "fused_raw": round(fused_raw, 8),
            "fused_calibrated": round(fused_calibrated, 6),
            "lexical_coverage": round(lexical_coverage, 6),
            "reranker_raw": reranker_raw,
            "reranker_rank": reranker_rank,
            "reranker_count": reranker_count,
            "reranker_margin": round(reranker_margin, 6),
            "anti_index": round(float(contextual.get("anti_index") or 0.0), 6),
            "hub_penalty": round(float(contextual.get("hub_penalty") or 0.0), 6),
            "hub_degree": int(float(contextual.get("hub_degree") or 0)),
            "query_specificity": round(
                float(contextual.get("query_specificity") or 0.0), 6
            ),
            "support_coverage": round(
                float(contextual.get("support_coverage") or lexical_coverage), 6
            ),
            "exact_match_protected": (contextual.get("exact_match_protected") is True),
        },
        reasons=tuple(reasons),
        created_at=datetime.now().isoformat(timespec="milliseconds"),
    )


def append_certificates(
    certificates: list[EvidenceCertificate],
    *,
    path: Path = CERTIFICATE_LEDGER,
) -> int:
    """Append private teacher evidence without persisting the raw query."""

    if not certificates:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = (
        CERTIFICATE_LEDGER_LOCK
        if path == CERTIFICATE_LEDGER
        else path.with_suffix(path.suffix + ".lock")
    )
    with exclusive_text_file_lock(lock_path):
        with path.open("a", encoding="utf-8") as handle:
            for certificate in certificates:
                handle.write(
                    json.dumps(
                        asdict(certificate),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    os.chmod(path, 0o600)
    return len(certificates)
