from __future__ import annotations

import json
import stat
from pathlib import Path

from chronovisor.core.search_types import ScoredPage
from chronovisor.recall import evidence_certificate
from chronovisor.recall.evidence_certificate import EvidenceCertificate


def candidate(page_id: str, *, score: float, snippet: str = "") -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id.replace("-", " "),
        folder="",
        updated="2026-07-30",
        score=score,
        snippet=snippet,
    )


def test_certificate_rejects_weak_rrf_without_support(
    tmp_path: Path, monkeypatch
) -> None:
    page = tmp_path / "unrelated.md"
    page.write_text(
        "---\ntitle: Unrelated\n---\n\nA film and travel discussion.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evidence_certificate,
        "find_page",
        lambda _page_id: page,
    )

    certificate = evidence_certificate.certify_candidate(
        "Which GPU runs the embedding model?",
        candidate("unrelated", score=0.0164),
        policy={"certificate_required": True},
    )

    assert certificate.outcome == "reject"
    assert certificate.features["fused_calibrated"] < 0.25
    assert "weak_lexical_support" in certificate.reasons


def test_certificate_passes_independent_retrieval_and_exact_span(
    tmp_path: Path, monkeypatch
) -> None:
    page = tmp_path / "reranker.md"
    page.write_text(
        "---\ntitle: Reranker\n---\n\n"
        "The BGE reranker service uses MPS for foreground inference.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evidence_certificate,
        "find_page",
        lambda _page_id: page,
    )

    certificate = evidence_certificate.certify_candidate(
        "BGE reranker MPS foreground inference",
        candidate("reranker", score=0.04),
        policy={"certificate_required": True},
    )

    assert certificate.outcome == "pass"
    assert certificate.source_line > 0
    assert "BGE reranker" in certificate.supporting_span
    assert certificate.query_sha256
    assert certificate.content_sha256


def test_certificate_ledger_is_private_and_omits_raw_query(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "certificates.jsonl"
    certificate = EvidenceCertificate(
        certificate_id="cert-1",
        page_id="page-a",
        outcome="pass",
        confidence=0.9,
        label_quality="strong",
        supporting_span="bounded evidence",
        source_line=4,
        query_sha256="query-hash",
        content_sha256="content-hash",
        policy_sha256="policy-hash",
        model_revision="bge-revision",
        features={"reranker_raw": 2.0},
        reasons=("independent_signals_agree",),
        created_at="2026-07-30T22:00:00",
    )

    written = evidence_certificate.append_certificates(
        [certificate],
        path=ledger,
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))

    assert written == 1
    assert payload["query_sha256"] == "query-hash"
    assert "query" not in payload
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600


def test_default_certificate_ledger_uses_its_exact_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "evidence-certificate-ledger.jsonl"
    lock = tmp_path / "evidence-certificate-ledger.jsonl.lock"
    monkeypatch.setattr(evidence_certificate, "CERTIFICATE_LEDGER", ledger)
    monkeypatch.setattr(evidence_certificate, "CERTIFICATE_LEDGER_LOCK", lock)
    certificate = EvidenceCertificate(
        certificate_id="cert-lock",
        page_id="page-a",
        outcome="pass",
        confidence=0.9,
        label_quality="strong",
        supporting_span="bounded evidence",
        source_line=4,
        query_sha256="query-hash",
        content_sha256="content-hash",
        policy_sha256="policy-hash",
        model_revision="bge-revision",
        features={"reranker_raw": 2.0},
        reasons=("independent_signals_agree",),
        created_at="2026-07-30T22:00:00",
    )

    evidence_certificate.append_certificates([certificate], path=ledger)

    assert lock.exists()
    assert not (tmp_path / "evidence-certificate-ledger.lock").exists()
