from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llm_wiki_mcp.evidence_bundle import (
    build_bundle,
    classify_claim,
    deterministic_citations,
    simple_assess_claims,
)
from llm_wiki_mcp.research_store import ResearchStore
from llm_wiki_mcp.research_types import ClaimKind, ClaimStatus, EvidenceArtifact


def _artifact(artifact_id: str, preview: str, *, trust: str = "official") -> EvidenceArtifact:
    return EvidenceArtifact(
        artifact_id=artifact_id,
        source_type="wiki_read",
        source_uri="wiki:fact",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        sha256=artifact_id.removeprefix("sha256:"),
        byte_length=len(preview),
        preview=preview,
        trust=trust,
        title="Fact",
        citation="https://example.com/fact",
        durable=True,
    )


def test_claim_classification_and_unknown_preservation() -> None:
    assert classify_claim("最新バージョンは何か") == ClaimKind.FRESHNESS_SENSITIVE
    assert classify_claim("私はこれを使う", user_reported=True) == ClaimKind.USER_REPORTED
    rows = simple_assess_claims([("unmatched claim", False)], [])
    assert rows[0].status == ClaimStatus.UNKNOWN


def test_source_backed_support_contradiction_and_citation() -> None:
    supported = _artifact("sha256:" + "a" * 64, "Python version is current and supported")
    contradicted = _artifact("sha256:" + "b" * 64, "Python version is not supported")
    rows = simple_assess_claims([("Python version is supported", False)], [supported])
    assert rows[0].status == ClaimStatus.SUPPORTED
    assert deterministic_citations(rows[0], {supported.artifact_id: supported}) == [
        "[Fact](https://example.com/fact)"
    ]
    rows = simple_assess_claims([("Python version is supported", False)], [contradicted])
    assert rows[0].status == ClaimStatus.CONTRADICTED


def test_bundle_is_durable_and_rebuildable(tmp_path: Path, monkeypatch) -> None:
    from llm_wiki_mcp import research_store

    monkeypatch.setattr(research_store, "WIKI_ROOT", tmp_path / "wiki")
    store = ResearchStore(root=tmp_path / "runtime")
    artifact = store.put_artifact(
        "evidence",
        source_type="wiki_read",
        source_uri="wiki:fact",
        durable=True,
    )
    assessment = simple_assess_claims([("evidence", False)], [artifact])
    bundle = build_bundle(run_id="run-1", claims=assessment, artifacts=[artifact], store=store)
    assert store.read_artifact(artifact.artifact_id) == b"evidence"
    assert (tmp_path / "wiki" / "research" / "bundles" / "run-1.json").exists()
    assert bundle.research_run_id == "run-1"
