from __future__ import annotations

from pathlib import Path

from llm_wiki_mcp.evidence_bundle import build_bundle, simple_assess_claims
from llm_wiki_mcp.research_auditor import audit_research_run
from llm_wiki_mcp.research_store import ResearchStore


def test_auditor_records_missing_evidence_without_mutation(tmp_path: Path, monkeypatch) -> None:
    from llm_wiki_mcp import research_store

    monkeypatch.setattr(research_store, "WIKI_ROOT", tmp_path / "wiki")
    store = ResearchStore(root=tmp_path / "research")
    bundle = build_bundle(
        run_id="run",
        claims=simple_assess_claims([("no evidence", False)], []),
        artifacts=[],
        store=store,
    )
    path = tmp_path / "audit.jsonl"
    audit = audit_research_run({"stop_reason": "completed"}, bundle, store=store, path=path)
    assert audit["status"] == "attention"
    assert audit["missing_evidence"] == ["no evidence"]
    assert path.exists()
