from __future__ import annotations

from pathlib import Path

from chronovisor.evidence_bundle import build_bundle, simple_assess_claims
from chronovisor.research_auditor import audit_research_run
from chronovisor.research_store import ResearchStore


def test_auditor_records_missing_evidence_without_mutation(tmp_path: Path, monkeypatch) -> None:
    from chronovisor import research_store

    monkeypatch.setattr(research_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
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
