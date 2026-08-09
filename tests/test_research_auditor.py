from __future__ import annotations

from pathlib import Path

from chronovisor.research.evidence_bundle import build_bundle, simple_assess_claims
from chronovisor.research.research_auditor import audit_research_run
from chronovisor.search.research_store import ResearchStore


def test_auditor_records_missing_evidence_without_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.search import research_store

    monkeypatch.setattr(research_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    store = ResearchStore(root=tmp_path / "research")
    bundle = build_bundle(
        run_id="run",
        claims=simple_assess_claims([("no evidence", False)], []),
        artifacts=[],
        store=store,
    )
    path = tmp_path / "audit.jsonl"
    audit = audit_research_run(
        {"stop_reason": "completed"}, bundle, store=store, path=path
    )
    assert audit["status"] == "attention"
    assert audit["missing_evidence"] == ["no evidence"]
    assert path.exists()


def test_web_backed_claim_does_not_report_deep_research_as_avoidable(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.search import research_store

    monkeypatch.setattr(research_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    store = ResearchStore(root=tmp_path / "research")
    artifact = store.put_artifact(
        "The official SearXNG repository is searxng/searxng.",
        source_type="web_search",
        source_uri="https://github.com/searxng/searxng",
        durable=True,
    )
    bundle = build_bundle(
        run_id="web-run",
        claims=simple_assess_claims(
            [("The official SearXNG repository is searxng/searxng", False)],
            [artifact],
        ),
        artifacts=[artifact],
        store=store,
    )
    store.append_event(
        "web-run",
        {
            "kind": "action",
            "iteration": 1,
            "action": {"type": "web_search", "arguments": {"query": "SearXNG"}},
        },
    )

    audit = audit_research_run(
        {"stop_reason": "completed"},
        bundle,
        store=store,
        path=tmp_path / "audit.jsonl",
    )

    assert audit["avoidable_deep"] is False
    assert audit["status"] == "ok"


def test_locally_supported_claim_reports_later_deep_research_as_avoidable(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.search import research_store

    monkeypatch.setattr(research_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    store = ResearchStore(root=tmp_path / "research")
    artifact = store.put_artifact(
        "The project repository is example/project.",
        source_type="chronovisor_read",
        source_uri="wiki:project",
        durable=True,
    )
    bundle = build_bundle(
        run_id="local-run",
        claims=simple_assess_claims(
            [("The project repository is example/project", False)],
            [artifact],
        ),
        artifacts=[artifact],
        store=store,
    )
    store.append_event(
        "local-run",
        {
            "kind": "action",
            "iteration": 2,
            "action": {"type": "web_search", "arguments": {"query": "project"}},
        },
    )

    audit = audit_research_run(
        {"stop_reason": "completed"},
        bundle,
        store=store,
        path=tmp_path / "audit.jsonl",
    )

    assert audit["avoidable_deep"] is True
    assert audit["status"] == "attention"
