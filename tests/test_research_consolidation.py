from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_mcp.research_config import ResearchConfig
from llm_wiki_mcp.research_consolidation import ALLOWED_OPERATIONS, run_consolidation
from llm_wiki_mcp.research_store import ResearchStore


def _receipt(store: ResearchStore, run_id: str, claim: str) -> None:
    store.append_event(
        run_id,
        {
            "kind": "post_answer_audit",
            "audit": {
                "missing_evidence": [claim],
                "unsupported_claim": [],
                "wasted_action": [],
            },
        },
    )
    store.append_event(run_id, {"kind": "durable_receipt", "bundle_id": f"bundle:{run_id}"})


def _config() -> ResearchConfig:
    return ResearchConfig(
        enabled=True,
        mode="explicit",
        consolidation_enabled=True,
        consolidation_min_interval_seconds=0,
        consolidation_min_new_sessions=1,
        consolidation_max_jobs=3,
    )


def test_receipt_gated_proposal_only_and_latest_wins(tmp_path: Path, monkeypatch) -> None:
    from llm_wiki_mcp import research_store

    monkeypatch.setattr(research_store, "WIKI_ROOT", tmp_path / "wiki")
    store = ResearchStore(root=tmp_path / "runs")
    state = tmp_path / "state.json"
    lock = tmp_path / "lease.lock"
    proposals = tmp_path / "proposals.jsonl"
    _receipt(store, "run-1", "latest model version")
    first = run_consolidation(
        config=_config(), store=store, state_path=state, lock_path=lock, proposal_path=proposals
    )
    assert first["status"] == "ok"
    rows = [json.loads(line) for line in proposals.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["operation"] in ALLOWED_OPERATIONS
    assert rows[-1]["mutation_mode"] == "proposal_only"

    _receipt(store, "run-2", "latest model version")
    second = run_consolidation(
        config=_config(), store=store, state_path=state, lock_path=lock, proposal_path=proposals
    )
    assert second["status"] == "ok"
    rows = [json.loads(line) for line in proposals.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["supersedes"] == rows[0]["proposal_id"]


def test_failure_does_not_advance_cursor(tmp_path: Path, monkeypatch) -> None:
    from llm_wiki_mcp import research_consolidation, research_store

    monkeypatch.setattr(research_store, "WIKI_ROOT", tmp_path / "wiki")
    store = ResearchStore(root=tmp_path / "runs")
    _receipt(store, "run", "latest fact")
    state = tmp_path / "state.json"
    monkeypatch.setattr(
        research_consolidation,
        "append_jsonl_durable",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        run_consolidation(
            config=_config(),
            store=store,
            state_path=state,
            lock_path=tmp_path / "lease.lock",
            proposal_path=tmp_path / "proposals.jsonl",
        )
    assert not state.exists()


def test_without_receipt_never_runs_even_when_forced(tmp_path: Path, monkeypatch) -> None:
    from llm_wiki_mcp import research_store

    monkeypatch.setattr(research_store, "WIKI_ROOT", tmp_path / "wiki")
    store = ResearchStore(root=tmp_path / "runs")
    store.append_event("run", {"kind": "post_answer_audit", "audit": {"missing_evidence": ["x"]}})
    result = run_consolidation(
        config=_config(),
        store=store,
        force=True,
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "lease.lock",
        proposal_path=tmp_path / "proposals.jsonl",
    )
    assert result["reason"] == "no_durable_receipts"
