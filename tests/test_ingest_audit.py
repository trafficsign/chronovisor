from __future__ import annotations

import json
from pathlib import Path

from chronovisor.core.runtime_config import IngestAuditConfig
from chronovisor.ingest.ingest_audit import (
    decide_ingest_audit,
    record_frontier_audit_outcome,
)


def _decision(
    tmp_path: Path,
    *,
    source_key: str = "f" * 64,
    raw_content: str = "ordinary durable observation",
    operations: list[dict] | None = None,
    failed: list[dict] | None = None,
    disposition: str = "operations_available",
    config: IngestAuditConfig | None = None,
):
    return decide_ingest_audit(
        source_key=source_key,
        raw_content=raw_content,
        operations=operations
        if operations is not None
        else [{"type": "create", "filename": "memory/ordinary-note.md"}],
        failed_operation_specs=failed or [],
        local_disposition=disposition,
        state_path=tmp_path / "audit-state.json",
        config=config or IngestAuditConfig(),
    )


def test_low_risk_ingest_is_locally_authorized(tmp_path: Path) -> None:
    decision = _decision(tmp_path)

    assert decision.required is False
    assert decision.mode == "local"
    assert decision.sample_rate == 0.05


def test_stable_hash_selects_quality_sample(tmp_path: Path) -> None:
    decision = _decision(tmp_path, source_key="0" * 64)

    assert decision.required is True
    assert decision.mode == "sampled"
    assert "deterministic quality sample" in decision.reasons


def test_correction_and_incomplete_generation_are_mandatory(tmp_path: Path) -> None:
    correction = _decision(tmp_path, raw_content="その記憶は違う。訂正して")
    incomplete = _decision(tmp_path, failed=[{"filename": "missing.md"}])

    assert correction.mode == "mandatory"
    assert incomplete.mode == "mandatory"


def test_operational_policy_target_is_mandatory(tmp_path: Path) -> None:
    decision = _decision(
        tmp_path,
        operations=[{"type": "update", "filename": "chronovisor-security-policy.md"}],
    )

    assert decision.required is True
    assert "operational or sensitive target" in decision.reasons


def test_ordinary_policy_page_is_not_privileged(tmp_path: Path) -> None:
    decision = _decision(
        tmp_path,
        source_key="f" * 64,
        operations=[{"type": "update", "filename": "ai/model-policy-notes.md"}],
    )

    assert decision.required is False
    assert decision.mode == "local"


def test_adaptive_sampling_rises_when_audits_catch_issues(tmp_path: Path) -> None:
    state_path = tmp_path / "audit-state.json"
    config = IngestAuditConfig(adaptive_min_audits=5)
    for index in range(5):
        record_frontier_audit_outcome(
            state_path=state_path,
            source_key=f"{index:064x}",
            approved=index >= 2,
            mode="sampled",
            reasons=["test"],
            config=config,
        )

    decision = decide_ingest_audit(
        source_key="0" * 64,
        raw_content="ordinary",
        operations=[{"type": "create", "filename": "memory/note.md"}],
        failed_operation_specs=[],
        local_disposition="operations_available",
        state_path=state_path,
        config=config,
    )

    assert decision.caught_issue_rate == 0.4
    assert decision.adaptive_sample_rate == 0.1
    assert decision.sample_rate == 0.1
    assert decision.mode == "sampled"


def test_caught_issue_remains_sticky_after_repaired_approval(tmp_path: Path) -> None:
    state_path = tmp_path / "audit-state.json"
    config = IngestAuditConfig()
    record_frontier_audit_outcome(
        state_path=state_path,
        source_key="a" * 64,
        approved=False,
        mode="sampled",
        reasons=["sample"],
        config=config,
    )
    record_frontier_audit_outcome(
        state_path=state_path,
        source_key="a" * 64,
        approved=True,
        mode="sampled",
        reasons=["sample"],
        config=config,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(state["outcomes"]) == 1
    assert state["outcomes"][0]["caught_issue"] is True
    assert state["outcomes"][0]["approved"] is True
