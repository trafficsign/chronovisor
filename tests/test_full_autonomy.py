from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.ops import burn_monitor
from chronovisor.decision import decision_authority
from chronovisor.decision import decision_router
from chronovisor.core import durable_state
from chronovisor.decision import failure_supervisor
from chronovisor.ops import health
from chronovisor.raw import raw_replay
from chronovisor.ingest import read_back_repair
from chronovisor.ops import repair_runbook
from chronovisor.ops.deadman import inspect_heartbeat, write_heartbeat
from chronovisor.decision.decision_artifact import DecisionArtifactStore, execution_fingerprint
from chronovisor.decision.decision_router import DecisionRouter
from chronovisor.core.durable_state import (
    DiskPressureError,
    StateSealError,
    atomic_write_bytes,
    read_sealed_json,
    write_sealed_json,
)
from chronovisor.decision.frontier_guard import EvidenceValidationError, RepairIncidentEvidence
from chronovisor.decision.local_structured import ChatRequest
from chronovisor.librarian.managed_hold import ManagedHoldStore
from chronovisor.recall.provisional_recall import search_provisional
from chronovisor.decision.quality_guard import (
    QualityThresholds,
    append_immutable_anchor,
    evaluate_quality,
    lane_is_frozen,
    register_last_known_good,
    run_quality_probe,
)
from chronovisor.raw.raw_semantic_projection import (
    PROJECTION_CHILD_SCHEMA,
    PROJECTION_POLICY_VERSION,
)
from chronovisor.ingest.read_back_integrity import scan_jsonl_prefix, verify_prior_prefix
from chronovisor.core.runtime_config import DecisionRouterConfig


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "reason"],
    "properties": {
        "decision": {"type": "string", "enum": ["approved", "rejected"]},
        "reason": {"type": "string"},
    },
}


def test_light_dashboard_probe_projects_aggregate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = {
        "runtime": {
            "commit_id": "commit",
            "expected_commit": "commit",
            "drift": False,
            "module_path": "/archive/runtime_config.py",
        },
        "status": {
            "state": "idle",
            "pending": 0,
            "raw_outstanding": 0,
            "batch": {"active": False},
            "llm": {"active": False},
            "local_consensus": {"active": False},
            "frontier_repair": {"active": False},
            "frontier_review": {"active": False},
            "semantic_deferred": {"count": 0},
            "operational_deferred": {"count": 0},
        },
        "save_history": {
            "totals": {"pending_bytes": 0},
            "days": [],
        },
        "health": {
            "autonomy_hardening": {
                "decision_artifacts": {
                    "count": 0,
                    "replay_definition": "sealed_execution_fingerprint",
                },
                "deadman": {
                    "main": {"status": "ok"},
                    "observer": {"status": "ok"},
                },
                "quality": {"frozen": 0, "probe": {"status": "ok"}},
                "managed_holds": {"total": 0},
                "provisional_recall": {
                    "entries": 0,
                    "mutation_evidence_allowed": False,
                },
                "frontier_semantic_audit_allowed": False,
            },
            "read_back": {"derived_view_integrity": {"status": "ok"}},
        },
    }
    monkeypatch.setattr(
        burn_monitor,
        "http_get",
        lambda _url, *, parse_json: {
            "ok": True,
            "status": 200,
            "payload": aggregate,
        },
    )

    snapshot = burn_monitor.dashboard_snapshot(
        "http://127.0.0.1:8765", all_endpoints=False
    )

    assert snapshot["snapshot"]["valid"] is True
    assert snapshot["save_history"]["valid"] is True
    assert snapshot["hardening"]["valid"] is True
    assert snapshot["idle_violations"] == []


def _config() -> DecisionRouterConfig:
    return DecisionRouterConfig(
        primary_model="primary:test",
        challenger_model="challenger:test",
        tie_break_model="tie:test",
        primary_keep_alive="1m",
        challenger_keep_alive="1m",
        tie_break_keep_alive="1m",
        num_ctx=32_768,
        num_predict=256,
        read_timeout_ms=5_000,
        max_input_chars=20_000,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
        quorum=2,
    )


class _Transport:
    def __init__(self) -> None:
        self.calls: list[ChatRequest] = []

    def __call__(self, request: ChatRequest) -> str:
        self.calls.append(request)
        return json.dumps({"decision": "approved", "reason": "local quorum"})


def test_durable_state_seal_and_backup_recovery(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_sealed_json(path, {"value": 1})
    write_sealed_json(path, {"value": 2})
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(StateSealError):
        read_sealed_json(path)
    recovered = read_sealed_json(path, recover_backup=True)

    assert recovered["value"] == 1
    assert read_sealed_json(path)["value"] == 1


def test_durable_write_rejects_disk_pressure_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(durable_state.shutil, "disk_usage", lambda _path: usage(1, 1, 0))
    path = tmp_path / "state.bin"

    with pytest.raises(DiskPressureError):
        atomic_write_bytes(path, b"payload")

    assert not path.exists()


def test_decision_artifact_is_content_addressed(tmp_path: Path) -> None:
    fingerprint, identity = execution_fingerprint(
        request_sha256="a" * 64,
        lane="lane",
        context_tier=32_768,
        authority={"epoch": "one"},
        router_policy={"artifact": "b" * 64},
        generation_policy_sha256="c" * 64,
        model_runtime={"models": ["a", "b"]},
    )
    store = DecisionArtifactStore(tmp_path)
    published = store.publish(
        fingerprint=fingerprint,
        identity=identity,
        decision={"decision": "approved"},
        agreement_sha256="d" * 64,
        quorum_proof=[
            {"role": "primary", "model": "a", "signature_sha256": "d" * 64},
            {"role": "challenger", "model": "b", "signature_sha256": "d" * 64},
        ],
        provenance={"frontier_calls": 0},
    )

    assert store.load(fingerprint) == published
    assert published["frontier_calls"] == 0


def test_router_replays_same_execution_without_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import store

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(
        decision_router,
        "bind_lane_contract_request",
        lambda _lane, prompt, _schema, system: (prompt, system),
    )
    monkeypatch.setattr(
        decision_authority,
        "current_semantic_authority",
        lambda lane: ({"source": "test", "lane": lane, "epoch": "one"}, None),
    )
    transport = _Transport()
    router = DecisionRouter(
        config=_config(),
        transport=transport,
        resolve_adoption=False,
        record_replay=False,
        live_resource_control=False,
        artifact_replay=True,
        decision_artifact_root=tmp_path / "artifacts",
    )

    first = router.decide("prompt", SCHEMA, decision_lane="test_lane")
    first_calls = len(transport.calls)
    second = router.decide("prompt", SCHEMA, decision_lane="test_lane")

    assert first.ok and second.ok
    assert first_calls == 2
    assert len(transport.calls) == first_calls
    assert second.residency["source"] == "canonical_artifact_replay"
    assert second.residency["model_invocations"] == 0


def test_router_does_not_replay_unfingerprintable_custom_agreement_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import store

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(
        decision_router,
        "bind_lane_contract_request",
        lambda _lane, prompt, _schema, system: (prompt, system),
    )
    transport = _Transport()
    router = DecisionRouter(
        config=_config(),
        transport=transport,
        resolve_adoption=False,
        record_replay=False,
        live_resource_control=False,
        artifact_replay=True,
        decision_artifact_root=tmp_path / "artifacts",
    )

    first = router.decide(
        "prompt",
        SCHEMA,
        decision_lane="test_lane",
        agreement_key=lambda value: value["decision"],
    )
    second = router.decide(
        "prompt",
        SCHEMA,
        decision_lane="test_lane",
        agreement_key=lambda value: value["decision"],
    )

    assert first.status == second.status
    assert len(transport.calls) == 4
    assert not list((tmp_path / "artifacts").glob("[0-9a-f][0-9a-f]/*.json"))


def test_deadman_heartbeat_detects_stale_and_bad_seal(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    base = datetime(2026, 7, 15, tzinfo=timezone.utc)
    write_heartbeat(
        path,
        role="main_watchdog",
        reported_status="ok",
        now=base,
    )

    stale = inspect_heartbeat(
        path,
        expected_role="main_watchdog",
        max_age_seconds=60,
        now=base + timedelta(seconds=61),
    )
    assert stale["status"] == "stale"
    assert inspect_heartbeat(
        path,
        expected_role="main_watchdog",
        max_age_seconds=60,
        now=base - timedelta(seconds=301),
    )["status"] == "clock_regression"
    assert inspect_heartbeat(
        tmp_path / "missing-observer.json",
        expected_role="independent_observer",
        max_age_seconds=60,
        now=base,
    )["status"] == "missing"

    payload = json.loads(path.read_text())
    payload["sequence"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert inspect_heartbeat(
        path,
        expected_role="main_watchdog",
        max_age_seconds=60,
        now=base,
    )["status"] == "invalid"


def test_independent_observer_has_no_package_import_and_cross_checks_main(
    tmp_path: Path,
) -> None:
    observer_path = (
        Path(__file__).parents[1]
        / "src"
        / "chronovisor"
        / "deadman_observer.py"
    )
    source = observer_path.read_text(encoding="utf-8")
    assert "import chronovisor" not in source
    assert "from chronovisor" not in source
    spec = importlib.util.spec_from_file_location("deadman_observer_test", observer_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write_heartbeat(
        tmp_path / "autonomy" / "watchdog-heartbeat.json",
        role="main_watchdog",
        reported_status="ok",
    )

    result = module.run_once(tmp_path, max_main_age_seconds=60)

    assert result["status"] == "ok"
    observed = inspect_heartbeat(
        tmp_path / "autonomy" / "observer-heartbeat.json",
        expected_role="independent_observer",
        max_age_seconds=60,
    )
    assert observed["status"] == "ok"
    assert observed["sequence"] == 1


def test_observer_threshold_debounces_dedupes_and_honors_cooldown(
    tmp_path: Path,
) -> None:
    observer_path = (
        Path(__file__).parents[1]
        / "src"
        / "chronovisor"
        / "deadman_observer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "deadman_observer_threshold_test", observer_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = datetime(2026, 7, 15, tzinfo=timezone.utc)
    main_path = tmp_path / "autonomy" / "watchdog-heartbeat.json"
    write_heartbeat(
        main_path,
        role="main_watchdog",
        reported_status="ok",
        now=base,
    )

    first = module.run_once(
        tmp_path,
        max_main_age_seconds=60,
        now=base + timedelta(seconds=61),
    )
    second = module.run_once(
        tmp_path,
        max_main_age_seconds=60,
        now=base + timedelta(seconds=62),
    )
    duplicate = module.run_once(
        tmp_path,
        max_main_age_seconds=60,
        now=base + timedelta(seconds=63),
    )
    after_cooldown = module.run_once(
        tmp_path,
        max_main_age_seconds=60,
        now=base + timedelta(seconds=3663),
    )

    assert first["status"] == "ok"
    assert first["incident_emitted"] is False
    assert second["status"] == "alert"
    assert second["incident_emitted"] is True
    assert duplicate["incident_emitted"] is False
    assert after_cooldown["incident_emitted"] is True
    incidents = (
        tmp_path / "autonomy" / "deadman-incidents.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(incidents) == 2
    state = module.read(tmp_path / "autonomy" / "observer-threshold-state.json")
    assert state["threshold_policy"]["minimum_failure_samples"] == 2
    assert state["threshold_policy"]["recovery_samples"] == 2
    assert state["incident_budget_remaining"] == 2

    write_heartbeat(
        main_path,
        role="main_watchdog",
        reported_status="ok",
        now=base + timedelta(seconds=3664),
    )
    recovering = module.run_once(
        tmp_path,
        max_main_age_seconds=60,
        now=base + timedelta(seconds=3664),
    )
    recovered = module.run_once(
        tmp_path,
        max_main_age_seconds=60,
        now=base + timedelta(seconds=3665),
    )
    assert recovering["status"] == "alert"
    assert recovered["status"] == "ok"


def test_quality_drift_freezes_locally_without_frontier(tmp_path: Path) -> None:
    thresholds = QualityThresholds(
        minimum_samples=1,
        trip_samples=2,
        incident_budget_per_day=1,
    )
    bad = {
        "epoch": "e1",
        "sample_count": 10,
        "anchor_match_rate": 0.5,
        "metamorphic_pass_rate": 0.5,
        "flip_rate": 0.5,
    }
    evaluate_quality(root=tmp_path, lane="lane", metrics=bad, thresholds=thresholds)
    state = evaluate_quality(
        root=tmp_path,
        lane="lane",
        metrics=bad,
        thresholds=thresholds,
    )

    assert lane_is_frozen(tmp_path, "lane") is True
    assert state["containment"]["frontier_semantic_audit"] == "prohibited"
    assert state["frontier_allowed"] is False


def _quality_artifact(*, wrong: bool = False) -> dict:
    from chronovisor.lab.local_model_eval import adoption_evidence_sha256

    cases = []
    for index in range(5):
        expected = {"decision": "approved", "case": index}
        actual = {"decision": "rejected", "case": index} if wrong else expected
        cases.append(
            {
                "case_id": hashlib.sha256(f"case-{index}".encode()).hexdigest(),
                "contract_id": f"contract-v1:{index}",
                "decision_lane": "test_lane",
                "expected_signature": expected,
                "expected_decision": "approved",
                "expected_effect": "page_mutation",
                "actual_signature": actual,
                "actual_decision": "rejected" if wrong else "approved",
                "actual_effect": "no_page_mutation" if wrong else "page_mutation",
            }
        )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "adopted": True,
        "evaluation_result_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "model_metadata_sha256": "c" * 64,
        "cases": cases,
    }
    payload["evidence_sha256"] = adoption_evidence_sha256(payload)
    return payload


def test_quality_corpora_are_separated_and_drift_rolls_back_without_frontier(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quality"
    artifact_path = tmp_path / "adoption.json"
    good = _quality_artifact()
    artifact_path.write_text(json.dumps(good), encoding="utf-8")
    thresholds = QualityThresholds(
        minimum_samples=5,
        trip_samples=2,
        recovery_samples=3,
        cooldown_seconds=0,
    )
    for _ in range(3):
        probe = run_quality_probe(
            root=root,
            adoption_artifact=artifact_path,
            thresholds=thresholds,
            artifact_validator=lambda _path: None,
        )
    anchor_before = (root / "immutable-anchor.jsonl").read_bytes()
    assert probe["corpus"]["behavior_promoted_to_anchor"] is False
    assert len(list((root / "behavior-snapshots").glob("[0-9a-f]*.json"))) == 1

    artifact_path.write_text(json.dumps(_quality_artifact(wrong=True)), encoding="utf-8")
    run_quality_probe(
        root=root,
        adoption_artifact=artifact_path,
        thresholds=thresholds,
        artifact_validator=lambda _path: None,
    )
    tripped = run_quality_probe(
        root=root,
        adoption_artifact=artifact_path,
        thresholds=thresholds,
        artifact_validator=lambda _path: None,
    )

    restored = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert restored["cases"][0]["actual_decision"] == "approved"
    assert tripped["frozen_lanes"] == ["test_lane"]
    assert tripped["rollback"]["status"] == "rolled_back"
    assert tripped["lane_status"]["test_lane"]["rollback_verified"] is True
    assert tripped["lane_status"]["test_lane"]["shadow_replay_verified"] is True
    assert tripped["frontier_calls"] == 0
    assert (root / "immutable-anchor.jsonl").read_bytes() == anchor_before


def test_immutable_anchor_rejects_same_identity_relabel(tmp_path: Path) -> None:
    arguments = {
        "root": tmp_path,
        "lane": "lane",
        "case_id": "case",
        "source_kind": "user_correction",
        "source_reference": "conversation-1",
        "expected_signature": {"decision": "approved"},
        "expected_decision": "approved",
        "expected_effect": "mutation",
    }
    append_immutable_anchor(**arguments)
    with pytest.raises(Exception, match="conflict"):
        append_immutable_anchor(
            **{
                **arguments,
                "expected_signature": {"decision": "rejected"},
                "expected_decision": "rejected",
            }
        )


def test_last_known_good_publication_is_bound_to_measured_authority(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "adoption.json"
    authority.write_text('{"epoch":1}\n', encoding="utf-8")
    measured = hashlib.sha256(authority.read_bytes()).hexdigest()
    authority.write_text('{"epoch":2}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed before LKG"):
        register_last_known_good(
            root=tmp_path / "quality",
            lane="lane",
            authority_artifact=authority,
            expected_authority_sha256=measured,
        )


def test_managed_hold_exact_lease_crash_recovery_and_aba(tmp_path: Path) -> None:
    store = ManagedHoldStore(tmp_path / "state.json")
    base = datetime(2026, 7, 15, tzinfo=timezone.utc)
    entry = store.register(
        hold_sha256="a" * 64,
        authority_epoch="b" * 64,
        raw_sha256="c" * 64,
        lane="ingest_reconciliation",
        raw_files=["one.md"],
        now=base,
    )
    store.reconcile_authorities(
        {"ingest_reconciliation": "d" * 64},
        now=base,
    )
    lease = store.acquire(
        owner="worker",
        lease_seconds=1,
        minimum_lane_interval_seconds=0,
        now=base,
    )
    assert lease and lease["state"] == "leased"
    assert store.acquire(owner="other", now=base) is None
    assert store.recover_expired(now=base + timedelta(seconds=2)) == [entry["identity"]]
    replacement = store.acquire(
        owner="other",
        minimum_lane_interval_seconds=0,
        now=base + timedelta(seconds=2),
    )
    assert replacement and replacement["lease_token"] != lease["lease_token"]
    with pytest.raises(Exception, match="token"):
        store.finish(
            identity=entry["identity"],
            lease_token=lease["lease_token"],
            outcome="resolved",
            observed_raw_sha256="c" * 64,
        )


def test_managed_hold_reheld_uses_backoff_and_lane_rate_limit(tmp_path: Path) -> None:
    store = ManagedHoldStore(tmp_path / "state.json")
    base = datetime(2026, 7, 15, tzinfo=timezone.utc)
    first = store.register(
        hold_sha256="a" * 64,
        authority_epoch="b" * 64,
        raw_sha256="c" * 64,
        lane="lane",
        now=base,
    )
    second = store.register(
        hold_sha256="d" * 64,
        authority_epoch="b" * 64,
        raw_sha256="e" * 64,
        lane="lane",
        now=base,
    )
    store.reconcile_authorities({"lane": "f" * 64}, now=base)
    lease = store.acquire(owner="worker", now=base)
    assert lease and lease["identity"] == first["identity"]
    reheld = store.finish(
        identity=lease["identity"],
        lease_token=lease["lease_token"],
        outcome="reheld",
        observed_raw_sha256="c" * 64,
        retry_base_seconds=60,
        now=base,
    )
    assert reheld["next_attempt_at"] == (base + timedelta(seconds=60)).isoformat(
        timespec="seconds"
    )
    assert store.acquire(owner="worker-2", now=base + timedelta(seconds=29)) is None
    second_lease = store.acquire(owner="worker-2", now=base + timedelta(seconds=30))
    assert second_lease and second_lease["identity"] == second["identity"]
    assert store.reconcile_authorities(
        {"lane": "f" * 64}, now=base + timedelta(seconds=59)
    )["count"] == 0


def test_managed_hold_resolves_once_from_retired_packet_evidence(tmp_path: Path) -> None:
    store = ManagedHoldStore(tmp_path / "state.json")
    entry = store.register(
        hold_sha256="a" * 64,
        authority_epoch="b" * 64,
        raw_sha256="c" * 64,
        lane="lane",
    )
    store.reconcile_authorities({"lane": "d" * 64})

    resolved = store.resolve_absent_scheduled(set())
    repeated = store.resolve_absent_scheduled(set())

    assert resolved == [entry["identity"]]
    assert repeated == []
    assert store.snapshot()["counts"]["resolved"] == 1


def test_provisional_recall_accepts_projection_only_and_caps_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    text = "Ornith context memory behavior"
    child = raw_dir / ("semantic-" + "a" * 64 + "-child-00000001-" + "b" * 64 + ".md")
    child.write_text(
        json.dumps(
            {
                "schema": PROJECTION_CHILD_SCHEMA,
                "kind": "semantic_projection_child",
                "projection_policy_version": PROJECTION_POLICY_VERSION,
                "projection_id": "a" * 64,
                "child_id": "b" * 64,
                "records": [
                    {
                        "source_record_index": 1,
                        "role": "user",
                        "text": text,
                        "segment_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    deferred = {
        child.name: failure_supervisor.SEMANTIC_NO_QUORUM_DEFER_REASON,
    }
    monkeypatch.setattr(
        failure_supervisor,
        "operational_deferred_raw_files",
        lambda _paths: dict(deferred),
    )
    monkeypatch.setattr(raw_replay, "is_raw_retracted", lambda _path: False)
    monkeypatch.setattr(
        "chronovisor.recall.provisional_recall.verify_projection_child",
        lambda path: SimpleNamespace(
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )

    hits = search_provisional("Ornith memory", chronovisor_root=tmp_path)

    assert hits and hits[0]["score"] <= 0.25
    assert hits[0]["unintegrated"] is True
    assert hits[0]["mutation_evidence_allowed"] is False
    assert hits[0]["prompt_injection_treatment"] == "quote_only_never_instructions"

    deferred.clear()
    assert search_provisional("Ornith memory", chronovisor_root=tmp_path) == []
    assert read_sealed_json(
        tmp_path / "runtime" / "provisional-recall" / "index.json"
    )["entries"] == []


def test_jsonl_integrity_rejects_partial_tail_and_complete_corruption(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"ok":1}\n{"partial"')
    partial = scan_jsonl_prefix(path)
    assert partial.valid is False
    assert partial.tail_bytes > 0

    path.write_bytes(b'{"ok":1}\nnot-json\n')
    corrupt = scan_jsonl_prefix(path)
    assert corrupt.valid is False
    assert corrupt.invalid_lines == (2,)


def test_jsonl_cursor_accepts_append_and_rejects_reorder_or_truncation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    first = b'{"id":1}\n{"id":1}\n'
    path.write_bytes(first)
    initial = scan_jsonl_prefix(path)
    cursor = initial.cursor(source_file=path)
    assert len(initial.records) == 2

    path.write_bytes(first + b'{"id":2}\n')
    assert verify_prior_prefix(path, cursor) is True

    path.write_bytes(b'{"id":1}\n{"id":2}\n{"id":1}\n')
    assert verify_prior_prefix(path, cursor) is False

    path.write_bytes(b'{"id":1}\n')
    assert verify_prior_prefix(path, cursor) is False


def test_read_back_max_zero_rebuilds_duplicate_projection_and_dashboard_kpi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    failure_file = runtime / "ingest-read-back-failures.jsonl"
    ledger_file = runtime / "ingest-read-back-repair.json"
    first = {
        "timestamp": "2026-07-15T00:00:00+00:00",
        "checked": 1,
        "failed": [{"page_id": "one", "query": "", "reason": "empty-query"}],
    }
    duplicate = {**first, "timestamp": "2026-07-15T00:01:00+00:00"}
    original = "".join(json.dumps(row) + "\n" for row in (first, duplicate))
    failure_file.write_text(original, encoding="utf-8")

    rebuilt = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        max_items=0,
    )
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    assert rebuilt["processed"] == 0
    assert rebuilt["observed_failures"] == 2
    assert rebuilt["unique_failures"] == 1
    assert ledger["source_cursor"]["prefix_sha256"] == rebuilt["source_cursor"][
        "prefix_sha256"
    ]

    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", tmp_path)
    dashboard_kpi = health.read_back_kpi()
    assert dashboard_kpi["checked"] == 2
    assert dashboard_kpi["failures"] == 2
    assert dashboard_kpi["derived_view_integrity"]["status"] == "ok"

    valid_ledger = ledger_file.read_bytes()
    malformed = json.loads(valid_ledger)
    malformed.pop("view_sha256")
    ledger_file.write_text(json.dumps(malformed), encoding="utf-8")
    invalid_view = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        max_items=0,
    )
    assert invalid_view["status"] == "ledger_integrity_error"
    ledger_file.write_bytes(valid_ledger)

    failure_file.write_text("".join(reversed(original.splitlines(keepends=True))), encoding="utf-8")
    rewritten = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        max_items=0,
    )
    assert rewritten["status"] == "source_history_rewritten"

def test_l1_runbook_restores_only_allowlisted_sealed_state(tmp_path: Path) -> None:
    target = tmp_path / "runtime" / "managed-holds" / "state.json"
    write_sealed_json(target, {"version": 1})
    write_sealed_json(target, {"version": 2})

    result = repair_runbook.run_l1(
        "restore-durable-state",
        chronovisor_root=tmp_path,
        target="managed-holds",
    )

    assert result["status"] == "ok"
    assert read_sealed_json(target)["version"] == 1
    with pytest.raises(ValueError, match="allowlisted"):
        repair_runbook.run_l1(
            "restore-durable-state",
            chronovisor_root=tmp_path,
            target="arbitrary-path",
        )


def test_l0_dashboard_plist_requires_keepalive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plistlib

    launchd = tmp_path / "launchd"
    launchd.mkdir()
    path = launchd / "com.trafficsign.chronovisor-dashboard.plist"
    with path.open("wb") as stream:
        plistlib.dump(
            {
                "Label": "com.trafficsign.chronovisor-dashboard",
                "KeepAlive": True,
                "RunAtLoad": True,
            },
            stream,
        )
    monkeypatch.setattr(repair_runbook, "runtime_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        repair_runbook.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    status = repair_runbook.launchd_status("dashboard")

    assert status["loaded"] is True
    assert status["keep_alive"] is True
    assert status["restart_policy_valid"] is True


def test_frontier_guard_rejects_semantic_payload_even_for_trusted_producer() -> None:
    with pytest.raises(EvidenceValidationError, match="semantic payload"):
        RepairIncidentEvidence(
            component="watchdog.health_snapshot",
            fingerprint="c" * 64,
            failure_class="system_health_snapshot_exception",
            occurrence_count=3,
            distinct_inputs=("d" * 64, "e" * 64),
            local_repair_attempts=2,
            local_repair_evidence=("a" * 64, "b" * 64),
            reproduction_command=(
                "uv",
                "run",
                "pytest",
                "-q",
                "tests/test_full_autonomy.py::test_frontier_guard_rejects_semantic_payload_even_for_trusted_producer",
            ),
            notes={
                "producer": "trusted_watchdog",
                "incident_key": "incident-one",
                "golden_cases": [{"memory": "forbidden"}],
            },
        )
