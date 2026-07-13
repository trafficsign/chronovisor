from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm_wiki_mcp.frontier_guard import EvidenceValidationError
from llm_wiki_mcp.system_incident_supervisor import (
    IncidentStateError,
    SystemIncidentSupervisor,
)


FAILURE_CLASS = "ingest.runtime_schema_invalid"
FINGERPRINT = "ingest.runtime_schema_invalid"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    raw_files: tuple[str, ...],
    failure_class: str = FAILURE_CLASS,
    fingerprint: str = FINGERPRINT,
    related_raw_files: dict[str, tuple[str, ...]] | None = None,
) -> tuple[SystemIncidentSupervisor, Path, Path, list[Path]]:
    failures_root = tmp_path / "failures"
    packet_dir = failures_root / "packets"
    source_path = packet_dir / "operational-source.json"
    source = {
        "failure_id": "operational-source",
        "failure_class": failure_class,
        "fingerprint": fingerprint,
        "status": "local_quarantined",
        "local_repair_attempts": 2,
        "operational_local_repair_evidence": ["a" * 64, "b" * 64],
        "job_id": None,
    }
    _write_json(source_path, source)
    state = {
        "failures": {
            raw_file: {
                "fingerprint": fingerprint,
                "failure_class": failure_class,
                "self_heal_queued": True,
                "packet_path": str(source_path),
                **(
                    {"related_raw_files": list(related_raw_files[raw_file])}
                    if related_raw_files is not None and raw_file in related_raw_files
                    else {}
                ),
            }
            for raw_file in raw_files
        }
    }
    state_path = failures_root / "state.json"
    _write_json(state_path, state)
    enqueued: list[Path] = []

    def enqueue(path: Path) -> dict[str, object]:
        enqueued.append(path)
        return {"job_id": "repair-job", "enqueued": True, "coalesced": False}

    supervisor = SystemIncidentSupervisor(
        tmp_path / "system-incidents",
        packet_dir=packet_dir,
        failure_state_file=state_path,
        enqueue=enqueue,
    )
    return supervisor, source_path, state_path, enqueued


def _attach_verified_reproduction(
    supervisor: SystemIncidentSupervisor,
    source_path: Path,
    *,
    failing_test: str = "tests/test_ingest.py::test_schema_contract",
) -> Path:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    artifact = supervisor.root / "reproduction-artifacts" / "failure.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b'{"status":"failed"}\n')
    receipt = supervisor.root / "reproduction-receipts" / "failure.json"
    _write_json(
        receipt,
        {
            "schema_version": 1,
            "producer": "trusted_system_incident_supervisor",
            "outcome": "reproducibly_failed",
            "source_packet_path": str(source_path.resolve()),
            "source_failure_class": source["failure_class"],
            "source_fingerprint": source["fingerprint"],
            "artifact": str(artifact),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "failing_test": failing_test,
        },
    )
    source["deterministic_reproduction_receipt"] = str(receipt)
    _write_json(source_path, source)
    return receipt


def test_one_raw_never_promotes_routine_operational_failure(tmp_path: Path) -> None:
    supervisor, source_path, _state_path, enqueued = _fixture(
        tmp_path,
        raw_files=("one.md",),
    )

    result = supervisor.observe_operational_failure_packet(source_path)

    assert result["status"] == "observed"
    assert result["reason"] == "deterministic_reproduction_not_verified"
    assert result["distinct_raw_count"] == 1
    assert enqueued == []
    assert not list((tmp_path / "failures" / "packets").glob("system-*.json"))


def test_two_distinct_raws_promote_once_across_supervisor_instances(
    tmp_path: Path,
) -> None:
    supervisor, source_path, state_path, enqueued = _fixture(
        tmp_path,
        raw_files=("one.md", "two.md"),
    )
    _attach_verified_reproduction(supervisor, source_path)

    first = supervisor.observe_operational_failure_packet(source_path)
    second_process = SystemIncidentSupervisor(
        tmp_path / "system-incidents",
        packet_dir=tmp_path / "failures" / "packets",
        failure_state_file=state_path,
        enqueue=lambda path: enqueued.append(path) or {"job_id": "duplicate"},
    )
    duplicate = second_process.observe_operational_failure_packet(source_path)

    assert first["status"] == "packet_created"
    assert first["distinct_raw_count"] == 2
    assert duplicate["status"] == "packet_exists"
    assert enqueued == [Path(first["packet_path"])]
    packets = list(
        (tmp_path / "failures" / "packets").glob("system-operational-*.json")
    )
    assert packets == [Path(first["packet_path"])]
    incident = json.loads(packets[0].read_text(encoding="utf-8"))
    assert incident["source_packet_paths"] == [str(source_path.resolve())]
    assert incident["raw_files"] == ["one.md", "two.md"]
    assert incident["source_fingerprint"] == FINGERPRINT
    assert incident["logical_raw_groups"] == [["one.md"], ["two.md"]]
    assert incident["repair_evidence"]["notes"]["producer"] == (
        "trusted_operational_failure_supervisor"
    )


def test_two_distinct_raws_without_receipt_are_observability_only(
    tmp_path: Path,
) -> None:
    supervisor, source_path, _state_path, enqueued = _fixture(
        tmp_path,
        raw_files=("one.md", "two.md"),
    )

    result = supervisor.observe_operational_failure_packet(source_path)

    assert result["status"] == "observed"
    assert result["reason"] == "deterministic_reproduction_not_verified"
    assert result["distinct_raw_count"] == 2
    assert result["cross_input_cluster_observed"] is True
    assert result["deterministic_reproduction_verified"] is False
    assert result["frontier_eligible"] is False
    assert enqueued == []
    assert not list(
        (tmp_path / "failures" / "packets").glob("system-operational-*.json")
    )


def test_two_fragments_in_one_related_group_count_as_one_logical_input(
    tmp_path: Path,
) -> None:
    fragments = ("fragment-a.md", "fragment-b.md")
    supervisor, source_path, _state_path, enqueued = _fixture(
        tmp_path,
        raw_files=fragments,
        related_raw_files={name: fragments for name in fragments},
    )

    result = supervisor.observe_operational_failure_packet(source_path)

    assert result["status"] == "observed"
    assert result["reason"] == "deterministic_reproduction_not_verified"
    assert result["linked_raw_file_count"] == 2
    assert result["distinct_raw_count"] == 1
    assert enqueued == []


def test_tampered_artifact_blocks_reenqueue_fail_closed(tmp_path: Path) -> None:
    supervisor, source_path, _state_path, _enqueued = _fixture(
        tmp_path,
        raw_files=("one.md", "two.md"),
    )
    _attach_verified_reproduction(supervisor, source_path)
    enqueue_calls: list[Path] = []

    def unavailable(path: Path) -> dict[str, object]:
        enqueue_calls.append(path)
        raise RuntimeError("queue unavailable")

    supervisor._enqueue = unavailable
    first = supervisor.observe_operational_failure_packet(source_path)
    artifact_path = Path(first["artifact_path"])
    artifact_path.unlink()

    with pytest.raises(IncidentStateError, match="incident_artifact_unreadable"):
        supervisor.observe_operational_failure_packet(source_path)

    assert first["status"] == "packet_enqueue_failed"
    assert enqueue_calls == [Path(first["packet_path"])]


def test_incident_success_releases_sources_but_human_required_does_not(
    tmp_path: Path,
) -> None:
    supervisor, source_path, _state_path, _enqueued = _fixture(
        tmp_path,
        raw_files=("one.md", "two.md"),
    )
    _attach_verified_reproduction(supervisor, source_path)
    created = supervisor.observe_operational_failure_packet(source_path)
    incident_path = Path(created["packet_path"])
    incident = json.loads(incident_path.read_text(encoding="utf-8"))

    incident["status"] = "human_required"
    _write_json(incident_path, incident)
    deferred = supervisor.sync_operational_incident_outcome(incident_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert deferred["repair_success"] is False
    assert source["status"] == "local_quarantined"
    assert source["system_incident_status"] == "human_required"

    incident["status"] = "frontier_approved"
    _write_json(incident_path, incident)
    released = supervisor.sync_operational_incident_outcome(incident_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert released["repair_success"] is True
    assert source["status"] == "frontier_approved"
    assert source["system_incident_packet_path"] == str(incident_path.resolve())
    assert source["quarantined_at"] is None


@pytest.mark.parametrize(
    "failure_class",
    [
        "ingest.runtime_auth_required",
        "ingest.runtime_billing_required",
        "ingest.runtime_keychain_permission_required",
    ],
)
def test_external_authority_failures_are_excluded(
    tmp_path: Path,
    failure_class: str,
) -> None:
    supervisor, source_path, _state_path, enqueued = _fixture(
        tmp_path,
        raw_files=("one.md", "two.md"),
        failure_class=failure_class,
        fingerprint=failure_class,
    )

    result = supervisor.observe_operational_failure_packet(source_path)

    assert result == {
        "status": "excluded",
        "reason": "source_failure_not_operational",
        "source_packet_path": str(source_path.resolve()),
        "dry_run": False,
    }
    assert enqueued == []


def test_one_raw_requires_read_back_verified_deterministic_receipt(
    tmp_path: Path,
) -> None:
    supervisor, source_path, _state_path, enqueued = _fixture(
        tmp_path,
        raw_files=("one.md",),
    )
    artifact = tmp_path / "system-incidents" / "reproduction-artifacts" / "failure.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b'{"status":"failed"}\n')
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt = tmp_path / "system-incidents" / "reproduction-receipts" / "schema.json"
    _write_json(
        receipt,
        {
            "schema_version": 1,
            "producer": "trusted_system_incident_supervisor",
            "outcome": "reproducibly_failed",
            "source_packet_path": str(source_path.resolve()),
            "source_failure_class": FAILURE_CLASS,
            "source_fingerprint": FINGERPRINT,
            "artifact": str(artifact),
            "artifact_sha256": artifact_sha,
            "failing_test": "tests/test_ingest.py::test_schema_contract",
        },
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["deterministic_reproduction_receipt"] = str(receipt)
    _write_json(source_path, source)

    promoted = supervisor.observe_operational_failure_packet(source_path)

    assert promoted["status"] == "packet_created"
    assert promoted["distinct_raw_count"] == 1
    assert enqueued == [Path(promoted["packet_path"])]

    incident_path = Path(promoted["packet_path"])
    original_incident = json.loads(incident_path.read_text(encoding="utf-8"))
    assert original_incident["repair_evidence"]["reproduction"]["command"] == [
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/test_ingest.py::test_schema_contract",
    ]
    for mutation in ("command", "failing_test", "evidence"):
        tampered = json.loads(json.dumps(original_incident))
        if mutation == "command":
            tampered["repair_evidence"]["reproduction"]["command"] = ["true"]
        elif mutation == "failing_test":
            tampered["repair_evidence"]["reproduction"]["failing_test"] = (
                "tests/test_other.py::test_unrelated"
            )
        else:
            tampered["repair_evidence"]["notes"][
                "deterministic_reproduction_evidence"
            ] = "f" * 64
        _write_json(incident_path, tampered)
        with pytest.raises(
            IncidentStateError,
            match="incident_deterministic_reproduction_binding_mismatch",
        ):
            supervisor.validate_operational_incident_packet(incident_path)
    _write_json(incident_path, original_incident)

    artifact.write_bytes(b'{"status":"tampered"}\n')
    rejected = supervisor.observe_operational_failure_packet(source_path)
    assert rejected["status"] == "observed"
    assert rejected["reason"] == "deterministic_reproduction_not_verified"
    assert enqueued == [Path(promoted["packet_path"])]


def test_deterministic_receipt_rejects_arbitrary_reproduction_argv(
    tmp_path: Path,
) -> None:
    supervisor, source_path, _state_path, enqueued = _fixture(
        tmp_path,
        raw_files=("one.md",),
    )
    artifact = tmp_path / "system-incidents" / "reproduction-artifacts" / "bad.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b'{"status":"failed"}\n')
    receipt = tmp_path / "system-incidents" / "reproduction-receipts" / "bad.json"
    _write_json(
        receipt,
        {
            "schema_version": 1,
            "producer": "trusted_system_incident_supervisor",
            "outcome": "reproducibly_failed",
            "source_packet_path": str(source_path.resolve()),
            "source_failure_class": FAILURE_CLASS,
            "source_fingerprint": FINGERPRINT,
            "artifact": str(artifact),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "failing_test": "tests/test_ingest.py::test_schema_contract",
            "command": ["true"],
        },
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["deterministic_reproduction_receipt"] = str(receipt)
    _write_json(source_path, source)

    result = supervisor.observe_operational_failure_packet(source_path)

    assert result["status"] == "observed"
    assert result["reason"] == "deterministic_reproduction_not_verified"
    assert enqueued == []


@pytest.mark.parametrize(
    ("failure_class", "fingerprint"),
    [
        (
            "ingest.runtime_semantic_projection_failure",
            "ingest.runtime_semantic_projection_failure:rawsemanticprojectionerror",
        ),
        (
            "ingest.runtime_semantic_projection_artifact_conflict",
            "ingest.runtime_semantic_projection_artifact_conflict:fileexistserror",
        ),
        (
            "ingest.runtime_semantic_projection_capacity",
            "ingest.runtime_semantic_projection_capacity:memoryerror",
        ),
        (
            "ingest.runtime_semantic_projection_internal_error",
            "ingest.runtime_semantic_projection_internal_error:runtimeerror",
        ),
    ],
)
def test_projection_exception_type_cluster_requires_deterministic_receipt(
    tmp_path: Path,
    failure_class: str,
    fingerprint: str,
) -> None:
    supervisor, source_path, _state_path, enqueued = _fixture(
        tmp_path,
        raw_files=("one.md", "two.md"),
        failure_class=failure_class,
        fingerprint=fingerprint,
    )

    result = supervisor.observe_operational_failure_packet(source_path)

    assert result["status"] == "observed"
    assert result["reason"] == "deterministic_reproduction_not_verified"
    assert enqueued == []


def test_projection_capacity_cause_cannot_use_two_logical_inputs(
    tmp_path: Path,
) -> None:
    failure_class = "ingest.runtime_semantic_projection_capacity"
    supervisor, source_path, _state_path, enqueued = _fixture(
        tmp_path,
        raw_files=("one.md", "two.md"),
        failure_class=failure_class,
        fingerprint=f"{failure_class}:memoryerror",
    )

    result = supervisor.observe_operational_failure_packet(source_path)

    assert result["status"] == "observed"
    assert result["reason"] == "deterministic_reproduction_not_verified"
    assert result["distinct_raw_count"] == 2
    assert enqueued == []


def test_operational_incident_enqueue_contract_accepts_only_bound_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import background_jobs, self_heal
    from llm_wiki_mcp import system_incident_supervisor as incident_module

    supervisor, source_path, _state_path, _enqueued = _fixture(
        tmp_path,
        raw_files=("one.md", "two.md"),
    )
    _attach_verified_reproduction(supervisor, source_path)
    incident_path = Path(
        supervisor.observe_operational_failure_packet(source_path)["packet_path"]
    )
    jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        background_jobs,
        "enqueue_job",
        lambda **kwargs: jobs.append(kwargs) or {"job_id": "job-1", "enqueued": True},
    )
    monkeypatch.setattr(
        incident_module,
        "validate_operational_incident_packet",
        supervisor.validate_operational_incident_packet,
    )

    self_heal.enqueue_system_code_repair(incident_path)

    assert len(jobs) == 1
    assert jobs[0]["name"] == "system-code-repair"
    assert jobs[0]["args"] == [
        "--packet",
        str(incident_path.resolve()),
        "--enable-frontier-repair",
    ]

    original = json.loads(incident_path.read_text(encoding="utf-8"))
    for field, value in (
        ("job_id", "routine-job"),
        ("producer", "untrusted_producer"),
        ("source_failure_class", "ingest.runtime_auth_required"),
    ):
        tampered = json.loads(json.dumps(original))
        if field in {"producer", "source_failure_class"}:
            tampered["repair_evidence"]["notes"][field] = value
        else:
            tampered[field] = value
        tampered_path = incident_path.with_name(f"tampered-{field}.json")
        _write_json(tampered_path, tampered)
        with pytest.raises(EvidenceValidationError):
            self_heal.enqueue_system_code_repair(tampered_path)

    assert len(jobs) == 1


def test_terminal_routine_self_heal_routes_through_incident_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import background_jobs, runtime_status, self_heal, wiki
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    wiki_root = tmp_path / "wiki"
    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(
        runtime_status, "safe_append_event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: LocalRepairDecision(
            status="escalate",
            action="propose_test_case",
            confidence=1.0,
            reason="deterministic system-contract failure",
            source="deterministic",
        ),
    )
    jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        background_jobs,
        "enqueue_job",
        lambda **kwargs: (
            jobs.append(kwargs) or {"job_id": "repair-1", "enqueued": True}
        ),
    )
    packet_path = wiki_root / "runtime" / "failures" / "packets" / "source.json"
    _write_json(
        packet_path,
        {
            "failure_id": "source",
            "failure_class": FAILURE_CLASS,
            "fingerprint": FINGERPRINT,
            "status": "pending_local_repair",
            "local_repair_attempts": 0,
            "job_id": None,
        },
    )
    _write_json(
        wiki_root / "runtime" / "failures" / "state.json",
        {
            "failures": {
                raw_file: {
                    "fingerprint": FINGERPRINT,
                    "failure_class": FAILURE_CLASS,
                    "self_heal_queued": True,
                    "packet_path": str(packet_path),
                }
                for raw_file in ("one.md", "two.md")
            }
        },
    )
    _attach_verified_reproduction(SystemIncidentSupervisor(), packet_path)

    first = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        max_attempts=2,
        backoff_base_seconds=0,
    )
    second = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        max_attempts=2,
        backoff_base_seconds=0,
    )

    assert first["status"] == "pending_local_repair"
    assert second["status"] == "local_quarantined"
    assert second["system_incident"]["status"] == "packet_created"
    source = json.loads(packet_path.read_text(encoding="utf-8"))
    assert len(source["operational_local_repair_evidence"]) == 2
    incident_path = Path(source["system_incident_packet_path"])
    assert incident_path.is_file()
    assert jobs == [
        {
            "name": "system-code-repair",
            "module": "llm_wiki_mcp.self_heal",
            "args": [
                "--packet",
                str(incident_path.resolve()),
                "--enable-frontier-repair",
            ],
            "env": {},
            "stdin_text": "",
        }
    ]

    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    Path(incident["reproduction_artifact"]).unlink()
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: pytest.fail(
            "tampered operational incident must not reach frontier"
        ),
    )
    blocked = self_heal.handle_packet(
        incident_path,
        use_qwen=False,
        enable_frontier=True,
    )
    assert blocked["status"] == "frontier_quarantined"
    assert blocked["reason"] == "operational_incident_evidence_invalid"
