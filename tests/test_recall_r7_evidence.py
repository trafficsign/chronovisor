from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_rollout as rollout
from chronovisor.recall import recall_distillation_store as store

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r7_evidence_test", ROOT / "src/chronovisor/recall/recall_r7_evidence.py"
)
assert SPEC is not None and SPEC.loader is not None
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


def _id(number: int) -> str:
    return f"{number:064x}"


def _poll(root: Path, stage: str, when: datetime, observation_id: str) -> str:
    poll_id, _, _ = store.write_immutable(
        root / "polls",
        {
            "kind": "r7-live-poll",
            "stage": stage,
            "run_id": _id(100),
            "captured_at": when.isoformat(),
            "monotonic_ns": 1,
            "identities": {},
            "source": {},
            "runtime": {},
            "process": {},
            "health": {},
            "api": {},
            "dom_sha256": _id(200),
            "observation_chain": {"records": 0, "head_sha256": ""},
            "observations_sha256": _id(201),
            "observations": [
                {
                    "observation_id": observation_id,
                    "host": "host-a",
                    "cohort": "cohort-a",
                }
            ],
            "producer": {
                "name": "chronovisor-r7-evidence",
                "version": 1,
                "synthetic_fixture": False,
            },
        },
        schema=EVIDENCE.POLL_SCHEMA,
    )
    return poll_id


def _ledger(root: Path, entries: list[tuple[str, str, datetime, int]]) -> None:
    prior = ""
    lines = []
    for poll_id, stage, when, monotonic_ns in entries:
        row = {
            "schema": EVIDENCE.LEDGER_SCHEMA,
            "namespace": "recall-distillation",
            "poll_id": poll_id,
            "poll_sha256": hashlib.sha256(
                (root / "polls" / f"{poll_id}.json").read_bytes()
            ).hexdigest(),
            "stage": stage,
            "observed_at": when.isoformat(),
            "monotonic_ns": monotonic_ns,
            "previous_sha256": prior,
        }
        row["entry_sha256"] = EVIDENCE._digest(row)
        prior = row["entry_sha256"]
        lines.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    (root / "poll-ledger.jsonl").write_text("\n".join(lines) + "\n")
    store.write_sealed_state(
        root / "poll-ledger-state.json",
        {
            "kind": "r7-poll-ledger-state",
            "count": len(entries),
            "head_sha256": prior,
        },
    )


def test_empty_or_short_real_collector_is_not_certified(tmp_path: Path) -> None:
    assert EVIDENCE.validate_collector(tmp_path)["certification"] is False
    now = datetime(2026, 8, 24, tzinfo=UTC)
    poll_id = _poll(tmp_path, "shadow", now, _id(1))
    _ledger(tmp_path, [(poll_id, "shadow", now, 1)])
    result = EVIDENCE.validate_collector(tmp_path)
    assert result["certification"] is False
    assert result["certification_reason"] == "collector_poll_provenance_invalid"


def test_ledger_tamper_backward_clock_and_cross_stage_reuse_fail_closed(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 24, tzinfo=UTC)
    first = _poll(tmp_path, "shadow", now, _id(1))
    second = _poll(tmp_path, "5", now + timedelta(seconds=1), _id(1))
    _ledger(
        tmp_path,
        [(first, "shadow", now, 2), (second, "5", now + timedelta(seconds=1), 1)],
    )
    assert (
        EVIDENCE.validate_collector(tmp_path)["certification_reason"]
        == "collector_ledger_invalid"
    )
    _ledger(
        tmp_path,
        [(first, "shadow", now, 1), (second, "5", now + timedelta(seconds=1), 2)],
    )
    result = EVIDENCE.validate_collector(tmp_path)
    assert result["certification_reason"] == "collector_poll_provenance_invalid"
    assert all(stage["certified"] is False for stage in result["stages"].values())
    text = (tmp_path / "poll-ledger.jsonl").read_text()
    (tmp_path / "poll-ledger.jsonl").write_text(
        text.replace('"stage":"shadow"', '"stage":"25"', 1)
    )
    assert (
        EVIDENCE.validate_collector(tmp_path)["certification_reason"]
        == "collector_ledger_invalid"
    )


def test_safe_input_rejects_symlink_and_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(EVIDENCE.EvidenceError, match="unsafe"):
        EVIDENCE._safe_json(link, "input")
    source.write_text("[]")
    with pytest.raises(EVIDENCE.EvidenceError, match="not object"):
        EVIDENCE._safe_json(source, "input")


def test_content_addressed_artifact_id_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    artifact_id, path, _ = store.write_immutable(
        tmp_path.resolve(),
        {"kind": "attacker-shaped"},
        schema="chronovisor.test-artifact.v1",
        artifact_id="2" * 64,
    )
    assert artifact_id == "2" * 64
    with pytest.raises(EVIDENCE.EvidenceError, match="artifact id/content"):
        EVIDENCE._read_sealed_artifact(
            path, "chronovisor.test-artifact.v1", "test artifact"
        )


def test_process_identity_rejects_a_spoofed_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "worker"
    executable.write_text("worker", encoding="utf-8")
    real = Path(sys.executable).resolve()

    class Result:
        stdout = f"{os.getpid()} Mon Aug 25 00:00:00 2026 {real}\n"

    monkeypatch.setattr(EVIDENCE.subprocess, "run", lambda *_args, **_kwargs: Result())
    with pytest.raises(EVIDENCE.EvidenceError, match="mismatch"):
        EVIDENCE._process_identity(executable, os.getpid())


def test_process_digest_bound_sparse_and_symlink_inputs(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "worker"
    executable.write_bytes(b"worker")
    digest, metadata = EVIDENCE._read_stable_sha256(executable, "executable", max_bytes=6)
    assert digest == hashlib.sha256(b"worker").hexdigest()
    assert metadata["size"] == 6

    sparse = tmp_path / "sparse-worker"
    with sparse.open("wb") as handle:
        handle.truncate(7)
    with pytest.raises(EVIDENCE.EvidenceError, match="too large"):
        EVIDENCE._read_stable_sha256(sparse, "executable", max_bytes=6)

    global_sparse = tmp_path / "global-sparse-worker"
    with global_sparse.open("wb") as handle:
        handle.truncate(EVIDENCE.MAX_PROCESS_BYTES + 1)
    with pytest.raises(EVIDENCE.EvidenceError, match="too large"):
        EVIDENCE._read_stable_sha256(global_sparse, "executable")

    link = tmp_path / "worker-link"
    link.symlink_to(executable)
    with pytest.raises(EVIDENCE.EvidenceError, match="unsafe"):
        EVIDENCE._read_stable_sha256(link, "executable", max_bytes=6)


def test_process_digest_rejects_path_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "worker"
    executable.write_bytes(b"A" * 4096)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"B" * 4096)
    original_read = EVIDENCE.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if chunk and not replaced:
            os.replace(replacement, executable)
            replaced = True
        return chunk

    monkeypatch.setattr(EVIDENCE.os, "read", replacing_read)
    with pytest.raises(EVIDENCE.EvidenceError, match="changed"):
        EVIDENCE._read_stable_sha256(executable, "executable", max_bytes=8192)


def test_launchctl_closed_schema_uses_header_and_rejects_unknown_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EVIDENCE.sys, "platform", "darwin")
    label = EVIDENCE._SERVICE_LABELS["dashboard"]
    domain = f"gui/{os.getuid()}"
    valid = (
        f"{domain}/{label} = {{\n"
        "program = /usr/bin/worker\n"
        "arguments = (\n"
        "    /usr/bin/worker\n"
        "    --mode\n"
        "    foreground\n"
        ")\n"
        "default environment = {\n"
        "    PATH => /usr/bin:/bin\n"
        "}\n"
        "active count = 1\n"
        "path = /one\n"
        "type = Submitted\n"
        "state = running\n"
        "pid = 42\n"
        "resource coalition = {\n"
        "    ID = 1\n"
        "    type = resource\n"
        "    state = active\n"
        "}\n"
        "}\n"
    )

    class Result:
        returncode = 0
        stdout = valid.encode()

    monkeypatch.setattr(EVIDENCE.subprocess, "run", lambda *_args, **_kwargs: Result())
    service = EVIDENCE._launchctl_probe("dashboard", 42)
    assert service["pid"] == 42
    assert service["child_pid"] is None
    assert service["parent_pid"] is None

    labelled = valid.replace(
        "pid = 42\n", f"label = {label}\npid = 42\n"
    )
    monkeypatch.setattr(
        EVIDENCE.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Result", (), {"returncode": 0, "stdout": labelled.encode()}
        )(),
    )
    assert EVIDENCE._launchctl_probe("dashboard", 42)["pid"] == 42

    optional = valid.replace(
        "pid = 42\n", "pid = 42\nparent pid = 7\nchild pid = 42\n"
    )
    monkeypatch.setattr(
        EVIDENCE.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Result", (), {"returncode": 0, "stdout": optional.encode()}
        )(),
    )
    optional_service = EVIDENCE._launchctl_probe("dashboard", 42)
    assert optional_service["parent_pid"] == 7
    assert optional_service["child_pid"] == 42

    invalid = {
        "duplicate known field": valid.replace(
            "path = /one\n", "path = /one\npath = /two\n"
        ),
        "duplicate singleton program": valid.replace(
            "program = /usr/bin/worker\n",
            "program = /usr/bin/worker\nprogram = /usr/bin/other\n",
        ),
        "wrong optional child": optional.replace(
            "child pid = 42\n", "child pid = 43\n"
        ),
        "duplicate optional parent": optional.replace(
            "parent pid = 7\n", "parent pid = 7\nparent pid = 8\n"
        ),
        "unknown assignment": valid.replace(
            "path = /one\n", "unknown = x\npath = /one\n"
        ),
        "unknown bare line": valid.replace(
            "path = /one\n", "evil\npath = /one\n"
        ),
        "unknown arrow": valid.replace(
            "path = /one\n", "evil => hacked\npath = /one\n"
        ),
        "unknown nested argument key": valid.replace(
            "arguments = (\n", "arguments = (\n    evil = x\n"
        ),
        "argv escape": valid.replace(
            "    --mode\n", "    bad\\\\ value\n"
        ),
        "argv arrow": valid.replace(
            "    --mode\n", "    evil => hacked\n"
        ),
        "wrong list close": valid.replace(
            "    --mode\n", "    --mode\n}\n"
        ),
        "extra list close": valid.replace(
            ")\ndefault environment", ")\n)\ndefault environment"
        ),
        "unbalanced list": valid.replace(
            "    foreground\n)\n", "    foreground\n"
        ),
        "program list misuse": valid.replace(
            "program = /usr/bin/worker\n", "program = (\n"
        ),
        "wrong label assignment": valid.replace(
            "pid = 42\n", "label = other\npid = 42\n"
        ),
        "duplicate label assignment": valid.replace(
            "pid = 42\n",
            f"label = {label}\nlabel = {label}\npid = 42\n",
        ),
    }
    for _name, payload in invalid.items():
        monkeypatch.setattr(
            EVIDENCE.subprocess,
            "run",
            lambda *_args, payload=payload, **_kwargs: type(
                "Result", (), {"returncode": 0, "stdout": payload.encode()}
            )(),
        )
        with pytest.raises(EVIDENCE.EvidenceError):
            EVIDENCE._launchctl_probe("dashboard", 42)


def test_launchctl_optional_parent_must_match_trusted_ps_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_at = datetime.now(UTC).isoformat()
    service = {
        "role": "dashboard",
        "domain": f"gui/{os.getuid()}",
        "label": EVIDENCE._SERVICE_LABELS["dashboard"],
        "state": "running",
        "pid": 42,
        "parent_pid": 7,
        "child_pid": None,
        "captured_at": captured_at,
        "raw_output_sha256": "0" * 64,
    }
    monkeypatch.setattr(EVIDENCE, "_launchctl_probe", lambda *_args, **_kwargs: dict(service))
    monkeypatch.setattr(EVIDENCE, "_ps_process_lineage", lambda _pid: (42, 8))
    monkeypatch.setattr(
        EVIDENCE,
        "_darwin_process_probe",
        lambda pid: (Path("/usr/bin/worker"), pid, 1, 2),
    )
    monkeypatch.setattr(
        EVIDENCE,
        "_process_identity",
        lambda *_args, **_kwargs: {
            "pid": 42,
            "started_at": "1.000002",
            "executable_path": "/usr/bin/worker",
            "executable_lstat": {},
            "executable_sha256": "0" * 64,
            "service": dict(service),
        },
    )
    with pytest.raises(EVIDENCE.EvidenceError, match="parent PID mismatch"):
        EVIDENCE._service_process_identity("dashboard")


def test_ps_process_lineage_is_single_pid_ppid_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EVIDENCE.sys, "platform", "darwin")

    class Result:
        returncode = 0
        stdout = b"42 7\n"

    monkeypatch.setattr(EVIDENCE.subprocess, "run", lambda *_args, **_kwargs: Result())
    assert EVIDENCE._ps_process_lineage(42) == (42, 7)

    for payload in (b"43 7\n", b"42 0\n", b"42 7\n44 8\n"):
        class InvalidResult:
            returncode = 0
            stdout = payload

        invalid_result = InvalidResult()
        monkeypatch.setattr(
            EVIDENCE.subprocess,
            "run",
            lambda *_args, result=invalid_result, **_kwargs: result,
        )
        with pytest.raises(EVIDENCE.EvidenceError):
            EVIDENCE._ps_process_lineage(42)


def test_orphan_poll_and_false_validation_exit_hold(tmp_path: Path) -> None:
    now = datetime(2026, 8, 24, tzinfo=UTC)
    _poll(tmp_path, "shadow", now, _id(1))
    assert EVIDENCE.validate_collector(tmp_path)["certification_reason"] == "collector_orphan_poll"
    assert EVIDENCE.main(["validate", "--evidence-root", str(tmp_path)]) == 1


def test_readonly_snapshot_does_not_create_missing_runtime_lock(tmp_path: Path) -> None:
    ledger = tmp_path / "shadow-observation-receipts.jsonl"
    lock = ledger.with_suffix(ledger.suffix + ".lock")
    with pytest.raises(EVIDENCE.EvidenceError, match="lock"):
        EVIDENCE._readonly_chain_snapshot(ledger)
    assert not lock.exists()


def test_missing_poll_timestamp_holds_every_stage(tmp_path: Path) -> None:
    now = datetime(2026, 8, 24, tzinfo=UTC)
    poll_id = _poll(tmp_path, "shadow", now, _id(1))
    _ledger(tmp_path, [(poll_id, "shadow", now, 1)])
    path = tmp_path / "polls" / f"{poll_id}.json"
    poll = json.loads(path.read_text())
    poll.pop("captured_at")
    poll["seal_sha256"] = EVIDENCE._digest(
        {key: value for key, value in poll.items() if key != "seal_sha256"}
    )
    path.write_text(json.dumps(poll, sort_keys=True, separators=(",", ":")))
    result = EVIDENCE.validate_collector(tmp_path)
    assert result["certification"] is False
    assert result["certification_reason"] == "collector_bundle_invalid"
    assert all(stage["certified"] is False for stage in result["stages"].values())


@pytest.mark.parametrize(
    "mutation", ["host", "observations", "producer", "run_id", "stage"]
)
def test_malformed_sealed_poll_never_escapes_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / mutation
    root.mkdir()
    now = datetime(2026, 8, 24, tzinfo=UTC)
    poll_id = _poll(root, "shadow", now, _id(1))
    _ledger(root, [(poll_id, "shadow", now, 1)])
    path = root / "polls" / f"{poll_id}.json"
    poll = json.loads(path.read_text())
    if mutation == "host":
        poll["observations"][0].pop("host")
    elif mutation == "observations":
        poll["observations"] = None
    elif mutation == "producer":
        poll.pop("producer")
    elif mutation == "run_id":
        poll["run_id"] = []
    else:
        poll["stage"] = "not-a-stage"
    poll["seal_sha256"] = EVIDENCE._digest(
        {key: value for key, value in poll.items() if key != "seal_sha256"}
    )
    path.write_text(json.dumps(poll, sort_keys=True, separators=(",", ":")))
    result = EVIDENCE.validate_collector(root)
    assert result["certification"] is False
    assert all(stage["certified"] is False for stage in result["stages"].values())


def test_collector_rejects_nonproduction_root_before_any_runtime_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(EVIDENCE.EvidenceError, match="not the production"):
        EVIDENCE.collect_poll(
            root=tmp_path,
            source_root=tmp_path,
            evidence_root=tmp_path / "evidence",
            stage="shadow",
            run_id=_id(1),
            dashboard_url="http://127.0.0.1:1",
            dom_capture_path=tmp_path / "dom.json",
            direct_url_path=tmp_path / "direct-url.json",
            executable=tmp_path / "worker",
            pid=1,
        )


def test_forged_full_window_and_resealed_flags_never_certify(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    entries: list[tuple[str, str, datetime, int]] = []
    number = 1
    for stage_index, stage in enumerate(EVIDENCE.STAGES):
        stage_start = start + timedelta(days=stage_index * 8)
        for _poll_index, when in enumerate(
            (stage_start, stage_start + timedelta(days=7))
        ):
            rows = [
                {
                    "observation_id": _id(number + offset),
                    "host": "host-a",
                    "cohort": "cohort-a",
                    "decision_sha256": _id(30_000 + number + offset),
                    "session_sha256": _id(40_000 + number + offset),
                    "query_sha256": _id(50_000 + number + offset),
                    "candidate_pool_sha256": _id(60_000 + number + offset),
                    "feature_bytes_sha256": _id(70_000),
                }
                for offset in range(250)
            ]
            poll_id, _, _ = store.write_immutable(
                tmp_path / "polls",
                {
                    "kind": "r7-live-poll",
                    "stage": stage,
                    "run_id": _id(10_000 + stage_index),
                    "captured_at": when.isoformat(),
                    "monotonic_ns": number,
                    "identities": {},
                    "source": {},
                    "runtime": {},
                    "process": {},
                    "health": {},
                    "api": {},
                    "dom_sha256": _id(20_000),
                    "observation_chain": {"records": 2_000, "head_sha256": _id(20_002)},
                    "observations_sha256": _id(20_001),
                    "observations": rows,
                    "producer": {
                        "name": "forged",
                        "version": 99,
                        "synthetic_fixture": False,
                    },
                },
                schema=EVIDENCE.POLL_SCHEMA,
            )
            entries.append((poll_id, stage, when, number))
            number += 250
    _ledger(tmp_path, entries)
    result = EVIDENCE.validate_collector(tmp_path)
    assert result["certification"] is False
    assert (
        result["certification_reason"] == "collector_poll_provenance_invalid"
    )
    assert (
        EVIDENCE.validate_collector(tmp_path, root=tmp_path)["certification"] is False
    )
    ledger_path = tmp_path / "poll-ledger.jsonl"
    ledger_path.write_text(ledger_path.read_text().splitlines()[0] + "\n")
    assert (
        EVIDENCE.validate_collector(tmp_path)["certification_reason"]
        == "collector_ledger_invalid"
    )


def test_rollback_never_accepts_forged_post_state_without_r7_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EVIDENCE.store, "CHRONOVISOR_ROOT", tmp_path)
    with pytest.raises(EVIDENCE.EvidenceError, match="production runtime"):
        EVIDENCE.validate_rollback(tmp_path, tmp_path / "forged-receipt.json")


@pytest.mark.darwin_contract
def test_test_rollback_receipt_rechecks_post_state_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    root, source, evidence = tmp_path / "runtime", tmp_path / "source", tmp_path / "evidence"
    root.mkdir()
    source.mkdir()
    subprocess = EVIDENCE.subprocess
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    (source / "tracked.txt").write_text("source", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "source"],
        cwd=source,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    policy = {
        "kind": "tiny-logistic-policy",
        "feature_revision": distill.TEXT_FEATURE_REVISION,
        "feature_keys": list(distill.FAST_FEATURE_KEYS),
        "weights": {key: 0.0 for key in distill.FAST_FEATURE_KEYS},
        "bias": 0.0,
        "threshold": 0.5,
        "abstain_margin": 0.0,
        "max_cards": 1,
    }
    lkg, _, _ = store.write_immutable(
        store.distillation_dir(root) / "policies", policy, schema=rollout.POLICY_SCHEMA
    )
    candidate, _, _ = store.write_immutable(
        store.distillation_dir(root) / "policies", {**policy, "bias": 0.1}, schema=rollout.POLICY_SCHEMA
    )
    baseline, _, _ = store.write_immutable(
        store.distillation_dir(root) / "baselines", {"kind": "baseline"}, schema=distill.BASELINE_SCHEMA
    )
    store.write_pointer(root, "lkg", lkg)
    store.write_pointer(root, "active", candidate)
    store.write_pointer(root, "candidate", candidate)
    run_id = _id(77)
    store.write_sealed_state(
        store.distillation_dir(root) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "canary",
            "rollout_percent": 100,
            "baseline_artifact_id": baseline,
            "stage_started_at": "2026-08-01T00:00:00Z",
            "stage_run_id": run_id,
            "candidate_policy_id": candidate,
            "lkg_policy_id": lkg,
        },
    )
    direct = tmp_path / "direct_url.json"
    direct.write_text(json.dumps({"vcs_info": {"commit_id": commit}}), encoding="utf-8")
    identities = EVIDENCE._stage_state(root, "100")
    poll_id, _, _ = store.write_immutable(
        evidence / "polls",
        {
            "kind": "r7-live-poll",
            "stage": "100",
            "run_id": run_id,
            "identities": identities,
        },
        schema=EVIDENCE.POLL_SCHEMA,
    )
    recorded = EVIDENCE.record_forced_rollback(
        root=root,
        evidence_root=evidence,
        source_root=source,
        direct_url_path=direct,
        executable=Path(sys.executable).resolve(),
        pid=os.getpid(),
        stage="100",
        run_id=run_id,
        poll_id=poll_id,
        failure_token="deterministic-test-failure",
    )
    receipt_path = evidence / "rollbacks" / f"{recorded['artifact_id']}.json"
    assert EVIDENCE.validate_rollback(root, receipt_path, allow_test_root=True) == recorded
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["post"]["state"]["active_policy_id"] = candidate
    receipt["seal_sha256"] = EVIDENCE._digest(
        {key: value for key, value in receipt.items() if key != "seal_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(EVIDENCE.EvidenceError, match="authoritative R7 binding"):
        EVIDENCE.validate_rollback(root, receipt_path, allow_test_root=True)
