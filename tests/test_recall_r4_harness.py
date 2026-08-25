from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r4_harness_test_module", ROOT / "scripts" / "recall_r4_harness.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)

from chronovisor.recall import recall_distillation as distill  # noqa: E402
from chronovisor.recall import recall_distillation_store as distill_store  # noqa: E402
from chronovisor.recall.recall_distillation_workset import (  # noqa: E402
    DistillationWorkset,
)


def _git_source(tmp_path: Path) -> tuple[Path, str]:
    # Keep positive fixtures outside macOS's /var -> /private symlink.
    tmp_path = tmp_path.resolve()
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("source\n")
    module = source / "src" / "chronovisor" / "recall"
    module.mkdir(parents=True)
    (module / "recall_distillation_remote_teacher.py").write_text(
        "OX_ALPHA_FIXED_IDENTITY = "
        + repr(
            {
                "revision": HARNESS.OX_IDENTITY_REVISION,
                "model_digest": HARNESS.OX_MODEL_SHA256,
                "route_digest": HARNESS.OX_ROUTE_SHA256,
                "prompt_template_sha256": HARNESS.OX_PROMPT_SHA256,
                "schema_revision_sha256": HARNESS.OX_SCHEMA_SHA256,
                "route_identity": {
                    "provider": "opencode-go",
                    "model": HARNESS.OX_ROUTE,
                    "location": "remote",
                },
            }
        )
        + "\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "r4@example.invalid"], cwd=source, check=True
    )
    subprocess.run(["git", "config", "user.name", "r4"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "add",
            "README.md",
            "src/chronovisor/recall/recall_distillation_remote_teacher.py",
        ],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, commit


def _receipt(payload: dict[str, object], index: int) -> dict[str, object]:
    unsigned = {
        "schema": HARNESS.RECEIPT_SCHEMA,
        "namespace": "recall-distillation",
        "receipt_id": HARNESS._sha256({"index": index, "payload": payload}),
        **payload,
    }
    return cast(dict[str, object], HARNESS._sealed(unsigned))


def _write_receipts(path: Path, receipts: list[dict[str, object]]) -> None:
    path.mkdir()
    # Keep each immutable receipt file below the harness's bounded input size;
    # production-sized local cohorts are intentionally represented by a set of
    # files rather than one unbounded JSON array.
    chunk_size = 100
    for index in range(0, len(receipts), chunk_size):
        chunk = receipts[index : index + chunk_size]
        (path / f"receipts-{index // chunk_size:03d}.json").write_text(
            json.dumps(chunk, sort_keys=True)
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("stage", "bogus"),
        ("payload_ref", "tampered-payload"),
        ("lease_owner", "forged-owner"),
    ],
)
def test_production_workset_rejects_row_field_tampering(
    tmp_path: Path, column: str, value: str
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    workset_path = production / HARNESS.PRODUCTION_WORKSET_RELATIVE
    with sqlite3.connect(workset_path) as connection:
        connection.execute(f"UPDATE work_items SET {column} = ? WHERE sequence = 1", (value,))
    with pytest.raises(HARNESS.R4Error):
        HARNESS._production_workset(workset_path)


def test_production_workset_rejects_watermark_tampering(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    workset_path = production / HARNESS.PRODUCTION_WORKSET_RELATIVE
    with sqlite3.connect(workset_path) as connection:
        connection.execute(
            "UPDATE workset_state SET value_json = ? WHERE key = 'watermark'",
            ('{"forged":true}',),
        )
    with pytest.raises(HARNESS.R4Error):
        HARNESS._production_workset(workset_path)


def test_production_workset_read_does_not_touch_checkpointed_shm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    workset_path = production / HARNESS.PRODUCTION_WORKSET_RELATIVE
    wal_path = workset_path.with_name(f"{workset_path.name}-wal")
    shm_path = workset_path.with_name(f"{workset_path.name}-shm")
    with sqlite3.connect(workset_path) as writer:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert wal_path.stat().st_size == 0
        assert shm_path.is_file()
        wal_bytes = wal_path.read_bytes()
        shm_bytes = shm_path.read_bytes()
    wal_path.write_bytes(wal_bytes)
    shm_path.write_bytes(shm_bytes)
    before = HARNESS._production_sqlite_state(
        workset_path, label="production workset"
    )
    opened: list[str] = []
    real_connect = sqlite3.connect

    def tracked_connect(database: object, *args: object, **kwargs: object) -> object:
        opened.append(str(database))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(HARNESS.sqlite3, "connect", tracked_connect)

    result = HARNESS._production_workset(workset_path)

    assert result["rows"] > 0
    assert opened[0] == f"file:{workset_path}?immutable=1"
    assert (
        HARNESS._production_sqlite_state(
            workset_path, label="production workset"
        )
        == before
    )
    with real_connect(workset_path) as writer:
        writer.execute("CREATE TABLE uncheckpointed_probe(value INTEGER)")
        writer.commit()
        assert wal_path.stat().st_size > 0
        opened.clear()
        with pytest.raises(
            HARNESS.R4Error, match="production workset has uncheckpointed WAL"
        ):
            HARNESS._production_workset(workset_path)
        assert opened == []


def test_collector_rejects_payload_ref_absent_from_candidate_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(production)
    )
    workset_path = production / HARNESS.PRODUCTION_WORKSET_RELATIVE
    with sqlite3.connect(workset_path) as connection:
        payload_ref = str(
            connection.execute(
                "SELECT payload_ref FROM work_items ORDER BY sequence LIMIT 1"
            ).fetchone()[0]
        )
        rally_id = payload_ref.split(":")[1]
        connection.execute(
            "UPDATE work_items SET payload_ref = ? WHERE sequence = 1",
            (f"candidate-snapshot:{rally_id}:{'f' * 64}",),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )
    assert result["passed"] is False
    assert result["reasons"] == [
        "production workset payload does not resolve in candidate ledger"
    ]


def test_collector_rejects_workset_mutation_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _path, **_kwargs: _fixture_candidate_anchor(production),
    )
    original = HARNESS._production_fault_scenarios

    def mutate_after_snapshot(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        workset_path = production / HARNESS.PRODUCTION_WORKSET_RELATIVE
        with sqlite3.connect(workset_path) as connection:
            connection.execute("PRAGMA user_version = 1")
        return result

    monkeypatch.setattr(HARNESS, "_production_fault_scenarios", mutate_after_snapshot)
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )

    assert result["passed"] is False
    assert result["reasons"] == ["production workset changed during validation"]


def test_collector_rejects_supplied_source_identity_not_matching_clean_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    source_snapshot = HARNESS._assert_source(source, commit)
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source={**source_snapshot, "tree_sha256": "0" * 64},
        production_root=production,
    )
    assert result["passed"] is False
    assert result["reasons"] == ["source_identity_mismatch"]


def test_production_candidate_ledger_rejects_garbage_chain(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    ledger = production / HARNESS.PRODUCTION_CANDIDATE_RELATIVE
    ledger.write_bytes(b"NOT-A-CHAIN\n")
    with pytest.raises(HARNESS.R4Error):
        HARNESS._production_ledger_checkpoint(
            ledger,
            production / HARNESS.PRODUCTION_CANDIDATE_CHECKPOINT_RELATIVE,
            ledger_name="candidate-ledger.jsonl",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ledger_name", "wrong-ledger.jsonl"),
        ("event_key", "0" * 64),
        ("event_binding_sha256", "1" * 64),
        ("record_sha256", "2" * 64),
        ("kind", "wrong-kind"),
    ],
)
def test_production_event_anchor_rejects_field_tampering(
    tmp_path: Path, field: str, value: str
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    distillation = production / HARNESS.PRODUCTION_DISTILLATION_RELATIVE
    state = json.loads((distillation / "state.json").read_text())
    anchor = next((distillation / "ox-event-anchors").glob("*.json"))
    payload = json.loads(anchor.read_text())
    payload[field] = value
    anchor.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(HARNESS.R4Error):
        HARNESS._production_ox_events(
            production,
            source=HARNESS._assert_source(source, commit),
            contract_id=str(state["profile_contract_id"]),
        )


def test_cap10_anchor_rejects_later_cohort_substitution(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    distillation = production / HARNESS.PRODUCTION_DISTILLATION_RELATIVE
    state = json.loads((distillation / "state.json").read_text())
    ramp_rows = [
        json.loads(line)
        for line in (distillation / "ox-ramp-receipts.jsonl").read_text().splitlines()
    ]
    cap10 = next(row for row in ramp_rows if row.get("cap") == 10)
    anchor = next(
        path
        for path in (distillation / "ox-event-anchors").glob("*.json")
        if json.loads(path.read_text()).get("record_sha256") == cap10["record_sha256"]
    )
    payload = json.loads(anchor.read_text())
    payload["record_sha256"] = ramp_rows[0]["record_sha256"]
    anchor.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(HARNESS.R4Error):
        HARNESS._production_ox_events(
            production,
            source=HARNESS._assert_source(source, commit),
            contract_id=str(state["profile_contract_id"]),
        )


def _local_rows(source: Path, commit: str) -> list[dict[str, object]]:
    tree = HARNESS._source_tree_digest(source)["tree_sha256"]
    rows: list[dict[str, object]] = []
    reasons = [
        "schema",
        "coverage",
        "capacity",
        "timeout",
        "preemption",
        "route_model_mismatch",
        "ok",
    ]
    for index, reason in enumerate(reasons):
        rally = f"rally-{index}"
        candidate = f"candidate-{index}"
        owner = HARNESS._expected_owner(rally, candidate)
        outcome_class = (
            "valid"
            if reason == "ok"
            else "deferred"
            if reason in HARNESS._DEFERRED_REASONS
            else "invalid"
        )
        outcome: dict[str, object] = {"class": outcome_class, "reason": reason}
        if outcome_class == "valid":
            outcome.update(schema_valid=True, coverage_valid=True)
        rows.append(
            {
                "profile": HARNESS.LOCAL_PROFILE,
                "source_commit": commit,
                "source_tree_sha256": tree,
                "rally_id": rally,
                "candidate_id": candidate,
                "primary_owner": owner,
                "probe": HARNESS._expected_probe(rally, candidate),
                "assignment_revision": HARNESS.LOCAL_ASSIGNMENT_REVISION,
                "probe_assignment_revision": HARNESS.LOCAL_PROBE_REVISION,
                "route_identity": {
                    "role": owner,
                    "provider": "local",
                    "model": {
                        role: f"model-{role[-1]}" for role in HARNESS.LOCAL_ROLES
                    }[owner],
                    "location": "local",
                },
                "lease": {
                    "kind": "LocalStructuredSession",
                    "foreground": True,
                    "inflight": 1,
                },
                "live_recall": {"unaffected": True, "remote_egress": 0},
                "max_inflight": 1,
                "failure_injection": True,
                "outcome": outcome,
            }
        )
    next_index = len(rows)
    while {row["primary_owner"] for row in rows} != set(HARNESS.LOCAL_ROLES):
        rally = f"rally-{next_index}"
        candidate = f"candidate-{next_index}"
        owner = HARNESS._expected_owner(rally, candidate)
        if owner in {row["primary_owner"] for row in rows}:
            next_index += 1
            continue
        rows.append(
            {
                "profile": HARNESS.LOCAL_PROFILE,
                "source_commit": commit,
                "source_tree_sha256": tree,
                "rally_id": rally,
                "candidate_id": candidate,
                "primary_owner": owner,
                "probe": HARNESS._expected_probe(rally, candidate),
                "assignment_revision": HARNESS.LOCAL_ASSIGNMENT_REVISION,
                "probe_assignment_revision": HARNESS.LOCAL_PROBE_REVISION,
                "route_identity": {
                    "role": owner,
                    "provider": "local",
                    "model": f"model-{owner[-1]}",
                    "location": "local",
                },
                "lease": {"kind": "LLMRuntime", "foreground": True, "inflight": 1},
                "live_recall": {"unaffected": True, "remote_egress": 0},
                "max_inflight": 1,
                "failure_injection": True,
                "outcome": {
                    "class": "valid",
                    "reason": "ok",
                    "schema_valid": True,
                    "coverage_valid": True,
                },
            }
        )
        next_index += 1
    for index in range(3000):
        rally = f"quality-rally-{index % 17}"
        candidate = f"quality-candidate-{index}"
        owner = HARNESS._expected_owner(rally, candidate)
        rows.append(
            {
                "profile": HARNESS.LOCAL_PROFILE,
                "source_commit": commit,
                "source_tree_sha256": tree,
                "rally_id": rally,
                "candidate_id": candidate,
                "primary_owner": owner,
                "probe": HARNESS._expected_probe(rally, candidate),
                "assignment_revision": HARNESS.LOCAL_ASSIGNMENT_REVISION,
                "probe_assignment_revision": HARNESS.LOCAL_PROBE_REVISION,
                "route_identity": {
                    "role": owner,
                    "provider": "local",
                    "model": f"model-{owner[-1]}",
                    "location": "local",
                },
                "lease": {"kind": "LLMRuntime", "foreground": True, "inflight": 1},
                "live_recall": {"unaffected": True, "remote_egress": 0},
                "max_inflight": 1,
                "outcome": {
                    "class": "valid",
                    "reason": "ok",
                    "schema_valid": True,
                    "coverage_valid": True,
                },
            }
        )
    return rows


def _ox_rows(source: Path, commit: str) -> list[dict[str, object]]:
    snapshot = HARNESS._source_tree_digest(source)
    tree = snapshot["tree_sha256"]
    source_identity = snapshot["ox_identity_sha256"]
    contract = {
        "route": HARNESS.OX_ROUTE,
        "model": HARNESS.OX_MODEL,
        "request_model": HARNESS.OX_MODEL,
        "required_returned_model": HARNESS.OX_MODEL,
        "request_revision": HARNESS.OX_REQUEST_REVISION,
        "prompt_sha256": HARNESS.OX_PROMPT_SHA256,
        "schema": HARNESS.OX_SCHEMA,
        "schema_sha256": HARNESS.OX_SCHEMA_SHA256,
        "route_sha256": HARNESS.OX_ROUTE_SHA256,
        "model_sha256": HARNESS.OX_MODEL_SHA256,
        "cohort": HARNESS.OX_COHORT,
        "identity_revision": HARNESS.OX_IDENTITY_REVISION,
        "fixed_identity": HARNESS.OX_FIXED_IDENTITY,
        "free_only": True,
        "no_paid_fallback": True,
        "kill_categories": list(HARNESS.OX_KILL_CATEGORIES),
        "live_recall_model_calls": 0,
        "expires_at": "2099-01-01T00:00:00Z",
        "source_commit": commit,
        "source_tree_sha256": tree,
        "source_ox_identity_sha256": source_identity,
    }
    contract_identity = {
        key: contract[key]
        for key in (
            "route",
            "model",
            "prompt_sha256",
            "schema",
            "schema_sha256",
            "route_sha256",
            "model_sha256",
            "cohort",
            "identity_revision",
            "request_revision",
            "request_model",
            "required_returned_model",
            "fixed_identity",
            "free_only",
            "no_paid_fallback",
            "kill_categories",
            "live_recall_model_calls",
            "expires_at",
            "source_commit",
            "source_tree_sha256",
            "source_ox_identity_sha256",
        )
    }
    contract["contract_id"] = HARNESS._sha256(contract_identity)
    rows: list[dict[str, object]] = []
    for cap in HARNESS.OX_STAGES:
        labels: list[dict[str, object]] = []
        for index in range(20):
            payload_source = {
                "candidate_id": f"candidate-{cap}-{index}",
                "rally_id": f"rally-{cap}-{index}",
            }
            payload_digest = HARNESS._sha256(payload_source)
            work_id = HARNESS._sha256(
                {
                    "kind": "ox-teacher-label-v1",
                    "profile": HARNESS.OX_PROFILE,
                    "cohort": HARNESS.OX_COHORT,
                    "route": HARNESS.OX_ROUTE,
                    "profile_contract_id": contract["contract_id"],
                    "payload_digest": payload_digest,
                }
            )
            labels.append(
                {
                    "label_id": HARNESS._sha256(f"label-{cap}-{index}"),
                    "commit_id": HARNESS._sha256(f"commit-{cap}-{index}"),
                    "work_id": work_id,
                    "profile_contract_id": contract["contract_id"],
                    "source_commit": commit,
                    "source_tree_sha256": tree,
                    "source_ox_identity_sha256": source_identity,
                    "payload_source": payload_source,
                    "payload_digest": payload_digest,
                    "request_sha256": HARNESS._expected_ox_request_sha256(
                        profile_contract_id=str(contract["contract_id"]),
                        payload_digest=payload_digest,
                    ),
                    "provider_request_sha256": HARNESS._expected_ox_provider_request_sha256(
                        profile_contract_id=str(contract["contract_id"]),
                        payload_digest=payload_digest,
                        work_id=work_id,
                        expires_at=str(contract["expires_at"]),
                    ),
                    "provider_response_request_sha256": HARNESS._expected_ox_provider_request_sha256(
                        profile_contract_id=str(contract["contract_id"]),
                        payload_digest=payload_digest,
                        work_id=work_id,
                        expires_at=str(contract["expires_at"]),
                    ),
                    "request_revision": HARNESS.OX_REQUEST_REVISION,
                    "expires_at": contract["expires_at"],
                }
            )
        rows.append(
            {
                "profile": HARNESS.OX_PROFILE,
                "captured_at": f"2026-08-24T00:00:{cap:02d}Z",
                "source_commit": commit,
                "source_tree_sha256": tree,
                "contract": contract,
                "negative_veto": {
                    "authenticated": True,
                    "exact_binding": True,
                    "conflicts": 0,
                },
                "blind_repeat": {
                    "revision": HARNESS.OX_PROBE_REVISION,
                    "complete": True,
                    "stability_passed": True,
                    "pairs": HARNESS.OX_MIN_BLIND_REPEAT_PAIRS,
                },
                "order_swap": {
                    "complete": True,
                    "pairs": HARNESS.OX_MIN_BLIND_REPEAT_PAIRS,
                },
                "rollback": {
                    "verified": True,
                    "active_unchanged": True,
                    "status": "not_rolled_back",
                },
                "control": {
                    "ox_enabled": True,
                    "free_only": True,
                    "no_paid_fallback": True,
                    "kill_switch_supported": True,
                    "kill_switch_tripped": False,
                },
                "stage": {
                    "cap": cap,
                    "valid_receipts": 20,
                    "attempts": 20,
                    "labels": labels,
                },
                "transition_receipts": [
                    {
                        "category": "429",
                        "before_cap": 10,
                        "after_cap": 5,
                        "status": "deferred",
                    },
                    {
                        "category": "5xx",
                        "attempts": 3,
                        "bounded": True,
                        "status": "deferred",
                    },
                    {
                        "category": "timeout",
                        "attempts": 3,
                        "bounded": True,
                        "status": "deferred",
                    },
                    {"category": "402", "status": "hard_stop"},
                    {"category": "paid", "status": "hard_stop"},
                    {"category": "model_drift", "status": "hard_stop"},
                ],
                "sensitive": 0,
                "raw": 0,
                "billable": 0,
                "unexpected_route": 0,
                "duplicate_label": 0,
                "duplicate_commit": 0,
                "lease_recovery": {"leased_after": 0},
            }
        )
    return rows


def _authoritative_production_root(tmp_path: Path, source: Path, commit: str) -> Path:
    """Seed a fixture through the real distillation writer pipeline.

    Domain records are synthetic, while config, Raw, candidate/label ledgers,
    Workset SQLite, OX events, checkpoints, and state are all published by the
    production APIs.  The only test seam replaces the R0-specific runtime
    identity projection for this tiny, independently anchored clone.
    """

    from dataclasses import replace
    from datetime import timedelta

    from chronovisor.core.llm_runtime import GenerationResult, RouteLocation
    from chronovisor.core.raw_segment import append_capture
    from chronovisor.core.store import RuntimeContext, init_chronovisor
    from chronovisor.recall import recall_distillation_remote_teacher as remote
    from chronovisor.recall.recall_distillation_remote_teacher import (
        OX_ALPHA_ENDPOINT,
        OpenCodeOxAlphaTeacher,
    )

    root = (tmp_path / "production").resolve()
    root.mkdir(parents=True)
    init_chronovisor(RuntimeContext(root))

    events: list[dict[str, object]] = []
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(240):
        stamp = (origin + timedelta(days=index)).isoformat().replace("+00:00", "Z")
        role = "user" if index % 2 == 0 else "assistant"
        content_type = "input_text" if role == "user" else "output_text"
        events.append(
            {
                "type": "response_item",
                "timestamp": stamp,
                "payload": {
                    "type": "message",
                    "role": role,
                    "content": [
                        {
                            "type": content_type,
                            "text": f"r4 fixture message {index} with durable evidence",
                        }
                    ],
                },
            }
        )
    source_path = root / "r4-fixture.jsonl"
    source_bytes = b"".join(
        json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n"
        for event in events
    )
    source_path.write_bytes(source_bytes)
    append_capture(
        raw_dir=root / "raw",
        raw_id="save-codex-test.md",
        idempotency_key="codex-test",
        host="codex",
        session_key="a" * 24,
        session_id="r4-fixture",
        source_file=source_path,
        after_line=0,
        until_line=len(events),
        source_bytes=source_bytes,
        record_count=len(events),
        now=datetime(2026, 8, 25, tzinfo=UTC),
    )

    config_path = root / "config.toml"
    config_path.write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "chunk_size = 200\n"
        "max_input_bytes = 4096\n"
        "max_candidates = 200\n"
        'teacher_profile = "ox-alpha-single-v1"\n'
        "teacher_max_inflight = 10\n"
        "teacher_claim_limit = 1\n"
        "ox_enabled = true\n"
        "ox_free_only = true\n"
        'ox_expires_at = "2099-01-01T00:00:00Z"\n'
    )

    source_snapshot = HARNESS._source_tree_digest(source)
    source_binding = {
        "source_commit": commit,
        "source_tree_sha256": source_snapshot["tree_sha256"],
        "source_ox_identity_sha256": source_snapshot["ox_identity_sha256"],
    }
    distill_dir = distill_store.distillation_dir(root)
    original_remote_binding = remote.ox_alpha_source_binding
    original_distill_binding = distill.ox_alpha_source_binding
    original_loader = distill.load_distillation_config
    original_prepare = distill._ox_prepare_tasks
    original_batches = distill._ox_prepare_batches
    original_identity = distill._r4_runtime_identity_projection

    class FixtureBackend:
        provider = "opencode-go"
        location = RouteLocation.REMOTE
        _profile = SimpleNamespace(endpoint=OX_ALPHA_ENDPOINT)

        def capabilities_for(self, _model: str) -> SimpleNamespace:
            return SimpleNamespace(structured_output=True)

        def generate(self, request: object, *, model: str) -> GenerationResult:
            request_format = getattr(request, "format", None)
            if not isinstance(request_format, Mapping):
                raise AssertionError("fixture request schema is unavailable")
            properties = request_format.get("properties")
            labels_schema = (
                properties.get("labels") if isinstance(properties, Mapping) else None
            )
            item_schema = (
                labels_schema.get("items")
                if isinstance(labels_schema, Mapping)
                else None
            )
            item_properties = (
                item_schema.get("properties")
                if isinstance(item_schema, Mapping)
                else None
            )
            candidate_ids = (
                item_properties.get("candidate_id", {}).get("enum")
                if isinstance(item_properties, Mapping)
                and isinstance(item_properties.get("candidate_id"), Mapping)
                else None
            )
            if not isinstance(candidate_ids, list) or not all(
                isinstance(value, str) for value in candidate_ids
            ):
                raise AssertionError("fixture request candidate IDs are unavailable")
            labels = [
                {
                    "candidate_id": candidate_id,
                    "verdict": "relevant",
                    "confidence": 0.9,
                    "rationale": "direct_match",
                }
                for candidate_id in candidate_ids
            ]
            return GenerationResult(
                content=json.dumps({"labels": labels}, separators=(",", ":")),
                provider="opencode-go",
                model=model,
                finish_reason="stop",
                metadata={
                    "returned_model": "ox-alpha-free",
                    "request_id": "r4-fixture",
                },
            )

    run_number = 0
    candidate_pool: list[tuple[Mapping[str, object], Mapping[str, object]]] = []

    def fixture_prepare_tasks(
        *,
        config: object,
        snapshots: Mapping[str, Mapping[str, object]],
        rally_by_id: Mapping[str, Mapping[str, object]],
        assignments: Mapping[str, object],
        split_plan_id: str,
        profile_contract_id: str,
        candidate_indexed: bool,
        candidate_state: Mapping[str, object],
        age_bands: Mapping[str, object] | None,
    ) -> dict[str, object]:
        del config, assignments, age_bands
        nonlocal run_number
        for rally_id, snapshot in snapshots.items():
            rally = rally_by_id.get(rally_id)
            candidates = snapshot.get("candidates")
            if isinstance(rally, Mapping) and isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(
                        candidate, Mapping
                    ) and not remote._contains_forbidden_text(
                        str(candidate.get("candidate_id") or "")
                    ):
                        candidate_pool.append((rally, candidate))
        run_number += 1
        if not candidate_pool:
            raise AssertionError("actual candidate writer produced no snapshots")
        start = (run_number - 1) * 20
        selected = [
            candidate_pool[(start + index) % len(candidate_pool)] for index in range(20)
        ]
        tasks: dict[str, dict[str, object]] = {}
        work_items: list[dict[str, object]] = []
        for index in range(20):
            source_rally, source_candidate = selected[index]
            candidate_id = str(source_candidate["candidate_id"])
            template_rally = {**source_rally, "context_refs": []}
            source_snapshot = snapshots.get(str(template_rally["rally_id"]), {})
            payload_source = {
                "rally_id": str(template_rally["rally_id"]),
                "candidate_id": candidate_id,
                "snapshot_sha256": str(source_snapshot.get("snapshot_sha256") or ""),
                "query_sha256": str(source_rally.get("query_sha256") or ""),
                "candidate_text_sha256": str(source_candidate.get("text_sha256") or ""),
                "context_sha256": [
                    ref.get("semantic_sha256", "")
                    for ref in source_rally.get("context_refs", [])
                    if isinstance(ref, Mapping)
                ],
            }
            payload_digest = HARNESS._sha256(payload_source)
            work_id = HARNESS._sha256(
                {
                    "kind": "ox-teacher-label-v1",
                    "profile": distill.OX_SINGLE_PROFILE,
                    "cohort": distill.OX_SINGLE_COHORT,
                    "route": HARNESS.OX_ROUTE,
                    "profile_contract_id": profile_contract_id,
                    "payload_digest": payload_digest,
                }
            )
            candidate = dict(source_candidate)
            rally_id = str(template_rally["rally_id"])
            pair_id = f"r4-pair-{run_number}-{index // 2}"
            assignment = {
                "revision": "single-teacher-v1",
                "owner": distill.OX_TEACHER_ROLE,
                "probe": False,
                "routes": [distill.OX_TEACHER_ROLE],
                "repeat_pair_id": pair_id,
                "fixed_repeat": True,
                "order_swap": True,
                "blind_order": "a_first" if index % 2 == 0 else "b_first",
            }
            temporal = {
                "as_of": template_rally["as_of"],
                "group_id": template_rally["session_cluster_id"],
                "split": "embargo",
                "split_plan_id": split_plan_id,
            }
            tasks[work_id] = {
                "rally": dict(template_rally),
                "candidate": candidate,
                "assignment": assignment,
                "temporal": temporal,
                "payload_source": payload_source,
            }
            work_items.append(
                {
                    "work_id": work_id,
                    "kind": "ox",
                    "payload_ref": f"candidate-snapshot:{rally_id}:{candidate_id}",
                    "payload_digest": payload_digest,
                    "priority": 0,
                    "temporal_split": temporal,
                    "provenance": {
                        "profile": distill.OX_SINGLE_PROFILE,
                        "cohort": distill.OX_SINGLE_COHORT,
                        "profile_contract_id": profile_contract_id,
                        "route": HARNESS.OX_ROUTE,
                        "teacher_role": distill.OX_TEACHER_ROLE,
                        "probe": False,
                        "repeat_pair_id": pair_id,
                        "fixed_repeat": True,
                        "order_swap": True,
                        "blind_order": assignment["blind_order"],
                    },
                }
            )

        def add_task(*_args: object, **_kwargs: object) -> None:
            return None

        candidate_records = candidate_state.get("record_count")
        if candidate_indexed:
            watermark: object = {
                "candidate_records": (
                    candidate_records
                    if isinstance(candidate_records, int)
                    and not isinstance(candidate_records, bool)
                    else 0
                ),
                "candidate_head": str(candidate_state.get("head_sha256") or ""),
                "split_plan_id": split_plan_id,
                "probe_revision": HARNESS.OX_PROBE_REVISION,
            }
        else:
            watermark = HARNESS._sha256({"work_ids": sorted(tasks)})
        return {
            "tasks": tasks,
            "work_items": work_items,
            "watermark": watermark,
            "add_task": add_task,
        }

    def fixture_batches(
        *,
        claims: Sequence[object],
        **_kwargs: object,
    ) -> tuple[list[list[object]], int]:
        return [[claim] for claim in claims], 0

    def fixture_config_loader(path: Path | None = None) -> object:
        loaded = original_loader(path)
        return replace(loaded, teacher_claim_limit=500, chunk_size=200)

    def fixture_runtime_identity(
        _root: Path,
        *,
        config_path: Path | None,
        source_binding: Mapping[str, str],
        profile_contract_id: str,
        candidate_path: Path,
        label_path: Path,
    ) -> dict[str, object]:
        del profile_contract_id
        config_file = config_path or root / "config.toml"
        candidate_checkpoint = distill_store.read_sealed(
            candidate_path.with_suffix(candidate_path.suffix + ".head.json")
        )
        label_checkpoint = distill_store.read_sealed(
            label_path.with_suffix(label_path.suffix + ".head.json")
        )
        workset_path = distill_dir / "ox-workset.sqlite3"
        receipt_head = DistillationWorkset(workset_path).audit_transition_receipts()
        anchor = {
            "candidate_anchor_artifact_id": "a" * 64,
            "candidate_anchor_head_sha256": candidate_checkpoint["head_sha256"],
            "candidate_anchor_records": candidate_checkpoint["records"],
            "candidate_anchor_bytes": candidate_checkpoint["file_state"]["size_bytes"],
        }
        return {
            "root": str(root.absolute()),
            "account_uid": HARNESS.ACCOUNT_UID,
            "account_home": str(HARNESS.ACCOUNT_HOME),
            **source_binding,
            "config_sha256": HARNESS._file_sha256(config_file),
            "workset_sha256": HARNESS._file_sha256(workset_path),
            "profile_contract_sha256": HARNESS._file_sha256(
                distill_dir
                / "ox-profile-contracts"
                / f"{str(distill_store.read_sealed(distill_dir / 'ox-profile-contract.json')['profile_contract_id'])}.json"
            ),
            "workset_receipt_head": receipt_head["head_sha256"],
            "candidate_checkpoint_head": candidate_checkpoint["head_sha256"],
            "candidate_checkpoint_records": candidate_checkpoint["records"],
            "candidate_checkpoint_file_state": candidate_checkpoint["file_state"],
            **anchor,
            "candidate_tail_records": 0,
            "candidate_tail_bytes": 0,
            "label_receipt_head": label_checkpoint["head_sha256"],
            "label_checkpoint_records": label_checkpoint["records"],
            "label_checkpoint_file_state": label_checkpoint["file_state"],
            "label_sha256": (
                HARNESS._file_sha256(label_path) if label_path.exists() else ""
            ),
            "os_identity": {
                "system": os.uname().sysname,
                "release": os.uname().release,
                "machine": os.uname().machine,
            },
        }

    try:
        remote.ox_alpha_source_binding = lambda: dict(source_binding)
        distill.ox_alpha_source_binding = lambda: dict(source_binding)
        distill.load_distillation_config = fixture_config_loader
        distill._ox_prepare_tasks = fixture_prepare_tasks
        distill._ox_prepare_batches = fixture_batches
        distill._r4_runtime_identity_projection = fixture_runtime_identity
        # Formal clone bootstrap: the production run must not create or alter
        # an OX workset implicitly.
        DistillationWorkset(distill_dir / "ox-workset.sqlite3")
        teacher = OpenCodeOxAlphaTeacher(
            FixtureBackend(), max_input_bytes=4096, test_only=True
        )
        for _ in range(4):
            distill.run_distillation_chunk(
                root=root,
                raw_dir=root / "raw",
                config_path=config_path,
                teachers={distill.OX_TEACHER_ROLE: teacher},
                max_elapsed_seconds=60,
            )
        # Failure categories are intentionally not injected into the sealed
        # event ledger.  They require separate provider-free dispatcher
        # scenarios (including hard-stop clones); this success fixture leaves
        # the production failure-coverage gate false until those scenarios are
        # independently collected.
    finally:
        remote.ox_alpha_source_binding = original_remote_binding
        distill.ox_alpha_source_binding = original_distill_binding
        distill.load_distillation_config = original_loader
        distill._ox_prepare_tasks = original_prepare
        distill._ox_prepare_batches = original_batches
        distill._r4_runtime_identity_projection = original_identity
    return root


def _fixture_candidate_anchor(production: Path) -> dict[str, object]:
    checkpoint_path = production / HARNESS.PRODUCTION_CANDIDATE_CHECKPOINT_RELATIVE
    checkpoint = json.loads(checkpoint_path.read_text())
    return {
        "artifact_id": "a" * 64,
        "candidate": {
            "head_sha256": checkpoint["head_sha256"],
            "records": checkpoint["records"],
            "bytes": checkpoint["file_state"]["size_bytes"],
            "file_state": dict(checkpoint["file_state"]),
        },
    }


def test_source_contract_passes_but_production_stays_false_without_attestation(
    tmp_path: Path,
) -> None:
    source, commit = _git_source(tmp_path)
    local_dir = tmp_path / "local"
    ox_dir = tmp_path / "ox"
    _write_receipts(
        local_dir,
        [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))],
    )
    _write_receipts(
        ox_dir,
        [
            _receipt(row, index + 100)
            for index, row in enumerate(_ox_rows(source, commit))
        ],
    )
    artifact, _ = HARNESS.run(
        source_root=source,
        source_commit=commit,
        output=tmp_path / "evidence",
        local_receipts=local_dir,
        ox_receipts=ox_dir,
    )
    assert artifact["source_contract"]["passed"] is True
    assert artifact["production_certification"]["passed"] is False
    assert (
        "independent_live_provider_attestation_unavailable"
        in artifact["production_certification"]["reasons"]
    )
    assert artifact["provider_calls"] == 0
    artifact_path = next((tmp_path / "evidence").glob("*.json"))
    assert (
        HARNESS.read_artifact(artifact_path)["artifact_id"] == artifact["artifact_id"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_revision", None),
        ("request_revision", "json-schema-core-label-abstain-16k-240s-v5"),
        ("fixed_identity", {}),
        ("kill_categories", []),
        ("live_recall_model_calls", 1),
        ("expires_at", None),
        ("expires_at", 9999999999.0),
        ("expires_at", True),
        ("expires_at", "2099-01-01T00:00:00"),
        ("expires_at", "2099-01-01T00:00:00.123Z"),
        ("expires_at", "2099-01-01T09:00:00+09:00"),
        ("expires_at", "2000-01-01T00:00:00Z"),
        ("expires_at", "9999-01-01T00:00:00Z"),
    ],
)
def test_source_ox_contract_rejects_identity_mutation(
    tmp_path: Path, field: str, value: object
) -> None:
    source, commit = _git_source(tmp_path)
    rows = _ox_rows(source, commit)
    for row in rows:
        contract = dict(cast(Mapping[str, object], row["contract"]))
        if value is None:
            contract.pop(field, None)
        else:
            contract[field] = value
        identity_keys = (
            "route",
            "model",
            "prompt_sha256",
            "schema",
            "schema_sha256",
            "route_sha256",
            "model_sha256",
            "cohort",
            "identity_revision",
            "request_revision",
            "request_model",
            "required_returned_model",
            "fixed_identity",
            "free_only",
            "no_paid_fallback",
            "kill_categories",
            "live_recall_model_calls",
            "expires_at",
            "source_commit",
            "source_tree_sha256",
            "source_ox_identity_sha256",
        )
        contract["contract_id"] = HARNESS._sha256(
            {key: contract.get(key) for key in identity_keys}
        )
        row["contract"] = contract

    result = HARNESS._validate_ox(rows, HARNESS._assert_source(source, commit))
    assert result["passed"] is False


def test_source_ox_contract_rejects_resealed_wrong_revision_and_mixed_labels(
    tmp_path: Path,
) -> None:
    source, commit = _git_source(tmp_path)
    rows = _ox_rows(source, commit)

    contract = dict(cast(Mapping[str, object], rows[1]["contract"]))
    contract["request_revision"] = "json-schema-core-label-abstain-16k-240s-v5"
    identity_keys = (
        "route",
        "model",
        "prompt_sha256",
        "schema",
        "schema_sha256",
        "route_sha256",
        "model_sha256",
        "cohort",
        "identity_revision",
        "request_revision",
        "request_model",
        "required_returned_model",
        "fixed_identity",
        "free_only",
        "no_paid_fallback",
        "kill_categories",
        "live_recall_model_calls",
        "expires_at",
        "source_commit",
        "source_tree_sha256",
        "source_ox_identity_sha256",
    )
    contract["contract_id"] = HARNESS._sha256(
        {key: contract.get(key) for key in identity_keys}
    )
    rows[1]["contract"] = contract
    rows[0]["stage"]["labels"][0]["request_revision"] = (
        "json-schema-core-label-abstain-16k-240s-v5"
    )

    result = HARNESS._validate_ox(rows, HARNESS._assert_source(source, commit))
    assert result["passed"] is False
    assert {
        "ox_contract_mixing",
        "ox_identity_mismatch",
        "ox_label_request_revision_invalid",
    } <= set(result["reasons"])


@pytest.mark.parametrize(
    "field",
    [
        "payload_digest",
        "request_sha256",
        "provider_request_sha256",
        "provider_response_request_sha256",
        "work_id",
        "profile_contract_id",
    ],
)
def test_source_ox_labels_reject_resealed_request_binding_mutation(
    tmp_path: Path, field: str
) -> None:
    source, commit = _git_source(tmp_path)
    rows = _ox_rows(source, commit)
    label = rows[0]["stage"]["labels"][0]  # type: ignore[index]
    assert isinstance(label, dict)
    label[field] = "f" * 64
    result = HARNESS._validate_ox(rows, HARNESS._assert_source(source, commit))
    assert result["passed"] is False
    assert "ox_label_binding_invalid" in result["reasons"]


def test_read_artifact_requires_canonical_closed_payload(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    artifact, artifact_path = HARNESS.run(
        source_root=source,
        source_commit=commit,
        output=tmp_path / "evidence",
    )
    original = artifact_path.read_bytes()
    artifact_path.write_bytes(original + b" ")
    with pytest.raises(HARNESS.R4Error, match="canonical"):
        HARNESS.read_artifact(artifact_path)
    artifact_path.write_bytes(original)

    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"artifact_id", "seal_sha256", "source_contract"}
    }
    forged_id = HARNESS._sha256(unsigned)
    forged = {"artifact_id": forged_id, **unsigned}
    forged["seal_sha256"] = HARNESS._sha256(forged)
    forged_path = artifact_path.with_name(f"{forged_id}.json")
    forged_path.write_bytes(HARNESS._json_bytes(forged) + b"\n")
    with pytest.raises(HARNESS.R4Error, match="payload shape|source contract"):
        HARNESS.read_artifact(forged_path)

    wrong_name = artifact_path.with_name("0" * 64 + ".json")
    wrong_name.write_bytes(original)
    with pytest.raises(HARNESS.R4Error, match="filename"):
        HARNESS.read_artifact(wrong_name)


def test_write_immutable_rejects_directory_swap_during_publish(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    displaced = tmp_path / "displaced"
    replacement = tmp_path / "replacement"
    replacement.mkdir()

    def replace_directory() -> None:
        os.rename(output, displaced)
        os.rename(replacement, output)

    with pytest.raises(HARNESS.R4Error, match="directory changed"):
        HARNESS._write_immutable(
            output,
            {"captured_at": "2026-08-25T00:00:00Z"},
            before_publish=replace_directory,
        )
    assert not list(output.iterdir())
    assert len(list(displaced.glob("*.json"))) == 1


def test_authority_reader_rejects_nested_symlink_and_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "nested" / "output"
    nested.mkdir(parents=True)
    receipt = nested / "receipt.json"
    receipt.write_text("{}\n")
    linked = tmp_path / "linked"
    linked.symlink_to(nested.parent, target_is_directory=True)
    with pytest.raises(HARNESS.R4Error, match="unsafe"):
        HARNESS._read_authority_regular(linked / "output" / receipt.name, label="test")

    original_states = HARNESS._authority_path_states
    displaced, replacement = tmp_path / "displaced", tmp_path / "replacement"
    replacement.mkdir()
    (replacement / receipt.name).write_text("{}\n")
    swapped = False

    def swap_after_snapshot(path: Path, *, label: str) -> list[dict[str, int]]:
        nonlocal swapped
        states = original_states(path, label=label)
        if not swapped:
            swapped = True
            os.rename(nested, displaced)
            os.rename(replacement, nested)
        return states

    monkeypatch.setattr(HARNESS, "_authority_path_states", swap_after_snapshot)
    with pytest.raises(HARNESS.R4Error, match="unsafe|changed"):
        HARNESS._read_authority_regular(receipt, label="test")


def test_source_bound_authority_receipt_uses_official_producer_and_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(production)
    )
    real_collector = HARNESS._collect_authoritative_production

    def certifying_collector(**kwargs: object) -> dict[str, object]:
        observed = real_collector(**kwargs)
        assert observed["provider_calls"] == 0
        return {**observed, "passed": True, "reasons": []}

    monkeypatch.setattr(HARNESS, "_collect_authoritative_production", certifying_collector)
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    artifact, artifact_path = HARNESS.run(
        source_root=source,
        source_commit=commit,
        output=tmp_path / "evidence",
        local_receipts=local_dir,
        ox_receipts=ox_dir,
        production_root=production,
    )
    reference = artifact["authority_receipt"]
    assert reference["available"] is True
    authority_path = artifact_path.parent / str(reference["relative_path"])
    authority_raw = authority_path.read_bytes()
    authority = json.loads(authority_raw)
    assert set(authority["input_payloads"]) == {"local", "ox"}
    assert authority["input_payloads"]["local"]
    assert authority["input_payloads"]["ox"]
    assert not (authority_path.parent / "authority-inputs").exists()
    assert HARNESS.validate_source_bound_authority_receipt(
        authority_path,
        artifact_path=artifact_path,
        source_root=source,
        source_commit=commit,
    )["artifact_id"] == reference["artifact_id"]

    forged = json.loads(authority_path.read_text())
    forged["captured_at"] = "2026-08-25T00:00:00+00:00"
    forged["artifact_id"] = HARNESS._sha256(
        {key: value for key, value in forged.items() if key not in {"artifact_id", "seal_sha256"}}
    )
    forged["seal_sha256"] = HARNESS._sha256(
        {key: value for key, value in forged.items() if key != "seal_sha256"}
    )
    authority_path.write_bytes(HARNESS._json_bytes(forged) + b"\n")
    with pytest.raises(HARNESS.R4Error, match="reference|path binding"):
        HARNESS.validate_source_bound_authority_receipt(
            authority_path,
            artifact_path=artifact_path,
            source_root=source,
            source_commit=commit,
        )


def test_authority_validator_rejects_resealed_embedded_duplicate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-sealed authority/R4 pair cannot inflate an embedded cohort."""

    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(production)
    )
    real_collector = HARNESS._collect_authoritative_production
    monkeypatch.setattr(
        HARNESS,
        "_collect_authoritative_production",
        lambda **kwargs: {**real_collector(**kwargs), "passed": True, "reasons": []},
    )
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    artifact, artifact_path = HARNESS.run(
        source_root=source,
        source_commit=commit,
        output=tmp_path / "evidence",
        local_receipts=local_dir,
        ox_receipts=ox_dir,
        production_root=production,
    )
    authority_path = artifact_path.parent / str(artifact["authority_receipt"]["relative_path"])
    authority = cast(dict[str, object], json.loads(authority_path.read_text()))
    inventory = cast(dict[str, object], authority["receipt_inventory"])
    payloads = cast(dict[str, object], authority["input_payloads"])
    local_payloads = cast(list[dict[str, object]], payloads["local"])
    local_files = cast(dict[str, object], inventory["local"])
    inventory_rows = cast(list[dict[str, object]], local_files["files"])

    first_payload = local_payloads[0]
    raw = base64.b64decode(cast(str, first_payload["payload_b64"]), validate=True)
    values = cast(list[dict[str, object]], json.loads(raw))
    values.append(dict(values[0]))
    duplicate_raw = HARNESS._json_bytes(values)
    duplicate_digest = hashlib.sha256(duplicate_raw).hexdigest()
    first_payload["payload_b64"] = base64.b64encode(duplicate_raw).decode("ascii")
    first_payload["sha256"] = duplicate_digest
    inventory_rows[0]["sha256"] = duplicate_digest
    file_state = cast(dict[str, int], inventory_rows[0]["file_state"])
    file_state["st_size"] = len(duplicate_raw)
    local_files["count"] = int(local_files["count"]) + 1

    local_values: list[dict[str, object]] = []
    for payload in local_payloads:
        embedded_raw = base64.b64decode(cast(str, payload["payload_b64"]), validate=True)
        local_values.extend(
            cast(
                list[dict[str, object]],
                HARNESS._authority_receipt_values(
                    embedded_raw, name=cast(str, payload["path"]), label="local receipt"
                ),
            )
        )
    assert len(local_values) == 3009
    source_snapshot = HARNESS._assert_source(source, commit)

    authority_unsigned = {
        key: value for key, value in authority.items() if key not in {"artifact_id", "seal_sha256"}
    }
    authority_id = HARNESS._sha256(authority_unsigned)
    forged_authority = HARNESS._sealed(
        {"artifact_id": authority_id, **authority_unsigned}
    )
    forged_authority_raw = HARNESS._json_bytes(forged_authority) + b"\n"
    forged_authority_path = authority_path.with_name(f"{authority_id}.authority.json")
    forged_authority_path.write_bytes(forged_authority_raw)

    forged_artifact = cast(dict[str, object], json.loads(artifact_path.read_text()))
    forged_artifact["authority_receipt"] = {
        "available": True,
        "artifact_id": authority_id,
        "seal_sha256": forged_authority["seal_sha256"],
        "relative_path": forged_authority_path.name,
        "file_sha256": hashlib.sha256(forged_authority_raw).hexdigest(),
        "parent_dev": authority_path.parent.stat().st_dev,
        "parent_ino": authority_path.parent.stat().st_ino,
    }
    forged_artifact["receipt_files"] = forged_authority["receipt_inventory"]
    source_contract = cast(dict[str, object], forged_artifact["source_contract"])
    source_contract["local"] = HARNESS._validate_local(local_values, source_snapshot)
    artifact_unsigned = {
        key: value for key, value in forged_artifact.items() if key not in {"artifact_id", "seal_sha256"}
    }
    artifact_id = HARNESS._sha256(artifact_unsigned)
    forged_artifact = HARNESS._sealed({"artifact_id": artifact_id, **artifact_unsigned})
    forged_artifact_path = artifact_path.with_name(f"{artifact_id}.json")
    forged_artifact_path.write_bytes(HARNESS._json_bytes(forged_artifact) + b"\n")

    with pytest.raises(HARNESS.R4Error, match="duplicate local receipt id"):
        HARNESS.validate_source_bound_authority_receipt(
            forged_authority_path,
            artifact_path=forged_artifact_path,
            source_root=source,
            source_commit=commit,
        )


def test_embedded_authority_inputs_reject_noncanonical_path_and_base64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(production)
    )
    real_collector = HARNESS._collect_authoritative_production
    monkeypatch.setattr(
        HARNESS,
        "_collect_authoritative_production",
        lambda **kwargs: {**real_collector(**kwargs), "passed": True, "reasons": []},
    )
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    artifact, artifact_path = HARNESS.run(
        source_root=source,
        source_commit=commit,
        output=tmp_path / "evidence",
        local_receipts=local_dir,
        ox_receipts=ox_dir,
        production_root=production,
    )
    authority_path = artifact_path.parent / str(artifact["authority_receipt"]["relative_path"])
    authority = json.loads(authority_path.read_text())

    for unsafe_path in ("", ".."):
        payloads = json.loads(json.dumps(authority["input_payloads"]))
        inventory = json.loads(json.dumps(authority["receipt_inventory"]))
        payloads["local"][0]["path"] = unsafe_path
        inventory["local"]["files"][0]["path"] = unsafe_path
        with pytest.raises(HARNESS.R4Error, match="embedded inputs"):
            HARNESS._read_embedded_authority_inputs(payloads, inventory=inventory)

    payloads = json.loads(json.dumps(authority["input_payloads"]))
    inventory = json.loads(json.dumps(authority["receipt_inventory"]))
    payloads["local"].append(json.loads(json.dumps(payloads["local"][0])))
    inventory["local"]["files"].append(
        json.loads(json.dumps(inventory["local"]["files"][0]))
    )
    with pytest.raises(HARNESS.R4Error, match="embedded inputs"):
        HARNESS._read_embedded_authority_inputs(payloads, inventory=inventory)

    payloads = json.loads(json.dumps(authority["input_payloads"]))
    inventory = json.loads(json.dumps(authority["receipt_inventory"]))
    row = payloads["local"][0]
    raw = base64.b64decode(row["payload_b64"], validate=True)
    while len(raw) % 3 == 0:
        raw += b" "
    canonical = base64.b64encode(raw).decode("ascii")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    altered_index = -3 if canonical.endswith("==") else -2
    value = alphabet.index(canonical[altered_index])
    lower_mask = 0x0F if canonical.endswith("==") else 0x03
    replacement = (value & ~lower_mask) | ((value + 1) & lower_mask)
    noncanonical = (
        canonical[:altered_index] + alphabet[replacement] + canonical[altered_index + 1 :]
    )
    assert base64.b64decode(noncanonical, validate=True) == raw
    assert noncanonical != canonical
    row["payload_b64"] = noncanonical
    row["sha256"] = hashlib.sha256(raw).hexdigest()
    inventory["local"]["files"][0]["sha256"] = row["sha256"]
    inventory["local"]["files"][0]["file_state"]["st_size"] = len(raw)
    with pytest.raises(HARNESS.R4Error, match="embedded inputs"):
        HARNESS._read_embedded_authority_inputs(payloads, inventory=inventory)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("st_dev", -1),
        ("st_ino", -1),
        ("st_mode", -1),
        ("st_mtime_ns", -1),
        ("st_size", -1),
        ("st_mode", 0o10000),
        ("st_size", True),
    ],
)
def test_embedded_authority_inputs_reject_invalid_file_state(
    field: str, value: object
) -> None:
    receipt = HARNESS._sealed(
        {
            "schema": HARNESS.RECEIPT_SCHEMA,
            "namespace": "recall-distillation",
            "receipt_id": "a" * 64,
        }
    )
    raw = HARNESS._json_bytes(receipt)
    digest = hashlib.sha256(raw).hexdigest()
    payloads = {
        "local": [
            {
                "path": "0000.json",
                "sha256": digest,
                "payload_b64": base64.b64encode(raw).decode("ascii"),
            }
        ],
        "ox": [],
    }
    file_state: dict[str, object] = {
        "st_dev": 1,
        "st_ino": 2,
        "st_mode": 0o600,
        "st_size": len(raw),
        "st_mtime_ns": 3,
    }
    file_state[field] = value
    inventory = {
        "local": {"files": [{"path": "0000.json", "sha256": digest, "file_state": file_state}], "count": 1},
        "ox": {"files": [], "count": 0},
    }
    with pytest.raises(HARNESS.R4Error, match="embedded inputs"):
        HARNESS._read_embedded_authority_inputs(payloads, inventory=inventory)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema", "forged-top-level"), ("namespace", "forged-namespace")],
)
def test_authority_validator_rejects_resealed_top_level_discriminator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(production)
    )
    real_collector = HARNESS._collect_authoritative_production
    monkeypatch.setattr(
        HARNESS,
        "_collect_authoritative_production",
        lambda **kwargs: {**real_collector(**kwargs), "passed": True, "reasons": []},
    )
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    artifact, artifact_path = HARNESS.run(
        source_root=source,
        source_commit=commit,
        output=tmp_path / "evidence",
        local_receipts=local_dir,
        ox_receipts=ox_dir,
        production_root=production,
    )
    authority_path = artifact_path.parent / str(artifact["authority_receipt"]["relative_path"])
    authority = cast(dict[str, object], json.loads(authority_path.read_text()))
    authority[field] = value
    authority_unsigned = {
        key: item for key, item in authority.items() if key not in {"artifact_id", "seal_sha256"}
    }
    authority_id = HARNESS._sha256(authority_unsigned)
    forged_authority = HARNESS._sealed({"artifact_id": authority_id, **authority_unsigned})
    authority_raw = HARNESS._json_bytes(forged_authority) + b"\n"
    forged_authority_path = authority_path.with_name(f"{authority_id}.authority.json")
    forged_authority_path.write_bytes(authority_raw)

    forged_artifact = cast(dict[str, object], json.loads(artifact_path.read_text()))
    forged_artifact["authority_receipt"] = {
        "available": True,
        "artifact_id": authority_id,
        "seal_sha256": forged_authority["seal_sha256"],
        "relative_path": forged_authority_path.name,
        "file_sha256": hashlib.sha256(authority_raw).hexdigest(),
        "parent_dev": authority_path.parent.stat().st_dev,
        "parent_ino": authority_path.parent.stat().st_ino,
    }
    artifact_unsigned = {
        key: item for key, item in forged_artifact.items() if key not in {"artifact_id", "seal_sha256"}
    }
    artifact_id = HARNESS._sha256(artifact_unsigned)
    forged_artifact = HARNESS._sealed({"artifact_id": artifact_id, **artifact_unsigned})
    forged_artifact_path = artifact_path.with_name(f"{artifact_id}.json")
    forged_artifact_path.write_bytes(HARNESS._json_bytes(forged_artifact) + b"\n")

    with pytest.raises(HARNESS.R4Error, match="authority receipt schema"):
        HARNESS.validate_source_bound_authority_receipt(
            forged_authority_path,
            artifact_path=forged_artifact_path,
            source_root=source,
            source_commit=commit,
        )


def test_authority_staging_rejects_output_inputs_and_kind_swaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    monkeypatch.setattr(
        HARNESS,
        "_collect_authoritative_production",
        lambda **_kwargs: {"passed": True, "provider_calls": 0, "collector": "test"},
    )

    output = tmp_path / "output-swap"
    displaced, replacement = tmp_path / "displaced", tmp_path / "replacement"

    def swap_output(phase: str) -> None:
        if phase == "output-opened":
            os.rename(output, displaced)
            replacement.mkdir()
            os.rename(replacement, output)

    with pytest.raises(HARNESS.R4Error, match="output changed"):
        HARNESS.produce_source_bound_authority_receipt(
            output, source_root=source, source_commit=commit,
            local_receipts=local_dir, ox_receipts=ox_dir, before_stage=swap_output,
        )
    assert not list(output.iterdir())
    assert not list(displaced.iterdir())

    output = tmp_path / "mutable-staging"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "authority-inputs").symlink_to(outside, target_is_directory=True)
    receipt, _inventory = HARNESS.produce_source_bound_authority_receipt(
        output, source_root=source, source_commit=commit,
        local_receipts=local_dir, ox_receipts=ox_dir,
    )
    assert receipt["available"] is True
    assert not list(outside.iterdir())


def test_authority_validator_takes_third_snapshot_after_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(production)
    )
    real_collector = HARNESS._collect_authoritative_production

    def certifying_collector(**kwargs: object) -> dict[str, object]:
        return {**real_collector(**kwargs), "passed": True, "reasons": []}

    monkeypatch.setattr(HARNESS, "_collect_authoritative_production", certifying_collector)
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    artifact, artifact_path = HARNESS.run(
        source_root=source, source_commit=commit, output=tmp_path / "evidence",
        local_receipts=local_dir, ox_receipts=ox_dir, production_root=production,
    )
    authority_path = artifact_path.parent / str(artifact["authority_receipt"]["relative_path"])
    authority_raw = authority_path.read_bytes()
    artifact_raw = artifact_path.read_bytes()
    calls = 0

    def corrupting_collector(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        observed = certifying_collector(**kwargs)
        if calls == 2:
            staged = authority_path.parent / "authority-inputs" / "local"
            staged.mkdir(parents=True)
            (staged / "forged.json").write_bytes(b"{}\n")
        return observed

    monkeypatch.setattr(HARNESS, "_collect_authoritative_production", corrupting_collector)
    assert HARNESS.validate_source_bound_authority_receipt(
        authority_path, artifact_path=artifact_path, source_root=source, source_commit=commit
    )["artifact_id"] == artifact["authority_receipt"]["artifact_id"]
    assert authority_path.read_bytes() == authority_raw
    assert artifact_path.read_bytes() == artifact_raw
    assert (authority_path.parent / "authority-inputs" / "local" / "forged.json").read_bytes() == b"{}\n"


def test_authority_validator_rechecks_after_final_authority_read_and_root_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(production)
    )
    real_collector = HARNESS._collect_authoritative_production
    monkeypatch.setattr(
        HARNESS, "_collect_authoritative_production",
        lambda **kwargs: {**real_collector(**kwargs), "passed": True, "reasons": []},
    )
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    artifact, artifact_path = HARNESS.run(
        source_root=source, source_commit=commit, output=tmp_path / "evidence",
        local_receipts=local_dir, ox_receipts=ox_dir, production_root=production,
    )
    authority_path = artifact_path.parent / str(artifact["authority_receipt"]["relative_path"])
    authority_raw = authority_path.read_bytes()
    artifact_raw = artifact_path.read_bytes()
    real_read = HARNESS._authority_read_fd
    authority_reads = 0

    def mutate_input_after_final_authority_read(
        directory_fd: int, name: str, *, label: str
    ) -> tuple[bytes, dict[str, int]]:
        nonlocal authority_reads
        result = real_read(directory_fd, name, label=label)
        if label == "authority receipt":
            authority_reads += 1
            if authority_reads == 2:
                staged = authority_path.parent / "authority-inputs" / "local"
                staged.mkdir(parents=True)
                (staged / "forged.json").write_bytes(b"{}\n")
        return result

    monkeypatch.setattr(HARNESS, "_authority_read_fd", mutate_input_after_final_authority_read)
    assert HARNESS.validate_source_bound_authority_receipt(
        authority_path, artifact_path=artifact_path, source_root=source, source_commit=commit
    )["artifact_id"] == artifact["authority_receipt"]["artifact_id"]
    assert authority_path.read_bytes() == authority_raw
    assert artifact_path.read_bytes() == artifact_raw
    assert (authority_path.parent / "authority-inputs" / "local" / "forged.json").read_bytes() == b"{}\n"

    monkeypatch.setattr(HARNESS, "_authority_read_fd", real_read)
    artifact, artifact_path = HARNESS.run(
        source_root=source, source_commit=commit, output=tmp_path / "second-evidence",
        local_receipts=local_dir, ox_receipts=ox_dir, production_root=production,
    )
    authority_path = artifact_path.parent / str(artifact["authority_receipt"]["relative_path"])
    displaced, replacement = tmp_path / "displaced", tmp_path / "replacement"
    authority_reads = 0

    def swap_root_after_final_authority_read(
        directory_fd: int, name: str, *, label: str
    ) -> tuple[bytes, dict[str, int]]:
        nonlocal authority_reads
        result = real_read(directory_fd, name, label=label)
        if label == "authority receipt":
            authority_reads += 1
            if authority_reads == 2:
                os.rename(authority_path.parent, displaced)
                replacement.mkdir()
                os.rename(replacement, authority_path.parent)
        return result

    monkeypatch.setattr(HARNESS, "_authority_read_fd", swap_root_after_final_authority_read)
    with pytest.raises(HARNESS.R4Error, match="output changed|output path"):
        HARNESS.validate_source_bound_authority_receipt(
            authority_path, artifact_path=artifact_path, source_root=source, source_commit=commit
        )
    assert not authority_path.exists()


def test_authority_staging_failure_cleans_owned_entries_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    monkeypatch.setattr(
        HARNESS,
        "_collect_authoritative_production",
        lambda **_kwargs: {"passed": True, "provider_calls": 0, "collector": "test"},
    )
    output = tmp_path / "evidence"
    real_read = HARNESS._authority_read_fd

    def fail_authority_readback(
        directory_fd: int, name: str, *, label: str
    ) -> tuple[bytes, dict[str, int]]:
        if label == "authority receipt":
            raise HARNESS.R4Error("synthetic authority readback failure")
        return real_read(directory_fd, name, label=label)

    monkeypatch.setattr(HARNESS, "_authority_read_fd", fail_authority_readback)
    with pytest.raises(HARNESS.R4Error, match="synthetic authority"):
        HARNESS.produce_source_bound_authority_receipt(
            output, source_root=source, source_commit=commit,
            local_receipts=local_dir, ox_receipts=ox_dir,
        )
    assert not list(output.glob("*.authority.json"))
    assert not (output / "authority-inputs").exists()
    monkeypatch.setattr(HARNESS, "_authority_read_fd", real_read)
    receipt, _inventory = HARNESS.produce_source_bound_authority_receipt(
        output, source_root=source, source_commit=commit,
        local_receipts=local_dir, ox_receipts=ox_dir,
    )
    assert receipt["available"] is True


def test_run_rejects_output_swap_between_authority_and_artifact_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(production)
    )
    real_collector = HARNESS._collect_authoritative_production
    monkeypatch.setattr(
        HARNESS,
        "_collect_authoritative_production",
        lambda **kwargs: {**real_collector(**kwargs), "passed": True, "reasons": []},
    )
    local_dir, ox_dir = tmp_path / "local", tmp_path / "ox"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    _write_receipts(ox_dir, [_receipt(row, index) for index, row in enumerate(_ox_rows(source, commit))])
    output, displaced, replacement = tmp_path / "evidence", tmp_path / "displaced", tmp_path / "replacement"
    real_publish = HARNESS._write_immutable_pinned

    def swap_before_artifact_publish(*args: object, **kwargs: object) -> tuple[str, str, dict[str, object], bool]:
        os.rename(output, displaced)
        replacement.mkdir()
        os.rename(replacement, output)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(HARNESS, "_write_immutable_pinned", swap_before_artifact_publish)
    with pytest.raises(HARNESS.R4Error, match="output changed"):
        HARNESS.run(
            source_root=source, source_commit=commit, output=output,
            local_receipts=local_dir, ox_receipts=ox_dir, production_root=production,
    )
    assert not list(output.iterdir())
    assert not [path for path in displaced.glob("*.json") if not path.name.endswith(".authority.json")]


def test_authority_cleanup_keeps_primary_error_and_closes_parent_after_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    local_dir = tmp_path / "local"
    _write_receipts(local_dir, [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))])
    monkeypatch.setattr(
        HARNESS,
        "_collect_authoritative_production",
        lambda **_kwargs: {"passed": True, "provider_calls": 0, "collector": "test"},
    )
    opened: tuple[int, int, str, tuple[int, int]] | None = None
    real_open = HARNESS._open_authority_output_root

    def capture_open(path: Path) -> tuple[int, int, str, tuple[int, int]]:
        nonlocal opened
        opened = real_open(path)
        return opened

    monkeypatch.setattr(HARNESS, "_open_authority_output_root", capture_open)
    real_close = os.close
    closed: list[int] = []
    failed = False

    def fail_once_on_output(fd: int) -> None:
        nonlocal failed
        if opened is not None and fd == opened[1] and not failed:
            failed = True
            raise OSError("synthetic close failure")
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(HARNESS.os, "close", fail_once_on_output)
    output = tmp_path / "evidence"
    with pytest.raises(HARNESS.R4Error, match="authority receipt inputs no longer satisfy"):
        HARNESS.produce_source_bound_authority_receipt(
            output, source_root=source, source_commit=commit,
            local_receipts=local_dir, ox_receipts=None,
        )
    assert opened is not None and opened[0] in closed
    assert not (output / "authority-inputs").exists()
    monkeypatch.setattr(HARNESS.os, "close", real_close)
    real_close(opened[1])


def test_run_rejects_source_mutation_during_production_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = tmp_path / "production"
    production.mkdir()
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)

    def mutating_collector(**kwargs: object) -> dict[str, object]:
        source_root = cast(Path, kwargs["source_root"])
        (source_root / "README.md").write_text("mutated during collection\n")
        return {
            "passed": False,
            "reasons": ["synthetic_collector"],
            "collector": "test",
            "provider_calls": 0,
        }

    monkeypatch.setattr(
        HARNESS, "_collect_authoritative_production", mutating_collector
    )
    with pytest.raises(HARNESS.R4Error, match="dirty|final evidence"):
        HARNESS.run(
            source_root=source,
            source_commit=commit,
            output=tmp_path / "evidence",
            production_root=production,
        )


def test_run_does_not_publish_when_source_changes_during_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    local_dir = tmp_path / "local"
    ox_dir = tmp_path / "ox"
    _write_receipts(
        local_dir,
        [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))],
    )
    _write_receipts(
        ox_dir,
        [
            _receipt(row, index + 100)
            for index, row in enumerate(_ox_rows(source, commit))
        ],
    )
    real_publish = HARNESS._authority_publish_fd
    mutated = False

    def mutating_publish(
        directory_fd: int, name: str, raw: bytes, *, label: str
    ) -> dict[str, int]:
        nonlocal mutated
        if label == "R4 artifact" and not mutated:
            mutated = True
            (source / "README.md").write_text("mutated during publication\n")
        return real_publish(directory_fd, name, raw, label=label)

    monkeypatch.setattr(HARNESS, "_authority_publish_fd", mutating_publish)
    with pytest.raises(HARNESS.R4Error, match="dirty|after artifact publication"):
        HARNESS.run(
            source_root=source,
            source_commit=commit,
            output=tmp_path / "evidence",
            local_receipts=local_dir,
            ox_receipts=ox_dir,
        )
    assert list((tmp_path / "evidence").glob("*.json")) == []


def test_public_cli_ignores_home_override_for_production_root(
    tmp_path: Path,
) -> None:
    source, commit = _git_source(tmp_path)
    output = tmp_path / "evidence"
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    environment = dict(os.environ)
    environment["HOME"] = str(fake_home)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "recall_r4_harness.py"),
            "--source-root",
            str(source),
            "--source-commit",
            commit,
            "--output",
            str(output),
            "--production",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    artifact_path = next(output.glob("*.json"))
    artifact = HARNESS.read_artifact(artifact_path)
    assert artifact["production_certification"]["root"] == str(
        HARNESS.ACCOUNT_HOME / ".chronovisor"
    )


def test_r4_collector_does_not_fall_back_to_checkout_handoff(tmp_path: Path) -> None:
    source = tmp_path / "source"
    anchor_path = source / HARNESS.R0_EVIDENCE_RELATIVE
    anchor_path.parent.mkdir(parents=True)
    tracked = ROOT / HARNESS.R0_EVIDENCE_RELATIVE
    anchor_path.write_bytes(tracked.read_bytes())
    # A copied tracked artifact is not runtime authority.  The collector must
    # require the explicit managed r4-candidate-anchor instead.
    with pytest.raises(HARNESS.R4Error, match="candidate anchor path is unavailable"):
        HARNESS._load_production_anchor(source, source={"commit": "a" * 40})


def test_production_anchor_requires_the_audited_source_commit(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    production = tmp_path / "production"
    anchor_path = production / HARNESS.PRODUCTION_CANDIDATE_ANCHOR_RELATIVE
    anchor_path.parent.mkdir(parents=True)
    unsigned = {
        "schema": HARNESS.R4_CANDIDATE_ANCHOR_SCHEMA,
        "namespace": "recall-distillation",
        "kind": "r4-candidate-anchor",
        "r0_artifact_id": HARNESS.R0_EVIDENCE_ID,
        "r0_file_sha256": "a" * 64,
        "bootstrap_source_commit": "0" * 40,
        "candidate_checkpoint": {},
        "critical_module_sha256": {},
    }
    payload = HARNESS._sealed({"artifact_id": HARNESS._sha256(unsigned), **unsigned})
    anchor_path.write_bytes(HARNESS._json_bytes(payload) + b"\n")
    with pytest.raises(HARNESS.R4Error, match="anchor identity"):
        HARNESS._load_production_anchor(
            production, source=HARNESS._assert_source(source, commit)
        )


def test_candidate_checkpoint_reseal_and_same_size_substitution_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    anchor = _fixture_candidate_anchor(production)
    monkeypatch.setattr(HARNESS, "_load_production_anchor", lambda _path, **_kwargs: anchor)
    candidate_path = production / HARNESS.PRODUCTION_CANDIDATE_RELATIVE
    checkpoint_path = production / HARNESS.PRODUCTION_CANDIDATE_CHECKPOINT_RELATIVE
    original = candidate_path.read_bytes()
    forged = original.replace(b"candidate-fixture", b"candidate-forge!!", 1)
    assert len(forged) == len(original)
    candidate_path.write_bytes(forged)
    checkpoint = json.loads(checkpoint_path.read_text())
    current = HARNESS._production_stat(candidate_path, label="forged candidates")
    checkpoint["file_state"] = {
        "size_bytes": current["st_size"],
        "st_dev": current["st_dev"],
        "st_ino": current["st_ino"],
        "st_mtime_ns": current["st_mtime_ns"],
        "st_ctime_ns": current["st_ctime_ns"],
    }
    checkpoint["seal_sha256"] = HARNESS._sha256(
        {key: value for key, value in checkpoint.items() if key != "seal_sha256"}
    )
    checkpoint_path.write_text(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))
    )
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )
    assert result["passed"] is False
    assert any("candidate" in reason for reason in result["reasons"])


def test_candidate_append_requires_a_fresh_offline_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    anchor = _fixture_candidate_anchor(production)
    monkeypatch.setattr(HARNESS, "_load_production_anchor", lambda _path, **_kwargs: anchor)
    candidate_path = production / HARNESS.PRODUCTION_CANDIDATE_RELATIVE
    checkpoint_path = production / HARNESS.PRODUCTION_CANDIDATE_CHECKPOINT_RELATIVE
    first = json.loads(candidate_path.read_text().splitlines()[0])
    second_unsigned = {
        "candidate_id": "candidate-next",
        "namespace": "recall-distillation",
        "previous_sha256": first["record_sha256"],
        "schema": "chronovisor.recall-distillation.v1",
    }
    second = {**second_unsigned, "record_sha256": HARNESS._sha256(second_unsigned)}
    candidate_path.write_bytes(
        candidate_path.read_bytes()
        + json.dumps(second, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    current = HARNESS._production_stat(candidate_path, label="appended candidates")
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint.update(
        records=2,
        head_sha256=second["record_sha256"],
        file_state={
            "size_bytes": current["st_size"],
            "st_dev": current["st_dev"],
            "st_ino": current["st_ino"],
            "st_mtime_ns": current["st_mtime_ns"],
            "st_ctime_ns": current["st_ctime_ns"],
        },
    )
    checkpoint["seal_sha256"] = HARNESS._sha256(
        {key: value for key, value in checkpoint.items() if key != "seal_sha256"}
    )
    checkpoint_path.write_text(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))
    )
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )
    assert result["passed"] is False
    assert any(
        reason in result["reasons"]
        for reason in (
            "production candidate ledger differs from sealed R0 anchor",
            "production candidate checkpoint precedes R0 anchor",
        )
    )


def test_authoritative_collector_rejects_persistent_root_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path / "first", source, commit)
    replacement = _authoritative_production_root(tmp_path / "second", source, commit)
    replacement_state_path = replacement / HARNESS.PRODUCTION_STATE_RELATIVE
    replacement_state = json.loads(replacement_state_path.read_text())
    replacement_state["runtime_identity"]["root"] = str(production.absolute())
    replacement_state["seal_sha256"] = HARNESS._sha256(
        {key: value for key, value in replacement_state.items() if key != "seal_sha256"}
    )
    replacement_state_path.write_text(
        json.dumps(replacement_state, sort_keys=True, separators=(",", ":"))
    )
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _path, **_kwargs: _fixture_candidate_anchor(production),
    )
    real_directory_identity = HARNESS._production_directory_identity
    identity_calls = 0
    swapped = False
    old_production = tmp_path / "production-old"

    def swap_before_final_identity(path: Path, *, label: str) -> dict[str, int]:
        nonlocal identity_calls, swapped
        identity_calls += 1
        if identity_calls == 2 and not swapped:
            swapped = True
            production.rename(old_production)
            replacement.rename(production)
        return cast(dict[str, int], real_directory_identity(path, label=label))

    monkeypatch.setattr(
        HARNESS, "_production_directory_identity", swap_before_final_identity
    )
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )
    assert result["passed"] is False
    assert "production root changed during validation" in result["reasons"]


def test_authoritative_collector_restores_cwd_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    before = Path.cwd()

    def fail_workset(_path: Path) -> dict[str, object]:
        raise HARNESS.R4Error("synthetic workset failure")

    monkeypatch.setattr(HARNESS, "_production_workset", fail_workset)
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )
    assert result["passed"] is False
    assert "synthetic workset failure" in result["reasons"]
    assert Path.cwd() == before


def test_authoritative_collector_fails_closed_on_cwd_restore_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    anchor = _fixture_candidate_anchor(production)
    monkeypatch.setattr(HARNESS, "_load_production_anchor", lambda _path, **_kwargs: anchor)
    emergency_fd = os.open(".", os.O_RDONLY)
    real_fchdir = HARNESS.os.fchdir
    calls = 0

    def fail_restore(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic cwd restore failure")
        real_fchdir(fd)

    monkeypatch.setattr(HARNESS.os, "fchdir", fail_restore)
    try:
        with pytest.raises(HARNESS.R4Error, match="cwd restore"):
            HARNESS.run(
                source_root=source,
                source_commit=commit,
                output=tmp_path / "evidence",
                production_root=production,
            )
    finally:
        real_fchdir(emergency_fd)
        os.close(emergency_fd)
    assert list((tmp_path / "evidence").glob("*.json")) == []


def test_authoritative_collector_can_certify_only_fixed_sealed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unit-only root injection exercises the closed schema; the public CLI
    # cannot supply this path and always reads the fixed managed root.
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _path, **_kwargs: _fixture_candidate_anchor(production),
    )
    source_snapshot = HARNESS._assert_source(source, commit)
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=source_snapshot,
        production_root=production,
    )
    # The persisted writer path is valid, but this fixture deliberately does
    # not synthesize the six provider-failure transitions.  Production
    # certification must therefore remain fail-closed while the ramp evidence
    # itself is still independently readable.
    assert result["passed"] is False
    assert set(result["reasons"]) == {
        "production_failure_coverage_incomplete",
        "production_fault_scenarios_missing",
        "production_label_identity_invalid",
        "production_lease_recovery_invalid",
        "production_runtime_archive_binding_invalid",
    }
    assert result["provider_calls"] == 0
    assert result["workset"]["receipts"]["verified"] is True
    assert result["quality"]["stages"] == {
        str(cap): {
            "valid_receipts": 20,
            "attempts": 20,
            "valid_rate": 1.0,
            "work_ids": result["quality"]["stages"][str(cap)]["work_ids"],
        }
        for cap in HARNESS.OX_STAGES
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payload_digest", "0" * 64),
        ("request_sha256", "1" * 64),
        ("provider_request_sha256", "2" * 64),
        ("provider_response_request_sha256", "3" * 64),
        ("work_id", "4" * 64),
        ("profile_contract_id", "5" * 64),
    ],
)
def test_production_quality_rederives_label_request_binding(
    tmp_path: Path, field: str, value: object
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    source_snapshot = HARNESS._assert_source(source, commit)
    state = json.loads((production / HARNESS.PRODUCTION_STATE_RELATIVE).read_text())
    contract_id = str(state["profile_contract_id"])
    contract, _, _ = HARNESS._production_json(
        production
        / HARNESS.PRODUCTION_CONTRACT_DIR_RELATIVE
        / f"{contract_id}.json",
        label="production profile contract",
        schema="chronovisor.recall-distill-ox-profile.v1",
    )
    workset = HARNESS._production_workset(
        production / HARNESS.PRODUCTION_WORKSET_RELATIVE
    )
    labels = HARNESS._production_chain(
        production / HARNESS.PRODUCTION_LABEL_RELATIVE,
        production / HARNESS.PRODUCTION_LABEL_CHECKPOINT_RELATIVE,
    )
    events = HARNESS._production_ox_events(
        production, source=source_snapshot, contract_id=contract_id
    )
    mutated = [dict(row) for row in labels["rows"]]
    mutated[0][field] = value
    reasons, _quality = HARNESS._production_quality(
        state=state,
        workset=workset,
        labels={**labels, "rows": mutated},
        events=events,
        source=source_snapshot,
        contract=contract,
        contract_id=contract_id,
    )
    assert "production_label_identity_invalid" in reasons


def test_production_quality_rejects_self_resealed_payload_source(
    tmp_path: Path,
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    source_snapshot = HARNESS._assert_source(source, commit)
    state = json.loads((production / HARNESS.PRODUCTION_STATE_RELATIVE).read_text())
    contract_id = str(state["profile_contract_id"])
    contract, _, _ = HARNESS._production_json(
        production
        / HARNESS.PRODUCTION_CONTRACT_DIR_RELATIVE
        / f"{contract_id}.json",
        label="production profile contract",
        schema="chronovisor.recall-distill-ox-profile.v1",
    )
    workset = HARNESS._production_workset(
        production / HARNESS.PRODUCTION_WORKSET_RELATIVE
    )
    labels = HARNESS._production_chain(
        production / HARNESS.PRODUCTION_LABEL_RELATIVE,
        production / HARNESS.PRODUCTION_LABEL_CHECKPOINT_RELATIVE,
    )
    candidate = HARNESS._production_chain(
        production / HARNESS.PRODUCTION_CANDIDATE_RELATIVE,
        production / HARNESS.PRODUCTION_CANDIDATE_CHECKPOINT_RELATIVE,
        ledger_name="candidate-ledger.jsonl",
    )
    from chronovisor.recall import recall_distillation as distill

    rallies = distill._materialization_rallies(production, None)
    events = HARNESS._production_ox_events(
        production, source=source_snapshot, contract_id=contract_id
    )
    forged = dict(labels["rows"][0])
    source_payload = dict(forged["payload_source"])
    source_payload["query_sha256"] = "0" * 64
    payload_digest = HARNESS._sha256(source_payload)
    work_id = HARNESS._sha256(
        {
            "kind": "ox-teacher-label-v1",
            "profile": HARNESS.OX_PROFILE,
            "cohort": HARNESS.OX_COHORT,
            "route": HARNESS.OX_ROUTE,
            "profile_contract_id": contract_id,
            "payload_digest": payload_digest,
        }
    )
    forged.update(
        {
            "payload_source": source_payload,
            "payload_digest": payload_digest,
            "work_id": work_id,
            "request_sha256": HARNESS._expected_ox_request_sha256(
                profile_contract_id=contract_id, payload_digest=payload_digest
            ),
            "provider_request_sha256": HARNESS._expected_ox_provider_request_sha256(
                profile_contract_id=contract_id,
                payload_digest=payload_digest,
                work_id=work_id,
                expires_at=str(contract["expires_at"]),
            ),
        }
    )
    forged["provider_response_request_sha256"] = forged["provider_request_sha256"]
    old_work_id = str(labels["rows"][0]["work_id"])
    forged_completed = dict(workset["completed"])
    completion = dict(forged_completed.pop(old_work_id))
    completion["work_id"] = work_id
    forged_completed[work_id] = completion
    forged_items = dict(workset["items"])
    item = dict(forged_items.pop(old_work_id))
    item["payload_digest"] = payload_digest
    forged_items[work_id] = item
    forged_workset = {**workset, "completed": forged_completed, "items": forged_items}
    reasons, _quality = HARNESS._production_quality(
        state=state,
        workset=forged_workset,
        labels={**labels, "rows": [forged, *labels["rows"][1:]]},
        events=events,
        source=source_snapshot,
        contract=contract,
        contract_id=contract_id,
        candidate_rows=candidate["rows"],
        rallies=rallies,
    )
    assert "production_label_identity_invalid" in reasons


def test_authoritative_collector_rejects_test_only_fault_scenario_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _path, **_kwargs: _fixture_candidate_anchor(production),
    )
    source_snapshot = HARNESS._assert_source(source, commit)
    state = json.loads((production / HARNESS.PRODUCTION_STATE_RELATIVE).read_text())
    contract_id = str(state["profile_contract_id"])
    workset = HARNESS._production_workset(
        production / HARNESS.PRODUCTION_WORKSET_RELATIVE
    )
    events = HARNESS._production_ox_events(
        production, source=source_snapshot, contract_id=contract_id
    )
    heads = {
        name: str(rows[-1]["record_sha256"]) if rows else ""
        for name, rows in events.items()
    }
    unsigned = {
        "schema": HARNESS.R4_FAULT_SCENARIO_SCHEMA,
        "namespace": "recall-distillation",
        "scenario": "http_429",
        "writer_path": "public-run-distillation-chunk-v1",
        "test_only": True,
        "source": {
            "source_commit": source_snapshot["commit"],
            "source_tree_sha256": source_snapshot["tree_sha256"],
            "source_ox_identity_sha256": source_snapshot["ox_identity_sha256"],
        },
        "profile_contract_id": contract_id,
        "outcome": {
            "profile_stopped": False,
            "backoff_bounded": True,
            "quarantined": 0,
            "ready": 0,
            "leased": 0,
            "duplicate_labels": 0,
            "provider_calls": 0,
        },
        "workset_receipt": {
            "generation": workset["receipts"]["generation"],
            "head_sha256": workset["receipts"]["head_sha256"],
        },
        "event_heads": heads,
    }
    artifact_id = HARNESS._sha256(unsigned)
    artifact = HARNESS._sealed({"artifact_id": artifact_id, **unsigned})
    directory = production / HARNESS.PRODUCTION_FAULT_SCENARIO_RELATIVE
    directory.mkdir()
    (directory / f"{artifact_id}.json").write_bytes(HARNESS._json_bytes(artifact) + b"\n")
    result = HARNESS._collect_authoritative_production(
        source_root=source, source=source_snapshot, production_root=production
    )
    assert result["passed"] is False
    assert "production_fault_scenario_invalid" in result["reasons"]
    assert "production_fault_scenarios_incomplete" in result["reasons"]


def test_state_ramp_claim_cannot_replace_missing_event_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _path, **_kwargs: _fixture_candidate_anchor(production),
    )
    distillation = production / HARNESS.PRODUCTION_DISTILLATION_RELATIVE
    for name in ("ox-ramp-receipts.jsonl", "ox-failure-receipts.jsonl"):
        path = distillation / name
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".head.json").unlink(missing_ok=True)
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )
    assert result["passed"] is False
    assert "production_ramp_stages_incomplete" in result["reasons"]
    assert "production_failure_coverage_incomplete" in result["reasons"]


def test_authoritative_collector_rejects_fake_root_and_external_receipts(
    tmp_path: Path,
) -> None:
    source, commit = _git_source(tmp_path)
    source_snapshot = HARNESS._assert_source(source, commit)
    fake = tmp_path / "fake-production"
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=source_snapshot,
        production_root=fake,
    )
    assert result["passed"] is False
    assert "production_root_not_authoritative" in result["reasons"]
    receipts = tmp_path / "forged-receipts"
    receipts.mkdir()
    artifact, _ = HARNESS.run(
        source_root=source,
        source_commit=commit,
        output=tmp_path / "evidence",
        production_receipts=receipts,
    )
    assert artifact["production_certification"]["passed"] is False
    assert artifact["production_certification"]["reasons"] == [
        "external_production_receipts_rejected"
    ]


def test_actual_writer_artifact_one_field_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collector rejects a resealed mutation of actual writer output."""

    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _path, **_kwargs: _fixture_candidate_anchor(production),
    )
    state_path = production / HARNESS.PRODUCTION_STATE_RELATIVE
    state = json.loads(state_path.read_text())
    state["source_commit"] = "0" * 40
    state["seal_sha256"] = HARNESS._sha256(
        {key: value for key, value in state.items() if key != "seal_sha256"}
    )
    state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )
    assert result["passed"] is False
    assert any("state" in reason for reason in result["reasons"])


def test_authoritative_collector_rejects_symlink_and_resealed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    target = tmp_path / "production-target"
    target.symlink_to(production, target_is_directory=True)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", target)
    source_snapshot = HARNESS._assert_source(source, commit)
    symlinked = HARNESS._collect_authoritative_production(
        source_root=source,
        source=source_snapshot,
        production_root=target,
    )
    assert symlinked["passed"] is False
    assert "production_root_unavailable" in symlinked["reasons"]
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _path, **_kwargs: _fixture_candidate_anchor(production),
    )
    state_path = production / HARNESS.PRODUCTION_STATE_RELATIVE
    state = json.loads(state_path.read_text())
    state["seal_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
    resealed = HARNESS._collect_authoritative_production(
        source_root=source,
        source=source_snapshot,
        production_root=production,
    )
    assert resealed["passed"] is False
    assert any("seal" in reason for reason in resealed["reasons"])


def test_authoritative_collector_rejects_truncation_route_spoof_and_veto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    source_snapshot = HARNESS._assert_source(source, commit)

    truncated = _authoritative_production_root(tmp_path / "truncated", source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", truncated)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _path, **_kwargs: _fixture_candidate_anchor(truncated),
    )
    label_path = truncated / HARNESS.PRODUCTION_LABEL_RELATIVE
    label_path.write_bytes(label_path.read_bytes()[:-1])
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=source_snapshot,
        production_root=truncated,
    )
    assert result["passed"] is False
    assert any(
        "truncated" in reason or "checkpoint" in reason for reason in result["reasons"]
    )

    spoofed = _authoritative_production_root(tmp_path / "spoofed", source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", spoofed)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(spoofed)
    )
    spoof_label_path = spoofed / HARNESS.PRODUCTION_LABEL_RELATIVE
    first, *rest = spoof_label_path.read_bytes().splitlines(keepends=True)
    first_row = json.loads(first)
    first_row["route"] = "paid/provider"
    spoof_label_path.write_bytes(
        json.dumps(first_row, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        + b"".join(rest)
    )
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=source_snapshot,
        production_root=spoofed,
    )
    assert result["passed"] is False

    vetoed = _authoritative_production_root(tmp_path / "vetoed", source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", vetoed)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _path, **_kwargs: _fixture_candidate_anchor(vetoed)
    )
    state_path = vetoed / HARNESS.PRODUCTION_STATE_RELATIVE
    state = json.loads(state_path.read_text())
    state["sensitive"] = 1
    state["seal_sha256"] = HARNESS._sha256(
        {key: value for key, value in state.items() if key != "seal_sha256"}
    )
    state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=source_snapshot,
        production_root=vetoed,
    )
    assert result["passed"] is False
    assert "production_sensitive_veto_invalid" in result["reasons"]


def test_authoritative_label_collector_uses_bounded_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    label_path = production / HARNESS.PRODUCTION_LABEL_RELATIVE
    checkpoint_path = production / HARNESS.PRODUCTION_LABEL_CHECKPOINT_RELATIVE
    monkeypatch.setattr(HARNESS, "PRODUCTION_MAX_FULL_LEDGER_BYTES", 1)
    monkeypatch.setattr(HARNESS, "PRODUCTION_MAX_LEDGER_TAIL_BYTES", 32 * 1024)
    view = HARNESS._production_chain(label_path, checkpoint_path)
    assert view["count"] == 80
    assert 0 < len(view["rows"]) < view["count"]
    assert view["sha256"] is None


def test_forged_seal_and_arbitrary_source_hash_fail_closed(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    directory = tmp_path / "receipts"
    row = _local_rows(source, commit)[0]
    row["source_tree_sha256"] = "f" * 64
    _write_receipts(directory, [_receipt(row, 1)])
    result = HARNESS._validate_local(
        HARNESS.load_receipts(directory)[0], HARNESS._source_tree_digest(source)
    )
    assert result["passed"] is False
    assert "source_binding_mismatch" in result["reasons"]
    receipt_path = directory / "receipts-000.json"
    payload = json.loads(receipt_path.read_text())
    payload[0]["seal_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(payload))
    with pytest.raises(HARNESS.R4Error, match="seal"):
        HARNESS.load_receipts(directory)


def test_root_matrix_rejects_symlink_and_overlap(tmp_path: Path) -> None:
    source, _ = _git_source(tmp_path)
    output = tmp_path / "output"
    HARNESS.assert_root_matrix(source, output)
    with pytest.raises(HARNESS.R4Error, match="overlap"):
        HARNESS.assert_root_matrix(source, source / "nested")
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(HARNESS.R4Error, match="symlink"):
        HARNESS.assert_root_matrix(link, output)


def test_run_rejects_output_symlink_before_resolve(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    output = tmp_path / "output"
    output.symlink_to(external, target_is_directory=True)
    with pytest.raises(HARNESS.R4Error, match="symlink"):
        HARNESS.run(source_root=source, source_commit=commit, output=output)


def test_tracked_source_symlink_is_rejected(tmp_path: Path) -> None:
    source, _ = _git_source(tmp_path)
    (source / "target.txt").write_text("target\n")
    (source / "tracked-link").symlink_to("target.txt")
    subprocess.run(["git", "add", "target.txt", "tracked-link"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "symlink"], cwd=source, check=True)
    with pytest.raises(HARNESS.R4Error, match="tracked symlink"):
        HARNESS._source_tree_digest(source)


def test_dirty_source_is_rejected_before_receipt_reads(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    (source / "README.md").write_text("dirty\n")
    with pytest.raises(HARNESS.R4Error, match="dirty"):
        HARNESS.run(
            source_root=source, source_commit=commit, output=tmp_path / "evidence"
        )


def test_local_skew_and_probe_revision_are_fixed_gates(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    rows = _local_rows(source, commit)
    quality = [row for row in rows if row.get("failure_injection") is not True]
    selected: list[dict[str, object]] = []
    for role, _limit in zip(HARNESS.LOCAL_ROLES, (102, 5, 1), strict=True):
        selected.extend(row for row in quality if row["primary_owner"] == role)
        selected = selected[: sum((102, 5, 1)[: HARNESS.LOCAL_ROLES.index(role) + 1])]
    skew_result = HARNESS._validate_local(selected, HARNESS._source_tree_digest(source))
    assert skew_result["passed"] is False
    assert "load_skew_above_p0" in skew_result["reasons"]
    probe_row = dict(rows[-1])
    probe_row["probe_assignment_revision"] = "probe-unknown"
    probe_result = HARNESS._validate_local(
        [*rows[:-1], probe_row], HARNESS._source_tree_digest(source)
    )
    assert probe_result["passed"] is False
    assert "probe_assignment_missing" in probe_result["reasons"]


def test_ox_negative_aliases_and_invalid_backoff_are_vetoes(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    source_snapshot = HARNESS._source_tree_digest(source)
    for alias in (
        "paid",
        "paid_calls",
        "drift",
        "route_model_drift",
        "model_drift",
        "dup",
        "duplicate",
    ):
        rows = _ox_rows(source, commit)
        rows[0][alias] = 1
        result = HARNESS._validate_ox(rows, source_snapshot)
        assert result["passed"] is False
        assert any(alias in reason for reason in result["reasons"])
    rows = _ox_rows(source, commit)
    rows[0]["transition_receipts"][0]["before_cap"] = -1  # type: ignore[index]
    result = HARNESS._validate_ox(rows, source_snapshot)
    assert result["passed"] is False
    assert "429_halving_invalid" in result["reasons"]


def test_ox_ramp_order_and_capture_time_are_fixed_gates(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    source_snapshot = HARNESS._source_tree_digest(source)
    rows = list(reversed(_ox_rows(source, commit)))
    result = HARNESS._validate_ox(rows, source_snapshot)
    assert result["passed"] is False
    assert "ox_ramp_order_invalid" in result["reasons"]
    assert "ox_captured_at_not_monotonic" in result["reasons"]


def test_production_boolean_is_not_an_attestation(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    result = HARNESS._validate_production_attestations(
        [{"production_certification": True, "producer": "fixture"}],
        HARNESS._source_tree_digest(source),
    )
    assert result["passed"] is False
    assert result["reasons"]


def test_self_reported_attestation_is_rejected(tmp_path: Path) -> None:
    source, _ = _git_source(tmp_path)
    result = HARNESS._validate_production_attestations(
        [{"production_certification": True, "producer": "fixture"}],
        HARNESS._source_tree_digest(source),
    )
    assert result["passed"] is False
    assert result["reasons"] == ["independent_live_provider_attestation_unavailable"]


def test_default_cli_requires_production_certification(tmp_path: Path) -> None:
    source, commit = _git_source(tmp_path)
    local_dir = tmp_path / "local"
    ox_dir = tmp_path / "ox"
    _write_receipts(
        local_dir,
        [_receipt(row, index) for index, row in enumerate(_local_rows(source, commit))],
    )
    _write_receipts(
        ox_dir,
        [
            _receipt(row, index + 100)
            for index, row in enumerate(_ox_rows(source, commit))
        ],
    )
    common = [
        "--source-root",
        str(source),
        "--source-commit",
        commit,
        "--local-receipts",
        str(local_dir),
        "--ox-receipts",
        str(ox_dir),
    ]
    result = HARNESS.main(
        [
            *common,
            "--output",
            str(tmp_path / "evidence"),
        ]
    )
    assert result == 3
    source_only = HARNESS.main(
        [
            *common,
            "--output",
            str(tmp_path / "source-only-evidence"),
            "--source-contract-only",
        ]
    )
    assert source_only == 0
    child = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "recall_r4_harness.py"),
            *common,
            "--output",
            str(tmp_path / "fresh-subprocess-evidence"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 3, child.stderr
    child_payload = json.loads(child.stdout)
    assert child_payload["source_contract"] is True
    assert child_payload["production_certification"] is False
