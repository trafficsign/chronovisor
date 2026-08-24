from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
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


def _git_source(tmp_path: Path) -> tuple[Path, str]:
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
        "prompt_sha256": HARNESS.OX_PROMPT_SHA256,
        "schema": HARNESS.OX_SCHEMA,
        "schema_sha256": HARNESS.OX_SCHEMA_SHA256,
        "route_sha256": HARNESS.OX_ROUTE_SHA256,
        "model_sha256": HARNESS.OX_MODEL_SHA256,
        "cohort": HARNESS.OX_COHORT,
        "identity_revision": HARNESS.OX_IDENTITY_REVISION,
        "expires_at": "2099-01-01T00:00:00Z",
        "source_identity_sha256": source_identity,
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
            "expires_at",
        )
    }
    contract["contract_id"] = HARNESS._sha256(contract_identity)
    rows: list[dict[str, object]] = []
    for cap in HARNESS.OX_STAGES:
        rows.append(
            {
                "profile": HARNESS.OX_PROFILE,
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
                    "labels": [
                        {
                            "label_id": f"label-{cap}-{index}",
                            "commit_id": f"commit-{cap}-{index}",
                            "work_id": f"work-{cap}-{index}",
                        }
                        for index in range(20)
                    ],
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
    """Build a tiny sealed production-shaped fixture for collector tests."""

    root = tmp_path / "production"
    distill = root / "runtime" / "recall-distillation"
    contracts = distill / "ox-profile-contracts"
    contracts.mkdir(parents=True)
    config = """[recall.distillation]
enabled = true
teacher_profile = "ox-alpha-single-v1"
teacher_max_inflight = 10
teacher_claim_limit = 1
ox_enabled = true
ox_free_only = true
max_input_bytes = 12000
max_candidates = 200
"""
    config_path = root / "config.toml"
    config_path.write_text(config)
    config_sha = HARNESS._file_sha256(config_path)
    relevant = {
        "teacher_profile": HARNESS.OX_PROFILE,
        "teacher_max_inflight": 10,
        "ox_enabled": True,
        "ox_free_only": True,
        "max_input_bytes": 12000,
        "max_candidates": 200,
    }
    contract_unsigned: dict[str, object] = {
        "cohort": HARNESS.OX_COHORT,
        "expires_at": "2099-01-01T00:00:00Z",
        "free_only": True,
        "kind": "ox-alpha-free-profile",
        "live_recall_model_calls": 0,
        "max_inflight": 10,
        "no_paid_fallback": True,
        "profile": HARNESS.OX_PROFILE,
        "relevant_config_sha256": HARNESS._sha256(relevant),
        "request_model": HARNESS.OX_MODEL,
        "required_returned_model": HARNESS.OX_MODEL,
        "route": HARNESS.OX_ROUTE,
        "schema": "chronovisor.recall-distill-ox-profile.v1",
    }
    contract_unsigned = {
        "schema": "chronovisor.recall-distill-ox-profile.v1",
        "namespace": "recall-distillation",
        **contract_unsigned,
    }
    contract_id = HARNESS._sha256(contract_unsigned)
    contract = {"artifact_id": contract_id, **contract_unsigned}
    contract["seal_sha256"] = HARNESS._sha256(
        {key: value for key, value in contract.items() if key != "seal_sha256"}
    )
    contract_path = contracts / f"{contract_id}.json"
    contract_path.write_text(
        json.dumps(contract, sort_keys=True, separators=(",", ":"))
    )

    tree = HARNESS._source_tree_digest(source)["tree_sha256"]
    work_rows: list[tuple[object, ...]] = []
    label_rows: list[dict[str, object]] = []
    previous = ""
    stage_work_ids: dict[int, list[str]] = {cap: [] for cap in HARNESS.OX_STAGES}
    for cap in HARNESS.OX_STAGES:
        for index in range(20):
            work_id = HARNESS.hashlib.sha256(f"work-{cap}-{index}".encode()).hexdigest()
            unsigned: dict[str, object] = {
                "cohort": HARNESS.OX_COHORT,
                "dimension": "relevance",
                "identity_revision": HARNESS.OX_IDENTITY_REVISION,
                "kind": "teacher-label",
                "model_digest": HARNESS.OX_MODEL_SHA256,
                "namespace": "recall-distillation",
                "previous_sha256": previous,
                "profile": HARNESS.OX_PROFILE,
                "profile_contract_id": contract_id,
                "prompt_sha256": HARNESS.OX_PROMPT_SHA256,
                "ramp_cap": cap,
                "route": HARNESS.OX_ROUTE,
                "route_digest": HARNESS.OX_ROUTE_SHA256,
                "route_identity": {
                    "location": "remote",
                    "model": HARNESS.OX_ROUTE,
                    "provider": "opencode-go",
                },
                "schema": "chronovisor.recall-distillation.v1",
                "schema_sha256": HARNESS.OX_SCHEMA_SHA256,
                "source_ox_identity_sha256": HARNESS._source_tree_digest(source)[
                    "ox_identity_sha256"
                ],
                "source_commit": commit,
                "source_tree_sha256": tree,
                "status": "completed",
                "teacher_role": "recall.distill.teacher.ox-alpha",
                "work_id": work_id,
                "attempt_count": 1,
                "label_id": HARNESS.hashlib.sha256(
                    f"label-{work_id}".encode()
                ).hexdigest(),
                "commit_id": HARNESS.hashlib.sha256(
                    f"commit-{work_id}".encode()
                ).hexdigest(),
            }
            digest = HARNESS._sha256(unsigned)
            row = {**unsigned, "record_sha256": digest}
            label_rows.append(row)
            previous = digest
            stage_work_ids[cap].append(work_id)
            payload_ref = f"candidate-ledger:{work_id}"
            payload_digest = HARNESS.hashlib.sha256(
                f"payload-{work_id}".encode()
            ).hexdigest()
            provenance = json.dumps(
                {
                    "cohort": HARNESS.OX_COHORT,
                    "profile": HARNESS.OX_PROFILE,
                    "profile_contract_id": contract_id,
                    "route": HARNESS.OX_ROUTE,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            work_rows.append(
                (
                    len(work_rows) + 1,
                    work_id,
                    "ox",
                    payload_ref,
                    payload_digest,
                    json.dumps(
                        {"split": "train", "split_plan_id": "split"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    provenance,
                    0,
                    "completed",
                    1,
                    f"label-ledger:{digest}",
                    digest,
                )
            )
    label_path = distill / "label-ledger.jsonl"
    label_path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in label_rows
        )
    )
    label_file_state = HARNESS._production_stat(label_path, label="fixture labels")
    label_checkpoint = {
        "kind": "ledger-chain-checkpoint",
        "ledger_name": "label-ledger.jsonl",
        "namespace": "recall-distillation",
        "schema": "chronovisor.recall-distillation.v1",
        "records": len(label_rows),
        "head_sha256": previous,
        "file_state": {
            "size_bytes": label_file_state["st_size"],
            "st_dev": label_file_state["st_dev"],
            "st_ino": label_file_state["st_ino"],
            "st_mtime_ns": label_file_state["st_mtime_ns"],
            "st_ctime_ns": label_file_state["st_ctime_ns"],
        },
    }
    label_checkpoint["seal_sha256"] = HARNESS._sha256(label_checkpoint)
    (distill / "label-ledger.jsonl.head.json").write_text(
        json.dumps(label_checkpoint, sort_keys=True, separators=(",", ":"))
    )
    candidate_row_unsigned = {
        "candidate_id": "candidate-fixture",
        "namespace": "recall-distillation",
        "previous_sha256": "",
        "schema": "chronovisor.recall-distillation.v1",
    }
    candidate_row = {
        **candidate_row_unsigned,
        "record_sha256": HARNESS._sha256(candidate_row_unsigned),
    }
    candidate_path = distill / "candidate-ledger.jsonl"
    candidate_path.write_bytes(
        json.dumps(candidate_row, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    candidate_file_state = HARNESS._production_stat(
        candidate_path, label="fixture candidates"
    )
    candidate_checkpoint = {
        "kind": "ledger-chain-checkpoint",
        "ledger_name": "candidate-ledger.jsonl",
        "namespace": "recall-distillation",
        "schema": "chronovisor.recall-distillation.v1",
        "records": 1,
        "head_sha256": candidate_row["record_sha256"],
        "file_state": {
            "size_bytes": candidate_file_state["st_size"],
            "st_dev": candidate_file_state["st_dev"],
            "st_ino": candidate_file_state["st_ino"],
            "st_mtime_ns": candidate_file_state["st_mtime_ns"],
            "st_ctime_ns": candidate_file_state["st_ctime_ns"],
        },
    }
    candidate_checkpoint["seal_sha256"] = HARNESS._sha256(candidate_checkpoint)
    (distill / "candidate-ledger.jsonl.head.json").write_text(
        json.dumps(candidate_checkpoint, sort_keys=True, separators=(",", ":"))
    )

    workset_path = distill / "ox-workset.sqlite3"
    watermark = {"label_head": previous, "source_commit": commit}
    with sqlite3.connect(workset_path) as connection:
        connection.executescript(
            """
            CREATE TABLE work_items (
                sequence INTEGER PRIMARY KEY,
                work_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                payload_ref TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                temporal_split_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                priority INTEGER NOT NULL,
                state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                completion_ref TEXT NOT NULL,
                completion_digest TEXT NOT NULL
            );
            CREATE TABLE workset_state (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            CREATE TABLE workset_receipts (
                generation INTEGER PRIMARY KEY,
                previous_sha256 TEXT NOT NULL,
                operation TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL UNIQUE
            );
            """
        )
        connection.executemany(
            "INSERT INTO work_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            work_rows,
        )
        connection.execute(
            "INSERT INTO workset_state VALUES ('watermark', ?)",
            (json.dumps(watermark, sort_keys=True, separators=(",", ":")),),
        )
        payload = {
            "after": {
                "counts": {"completed": 80, "leased": 0, "quarantined": 0, "ready": 0},
                "watermark": watermark,
            },
            "before": {
                "counts": {"completed": 0, "leased": 80, "quarantined": 0, "ready": 0},
                "watermark": None,
            },
            "delta": {"completed": 80, "leased": -80, "quarantined": 0, "ready": 0},
            "details": {
                "completed": 80,
                "quarantined": 0,
                "retry": 0,
                "selection_sha256": "0" * 64,
            },
        }
        envelope = {
            "generation": 1,
            "previous_sha256": "",
            "operation": "commit",
            "payload": payload,
        }
        connection.execute(
            "INSERT INTO workset_receipts VALUES (?, ?, ?, ?, ?)",
            (
                1,
                "",
                "commit",
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                HARNESS._sha256(envelope),
            ),
        )

    workset_sha = HARNESS._file_sha256(workset_path)
    label_sha = HARNESS._file_sha256(label_path)
    contract_sha = HARNESS._file_sha256(contract_path)
    os_identity = {
        "system": __import__("os").uname().sysname,
        "release": __import__("os").uname().release,
        "machine": __import__("os").uname().machine,
    }
    state: dict[str, object] = {
        "kind": "worker-state",
        "namespace": "recall-distillation",
        "profile_contract_id": contract_id,
        "source_commit": commit,
        "source_tree_sha256": tree,
        "runtime_identity": {
            "root": str(root.absolute()),
            "account_uid": HARNESS.ACCOUNT_UID,
            "account_home": str(HARNESS.ACCOUNT_HOME),
            "source_commit": commit,
            "source_tree_sha256": tree,
            "source_ox_identity_sha256": HARNESS._source_tree_digest(source)[
                "ox_identity_sha256"
            ],
            "config_sha256": config_sha,
            "workset_sha256": workset_sha,
            "label_sha256": label_sha,
            "label_checkpoint_records": len(label_rows),
            "label_checkpoint_file_state": label_checkpoint["file_state"],
            "candidate_checkpoint_head": candidate_checkpoint["head_sha256"],
            "candidate_checkpoint_records": 1,
            "candidate_checkpoint_file_state": candidate_checkpoint["file_state"],
            "candidate_anchor_artifact_id": "a" * 64,
            "candidate_anchor_head_sha256": candidate_checkpoint["head_sha256"],
            "candidate_anchor_records": 1,
            "candidate_anchor_bytes": candidate_file_state["st_size"],
            "candidate_tail_records": 0,
            "candidate_tail_bytes": 0,
            "profile_contract_sha256": contract_sha,
            "workset_receipt_head": HARNESS._sha256(envelope),
            "label_receipt_head": previous,
            "os_identity": os_identity,
        },
        "ramp_receipts": [
            {
                "cap": cap,
                "valid_receipts": 20,
                "attempts": 20,
                "work_ids": stage_work_ids[cap],
                "source_commit": commit,
                "profile_contract_id": contract_id,
            }
            for cap in HARNESS.OX_STAGES
        ],
        "quality_gates": {
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
        },
        "failure_receipts": [
            {"category": "429", "before_cap": 10, "after_cap": 5, "status": "deferred"},
            {"category": "5xx", "attempts": 3, "bounded": True, "status": "deferred"},
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
        "lease_recovery": {"leased_after": 0, "recovered": 0},
    }
    state["schema"] = "chronovisor.recall-distillation.v1"
    state["seal_sha256"] = HARNESS._sha256(
        {key: value for key, value in state.items() if key != "seal_sha256"}
    )
    (distill / "state.json").write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":"))
    )
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

    monkeypatch.setattr(HARNESS, "_collect_authoritative_production", mutating_collector)
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
    real_replace = HARNESS.os.replace
    mutated = False

    def mutating_replace(source_path: str | bytes, destination: str | bytes) -> None:
        nonlocal mutated
        if not mutated:
            mutated = True
            (source / "README.md").write_text("mutated during publication\n")
        real_replace(source_path, destination)

    monkeypatch.setattr(HARNESS.os, "replace", mutating_replace)
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


def test_r0_candidate_anchor_is_sealed_and_tamper_evident(tmp_path: Path) -> None:
    source = tmp_path / "source"
    anchor_path = source / HARNESS.R0_EVIDENCE_RELATIVE
    anchor_path.parent.mkdir(parents=True)
    tracked = ROOT / HARNESS.R0_EVIDENCE_RELATIVE
    anchor_path.write_bytes(tracked.read_bytes())
    anchor = HARNESS._load_production_anchor(source)
    assert anchor["artifact_id"] == HARNESS.R0_EVIDENCE_ID
    assert anchor["candidate"]["records"] == 8050
    original = anchor_path.read_bytes()
    anchor_path.write_bytes(original.replace(b'"records":8050', b'"records":8051', 1))
    with pytest.raises(HARNESS.R4Error, match="seal|digest"):
        HARNESS._load_production_anchor(source)


def test_candidate_checkpoint_reseal_and_same_size_substitution_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    anchor = _fixture_candidate_anchor(production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _: anchor
    )
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
    checkpoint_path.write_text(json.dumps(checkpoint, sort_keys=True, separators=(",", ":")))
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
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _: anchor
    )
    candidate_path = production / HARNESS.PRODUCTION_CANDIDATE_RELATIVE
    checkpoint_path = production / HARNESS.PRODUCTION_CANDIDATE_CHECKPOINT_RELATIVE
    first = json.loads(candidate_path.read_text())
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
    checkpoint_path.write_text(json.dumps(checkpoint, sort_keys=True, separators=(",", ":")))
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=HARNESS._assert_source(source, commit),
        production_root=production,
    )
    assert result["passed"] is False
    assert "production candidate ledger differs from sealed R0 anchor" in result["reasons"]


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
        {
            key: value
            for key, value in replacement_state.items()
            if key != "seal_sha256"
        }
    )
    replacement_state_path.write_text(
        json.dumps(replacement_state, sort_keys=True, separators=(",", ":"))
    )
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS,
        "_load_production_anchor",
        lambda _: _fixture_candidate_anchor(production),
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
        return real_directory_identity(path, label=label)

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


def test_authoritative_collector_can_certify_only_fixed_sealed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unit-only root injection exercises the closed schema; the public CLI
    # cannot supply this path and always reads the fixed managed root.
    source, commit = _git_source(tmp_path)
    production = _authoritative_production_root(tmp_path, source, commit)
    monkeypatch.setattr(HARNESS, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(
        HARNESS, "_load_production_anchor", lambda _: _fixture_candidate_anchor(production)
    )
    source_snapshot = HARNESS._assert_source(source, commit)
    result = HARNESS._collect_authoritative_production(
        source_root=source,
        source=source_snapshot,
        production_root=production,
    )
    assert result["passed"] is True
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
        HARNESS, "_load_production_anchor", lambda _: _fixture_candidate_anchor(production)
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
        HARNESS, "_load_production_anchor", lambda _: _fixture_candidate_anchor(truncated)
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
        HARNESS, "_load_production_anchor", lambda _: _fixture_candidate_anchor(spoofed)
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
        HARNESS, "_load_production_anchor", lambda _: _fixture_candidate_anchor(vetoed)
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
