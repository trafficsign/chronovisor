from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
import tomllib
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from chronovisor.core import canonical_json
from chronovisor.core.legacy_archive import write_legacy_archive
from chronovisor.core.raw_segment import append_capture
from chronovisor.core.store import RuntimeContext, init_chronovisor
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall import recall_distillation_workset as workset
from chronovisor.recall.recall_distillation_remote_teacher import (
    ox_alpha_response_metadata,
)


@pytest.fixture(autouse=True)
def _trusted_ox_test_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy in-process stubs outside the production egress authority."""

    binding = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    monkeypatch.setattr(distill, "ox_alpha_source_binding", lambda: dict(binding))
    monkeypatch.setattr(
        distill, "_ox_teacher_source_binding", lambda _teacher: dict(binding)
    )
    monkeypatch.setattr(distill, "_ox_source_binding_matches", lambda *_args: True)


@pytest.fixture(autouse=True)
def _bootstrap_ox_workset_for_direct_batch_tests(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Direct batch tests own an explicit queue bootstrap; workers never migrate."""

    nested_batch_tests = {
        "test_ox_single_teacher_uncertain_output_completes_as_non_training_abstention",
        "test_ox_missing_payload_quarantines_without_remote_call",
    }
    if (
        "_run_teacher_batch" in request.node.function.__code__.co_names
        or request.node.originalname in nested_batch_tests
    ):
        workset.DistillationWorkset(
            store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
        )


def _ox_metadata(payload: object) -> dict[str, object]:
    assert isinstance(payload, dict)
    metadata = ox_alpha_response_metadata(payload)
    assert metadata is not None
    return {**metadata, "_provider_receipt_sha256": "e" * 64}


def _ox_projection_payload_source(candidate_id: str = "candidate") -> dict[str, str]:
    return {"candidate_id": candidate_id, "rally_id": "rally"}


def _ox_projection_label_identity() -> dict[str, Any]:
    return {
        "cohort": distill.OX_SINGLE_COHORT,
        "route": "opencode-go/deepseek-v4-flash",
        "teacher_role": distill.OX_TEACHER_ROLE,
        "identity_revision": distill.OX_ALPHA_FIXED_IDENTITY["revision"],
        "route_identity": dict(distill.OX_ALPHA_FIXED_IDENTITY["route_identity"]),
        "route_digest": distill.OX_ALPHA_FIXED_IDENTITY["route_digest"],
        "model_digest": distill.OX_ALPHA_FIXED_IDENTITY["model_digest"],
        "prompt_sha256": distill.OX_ALPHA_FIXED_IDENTITY["prompt_template_sha256"],
        "schema_sha256": distill.OX_ALPHA_FIXED_IDENTITY["schema_revision_sha256"],
        "test_only": False,
    }


def _ox_projection_work(
    tmp_path: Path, *, max_inflight: int = 10
) -> tuple[dict[str, str], dict[str, Any], str, str]:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            teacher_claim_limit=1,
            teacher_max_inflight=max_inflight,
        ),
        source_binding=source,
    )
    payload_digest = canonical_json.canonical_json_sha256_strict(
        _ox_projection_payload_source()
    )
    work_id = canonical_json.canonical_json_sha256_strict(
        {
            "kind": "ox-teacher-label-v1",
            "profile": distill.OX_SINGLE_PROFILE,
            "cohort": distill.OX_SINGLE_COHORT,
            "route": "opencode-go/deepseek-v4-flash",
            "profile_contract_id": contract["artifact_id"],
            "payload_digest": payload_digest,
        }
    )
    workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    ).advance(
        [
            {
                "work_id": work_id,
                "kind": "ox",
                "payload_ref": "candidate-snapshot:rally:candidate",
                "payload_digest": payload_digest,
                "temporal_split": {"split": "embargo"},
                "provenance": {
                    "profile": distill.OX_SINGLE_PROFILE,
                    "cohort": distill.OX_SINGLE_COHORT,
                    "route": "opencode-go/deepseek-v4-flash",
                    "teacher_role": distill.OX_TEACHER_ROLE,
                    "profile_contract_id": contract["artifact_id"],
                    "probe": False,
                },
            }
        ],
        {"candidate_records": 1},
    )
    return source, contract, work_id, payload_digest


def _complete_projection_label(tmp_path: Path, work_id: str, label: Mapping[str, Any]) -> None:
    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    )
    claim = queue.claim("ox", 1, "test", 60)[0]
    assert claim.work_id == work_id
    record_sha256 = str(label["record_sha256"])
    queue.commit(
        [claim],
        [
            {
                "status": "completed",
                "completion_ref": f"label-ledger:{record_sha256}",
                "completion_digest": record_sha256,
            }
        ],
    )


def _ox_projection_label_record(
    source: Mapping[str, str], contract: Mapping[str, Any], work_id: str, payload_digest: str
) -> dict[str, Any]:
    return {
        "kind": "teacher-label",
        "status": "completed",
        "work_id": work_id,
        "payload_digest": payload_digest,
        "payload_source": _ox_projection_payload_source(),
        "attempt_count": 1,
        "ramp_cap": 1,
        "profile": distill.OX_SINGLE_PROFILE,
        "profile_contract_id": contract["artifact_id"],
        **_ox_projection_label_identity(),
        **source,
        "request_revision": distill.OX_RAMP_REQUEST_REVISION,
        "expires_at": contract["expires_at"],
        "provider_request_sha256": distill.expected_ox_provider_request_sha256(
            profile_contract_id=str(contract["artifact_id"]),
            payload_digest=payload_digest,
            work_id=work_id,
            expires_at=str(contract["expires_at"]),
        ),
        "provider_receipt_sha256": "f" * 64,
        "request_sha256": distill.expected_ox_request_sha256(
            profile_contract_id=str(contract["artifact_id"]),
            payload_digest=payload_digest,
        ),
    }


def test_ox_receipt_projection_cannot_promote_shallow_forgery(tmp_path: Path) -> None:
    """One forged label/pointer must not replace the authoritative gate."""

    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
        ),
        source_binding=source,
    )
    profile_contract_id = str(contract["artifact_id"])
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile_contract_id": profile_contract_id,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            **source,
            "assignment": {"repeat_pair_id": "fake", "fixed_repeat": True},
        },
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "active-policy.json",
        {"kind": "shallow-forgery", "artifact_id": "e" * 64},
    )
    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=profile_contract_id,
        source_binding=source,
        workset={"leased": 0},
        label_path=label_path,
        authoritative_gate={"passed": True, "reasons": []},
    )
    quality = projection["quality_gates"]
    assert quality["passed"] is False
    assert "ramp_receipts_incomplete" in quality["reasons"]
    assert "failure_receipts_incomplete" in quality["reasons"]


def test_ox_event_projection_keeps_legacy_provider_receipts_noncertifying(
    tmp_path: Path,
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        ox_free_only=False,
        ox_expires_at="2099-01-01T00:00:00Z",
        teacher_claim_limit=1,
    )
    contract = distill._ensure_ox_profile_contract(
        tmp_path, config, source_binding=source
    )
    payload_digest = "d" * 64
    work_id = canonical_json.canonical_json_sha256_strict(
        {
            "kind": "ox-teacher-label-v1",
            "profile": distill.OX_SINGLE_PROFILE,
            "cohort": distill.OX_SINGLE_COHORT,
            "route": "opencode-go/deepseek-v4-flash",
            "profile_contract_id": contract["artifact_id"],
            "payload_digest": payload_digest,
        }
    )
    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    )
    queue.advance(
        [
            {
                "work_id": work_id,
                "kind": "ox",
                "payload_ref": "candidate-snapshot:rally:candidate",
                "payload_digest": payload_digest,
                "temporal_split": {
                    "as_of": "2026-01-01T00:00:00Z",
                    "group_id": "group",
                    "split": "embargo",
                    "split_plan_id": "",
                },
                "provenance": {
                    "profile": distill.OX_SINGLE_PROFILE,
                    "cohort": distill.OX_SINGLE_COHORT,
                    "route": "opencode-go/deepseek-v4-flash",
                    "teacher_role": distill.OX_TEACHER_ROLE,
                    "profile_contract_id": contract["artifact_id"],
                    "probe": False,
                },
            }
        ],
        {
            "candidate_records": 1,
            "candidate_head": "e" * 64,
            "split_plan_id": "",
            "probe_revision": distill.OX_PROBE_REVISION,
        },
    )
    forged_receipt = "f" * 64
    distill._append_ox_event(
        tmp_path,
        "ox-failure-receipts.jsonl",
        {
            "kind": "ox-provider-failure",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "category": "5xx",
            "status": "deferred",
            "attempts": 1,
            "bounded": True,
            "before_cap": 1,
            "after_cap": 1,
            "work_ids": [work_id],
            "attempts_by_work": {work_id: 1},
            "provider_receipts": {work_id: forged_receipt},
            "captured_at": "2026-08-25T00:00:00Z",
        },
    )
    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        workset={"leased": 0},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        authoritative_gate={"passed": True, "reasons": []},
    )
    assert projection["failure_receipts"] == []


def test_ox_event_projection_keeps_legacy_workset_noncertifying(
    tmp_path: Path,
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
        ),
        source_binding=source,
    )
    workset_path = store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    queue = workset.DistillationWorkset(workset_path)
    queue.advance([], {"candidate_records": 0})
    with sqlite3.connect(workset_path) as connection:
        connection.execute("DELETE FROM workset_receipts")

    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        workset={"leased": 0},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        authoritative_gate={"passed": False, "reasons": ["test"]},
    )

    assert "workset_receipts_noncertifying" in projection["quality_gates"]["reasons"]


def test_ox_label_projection_ignores_legacy_contract_before_validating_current_labels(
    tmp_path: Path,
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": "f" * 64,
        },
    )

    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        workset={"leased": 0},
        label_path=label_path,
        authoritative_gate={"passed": False, "reasons": ["test"]},
    )

    assert projection["quality_gates"]["negative_veto"]["authenticated"] is False
    store.append_chain(
        label_path,
        {
            "kind": "other",
            "status": "completed",
            "work_id": work_id,
            "payload_digest": payload_digest,
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
        },
    )

    with pytest.raises(distill.DistillationError, match="kind or status is invalid"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize("contract_id", [None, "", "not-a-digest", "F" * 64])
def test_ox_label_projection_rejects_malformed_contract_ids(
    tmp_path: Path, contract_id: object
) -> None:
    source, contract, _, _ = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract_id,
        },
    )

    with pytest.raises(distill.DistillationError, match="profile contract is invalid"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": False, "reasons": ["test"]},
        )


def test_ox_label_projection_rejects_retired_provider_receipt_key(
    tmp_path: Path,
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            teacher_claim_limit=1,
        ),
        source_binding=source,
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            **source,
            "provider_response_request_sha256": "d" * 64,
        },
    )

    with pytest.raises(distill.DistillationError, match="retired provider receipt"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize("synthetic", [True, False])
def test_ox_failure_projection_requires_distinct_actual_receipt(
    tmp_path: Path, synthetic: bool
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    distill._append_ox_event(
        tmp_path,
        "ox-failure-receipts.jsonl",
        {
                "event_version": 2,
                "kind": "ox-provider-failure",
                "profile_contract_id": contract["artifact_id"],
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                "cap": 1,
                "category": "5xx",
                "status": "deferred",
                "attempts": 1,
                "bounded": True,
                "work_ids": [work_id],
                "attempts_by_work": {work_id: 1},
                "provider_requests": {work_id: provider_request},
                "provider_receipts": {
                    work_id: provider_request if synthetic else "f" * 64
                },
                "captured_at": "2026-08-25T00:00:00Z",
        },
    )
    kwargs = {
        "profile_contract_id": str(contract["artifact_id"]),
        "source_binding": source,
        "workset": {"leased": 0},
        "label_path": store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        "authoritative_gate": {"passed": True, "reasons": []},
    }
    if synthetic:
        with pytest.raises(distill.DistillationError, match="receipt is synthetic"):
            distill._ox_event_projection(tmp_path, **kwargs)
    else:
        assert len(distill._ox_event_projection(tmp_path, **kwargs)["failure_receipts"]) == 1


@pytest.mark.parametrize("category", ["402", "paid", "model_drift"])
@pytest.mark.parametrize("status", ["deferred", "hard_stop"])
def test_ox_failure_projection_requires_hard_stop_for_terminal_categories(
    tmp_path: Path, category: str, status: str
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    distill._append_ox_event(
        tmp_path,
        "ox-failure-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-provider-failure",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": 1,
            "category": category,
            "status": status,
            "attempts": 1,
            "work_ids": [work_id],
            "attempts_by_work": {work_id: 1},
            "provider_requests": {work_id: provider_request},
            "provider_receipts": {work_id: "f" * 64},
            "captured_at": "2026-08-25T00:00:00Z",
        },
    )
    kwargs = {
        "profile_contract_id": str(contract["artifact_id"]),
        "source_binding": source,
        "workset": {"leased": 0},
        "label_path": store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        "authoritative_gate": {"passed": True, "reasons": []},
    }
    if status == "deferred":
        with pytest.raises(distill.DistillationError, match="failure event fields"):
            distill._ox_event_projection(tmp_path, **kwargs)
    else:
        assert distill._ox_event_projection(tmp_path, **kwargs)["failure_receipts"][0][
            "status"
        ] == "hard_stop"


@pytest.mark.parametrize("cap, wrong_next_cap", [(1, 1), (2, 1), (5, 2), (10, 5)])
def test_ox_ramp_projection_rejects_wrong_next_cap(
    tmp_path: Path, cap: int, wrong_next_cap: int
) -> None:
    source, contract, _work_id, _payload_digest = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    label = store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )
    distill._append_ox_event(
        tmp_path,
        "ox-ramp-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-ramp-stage",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": cap,
            "next_cap": wrong_next_cap,
            "valid_receipts": 0,
            "attempts": 0,
            "work_ids": [],
            "label_count": 1,
            "label_head_sha256": label["record_sha256"],
            "failure_record_count": 0,
            "failure_head_sha256": "",
            "captured_at": f"2026-08-25T00:00:0{cap}Z",
        },
    )

    with pytest.raises(distill.DistillationError, match="ramp event fields"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize("synthetic", [True, False])
def test_ox_label_projection_requires_distinct_actual_receipt(
    tmp_path: Path, synthetic: bool
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    label = store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "work_id": work_id,
            "payload_digest": payload_digest,
            "attempt_count": 1,
            "ramp_cap": 1,
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            "payload_source": _ox_projection_payload_source(),
            **_ox_projection_label_identity(),
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "provider_request_sha256": provider_request,
            "provider_receipt_sha256": provider_request if synthetic else "f" * 64,
            "request_sha256": distill.expected_ox_request_sha256(
                profile_contract_id=str(contract["artifact_id"]),
                payload_digest=payload_digest,
            ),
        },
    )
    if not synthetic:
        _complete_projection_label(tmp_path, work_id, label)
    kwargs = {
        "profile_contract_id": str(contract["artifact_id"]),
        "source_binding": source,
        "workset": {"leased": 0},
        "label_path": label_path,
        "authoritative_gate": {"passed": True, "reasons": []},
    }
    if synthetic:
        with pytest.raises(distill.DistillationError, match="receipt is synthetic"):
            distill._ox_event_projection(tmp_path, **kwargs)
    else:
        assert distill._ox_event_projection(tmp_path, **kwargs)["quality_gates"][
            "negative_veto"
        ]["authenticated"] is True


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("cohort", "other", "producer identity"),
        ("route", "other", "producer identity"),
        ("teacher_role", "other", "producer identity"),
        ("identity_revision", "other", "producer identity"),
        ("route_identity", {}, "producer identity"),
        ("route_digest", "0" * 64, "producer identity"),
        ("model_digest", "0" * 64, "producer identity"),
        ("prompt_sha256", "0" * 64, "producer identity"),
        ("schema_sha256", "0" * 64, "producer identity"),
        ("test_only", True, "producer identity"),
        ("test_only_missing", None, "producer identity"),
        ("payload_source", {"candidate_id": "other", "rally_id": "rally"}, "payload binding"),
        ("work_id", "0" * 64, "provider request intent"),
    ],
)
def test_ox_label_projection_rejects_producer_identity_or_payload_binding_mutation(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    label = _ox_projection_label_record(source, contract, work_id, payload_digest)
    if field == "test_only_missing":
        label.pop("test_only")
    else:
        label[field] = value
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(label_path, label)

    with pytest.raises(distill.DistillationError, match=error):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize("mutation", ["extra", "missing", "wrong", "nonprobe_optional"])
def test_ox_label_projection_requires_exact_workset_provenance(
    tmp_path: Path, mutation: str
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    workset_path = store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    with sqlite3.connect(workset_path) as connection:
        provenance = json.loads(
            connection.execute(
                "SELECT provenance_json FROM work_items WHERE work_id = ?", (work_id,)
            ).fetchone()[0]
        )
        if mutation == "extra":
            provenance["extra"] = "forged"
        elif mutation == "missing":
            provenance.pop("route")
        elif mutation == "nonprobe_optional":
            provenance["repeat_pair_id"] = "d" * 64
        else:
            provenance["route"] = "forged"
        connection.execute(
            "UPDATE work_items SET provenance_json = ? WHERE work_id = ?",
            (json.dumps(provenance), work_id),
        )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        _ox_projection_label_record(source, contract, work_id, payload_digest),
    )

    with pytest.raises(distill.DistillationError, match="payload inventory|request intent"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


def test_ox_label_projection_accepts_exact_probe_workset_provenance(
    tmp_path: Path,
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    workset_path = store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    with sqlite3.connect(workset_path) as connection:
        connection.execute(
            "UPDATE work_items SET provenance_json = ? WHERE work_id = ?",
            (
                json.dumps(
                    {
                        "profile": distill.OX_SINGLE_PROFILE,
                        "cohort": distill.OX_SINGLE_COHORT,
                        "route": "opencode-go/deepseek-v4-flash",
                        "teacher_role": distill.OX_TEACHER_ROLE,
                        "profile_contract_id": contract["artifact_id"],
                        "probe": True,
                        "probe_revision": distill.OX_PROBE_REVISION,
                        "repeat_pair_id": "d" * 64,
                        "fixed_repeat": True,
                        "order_swap": True,
                        "blind_order": "a_first",
                        "probe_batch_id": "e" * 64,
                        "order_variant": 1,
                        "candidate_position": 0,
                    }
                ),
                work_id,
            ),
        )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        _ox_projection_label_record(source, contract, work_id, payload_digest),
    )

    with pytest.raises(distill.DistillationError, match="completion identity is unavailable"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


def test_ox_label_projection_rejects_missing_current_receipt(tmp_path: Path) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "work_id": work_id,
            "payload_digest": payload_digest,
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
        },
    )

    with pytest.raises(distill.DistillationError, match="provider receipt is invalid"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


def test_ox_label_projection_rejects_ready_workset_item(tmp_path: Path) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "work_id": work_id,
            "payload_digest": payload_digest,
            "attempt_count": 1,
            "ramp_cap": 1,
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            "payload_source": _ox_projection_payload_source(),
            **_ox_projection_label_identity(),
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "provider_request_sha256": provider_request,
            "provider_receipt_sha256": "f" * 64,
            "request_sha256": distill.expected_ox_request_sha256(
                profile_contract_id=str(contract["artifact_id"]),
                payload_digest=payload_digest,
            ),
        },
    )

    with pytest.raises(distill.DistillationError, match="completion identity is unavailable"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize("ramp_cap", [None, 3, "1"])
def test_ox_label_projection_rejects_missing_or_invalid_ramp_cap(
    tmp_path: Path, ramp_cap: object
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "work_id": work_id,
            "payload_digest": payload_digest,
            "attempt_count": 1,
            "ramp_cap": ramp_cap,
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            "payload_source": _ox_projection_payload_source(),
            **_ox_projection_label_identity(),
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "provider_request_sha256": provider_request,
            "provider_receipt_sha256": "f" * 64,
            "request_sha256": distill.expected_ox_request_sha256(
                profile_contract_id=str(contract["artifact_id"]),
                payload_digest=payload_digest,
            ),
        },
    )

    with pytest.raises(distill.DistillationError, match="ramp cap is invalid"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize(("kind", "status"), [("other", "completed"), ("teacher-label", "retry")])
def test_ox_label_projection_rejects_wrong_current_kind_or_status(
    tmp_path: Path, kind: str, status: str
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": kind,
            "status": status,
            "work_id": work_id,
            "payload_digest": payload_digest,
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
        },
    )

    with pytest.raises(distill.DistillationError, match="kind or status is invalid"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize(
    ("max_inflight", "ramp_caps"), [(10, (1, 2)), (1, (1, 1))]
)
def test_ox_projection_rejects_actual_receipt_reused_across_stages(
    tmp_path: Path, max_inflight: int, ramp_caps: tuple[int, int]
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(
        tmp_path, max_inflight=max_inflight
    )
    second_payload_source = _ox_projection_payload_source("candidate-2")
    second_payload_digest = canonical_json.canonical_json_sha256_strict(
        second_payload_source
    )
    second_work_id = canonical_json.canonical_json_sha256_strict(
        {
            "kind": "ox-teacher-label-v1",
            "profile": distill.OX_SINGLE_PROFILE,
            "cohort": distill.OX_SINGLE_COHORT,
            "route": "opencode-go/deepseek-v4-flash",
            "profile_contract_id": contract["artifact_id"],
            "payload_digest": second_payload_digest,
        }
    )
    workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    ).advance(
        [
            {
                "work_id": second_work_id,
                "kind": "ox",
                "payload_ref": "candidate-snapshot:rally:candidate-2",
                "payload_digest": second_payload_digest,
                "temporal_split": {"split": "embargo"},
                "provenance": {
                    "profile": distill.OX_SINGLE_PROFILE,
                    "cohort": distill.OX_SINGLE_COHORT,
                    "route": "opencode-go/deepseek-v4-flash",
                    "teacher_role": distill.OX_TEACHER_ROLE,
                    "profile_contract_id": contract["artifact_id"],
                    "probe": False,
                },
            }
        ],
        {"candidate_records": 2},
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    work_items = (
        (work_id, payload_digest, _ox_projection_payload_source()),
        (second_work_id, second_payload_digest, second_payload_source),
    )
    labels = []
    for (item_work_id, item_payload_digest, item_payload_source), ramp_cap in zip(
        work_items, ramp_caps, strict=True
    ):
        provider_request = distill.expected_ox_provider_request_sha256(
            profile_contract_id=str(contract["artifact_id"]),
            payload_digest=item_payload_digest,
            work_id=item_work_id,
            expires_at=str(contract["expires_at"]),
        )
        label = store.append_chain(
            label_path,
            {
                "kind": "teacher-label",
                "status": "completed",
                "work_id": item_work_id,
                "payload_digest": item_payload_digest,
                "attempt_count": 1,
                "profile": distill.OX_SINGLE_PROFILE,
                "profile_contract_id": contract["artifact_id"],
                "payload_source": item_payload_source,
                **_ox_projection_label_identity(),
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                "ramp_cap": ramp_cap,
                "provider_request_sha256": provider_request,
                "provider_receipt_sha256": "f" * 64,
                "request_sha256": distill.expected_ox_request_sha256(
                    profile_contract_id=str(contract["artifact_id"]),
                    payload_digest=item_payload_digest,
                ),
            },
        )
        labels.append(label)
        _complete_projection_label(tmp_path, item_work_id, label)

    for index, (ramp_cap, item_work_id, label) in enumerate(
        zip(ramp_caps, (work_id, second_work_id), labels, strict=True), start=1
    ):
        distill._append_ox_event(
            tmp_path,
            "ox-ramp-receipts.jsonl",
            {
                "event_version": 2,
                "kind": "ox-ramp-stage",
                "profile_contract_id": contract["artifact_id"],
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                "cap": ramp_cap,
                "next_cap": distill._next_ox_ramp_cap(ramp_cap, max_inflight),
                "valid_receipts": 1,
                "attempts": 1,
                "work_ids": [item_work_id],
                "label_count": index,
                "label_head_sha256": label["record_sha256"],
                "failure_record_count": 0,
                "failure_head_sha256": "",
                "captured_at": f"2026-08-25T00:00:0{index}Z",
            },
        )

    kwargs = {
        "profile_contract_id": str(contract["artifact_id"]),
        "source_binding": source,
        "workset": {"leased": 0},
        "label_path": label_path,
        "authoritative_gate": {"passed": True, "reasons": []},
    }
    with pytest.raises(distill.DistillationError, match="crosses ramp stages"):
        distill._ox_event_projection(tmp_path, **kwargs)


def test_ox_projection_rejects_current_label_work_id_reuse(tmp_path: Path) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    for attempt, receipt in ((1, "f" * 64), (2, "e" * 64)):
        store.append_chain(
            label_path,
            {
                "kind": "teacher-label",
                "status": "completed",
                "work_id": work_id,
                "payload_digest": payload_digest,
                "attempt_count": attempt,
                "ramp_cap": 1,
                "profile": distill.OX_SINGLE_PROFILE,
                "profile_contract_id": contract["artifact_id"],
                "payload_source": _ox_projection_payload_source(),
                **_ox_projection_label_identity(),
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                "provider_request_sha256": provider_request,
                "provider_receipt_sha256": receipt,
                "request_sha256": distill.expected_ox_request_sha256(
                    profile_contract_id=str(contract["artifact_id"]),
                    payload_digest=payload_digest,
                ),
            },
        )

    with pytest.raises(distill.DistillationError, match="work_id is not globally unique"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


def test_ox_projection_rejects_actual_receipt_reused_by_failure(
    tmp_path: Path,
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    receipt = "f" * 64
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "work_id": work_id,
            "payload_digest": payload_digest,
            "attempt_count": 1,
            "ramp_cap": 1,
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            "payload_source": _ox_projection_payload_source(),
            **_ox_projection_label_identity(),
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "provider_request_sha256": provider_request,
            "provider_receipt_sha256": receipt,
            "request_sha256": distill.expected_ox_request_sha256(
                profile_contract_id=str(contract["artifact_id"]),
                payload_digest=payload_digest,
            ),
        },
    )
    distill._append_ox_event(
        tmp_path,
        "ox-failure-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-provider-failure",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": 1,
            "category": "5xx",
            "status": "deferred",
            "attempts": 1,
            "bounded": True,
            "work_ids": [work_id],
            "attempts_by_work": {work_id: 2},
            "provider_requests": {work_id: provider_request},
            "provider_receipts": {work_id: receipt},
            "captured_at": "2026-08-25T00:00:00Z",
        },
    )

    with pytest.raises(distill.DistillationError, match="crosses success and failure"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


def test_ox_projection_rejects_success_and_failure_same_attempt(
    tmp_path: Path,
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "work_id": work_id,
            "payload_digest": payload_digest,
            "attempt_count": 1,
            "ramp_cap": 1,
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            "payload_source": _ox_projection_payload_source(),
            **_ox_projection_label_identity(),
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "provider_request_sha256": provider_request,
            "provider_receipt_sha256": "f" * 64,
            "request_sha256": distill.expected_ox_request_sha256(
                profile_contract_id=str(contract["artifact_id"]),
                payload_digest=payload_digest,
            ),
        },
    )
    distill._append_ox_event(
        tmp_path,
        "ox-failure-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-provider-failure",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": 1,
            "category": "5xx",
            "status": "deferred",
            "attempts": 1,
            "bounded": True,
            "work_ids": [work_id],
            "attempts_by_work": {work_id: 1},
            "provider_requests": {work_id: provider_request},
            "provider_receipts": {work_id: "e" * 64},
            "captured_at": "2026-08-25T00:00:00Z",
        },
    )

    with pytest.raises(distill.DistillationError, match="attempt is not globally unique"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


def test_ox_projection_rejects_actual_receipt_reused_by_failure_events(
    tmp_path: Path,
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    for cap in (1, 2):
        distill._append_ox_event(
            tmp_path,
            "ox-failure-receipts.jsonl",
            {
                "event_version": 2,
                "kind": "ox-provider-failure",
                "profile_contract_id": contract["artifact_id"],
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                "cap": cap,
                "category": "5xx",
                "status": "deferred",
                "attempts": 1,
                "bounded": True,
                "work_ids": [work_id],
                "attempts_by_work": {work_id: 1},
                "provider_requests": {work_id: provider_request},
                "provider_receipts": {work_id: "f" * 64},
                "captured_at": f"2026-08-25T00:00:0{cap}Z",
            },
        )

    with pytest.raises(distill.DistillationError, match="crosses failure events"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
            authoritative_gate={"passed": True, "reasons": []},
        )


def test_ox_projection_rejects_failure_attempt_reused_with_distinct_receipts(
    tmp_path: Path,
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    for cap, receipt in ((1, "f" * 64), (2, "e" * 64)):
        distill._append_ox_event(
            tmp_path,
            "ox-failure-receipts.jsonl",
            {
                "event_version": 2,
                "kind": "ox-provider-failure",
                "profile_contract_id": contract["artifact_id"],
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                "cap": cap,
                "category": "5xx",
                "status": "deferred",
                "attempts": 1,
                "bounded": True,
                "work_ids": [work_id],
                "attempts_by_work": {work_id: 1},
                "provider_requests": {work_id: provider_request},
                "provider_receipts": {work_id: receipt},
                "captured_at": f"2026-08-25T00:00:0{cap}Z",
            },
        )

    with pytest.raises(distill.DistillationError, match="attempt is not globally unique"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
            authoritative_gate={"passed": True, "reasons": []},
        )


def test_ox_projection_accepts_post_ramp_label_tail(tmp_path: Path) -> None:
    source, contract, work_id, _payload_digest = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    first_label = store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )
    distill._append_ox_event(
        tmp_path,
        "ox-ramp-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-ramp-stage",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": 10,
            "next_cap": 10,
            "valid_receipts": 1,
            "attempts": 1,
            "work_ids": [work_id],
            "label_count": 1,
            "label_head_sha256": first_label["record_sha256"],
            "failure_record_count": 0,
            "failure_head_sha256": "",
            "captured_at": "2026-08-25T00:00:00Z",
        },
    )
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )

    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        workset={"leased": 0},
        label_path=label_path,
        authoritative_gate={"passed": True, "reasons": []},
    )

    assert len(projection["ramp_receipts"]) == 1


def test_ox_projection_accepts_historical_terminal_before_429_requalification(
    tmp_path: Path,
) -> None:
    source, contract, work_id, payload_digest = _ox_projection_work(tmp_path)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    label = store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )
    failure_path = store.distillation_dir(tmp_path) / "ox-failure-receipts.jsonl"

    def append_stage(cap: int, failure_count: int) -> None:
        failures = store.read_chain(failure_path)
        distill._append_ox_event(
            tmp_path,
            "ox-ramp-receipts.jsonl",
            {
                "event_version": 2,
                "kind": "ox-ramp-stage",
                "profile_contract_id": contract["artifact_id"],
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                "cap": cap,
                "next_cap": distill._next_ox_ramp_cap(cap, 10),
                "valid_receipts": 0,
                "attempts": 0,
                "work_ids": [],
                "label_count": 1,
                "label_head_sha256": label["record_sha256"],
                "failure_record_count": failure_count,
                "failure_head_sha256": (
                    failures[-1]["record_sha256"] if failure_count else ""
                ),
                "captured_at": f"2026-08-25T00:00:0{cap}Z",
            },
        )

    for cap in (1, 2, 5, 10):
        append_stage(cap, 0)
    provider_request = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(contract["artifact_id"]),
        payload_digest=payload_digest,
        work_id=work_id,
        expires_at=str(contract["expires_at"]),
    )
    distill._append_ox_event(
        tmp_path,
        "ox-failure-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-provider-failure",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": 10,
            "category": "429",
            "status": "deferred",
            "attempts": 1,
            "before_cap": 10,
            "after_cap": 5,
            "work_ids": [work_id],
            "attempts_by_work": {work_id: 1},
            "provider_requests": {work_id: provider_request},
            "provider_receipts": {work_id: "f" * 64},
            "captured_at": "2026-08-25T00:00:10Z",
        },
    )
    for cap in (5, 10):
        append_stage(cap, 1)

    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        workset={"leased": 0},
        label_path=label_path,
        authoritative_gate={"passed": True, "reasons": []},
    )
    assert "ramp_receipts_incomplete" in projection["quality_gates"]["reasons"]



def test_ox_terminal_ramp_receipt_is_idempotent(tmp_path: Path) -> None:
    """A resumed cap-10 completion reuses its immutable receipt."""

    payload = {
        "kind": "ox-ramp-stage",
        "profile_contract_id": "a" * 64,
        "source_commit": "b" * 40,
        "source_tree_sha256": "c" * 64,
        "source_ox_identity_sha256": "d" * 64,
        "request_revision": distill.OX_RAMP_REQUEST_REVISION,
        "cap": 10,
        "valid_receipts": 20,
        "attempts": 20,
        "work_ids": [f"{index:064x}" for index in range(20)],
        "label_count": 20,
        "label_head_sha256": "e" * 64,
        "captured_at": "2026-08-25T00:00:00Z",
    }
    distill._append_ox_event(tmp_path, "ox-ramp-receipts.jsonl", payload)
    distill._append_ox_event(tmp_path, "ox-ramp-receipts.jsonl", payload)
    assert (
        store.chain_head(store.distillation_dir(tmp_path) / "ox-ramp-receipts.jsonl")[
            "records"
        ]
        == 1
    )


def test_ox_event_versions_preserve_legacy_and_bind_all_v2_fields(
    tmp_path: Path,
) -> None:
    legacy = {
        "kind": "ox-ramp-stage",
        "profile_contract_id": "a" * 64,
        "source_commit": "b" * 40,
        "source_tree_sha256": "c" * 64,
        "source_ox_identity_sha256": "d" * 64,
        "request_revision": distill.OX_RAMP_REQUEST_REVISION,
        "cap": 1,
    }
    distill._append_ox_event(tmp_path, "ox-ramp-receipts.jsonl", legacy)
    v2 = {
        **legacy,
        "event_version": 2,
        "expires_at": "2099-01-01T00:00:00Z",
        "valid_receipts": 20,
        "attempts": 20,
        "work_ids": [f"{index:064x}" for index in range(20)],
        "label_count": 20,
        "label_head_sha256": "e" * 64,
        "captured_at": "2026-08-25T00:00:00Z",
    }
    distill._append_ox_event(tmp_path, "ox-ramp-receipts.jsonl", v2)
    distill._append_ox_event(
        tmp_path, "ox-ramp-receipts.jsonl", {**v2, "attempts": 21}
    )
    assert (
        store.chain_head(store.distillation_dir(tmp_path) / "ox-ramp-receipts.jsonl")[
            "records"
        ]
        == 3
    )
    with pytest.raises(distill.DistillationError, match="version"):
        distill._append_ox_event(
            tmp_path, "ox-ramp-receipts.jsonl", {**v2, "event_version": 3}
        )


@pytest.mark.parametrize(
    "request_revision",
    [None, "json-schema-core-label-abstain-16k-240s-v5"],
)
def test_ox_event_projection_rejects_revision_drift(
    tmp_path: Path, request_revision: str | None
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
        ),
        source_binding=source,
    )
    event = {
        "kind": "ox-ramp-stage",
        "profile_contract_id": contract["artifact_id"],
        **source,
        "cap": 1,
        "valid_receipts": 20,
        "attempts": 20,
        "work_ids": [f"{index:064x}" for index in range(20)],
                    "label_count": 20,
                    "label_head_sha256": "d" * 64,
                    "failure_record_count": 0,
                    "failure_head_sha256": "",
        "expires_at": contract["expires_at"],
        "captured_at": "2026-08-25T00:00:00Z",
    }
    if request_revision is not None:
        event["request_revision"] = request_revision
    distill._append_ox_event(tmp_path, "ox-ramp-receipts.jsonl", event)

    with pytest.raises(
        distill.DistillationError, match="request revision|contract binding"
    ):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize("max_inflight", [3, 5])
def test_ox_event_projection_accepts_configured_terminal_cap(
    tmp_path: Path, max_inflight: int
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
            teacher_max_inflight=max_inflight,
        ),
        source_binding=source,
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    label = store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )
    distill._append_ox_event(
        tmp_path,
        "ox-ramp-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-ramp-stage",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": max_inflight,
            "next_cap": max_inflight,
            "valid_receipts": 0,
            "attempts": 0,
            "work_ids": [],
            "label_count": 1,
            "label_head_sha256": label["record_sha256"],
            "failure_record_count": 0,
            "failure_head_sha256": "",
            "captured_at": "2026-08-25T00:00:00Z",
        },
    )

    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        workset={"leased": 0},
        label_path=label_path,
        authoritative_gate={"passed": False, "reasons": ["test"]},
    )

    assert projection["ramp_receipts"][0]["cap"] == max_inflight


@pytest.mark.parametrize(
    ("max_inflight", "teacher_claim_limit", "certifying"),
    [(3, 1, False), (5, 1, False), (10, 2, False), (10, 1, True)],
)
def test_ox_event_projection_binds_certifying_terminal_tail_to_formal_contract(
    tmp_path: Path,
    max_inflight: int,
    teacher_claim_limit: int,
    certifying: bool,
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
            teacher_max_inflight=max_inflight,
            teacher_claim_limit=teacher_claim_limit,
        ),
        source_binding=source,
    )
    caps = tuple(sorted({min(cap, max_inflight) for cap in (1, 2, 5, 10)}))
    work_items: list[dict[str, Any]] = []
    labels_by_cap: dict[int, list[dict[str, Any]]] = {cap: [] for cap in caps}
    for cap in caps:
        for index in range(20):
            payload_source = {
                "candidate_id": f"candidate-{cap}-{index}",
                "rally_id": "rally",
            }
            payload_digest = canonical_json.canonical_json_sha256_strict(payload_source)
            work_id = canonical_json.canonical_json_sha256_strict(
                {
                    "kind": "ox-teacher-label-v1",
                    "profile": distill.OX_SINGLE_PROFILE,
                    "cohort": distill.OX_SINGLE_COHORT,
                    "route": "opencode-go/deepseek-v4-flash",
                    "profile_contract_id": contract["artifact_id"],
                    "payload_digest": payload_digest,
                }
            )
            work_items.append(
                {
                    "work_id": work_id,
                    "kind": "ox",
                    "payload_ref": f"candidate-snapshot:rally:{index}",
                    "payload_digest": payload_digest,
                    "temporal_split": {"split": "embargo"},
                    "provenance": {
                        "profile": distill.OX_SINGLE_PROFILE,
                        "cohort": distill.OX_SINGLE_COHORT,
                        "route": "opencode-go/deepseek-v4-flash",
                        "teacher_role": distill.OX_TEACHER_ROLE,
                        "profile_contract_id": contract["artifact_id"],
                        "probe": False,
                    },
                }
            )
            label = _ox_projection_label_record(
                source, contract, work_id, payload_digest
            )
            label["payload_source"] = payload_source
            label["ramp_cap"] = cap
            label["provider_receipt_sha256"] = f"{len(work_items):064x}"
            labels_by_cap[cap].append(label)

    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    )
    queue.advance(work_items, {"candidate_records": len(work_items)})
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    labels = [
        store.append_chain(label_path, label)
        for cap in caps
        for label in labels_by_cap[cap]
    ]
    labels_by_work = {str(label["work_id"]): label for label in labels}
    claims = queue.claim("ox", len(work_items), "test", 60)
    queue.commit(
        claims,
        [
            {
                "status": "completed",
                "completion_ref": f"label-ledger:{labels_by_work[claim.work_id]['record_sha256']}",
                "completion_digest": labels_by_work[claim.work_id]["record_sha256"],
            }
            for claim in claims
        ],
    )
    for stage_index, cap in enumerate(caps, start=1):
        label_count = stage_index * 20
        stage_labels = labels[(stage_index - 1) * 20 : label_count]
        distill._append_ox_event(
            tmp_path,
            "ox-ramp-receipts.jsonl",
            {
                "event_version": 2,
                "kind": "ox-ramp-stage",
                "profile_contract_id": contract["artifact_id"],
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                "cap": cap,
                "next_cap": distill._next_ox_ramp_cap(cap, max_inflight),
                "valid_receipts": 20,
                "attempts": 20,
                "work_ids": [str(label["work_id"]) for label in stage_labels],
                "label_count": label_count,
                "label_head_sha256": labels[label_count - 1]["record_sha256"],
                "failure_record_count": 0,
                "failure_head_sha256": "",
                "captured_at": f"2026-08-25T00:00:0{stage_index}Z",
            },
        )

    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        workset={"leased": 0},
        label_path=label_path,
        authoritative_gate={"passed": False, "reasons": ["test"]},
    )
    quality = projection["quality_gates"]
    assert quality["ramp_complete"] is certifying
    assert ("ramp_receipts_incomplete" not in quality["reasons"]) is certifying


@pytest.mark.parametrize("max_inflight", [3, 5])
def test_ox_event_projection_accepts_post_ramp_tail_for_configured_terminal_cap(
    tmp_path: Path, max_inflight: int
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
            teacher_max_inflight=max_inflight,
        ),
        source_binding=source,
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    label = store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )
    distill._append_ox_event(
        tmp_path,
        "ox-ramp-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-ramp-stage",
            "profile_contract_id": contract["artifact_id"],
            **source,
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": max_inflight,
            "next_cap": max_inflight,
            "valid_receipts": 0,
            "attempts": 0,
            "work_ids": [],
            "label_count": 1,
            "label_head_sha256": label["record_sha256"],
            "failure_record_count": 0,
            "failure_head_sha256": "",
            "captured_at": "2026-08-25T00:00:00Z",
        },
    )
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )

    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        workset={"leased": 0},
        label_path=label_path,
        authoritative_gate={"passed": False, "reasons": ["test"]},
    )

    assert projection["quality_gates"]["ramp_complete"] is False


def test_ox_event_projection_validates_historical_contracts_before_partition(
    tmp_path: Path,
) -> None:
    source_a = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    source_b = {
        "source_commit": "d" * 40,
        "source_tree_sha256": "e" * 64,
        "source_ox_identity_sha256": "f" * 64,
    }
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        ox_free_only=False,
        ox_expires_at="2099-01-01T00:00:00Z",
    )
    contract_a = distill._ensure_ox_profile_contract(
        tmp_path, config, source_binding=source_a
    )
    config_b = replace(config, ox_expires_at="2099-01-02T00:00:00Z")
    contract_b = distill._ensure_ox_profile_contract(
        tmp_path, config_b, source_binding=source_b
    )
    for contract, source, cap in (
        (contract_a, source_a, 1),
        (contract_b, source_b, 2),
    ):
        distill._append_ox_event(
            tmp_path,
            "ox-ramp-receipts.jsonl",
            {
                "event_version": 2,
                "kind": "ox-ramp-stage",
                "profile_contract_id": contract["artifact_id"],
                **source,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                    "cap": cap,
                    "next_cap": distill._next_ox_ramp_cap(cap, 10),
                "valid_receipts": 20,
                "attempts": 20,
                "work_ids": [f"{cap}{index:063d}" for index in range(20)],
                    "label_count": 20,
                    "label_head_sha256": "d" * 64,
                    "failure_record_count": 0,
                    "failure_head_sha256": "",
                    "captured_at": "2026-08-25T00:00:00Z",
            },
        )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    for contract, source in ((contract_a, source_a), (contract_b, source_b)):
        store.append_chain(
            label_path,
            {
                "kind": "teacher-label",
                "status": "completed",
                "teacher_profile": distill.LOCAL_TRIAD_PROFILE,
                "profile": distill.LOCAL_TRIAD_PROFILE,
                "profile_contract_id": contract["artifact_id"],
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "expires_at": contract["expires_at"],
                **source,
            },
        )

    projection = distill._ox_event_projection(
        tmp_path,
        profile_contract_id=str(contract_b["artifact_id"]),
        source_binding=source_b,
        workset={"leased": 0},
        label_path=label_path,
        authoritative_gate={"passed": True, "reasons": []},
    )
    assert [row["profile_contract_id"] for row in projection["ramp_receipts"]] == [
        contract_b["artifact_id"]
    ]
    assert projection["quality_gates"]["ramp_complete"] is False


@pytest.mark.parametrize(
    "expires_at",
    [
        None,
        9999999999.0,
        True,
        "2099-01-01",
        "2099-01-01T09:00:00+09:00",
        "2099-01-01T00:00:00.123Z",
        "9999-01-01T00:00:00Z",
    ],
)
def test_ox_event_projection_rejects_noncanonical_expiry(
    tmp_path: Path, expires_at: object
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
        ),
        source_binding=source,
    )
    event = {
        "kind": "ox-ramp-stage",
        "profile_contract_id": contract["artifact_id"],
        **source,
        "request_revision": distill.OX_RAMP_REQUEST_REVISION,
        "expires_at": expires_at,
        "cap": 1,
        "valid_receipts": 20,
        "attempts": 20,
        "work_ids": [f"{index:064x}" for index in range(20)],
        "label_count": 20,
        "label_head_sha256": "d" * 64,
        "captured_at": "2026-08-25T00:00:00Z",
    }
    distill._append_ox_event(tmp_path, "ox-ramp-receipts.jsonl", event)
    with pytest.raises(distill.DistillationError, match="expiry|contract binding"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
            authoritative_gate={"passed": True, "reasons": []},
        )


@pytest.mark.parametrize(
    "expires_at",
    [9999999999.0, True, "2099-01-01T09:00:00+09:00", "2099-01-01"],
)
def test_ox_label_projection_rejects_noncanonical_expiry(
    tmp_path: Path, expires_at: object
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
        ),
        source_binding=source,
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain(
        label_path,
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.OX_SINGLE_PROFILE,
            "profile_contract_id": contract["artifact_id"],
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": expires_at,
            **source,
        },
    )
    with pytest.raises(distill.DistillationError, match="expiry|contract binding"):
        distill._ox_event_projection(
            tmp_path,
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
            workset={"leased": 0},
            label_path=label_path,
            authoritative_gate={"passed": True, "reasons": []},
        )


def _config(root: Path, **overrides: object) -> Path:
    values = {
        "enabled": True,
        "chunk_size": 10,
        "max_input_bytes": 4096,
        "max_candidates": 20,
        "hard_floor_rallies": 100,
        "hard_floor_days": 30,
        "hard_floor_windows": 3,
        "hard_floor_teacher_labels": 100,
        "hard_floor_teacher_per_class": 10,
        "hard_floor_probe_pairs": 10,
        "hard_floor_counterfactual_pairs": 10,
        "canary_min_days": 7,
        **overrides,
    }
    lines = ["[recall.distillation]"]
    for key, value in values.items():
        lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
    lines.append("rollout_stages = [5, 25, 100]")
    path = root / "config.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _message(role: str, text: str, timestamp: str) -> dict[str, object]:
    content_type = "input_text" if role == "user" else "output_text"
    return {
        "type": "response_item",
        "timestamp": timestamp,
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": text}],
        },
    }


def _raw(root: Path) -> Path:
    init_chronovisor(RuntimeContext(root))
    raw_dir = root / "raw"
    events = [
        _message("user", "alpha memory", "2026-08-01T00:00:00Z"),
        _message("assistant", "alpha evidence", "2026-08-01T00:00:01Z"),
        {
            "type": "response_item",
            "timestamp": "2026-08-01T00:00:02Z",
            "payload": {"type": "function_call", "name": "search"},
        },
        _message("assistant", "alpha detail", "2026-08-01T00:00:03Z"),
        _message("user", "alpha question", "2026-08-02T00:00:00Z"),
        _message("assistant", "alpha future", "2026-08-02T00:00:01Z"),
        _message("user", "unanswered", "2026-08-03T00:00:00Z"),
    ]
    payload = b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
    )
    source = root / "session.jsonl"
    source.write_bytes(payload)
    append_capture(
        raw_dir=raw_dir,
        raw_id="save-codex-test.md",
        idempotency_key="codex-test",
        host="codex",
        session_key="a" * 24,
        session_id="session-one",
        source_file=source,
        after_line=0,
        until_line=len(events),
        source_bytes=payload,
        record_count=len(events),
        now=datetime(2026, 8, 3, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    return raw_dir


def _baseline_identity(root: Path) -> str:
    identity, _, _ = store.write_immutable(
        store.distillation_dir(root) / "baselines",
        {"kind": "test-incumbent"},
        schema=distill.BASELINE_SCHEMA,
    )
    return identity


def _fixture_candidate(
    root: Path, *, baseline_artifact_id: str = "", model_cohort_sha256: str = ""
) -> dict[str, object]:
    """Rollout fixtures bypass publication deliberately; production cannot."""

    _, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "policies",
        {
            "kind": "tiny-logistic-policy",
            **distill.train_tiny_policy([]),
            "lineage": {
                "fixture": True,
                "baseline_artifact_id": baseline_artifact_id,
                "model_cohort_sha256": model_cohort_sha256,
            },
        },
        schema=distill.POLICY_SCHEMA,
    )
    store.write_pointer(root, "candidate", str(artifact["artifact_id"]))
    store.write_sealed_state(
        store.distillation_dir(root) / store.STATE_FILE,
        {"kind": "worker-state", "status": "replay", "rollout_percent": 0},
    )
    return artifact


def test_config_is_off_by_default_and_environment_is_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.toml"
    assert not distill.distillation_enabled(missing)
    defaults = distill.load_distillation_config(missing)
    assert (defaults.chunk_size, defaults.max_input_bytes) == (25, 12_000)
    configured = _config(tmp_path)
    assert distill.distillation_enabled(configured)
    monkeypatch.setenv("CHRONOVISOR_RECALL_DISTILLATION", "false")
    assert not distill.distillation_enabled(configured)
    monkeypatch.setenv("CHRONOVISOR_RECALL_DISTILLATION", "true")
    assert distill.distillation_enabled(missing)


def test_migrate_distillation_config_dry_run_apply_and_idempotence(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    original = b'[runtime]\nsource = "keep"\n'
    config.write_bytes(original)

    dry_run = distill.migrate_distillation_config(config)
    assert dry_run["status"] == "dry_run"
    assert set(dry_run["additions"]) == {
        "recall.distillation",
        *distill._DISTILLATION_ROLES,
    }
    assert config.read_bytes() == original
    assert not config.with_name("config.toml.bak").exists()

    applied = distill.migrate_distillation_config(config, apply=True)
    parsed = tomllib.loads(config.read_text(encoding="utf-8"))
    assert applied == dry_run | {"status": "applied"}
    assert parsed["recall"]["distillation"] == distill._DISTILLATION_CONFIG
    assert parsed["llm"]["roles"] == distill._DISTILLATION_ROLES
    assert config.with_name("config.toml.bak").read_bytes() == original
    assert b"enabled = false\nchunk_size" in config.read_bytes()
    assert b'teacher_profile = "local-triad-v1"' in config.read_bytes()
    assert b"ox_enabled = false" in config.read_bytes()
    assert distill.migrate_distillation_config(config) == {
        "status": "noop",
        "additions": [],
    }


def test_legacy_distillation_config_migrates_profile_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    legacy = {
        key: value
        for key, value in distill._DISTILLATION_CONFIG.items()
        if key not in distill._OPTIONAL_PROFILE_CONFIG
    }
    config.write_text(
        "[recall.distillation]\n"
        + "".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}\n"
            for key, value in legacy.items()
        ),
        encoding="utf-8",
    )

    migrated = distill.migrate_distillation_config(config, apply=True)

    assert set(migrated["additions"]) == {
        *(f"recall.distillation.{key}" for key in distill._OPTIONAL_PROFILE_CONFIG),
        *distill._DISTILLATION_ROLES,
    }
    assert (
        tomllib.loads(config.read_text(encoding="utf-8"))["recall"]["distillation"]
        == distill._DISTILLATION_CONFIG
    )
    operator_enabled = config.read_text(encoding="utf-8").replace(
        "enabled = false", "enabled = true", 1
    )
    config.write_text(operator_enabled, encoding="utf-8")
    assert distill.migrate_distillation_config(config) == {
        "status": "noop",
        "additions": [],
    }


def test_ox_profile_contract_is_stable_and_fail_closed(tmp_path: Path) -> None:
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        ox_free_only=False,
        ox_expires_at="2099-01-01T00:00:00Z",
        teacher_max_inflight=10,
        teacher_claim_limit=1,
    )
    first = distill._ensure_ox_profile_contract(tmp_path, config)
    second = distill._ensure_ox_profile_contract(tmp_path, config)
    assert first["artifact_id"] == second["artifact_id"]
    assert first["endpoint"] == "https://opencode.ai/zen/go/v1/chat/completions"
    assert first["request_model"] == first["required_returned_model"] == "deepseek-v4-flash"
    assert first["request_revision"] == distill.OX_RAMP_REQUEST_REVISION
    assert first["expires_at"] == "2099-01-01T00:00:00Z"
    assert first["live_recall_model_calls"] == 0
    assert first["kill_categories"] == [
        "402",
        "payment_required",
        "model_unavailable",
        "route_model_drift",
        "privacy_gate",
    ]
    assert first["teacher_claim_limit"] == 1
    with pytest.raises(distill.DistillationError, match="unsafe"):
        distill._ensure_ox_profile_contract(
            tmp_path,
            distill.DistillationConfig(
                teacher_profile=distill.OX_SINGLE_PROFILE,
                teacher_max_inflight=11,
            ),
        )


def test_ox_expiry_normalizes_offsets_and_rejects_noncanonical_or_far_future() -> None:
    assert (
        distill._ox_expiry("2099-01-01T09:00:00+09:00")
        == "2099-01-01T00:00:00Z"
    )
    for value in (
        None,
        True,
        9999999999.0,
        "2099-01-01",
        "2099-01-01T00:00:00.123Z",
        "2000-01-01T00:00:00Z",
        "9999-01-01T00:00:00Z",
    ):
        with pytest.raises(distill.DistillationError):
            distill._ox_expiry(value)


def test_ox_profile_contract_reader_accepts_previous_remote_revision(
    tmp_path: Path,
) -> None:
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
        ),
        source_binding={
            "source_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "source_ox_identity_sha256": "c" * 64,
        },
    )
    payload = {
        key: value
        for key, value in contract.items()
        if key not in {"schema", "namespace", "artifact_id", "seal_sha256"}
    }
    payload["request_revision"] = "json-schema-core-label-abstain-16k-240s-v7"
    _, _, historical = store.write_immutable(
        store.distillation_dir(tmp_path) / "ox-profile-contracts",
        payload,
        schema=distill.OX_PROFILE_SCHEMA,
    )

    assert (
        distill._read_ox_profile_contract(tmp_path, historical["artifact_id"])
        == historical
    )


@pytest.mark.parametrize(
    "request_revision",
    [None, "json-schema-core-label-abstain-16k-240s-v5"],
)
def test_ox_profile_contract_rejects_resealed_revision_drift(
    tmp_path: Path, request_revision: str | None
) -> None:
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        ox_free_only=False,
        ox_expires_at="2099-01-01T00:00:00Z",
    )
    contract = distill._ensure_ox_profile_contract(tmp_path, config)
    contract_path = (
        store.distillation_dir(tmp_path)
        / "ox-profile-contracts"
        / f"{contract['artifact_id']}.json"
    )
    payload = json.loads(contract_path.read_text())
    if request_revision is None:
        payload.pop("request_revision")
    else:
        payload["request_revision"] = request_revision
    unsigned = {
        key: value for key, value in payload.items() if key != "seal_sha256"
    }
    payload["seal_sha256"] = canonical_json.canonical_json_sha256_strict(unsigned)
    contract_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    assert distill._ox_contract_source_binding(tmp_path, str(contract["artifact_id"])) == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("route", "evil-route"),
        ("request_model", "evil-model"),
        ("required_returned_model", "evil-model"),
        ("fixed_identity", {}),
        ("free_only", True),
        ("no_paid_fallback", False),
        ("kill_categories", []),
        ("live_recall_model_calls", 99),
    ],
)
def test_ox_contract_reader_rejects_self_consistent_identity_forgery(
    tmp_path: Path, field: str, value: object
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    contract = distill._ensure_ox_profile_contract(
        tmp_path,
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            ox_free_only=False,
            ox_expires_at="2099-01-01T00:00:00Z",
        ),
        source_binding=source,
    )
    original_path = (
        store.distillation_dir(tmp_path)
        / "ox-profile-contracts"
        / f"{contract['artifact_id']}.json"
    )
    forged = json.loads(original_path.read_text())
    forged[field] = value
    unsigned = {key: item for key, item in forged.items() if key not in {"artifact_id", "seal_sha256"}}
    forged_id = canonical_json.canonical_json_sha256_strict(unsigned)
    forged = {"artifact_id": forged_id, **unsigned}
    forged["seal_sha256"] = canonical_json.canonical_json_sha256_strict(forged)
    forged_path = (
        store.distillation_dir(tmp_path) / "ox-profile-contracts" / f"{forged_id}.json"
    )
    forged_path.write_bytes(canonical_json.canonical_json_bytes_strict(forged) + b"\n")
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "ox-profile-contract.json",
        {"kind": "ox-profile-contract-pointer", "profile_contract_id": forged_id},
    )

    assert distill._read_ox_profile_contract(tmp_path, forged_id) == {}
    assert distill._ox_contract_source_binding(tmp_path, forged_id) == {}


@pytest.mark.parametrize(
    "contents",
    (
        b"[recall.distillation]\nenabled = false\n",
        b'[llm.roles."recall.distill.teacher.a"]\nmodel = "wrong"\n',
    ),
)
def test_migrate_distillation_config_rejects_partial_or_conflicting_sections(
    tmp_path: Path, contents: bytes
) -> None:
    config = tmp_path / "config.toml"
    config.write_bytes(contents)

    with pytest.raises(distill.DistillationError):
        distill.migrate_distillation_config(config, apply=True)

    assert config.read_bytes() == contents
    assert not config.with_name("config.toml.bak").exists()


def test_local_worker_metadata_and_transient_failure_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama, research_scheduler

    route = SimpleNamespace(
        role="recall.distill.teacher.a",
        provider="ollama",
        model="local-model",
        location="local",
        structured_output=True,
    )
    identity = {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location,
    }
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))

    lane_kwargs: list[dict[str, object]] = []

    @contextmanager
    def lane(*_args: object, **kwargs: object):
        lane_kwargs.append(kwargs)
        yield object()

    monkeypatch.setattr(research_scheduler, "research_lane", lane)
    failure = {"value": ""}
    reported_identity: dict[str, object] = dict(identity)

    def command(
        _argv: object, encoded: str, _lease: object, **_kwargs: object
    ) -> SimpleNamespace:
        request = json.loads(encoded)
        if failure["value"]:
            value = {
                "schema": "chronovisor.recall-distillation-worker.v1",
                "ok": False,
                "operation": request["operation"],
                "role": request["role"],
                "request_id": request["request_id"],
                "route_identity": reported_identity,
                "model_digest": "d" * 64,
                "result": {},
                "failure_class": failure["value"],
            }
        else:
            value = {
                "schema": "chronovisor.recall-distillation-worker.v1",
                "ok": True,
                "operation": request["operation"],
                "role": request["role"],
                "request_id": request["request_id"],
                "route_identity": reported_identity,
                "model_digest": "d" * 64,
                "result": {"labels": []},
                "failure_class": "",
            }
        return SimpleNamespace(status="completed", value=value)

    monkeypatch.setattr(research_scheduler, "run_cancellable_command", command)
    result = distill._worker_call(
        "teacher",
        route.role,
        {"candidates": []},
        max_input_bytes=12_000,
        expected_route=identity,
        expected_digest="d" * 64,
    )
    assert result["_route_identity"] == identity
    assert lane_kwargs[-1]["mode"] == "sleep"
    assert lane_kwargs[-1]["purpose"] == "sleep"
    failure["value"] = "backend_error"
    with pytest.raises(distill.DistillationDeferred):
        distill._worker_call(
            "teacher",
            route.role,
            {"candidates": []},
            max_input_bytes=12_000,
            expected_route=identity,
            expected_digest="d" * 64,
        )
    failure["value"] = "output_invalid"
    with pytest.raises(distill.DistillationError, match="output"):
        distill._worker_call(
            "teacher",
            route.role,
            {"candidates": []},
            max_input_bytes=12_000,
            expected_route=identity,
            expected_digest="d" * 64,
        )
    failure["value"] = ""
    reported_identity["model"] = "wrong-model"
    with pytest.raises(distill._LocalR4ClassifiedFailure, match="route_model_mismatch"):
        distill._worker_call(
            "teacher",
            route.role,
            {"candidates": []},
            max_input_bytes=12_000,
            expected_route=identity,
            expected_digest="d" * 64,
        )
    with pytest.raises(distill.DistillationError, match="response"):
        distill._worker_call(
            "utility",
            route.role,
            {"candidates": []},
            max_input_bytes=12_000,
            expected_route=identity,
            expected_digest="d" * 64,
        )


def test_default_workers_keep_cold_teacher_and_counterfactual_budgets_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    roles = (
        *distill.TEACHER_ROLES,
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
    )
    routes = tuple(
        SimpleNamespace(
            role=role,
            provider="ollama",
            model=f"model-{index}",
            location="local",
            structured_output=True,
        )
        for index, role in enumerate(roles)
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda models: {
            model: f"{index + 1:064x}" for index, model in enumerate(models)
        },
    )

    teachers, counterfactual = distill._default_workers(
        distill.DistillationConfig(),
        teacher_deadline_ms=120_000,
        counterfactual_deadline_ms=45_000,
    )

    assert {worker.deadline_ms for worker in teachers.values()} == {120_000}
    assert counterfactual is not None
    assert counterfactual.deadline_ms == 45_000


def test_default_workers_accept_revision_pinned_local_omlx_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    roles = (
        *distill.TEACHER_ROLES,
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
    )
    models = ("qwen", "muse", "gemma", "qwen", "gemma")
    revisions = ("qwen-rev", "muse-rev", "gemma-rev", "qwen-rev", "gemma-rev")
    routes = tuple(
        ollama.RuntimeGenerationRoute(
            role=role,
            provider="omlx",
            model=model,
            location="local",
            structured_output=True,
            protocol="openai-compatible",
            endpoint_sha256="e" * 64,
            revision=revision,
        )
        for role, model, revision in zip(roles, models, revisions, strict=True)
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: (_ for _ in ()).throw(AssertionError("Ollama queried")),
    )

    teachers, counterfactual = distill._default_workers(distill.DistillationConfig())

    assert len(teachers) == 3
    assert counterfactual is not None
    assert len({worker.expected_digest for worker in teachers.values()}) == 3
    assert all(len(worker.expected_digest) == 64 for worker in teachers.values())
    assert (
        teachers["recall.distill.teacher.a"].expected_digest
        == counterfactual.digests["recall.distill.answer_generator"]
    )


def test_ox_profile_requires_explicit_enable_and_builds_one_remote_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import (
        llm_config,
        llm_security,
        ollama,
    )
    from chronovisor.recall import recall_distillation_remote_teacher as remote

    local_roles = (
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
    )
    local_routes = tuple(
        SimpleNamespace(
            role=role,
            provider="ollama",
            model=f"model-{index}",
            location="local",
            structured_output=True,
        )
        for index, role in enumerate(local_roles)
    )
    monkeypatch.setattr(
        ollama, "runtime_generation_routes", lambda _roles: local_routes
    )
    monkeypatch.setattr(
        ollama,
        "runtime_generation_route_fingerprints",
        lambda _routes: {
            local_roles[0]: "a" * 64,
            local_roles[1]: "b" * 64,
        },
    )

    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(
        llm_config,
        "compose_remote_generation_backend",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(remote, "OpenCodeOxAlphaTeacher", RemoteTeacher)
    monkeypatch.setattr(
        llm_security.CredentialResolver,
        "resolve",
        lambda *_args: object(),
    )

    disabled, counterfactual = distill._default_workers(
        distill.DistillationConfig(teacher_profile=distill.OX_SINGLE_PROFILE)
    )
    teachers, counterfactual_enabled = distill._default_workers(
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_max_inflight=10,
        )
    )

    assert disabled == {}
    assert counterfactual is None
    assert set(teachers) == {distill.OX_TEACHER_ROLE}
    assert teachers[distill.OX_TEACHER_ROLE].local is False
    assert teachers[distill.OX_TEACHER_ROLE].kwargs["timeout_ms"] == 600_000
    assert counterfactual_enabled is not None
    assert counterfactual_enabled.local is True


def test_ox_default_worker_is_unavailable_without_keyring_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import llm_security

    def missing(*_args: object) -> object:
        raise RuntimeError("missing keyring credential")

    monkeypatch.setattr(llm_security.CredentialResolver, "resolve", missing)

    teachers, counterfactual = distill._default_workers(
        distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
        )
    )

    assert teachers == {}
    assert counterfactual is None


def test_materialized_row_allocation_orders_normalized_utc_instants() -> None:
    rows = [
        {"rally_id": "later", "as_of": "2026-01-01T00:30:00Z"},
        {"rally_id": "earlier", "as_of": "2026-01-01T09:00:00+09:00"},
    ]
    assert [
        row["rally_id"] for row in distill._allocate_materialized_rows(rows, 2)
    ] == [
        "earlier",
        "later",
    ]


def test_ox_profile_config_is_toml_safe_and_capped(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        ox_expires_at="2099-01-01T00:00:00Z",
        teacher_max_inflight=10,
    )

    loaded = distill.load_distillation_config(config)

    assert loaded.teacher_profile == distill.OX_SINGLE_PROFILE
    assert loaded.ox_enabled is True
    assert loaded.teacher_max_inflight == 10
    assert loaded.teacher_claim_limit == 500
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "teacher_max_inflight = 10", "teacher_max_inflight = 11"
        ),
        encoding="utf-8",
    )
    with pytest.raises(distill.DistillationError, match="at most 10"):
        distill.load_distillation_config(config)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "teacher_max_inflight = 11",
            "teacher_max_inflight = 10\nteacher_claim_limit = 501",
        ),
        encoding="utf-8",
    )
    with pytest.raises(distill.DistillationError, match="at most 500"):
        distill.load_distillation_config(config)


def test_ox_disabled_profile_remains_capture_only(tmp_path: Path) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(
        tmp_path,
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=False,
        ox_expires_at="",
    )

    result = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )

    assert result["status"] in {"capture_only", "deferred"}


def test_rally_v1_folds_assistant_and_tool_refs_without_copying_text(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    rallies = distill.extract_rallies(raw_dir, root=tmp_path)
    assert len(rallies) == 3
    assert len(rallies[0]["actual_answer_refs"]) == 2
    assert len(rallies[0]["tool_refs"]) == 1
    assert rallies[1]["context_refs"][-1]["role"] == "assistant"
    assert rallies[2]["eligibility"]["reason"] == "missing_answer"
    assert "alpha" not in json.dumps(rallies)
    assert "session-one" not in json.dumps(rallies)
    assert "a" * 24 not in json.dumps(rallies)
    assert distill.extract_rallies(raw_dir, root=tmp_path) == rallies


def test_public_raw_watermark_preserves_receipt_inventory_bytes(tmp_path: Path) -> None:
    from chronovisor.core.raw_store import (
        RawStore,
        committed_raw_watermark,
    )
    from chronovisor.research.evidence_reconstruction import (
        committed_raw_watermark as evidence_watermark,
    )

    raw_dir = _raw(tmp_path)
    rows = []
    for unit in RawStore(raw_dir, mode="v2").iter_segment_units():
        assert unit.commit is not None
        rows.append(
            {
                "raw_id": unit.raw_id,
                "byte_range": [0, unit.length],
                "byte_coordinate_space": "logical_raw",
                "raw_sha256": unit.sha256,
                "receipt_sha256": distill.canonical_json.canonical_json_sha256_strict(
                    unit.commit.to_dict()
                ),
                "captured_at": unit.captured_at,
                "host": unit.commit.host,
                "session_key": unit.commit.session_key,
                "source_line_range": [
                    unit.commit.after_line,
                    unit.commit.until_line,
                ],
            }
        )
    expected = distill.canonical_json.canonical_json_sha256_strict(rows)
    assert committed_raw_watermark(raw_dir) == expected
    assert evidence_watermark(raw_dir) == expected


def test_rally_extraction_skips_archived_legacy_but_rejects_malformed_native(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    source_root = tmp_path / "legacy-source"
    source_root.mkdir()
    semantic_child = source_root / "semantic-child.md"
    semantic_child.write_text("archived semantic child\n", encoding="utf-8")
    day_dir = raw_dir / "2026" / "08" / "11"
    manifest = write_legacy_archive(
        [semantic_child],
        archive_path=day_dir / "legacy-part-001.tar.zst",
        captured_date="2026/08/11",
    )
    archive_path = day_dir / str(manifest["archive"])
    legacy = b"---\nraw_keywords: [historical]\n---\nLegacy transcript envelope.\n"
    append_capture(
        raw_dir=raw_dir,
        raw_id="save-codex-legacy-envelope.md",
        idempotency_key="codex-legacy-envelope",
        host="codex",
        session_key="b" * 24,
        session_id=None,
        source_file=archive_path,
        after_line=10,
        until_line=11,
        source_bytes=legacy,
        record_count=1,
        now=datetime(2026, 8, 11, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    rallies = distill.extract_rallies(raw_dir, root=tmp_path)
    assert len(rallies) == 3
    assert "Legacy transcript" not in json.dumps(rallies)

    bad_root = tmp_path / "bad"
    bad_raw = bad_root / "raw"
    init_chronovisor(RuntimeContext(bad_root))
    malformed = b"---\nnot native JSON\n"
    source = bad_root / "native.jsonl"
    source.write_bytes(malformed)
    append_capture(
        raw_dir=bad_raw,
        raw_id="save-codex-malformed.md",
        idempotency_key="codex-malformed",
        host="codex",
        session_key="c" * 24,
        session_id=None,
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=malformed,
        record_count=1,
        now=datetime(2026, 8, 11, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    with pytest.raises(distill.DistillationError, match="invalid JSON"):
        distill.extract_rallies(bad_raw, root=bad_root)


def test_historical_fts_is_assistant_only_and_strictly_point_in_time(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    index = tmp_path / "runtime" / "recall-distillation" / "historical.sqlite"
    first_digest = distill.build_historical_index(raw_dir, index)
    first_inode = index.stat().st_ino
    second_digest = distill.build_historical_index(raw_dir, index)
    assert first_digest == second_digest
    assert index.stat().st_ino == first_inode
    assert stat.S_IMODE(index.stat().st_mode) == 0o600
    rally = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    snapshot = distill.candidate_snapshot(
        index,
        rally,
        "alpha",
        limit=20,
        candidate_texts={
            hashlib.sha256(text.encode()).hexdigest(): text
            for text in ("alpha evidence", "alpha detail", "alpha future")
        },
    )
    digests = {row["text_sha256"] for row in snapshot["candidates"]}
    assert hashlib.sha256(b"alpha evidence").hexdigest() in digests
    assert hashlib.sha256(b"alpha detail").hexdigest() in digests
    assert hashlib.sha256(b"alpha future").hexdigest() not in digests
    assert all("text" not in row for row in snapshot["candidates"])
    assert all(
        row["feature_revision"] == distill.TEXT_FEATURE_REVISION
        and set(row["features"]) == set(distill.FAST_FEATURE_KEYS)
        and len(row["candidate_feature_text_sha256"]) == 64
        for row in snapshot["candidates"]
    )
    assert len(snapshot["query_feature_text_sha256"]) == 64


def test_historical_cutoff_rejects_later_time_even_with_earlier_source_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical.sqlite"
    atoms = [
        {
            "atom_id": "future",
            "host": "codex",
            "session_cluster_id": "s",
            "source_index": 1,
            "timestamp_us": 200,
            "text_sha256": "a" * 64,
            "ref": {"raw_id": "future"},
            "text": "future marker",
        },
        {
            "atom_id": "same-time-prior",
            "host": "codex",
            "session_cluster_id": "s",
            "source_index": 2,
            "timestamp_us": 100,
            "text_sha256": "b" * 64,
            "ref": {"raw_id": "prior"},
            "text": "prior marker",
        },
    ]
    store.create_historical_index(path, atoms)
    assert not store.search_historical_index(
        path,
        query="future",
        as_of_us=100,
        host="codex",
        session_cluster_id="s",
        source_index=3,
        limit=10,
    )
    assert (
        store.search_historical_index(
            path,
            query="prior",
            as_of_us=100,
            host="codex",
            session_cluster_id="s",
            source_index=3,
            limit=10,
        )[0]["candidate_id"]
        == "same-time-prior"
    )


def test_historical_index_finds_whitespace_free_japanese(tmp_path: Path) -> None:
    path = tmp_path / "historical.sqlite"
    store.create_historical_index(
        path,
        [
            {
                "atom_id": "jp",
                "host": "codex",
                "session_cluster_id": "s",
                "source_index": 1,
                "timestamp_us": 1,
                "text_sha256": "c" * 64,
                "ref": {"raw_id": "jp"},
                "text": "クロノバイザーの検索精度を改善する",
            }
        ],
    )
    rows = store.search_historical_index(
        path,
        query="検索精度",
        as_of_us=2,
        host="codex",
        session_cluster_id="other",
        source_index=1,
        limit=10,
    )
    assert rows[0]["candidate_id"] == "jp"


def test_context_is_a_fixed_event_suffix_and_full_prefix_is_only_a_digest(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    rally = distill.extract_rallies(raw_dir, root=tmp_path, max_context_bytes=5)[1]
    assert rally["context_refs"] == []
    assert rally["full_context"]["event_count"] > 0
    assert len(rally["full_context"]["refs_sha256"]) == 64


def test_assignment_label_authority_and_live_feature_boundary() -> None:
    assignment = distill.teacher_assignment("rally", "candidate")
    assert assignment == distill.teacher_assignment("rally", "candidate")
    assert assignment["owner"] in distill.TEACHER_ROLES
    assert (
        distill.adjudicate_label("helpful", closed_predicate="exact_claim_supported")[
            "authority"
        ]
        == "teacher-only"
    )
    assert (
        distill.adjudicate_label("helpful", closed_predicate="exact_claim_supported")[
            "authority"
        ]
        == "teacher-only"
    )
    assert (
        distill.adjudicate_label("helpful", closed_predicate="exact_test_outcome")[
            "authority"
        ]
        == "teacher-only"
    )
    with pytest.raises(distill.DistillationError, match="not whitelisted"):
        distill.build_fast_features({"answer_delta": 1})
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=0.4
    )
    policy = distill.train_tiny_policy(
        [{"features": features, "verdict": "helpful", "authority": "verified"}]
    )
    assert 0 <= distill.score_fast_features(features, policy) <= 1
    assert distill.policy_decision(0.8, policy, runner_up_score=0.2) == {
        "decision": "read",
        "max_cards": 3,
    }
    teacher_policy = distill.train_tiny_policy(
        [
            {
                "rally_id": "r",
                "candidate_id": "c",
                "dimension": "relevance",
                "features": features,
                "verdict": "relevant",
                "authority": "teacher-only",
            }
        ]
    )
    assert teacher_policy["training_rows"] == 1
    assert teacher_policy["weights"] != distill.train_tiny_policy([])["weights"]


def test_model_lane_scheduler_is_fair_and_reserves_counterfactual_turn() -> None:
    pending = {role: [{}] for role in distill.TEACHER_ROLES}
    labels: list[dict[str, str]] = []
    visited = []
    for _ in range(3):
        route = distill._ordered_teacher_routes(pending, labels)[0]
        visited.append(route)
        labels.append({"route": route})
    assert visited == list(distill.TEACHER_ROLES)
    assert not distill._is_counterfactual_turn(2, 0, available=True)
    assert distill._is_counterfactual_turn(3, 0, available=True)
    assert not distill._is_counterfactual_turn(3, 1, available=True)
    assert not distill._is_counterfactual_turn(6, 1, available=False)


@pytest.mark.parametrize(
    ("state_contract", "current_contract", "expected"),
    [
        ("a" * 64, "a" * 64, (170, 1)),
        ("a" * 64, "b" * 64, (0, 0)),
        (None, "b" * 64, (0, 0)),
        (None, "", (170, 1)),
    ],
)
def test_model_lane_scheduler_resets_counters_for_a_new_ox_contract(
    state_contract: object,
    current_contract: str,
    expected: tuple[int, int],
) -> None:
    state = {
        "teacher_model_calls": 170,
        "counterfactual_model_calls": 1,
        "ox_profile_contract_id": state_contract,
    }

    assert distill._scheduler_model_calls(state, current_contract) == expected


def test_exposure_receipt_is_exact_prospective_and_hash_chained(tmp_path: Path) -> None:
    digest = "d" * 64
    receipt = distill.record_exposure(
        decision_id="decision",
        host="codex",
        session_id="session-one",
        prompt_hash="prompt",
        policy_id="policy",
        candidate_ids=["a", "b"],
        candidate_snapshot_sha256=digest,
        observed_at="2026-08-14T00:00:00Z",
        root=tmp_path,
    )
    assert "session-one" not in json.dumps(receipt)
    path = store.distillation_dir(tmp_path) / "exposure-receipts.jsonl"
    assert store.verify_chain(path)["records"] == 1
    path.write_text(path.read_text().replace('"policy"', '"tampered"'))
    with pytest.raises(store.DistillationStoreError, match="chain mismatch"):
        store.verify_chain(path)


def test_exposure_receipt_retry_is_atomic_and_conflicts_fail(tmp_path: Path) -> None:
    def write() -> dict[str, object]:
        return distill.record_exposure(
            decision_id="one-decision",
            host="codex",
            session_id="session",
            prompt_hash="prompt",
            policy_id="policy",
            candidate_ids=["candidate"],
            candidate_snapshot_sha256="d" * 64,
            observed_at="2026-08-14T00:00:00Z",
            root=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda _index: write(), range(8)))
    assert len({row["record_sha256"] for row in rows}) == 1
    later = distill.record_exposure(
        decision_id="one-decision",
        host="codex",
        session_id="session",
        prompt_hash="prompt",
        policy_id="policy",
        candidate_ids=["candidate"],
        candidate_snapshot_sha256="d" * 64,
        observed_at="2026-08-14T00:00:01Z",
        root=tmp_path,
    )
    assert later["record_sha256"] == rows[0]["record_sha256"]
    path = store.distillation_dir(tmp_path) / "exposure-receipts.jsonl"
    assert store.verify_chain(path)["records"] == 1
    with pytest.raises(store.DistillationStoreError, match="identity conflict"):
        distill.record_exposure(
            decision_id="one-decision",
            host="codex",
            session_id="session",
            prompt_hash="different",
            policy_id="policy",
            candidate_ids=[],
            candidate_snapshot_sha256="d" * 64,
            observed_at="2026-08-14T00:00:00Z",
            root=tmp_path,
        )


def test_nonblocking_page_and_exact_receipts_write_nothing_when_busy(
    tmp_path: Path,
) -> None:
    identity = _baseline_identity(tmp_path)
    runtime_dir = store.distillation_dir(tmp_path)
    ledger = runtime_dir / "exposure-receipts.jsonl"
    lock = store.acquire_nonblocking_lock(ledger.with_suffix(".jsonl.lock"))
    assert lock is not None
    try:
        page = distill.record_exposure(
            decision_id="busy-page",
            host="codex",
            session_id="session",
            prompt_hash="prompt",
            policy_id=identity,
            candidate_ids=[],
            candidate_snapshot_sha256="a" * 64,
            observed_at="2026-08-14T00:00:00Z",
            nonblocking=True,
            root=tmp_path,
        )
        exact = distill.record_exact_exposure(
            decision_id="busy-exact",
            host="codex",
            session_id="session",
            query_semantic_sha256="b" * 64,
            policy_id=identity,
            candidate_refs=[],
            render_sha256="c" * 64,
            candidate_snapshot_sha256="d" * 64,
            observed_at="2026-08-14T00:00:00Z",
            nonblocking=True,
            root=tmp_path,
        )
    finally:
        store.release_lock(lock)
    assert (
        page
        == exact
        == {
            "status": "deferred",
            "reason": "receipt_ledger_busy",
        }
    )
    assert not ledger.exists()
    assert not (runtime_dir / "exposures").exists()
    artifact_lock = store.acquire_nonblocking_lock(
        runtime_dir / "exposures" / ".immutable.lock"
    )
    assert artifact_lock is not None
    try:
        artifact_busy = distill.record_exact_exposure(
            decision_id="busy-artifact",
            host="codex",
            session_id="session",
            query_semantic_sha256="b" * 64,
            policy_id=identity,
            candidate_refs=[],
            render_sha256="c" * 64,
            candidate_snapshot_sha256="d" * 64,
            observed_at="2026-08-14T00:00:00Z",
            nonblocking=True,
            root=tmp_path,
        )
    finally:
        store.release_lock(artifact_lock)
    assert artifact_busy == {
        "status": "deferred",
        "reason": "receipt_ledger_busy",
    }
    assert not ledger.exists()
    assert list((runtime_dir / "exposures").glob("*.json")) == []


def test_exposure_join_requires_one_receipt_bound_inside_the_rally(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    rally = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    identity = _baseline_identity(tmp_path)
    assert not rally["eligibility"]["answer_utility"]
    evidence_ref = distill.extract_rallies(raw_dir, root=tmp_path)[0][
        "actual_answer_refs"
    ][0]
    distill.record_exact_exposure(
        decision_id="inside",
        host="codex",
        session_id="session-one",
        query_semantic_sha256=rally["query_sha256"],
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "candidate",
                "content_sha256": evidence_ref["semantic_sha256"],
                "evidence_refs": [evidence_ref],
            }
        ],
        render_sha256="f" * 64,
        candidate_snapshot_sha256="e" * 64,
        observed_at="2026-08-02T00:00:00.500000Z",
        root=tmp_path,
    )
    joined = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    assert joined["eligibility"]["answer_utility"]
    distill.record_exact_exposure(
        decision_id="duplicate",
        host="codex",
        session_id="session-one",
        query_semantic_sha256=rally["query_sha256"],
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "candidate",
                "content_sha256": evidence_ref["semantic_sha256"],
                "evidence_refs": [evidence_ref],
            }
        ],
        render_sha256="f" * 64,
        candidate_snapshot_sha256="e" * 64,
        observed_at="2026-08-02T00:00:00.600000Z",
        root=tmp_path,
    )
    ambiguous = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    assert ambiguous["eligibility"]["reason"] == "ambiguous_exact_exposure"


def test_exact_page_exposure_seals_canonical_live_features(tmp_path: Path) -> None:
    identity = _baseline_identity(tmp_path)
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=0.25
    )
    rendered = "exact rendered card"
    receipt = distill.record_exact_exposure(
        decision_id="page-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="a" * 64,
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "page-v1",
                "page_id": "page",
                "page_content_sha256": "9" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        candidate_feature_snapshot=[
            {"candidate_id": "page-v1", "features": features},
            {"candidate_id": "unselected", "features": distill.build_fast_features()},
        ],
        candidate_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": True,
                "page_id": "page",
                "page_content_sha256": "9" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            },
            {
                "candidate_id": "unselected",
                "selected": False,
                "page_id": "other",
                "page_content_sha256": "8" * 64,
                "rendered_context": "other context",
                "rendered_context_sha256": hashlib.sha256(b"other context").hexdigest(),
            },
        ],
        render_sha256="b" * 64,
        candidate_snapshot_sha256="c" * 64,
        observed_at="2026-08-14T00:00:00Z",
        root=tmp_path,
    )
    artifact_path = (
        store.distillation_dir(tmp_path)
        / "exposures"
        / f"{receipt['exposure_artifact_id']}.json"
    )
    artifact = store.read_sealed(
        artifact_path, schema="chronovisor.recall-exact-exposure.v1"
    )
    assert artifact["candidate_feature_snapshot"][0]["features"] == features
    assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600
    assert "private-session" not in json.dumps(receipt)
    empty = distill.record_exact_exposure(
        decision_id="empty-et",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=identity,
        candidate_refs=[],
        candidate_feature_snapshot=[
            {"candidate_id": "unselected", "features": distill.build_fast_features()}
        ],
        candidate_pool_refs=[
            {
                "candidate_id": "unselected",
                "selected": False,
                "page_id": "other",
                "page_content_sha256": "8" * 64,
                "rendered_context": "other context",
                "rendered_context_sha256": hashlib.sha256(b"other context").hexdigest(),
            }
        ],
        render_sha256="f" * 64,
        candidate_snapshot_sha256="0" * 64,
        observed_at="2026-08-14T00:00:01Z",
        root=tmp_path,
    )
    assert empty["candidate_ids"] == []
    with pytest.raises(distill.DistillationError, match="source version"):
        distill.record_exact_exposure(
            decision_id="version-mismatch",
            host="codex",
            session_id="private-session",
            query_semantic_sha256="7" * 64,
            policy_id=identity,
            candidate_refs=[
                {
                    "candidate_id": "page-v1",
                    "page_id": "page",
                    "page_content_sha256": "9" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            candidate_feature_snapshot=[
                {"candidate_id": "page-v1", "features": features}
            ],
            candidate_pool_refs=[
                {
                    "candidate_id": "page-v1",
                    "selected": True,
                    "page_id": "page",
                    "page_content_sha256": "7" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            render_sha256="6" * 64,
            candidate_snapshot_sha256="5" * 64,
            observed_at="2026-08-14T00:00:02Z",
            root=tmp_path,
        )
    with pytest.raises(distill.DistillationError, match="canonical"):
        distill.record_exact_exposure(
            decision_id="bad",
            host="codex",
            session_id="private-session",
            query_semantic_sha256="a" * 64,
            policy_id=identity,
            candidate_refs=[],
            candidate_feature_snapshot=[
                {
                    "candidate_id": "bad",
                    "features": {"query_chargram_coverage": 2},
                }
            ],
            render_sha256="b" * 64,
            candidate_snapshot_sha256="c" * 64,
            observed_at="2026-08-14T00:00:00Z",
            root=tmp_path,
        )


def test_structural_verifier_accepts_exact_anchor_not_near_match() -> None:
    commit = "a" * 40
    rally = {"query_ref": {"structural": {"commit": [commit], "path": []}}}
    exact = {"ref": {"structural": {"commit": [commit], "path": []}}}
    near = {"ref": {"structural": {"commit": [commit[:-1] + "b"], "path": []}}}
    assert (
        distill._default_structural_verifier(rally, exact, {}) == "exact_commit_overlap"
    )
    assert distill._default_structural_verifier(rally, near, {}) is None
    label = distill._teacher_label(
        {"verdict": "irrelevant"},
        verified_predicate="exact_commit_overlap",
    )
    assert label["authority"] == "teacher-only"


def test_grouped_rolling_split_never_separates_a_session() -> None:
    rows = [
        {
            "rally_id": f"r{index}",
            "session_cluster_id": f"s{index // 2}",
            "as_of": f"2026-08-{index + 1:02}T00:00:00Z",
        }
        for index in range(12)
    ]
    split = distill.grouped_rolling_split(rows)
    for index in range(0, 12, 2):
        assert split[f"r{index}"] == split[f"r{index + 1}"]
    assert {"train", "validation", "test"}.issubset(set(split.values()))


def test_split_plan_growth_preserves_cohort_and_embargoes_new_rallies(
    tmp_path: Path,
) -> None:
    rallies = [
        {
            "rally_id": f"r{index}",
            "session_cluster_id": f"s{index}",
            "as_of": f"2026-08-{index + 1:02}T00:00:00Z",
        }
        for index in range(10)
    ]
    first = distill._ensure_split_plan(
        tmp_path,
        rallies,
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    expanded = [
        *rallies,
        {
            "rally_id": "r10",
            "session_cluster_id": "s10",
            "as_of": "2026-08-11T00:00:00Z",
        },
    ]
    second = distill._ensure_split_plan(
        tmp_path,
        expanded,
        raw_watermark="c" * 64,
        model_cohort_sha256="b" * 64,
    )
    assert second["artifact_id"] != first["artifact_id"]
    assert {
        rally_id: second["assignments"][rally_id] for rally_id in first["assignments"]
    } == first["assignments"]
    assert second["assignments"]["r10"] == "embargo"

    next_cohort = distill._ensure_split_plan(
        tmp_path,
        expanded,
        raw_watermark="c" * 64,
        model_cohort_sha256="d" * 64,
    )
    assert next_cohort["assignments"] == distill.grouped_rolling_split(expanded)
    with pytest.raises(distill.DistillationError, match="rally set regressed"):
        distill._ensure_split_plan(
            tmp_path,
            expanded[1:],
            raw_watermark="e" * 64,
            model_cohort_sha256="d" * 64,
        )


def test_growth_keeps_work_plan_and_age_receipt_frozen(tmp_path: Path) -> None:
    rallies = [
        {
            "rally_id": f"r{index}",
            "session_cluster_id": f"s{index}",
            "as_of": f"2026-08-{index + 1:02}T00:00:00Z",
            "query_sha256": f"q{index}",
            "context_refs": [],
            "actual_answer_refs": [{"sha256": f"answer-{index}"}],
        }
        for index in range(10)
    ]
    first = distill._ensure_split_plan(
        tmp_path,
        rallies,
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    snapshots = {
        rally["rally_id"]: {
            "snapshot_sha256": f"{index + 1:064x}",
            "candidates": [{"candidate_id": "c", "text_sha256": "text"}],
        }
        for index, rally in enumerate(rallies)
    }
    _, before = distill._prepare_local_teacher_work(
        snapshots=snapshots,
        rally_by_id={str(rally["rally_id"]): rally for rally in rallies},
        split_assignments=first["assignments"],
        split_plan_id=first["artifact_id"],
        age_bands=first["age_bands"],
    )
    before_ox = distill._ox_prepare_tasks(
        config=distill.DistillationConfig(),
        snapshots=snapshots,
        rally_by_id={str(rally["rally_id"]): rally for rally in rallies},
        assignments=first["assignments"],
        split_plan_id=first["artifact_id"],
        profile_contract_id="c" * 64,
        candidate_indexed=False,
        candidate_state={},
        age_bands=first["age_bands"],
    )
    before_counterfactual, _ = distill._prepare_counterfactual_work(
        root=tmp_path,
        snapshots=snapshots,
        rally_by_id={str(rally["rally_id"]): rally for rally in rallies},
    )
    expanded = [
        *rallies,
        {
            "rally_id": "new",
            "session_cluster_id": "new",
            "as_of": "2026-09-01T00:00:00Z",
            "query_sha256": "q-new",
            "context_refs": [],
            "actual_answer_refs": [{"sha256": "answer-new"}],
        },
    ]
    second = distill._ensure_split_plan(
        tmp_path,
        expanded,
        raw_watermark="c" * 64,
        model_cohort_sha256="b" * 64,
    )
    assert second["scheduling_split_plan_id"] == first["artifact_id"]
    frozen_bands = distill._scheduling_age_bands(tmp_path, second)
    assert frozen_bands == first["age_bands"]
    snapshots["new"] = {
        "snapshot_sha256": "snapshot-new",
        "candidates": [{"candidate_id": "c", "text_sha256": "text"}],
    }
    _, after = distill._prepare_local_teacher_work(
        snapshots=snapshots,
        rally_by_id={str(rally["rally_id"]): rally for rally in expanded},
        split_assignments=second["assignments"],
        split_plan_id=distill._scheduling_split_plan_id(second),
        age_bands=frozen_bands,
    )
    after_ox = distill._ox_prepare_tasks(
        config=distill.DistillationConfig(),
        snapshots=snapshots,
        rally_by_id={str(rally["rally_id"]): rally for rally in expanded},
        assignments=second["assignments"],
        split_plan_id=distill._scheduling_split_plan_id(second),
        profile_contract_id="c" * 64,
        candidate_indexed=False,
        candidate_state={},
        age_bands=frozen_bands,
    )
    after_counterfactual, _ = distill._prepare_counterfactual_work(
        root=tmp_path,
        snapshots=snapshots,
        rally_by_id={str(rally["rally_id"]): rally for rally in expanded},
    )
    assert [item["work_id"] for item in after] == [item["work_id"] for item in before]
    assert [item["temporal_split"] for item in after] == [
        item["temporal_split"] for item in before
    ]
    assert [item["work_id"] for item in after_ox["work_items"]] == [
        item["work_id"] for item in before_ox["work_items"]
    ]
    assert [item["temporal_split"] for item in after_ox["work_items"]] == [
        item["temporal_split"] for item in before_ox["work_items"]
    ]
    assert [item["work_id"] for item in after_counterfactual] == [
        item["work_id"] for item in before_counterfactual
    ]


def test_all_teacher_schedulers_order_source_by_normalized_utc(tmp_path: Path) -> None:
    rallies = [
        {
            "rally_id": "z",
            "session_cluster_id": "z",
            "as_of": "2026-01-01T00:00:00Z",
            "query_sha256": "q-z",
            "context_refs": [],
            "actual_answer_refs": [{"sha256": "answer-z"}],
        },
        {
            "rally_id": "a",
            "session_cluster_id": "a",
            "as_of": "2026-01-01T00:00:00+09:00",
            "query_sha256": "q-a",
            "context_refs": [],
            "actual_answer_refs": [{"sha256": "answer-a"}],
        },
        {
            "rally_id": "old",
            "session_cluster_id": "old",
            "as_of": "2025-12-30T00:00:00Z",
            "query_sha256": "q-old",
            "context_refs": [],
            "actual_answer_refs": [{"sha256": "answer-old"}],
        },
    ]
    rally_by_id = {str(rally["rally_id"]): rally for rally in rallies}
    snapshots = {
        rally_id: {
            "snapshot_sha256": f"snapshot-{rally_id}",
            "candidates": [{"candidate_id": "c", "text_sha256": f"text-{rally_id}"}],
        }
        for rally_id in rally_by_id
    }
    assignments = {rally_id: "train" for rally_id in rally_by_id}
    expected = ["old", "a", "z"]
    ox = distill._ox_prepare_tasks(
        config=distill.DistillationConfig(),
        snapshots=snapshots,
        rally_by_id=rally_by_id,
        assignments=assignments,
        split_plan_id="a" * 64,
        profile_contract_id="b" * 64,
        candidate_indexed=False,
        candidate_state={},
    )
    assert [item["payload_ref"].split(":")[1] for item in ox["work_items"]] == expected
    _, local = distill._prepare_local_teacher_work(
        snapshots=snapshots,
        rally_by_id=rally_by_id,
        split_assignments=assignments,
        split_plan_id="a" * 64,
    )
    assert (
        list(dict.fromkeys(item["payload_ref"].split(":")[1] for item in local))
        == expected
    )
    counterfactual, _ = distill._prepare_counterfactual_work(
        root=tmp_path, snapshots=snapshots, rally_by_id=rally_by_id
    )
    assert [item["payload_ref"].split(":")[1] for item in counterfactual] == expected


def test_sealed_policy_pointer_and_nested_rollout_selection(tmp_path: Path) -> None:
    _config(tmp_path)
    with pytest.raises(
        distill.DistillationError, match="candidate lineage is incomplete"
    ):
        distill.publish_policy(
            distill.train_tiny_policy([]), lineage={"ledger_head": "x"}, root=tmp_path
        )
    candidate = _fixture_candidate(tmp_path)
    candidate_id = candidate["artifact_id"]
    store.write_pointer(tmp_path, "active", candidate_id)
    store.write_pointer(tmp_path, "lkg", candidate_id)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "canary",
            "worker_status": "idle",
            "rollout_percent": 25,
            "stage_started_at": "2026-08-01T00:00:00Z",
        },
    )
    assert distill.load_active_policy(tmp_path)["artifact_id"] == candidate_id
    assert (
        distill.load_policy_for_session("session", tmp_path)["artifact_id"]
        == candidate_id
    )
    health = store.snapshot(tmp_path)
    assert health["rollout"] == 25
    assert health["active_policy_id"] == candidate_id[:12]


def test_publish_policy_rejects_unsealed_zero_row_candidate(tmp_path: Path) -> None:
    with pytest.raises(
        distill.DistillationError, match="candidate lineage is incomplete"
    ):
        distill.publish_policy(distill.train_tiny_policy([]), lineage={}, root=tmp_path)


def test_bootstrap_is_automatic_and_never_replaces_legacy_serving(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    active_id = store.read_pointer(tmp_path, "active")["policy_id"]
    assert store.read_pointer(tmp_path, "lkg")["policy_id"] == active_id
    bootstrap = store.read_sealed(
        store.distillation_dir(tmp_path) / "policies" / f"{active_id}.json",
        schema=distill.POLICY_SCHEMA,
    )
    assert bootstrap["serve_mode"] == "legacy"
    assert distill.load_active_policy(tmp_path) == {}

    candidate = _fixture_candidate(tmp_path)
    candidate_id = candidate["artifact_id"]
    state_path = store.distillation_dir(tmp_path) / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "shadow",
            "rollout_percent": 0,
            "learning_halted": False,
        },
    )
    assert distill.load_policy_for_session("shadow", tmp_path) == {}

    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "canary",
            "rollout_percent": 5,
            "learning_halted": False,
        },
    )
    selected = []
    legacy = []
    for index in range(200):
        value = distill.load_policy_for_session(f"session-{index}", tmp_path)
        (selected if value else legacy).append(value)
    assert any(value.get("artifact_id") == candidate_id for value in selected)
    assert legacy

    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "rolled_back",
            "rollout_percent": 0,
            "learning_halted": True,
            "lkg_policy_id": active_id,
        },
    )
    assert distill.load_policy_for_session("rollback", tmp_path) == {}
    assert distill.load_active_policy(tmp_path) == {}


def test_reserved_store_fields_are_rejected_without_corrupting_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    store.append_chain(path, {"kind": "safe"})
    with pytest.raises(store.DistillationStoreError, match="reserved"):
        store.append_chain(path, {"previous_sha256": "forged"})
    assert store.verify_chain(path)["records"] == 1
    with pytest.raises(store.DistillationStoreError, match="reserved"):
        store.write_sealed_state(tmp_path / "state.json", {"schema": "forged"})


def test_preflight_and_chunk_are_deterministic_capture_only_below_floor(
    tmp_path: Path,
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    first = distill.preflight(
        raw_dir=raw_dir, root=tmp_path, config_path=config, runtime_commit="abcdef0"
    )
    second = distill.preflight(
        raw_dir=raw_dir, root=tmp_path, config_path=config, runtime_commit="abcdef0"
    )
    assert first == second
    assert not first["hard_floor"]["p5_allowed"]
    result = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    assert result["status"] == "capture_only"
    assert result["processed"] == 3
    retry = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    assert retry["processed"] == 0
    assert (
        store.verify_chain(store.distillation_dir(tmp_path) / "rally-manifest.jsonl")[
            "records"
        ]
        == 3
    )


def test_matching_p5_baseline_rejects_current_hold_and_runtime_drift(
    tmp_path: Path,
) -> None:
    _, _, baseline = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {
            "kind": "privacy-safe-baseline",
            "raw_watermark": "a" * 64,
            "config_sha256": "b" * 64,
            "runtime_commit": "abcdef0",
            "metrics": {"archive_commit": "abcdef0", "drift": False},
            "frozen_contract": {"feature_revision": distill.TEXT_FEATURE_REVISION},
            "offline_training_gate": {"passed": True},
            "hard_floor": {"p5_allowed": True, "reasons": []},
        },
        schema=distill.BASELINE_SCHEMA,
    )
    assert distill._matching_p5_baseline(tmp_path, baseline) == baseline
    held = {**baseline, "hard_floor": {"p5_allowed": False, "reasons": ["drift"]}}
    assert distill._matching_p5_baseline(tmp_path, held) is None
    changed_commit = {**baseline, "runtime_commit": "1234567"}
    assert distill._matching_p5_baseline(tmp_path, changed_commit) is None
    changed_metrics = {
        **baseline,
        "metrics": {"archive_commit": "abcdef0", "drift": True},
    }
    assert distill._matching_p5_baseline(tmp_path, changed_metrics) is None
    changed_contract = {**baseline, "frozen_contract": {"feature_revision": "other"}}
    assert distill._matching_p5_baseline(tmp_path, changed_contract) is None


def test_chunk_hard_timeout_preserves_state_counters_and_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    runtime_dir = store.distillation_dir(tmp_path)
    ledger_path = runtime_dir / "rally-manifest.jsonl"
    before_ledger = ledger_path.read_bytes()

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("steady chunk must use catalog and chain_head")

    monkeypatch.setattr(distill, "_events", unavailable)
    monkeypatch.setattr(distill, "build_historical_index", unavailable)
    monkeypatch.setattr(store, "verify_chain", unavailable)
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        max_elapsed_seconds=60,
    )
    assert result["status"] == "capture_only"
    assert result["processed"] == 0
    assert ledger_path.read_bytes() == before_ledger
    lock = store.acquire_nonblocking_lock(runtime_dir / "distillation-worker.lock")
    assert lock is not None
    store.release_lock(lock)


def test_chunk_reuses_loaded_candidates_for_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    candidate_path = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    original_read_chain = distill._read_chain
    candidate_reads = 0

    def counted_read_chain(path: Path) -> list[dict[str, Any]]:
        nonlocal candidate_reads
        if path == candidate_path:
            candidate_reads += 1
        return original_read_chain(path)

    monkeypatch.setattr(distill, "_read_chain", counted_read_chain)
    result = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    assert result["status"] == "capture_only"
    assert candidate_reads == 1


def test_timeout_after_atomic_batch_resumes_without_duplicate_or_cursor_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    runtime_dir = store.distillation_dir(tmp_path)
    state_path = runtime_dir / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "capture_only",
            "rollout_percent": 0,
            "raw_watermark": "0" * 64,
            "cold_start_lane_turn": 7,
            "teacher_model_calls": 5,
            "counterfactual_model_calls": 2,
        },
    )
    monkeypatch.setattr(
        distill,
        "build_historical_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("steady chunk must use catalog historical index")
        ),
    )
    first = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        cold_start=True,
        max_elapsed_seconds=60,
    )
    assert first["processed"] == 3
    manifest = runtime_dir / "rally-manifest.jsonl"
    committed = store.verify_chain(manifest)
    assert committed["records"] == 3

    resumed = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        cold_start=True,
        max_elapsed_seconds=300,
    )
    assert resumed["processed"] == 0
    assert store.verify_chain(manifest) == committed
    state = store.read_sealed(state_path)
    assert state["teacher_model_calls"] == 5
    assert state["counterfactual_model_calls"] == 2
    assert state["cold_start_lane_turn"] == 9
    assert state["raw_watermark"] == distill.committed_raw_watermark(raw_dir)


def test_final_state_is_last_commit_and_binds_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall.recall_runtime import RecallWallClockTimeout

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    runtime_dir = store.distillation_dir(tmp_path)
    state_path = runtime_dir / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "capture_only",
            "rollout_percent": 0,
            "raw_watermark": "0" * 64,
            "cold_start_lane_turn": 4,
            "teacher_model_calls": 2,
            "counterfactual_model_calls": 1,
        },
    )
    before_state = state_path.read_bytes()
    write_immutable = store.write_immutable

    def timeout_before_run_commit(
        directory: Path, *args: object, **kwargs: object
    ) -> object:
        if directory.name == "runs":
            raise RecallWallClockTimeout("former run boundary")
        return write_immutable(directory, *args, **kwargs)

    monkeypatch.setattr(store, "write_immutable", timeout_before_run_commit)
    deferred = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        max_elapsed_seconds=60,
    )
    assert deferred["status"] == "deferred"
    assert deferred["atomic_progress_may_be_present"] is True
    assert state_path.read_bytes() == before_state

    monkeypatch.setattr(store, "write_immutable", write_immutable)
    setitimer = distill.signal.setitimer
    deadline_cancelled = {"value": False}

    def track_timer(which: int, seconds: float, *args: object) -> object:
        if which == distill.signal.ITIMER_REAL and seconds == 0:
            deadline_cancelled["value"] = True
        return setitimer(which, seconds, *args)

    write_state = store.write_sealed_state

    def require_cancelled(path: Path, payload: object) -> dict[str, object]:
        artifact = write_state(path, payload)  # type: ignore[arg-type]
        if path == state_path and not deadline_cancelled["value"]:
            raise RecallWallClockTimeout("injected immediately after state")
        return artifact

    monkeypatch.setattr(distill.signal, "setitimer", track_timer)
    monkeypatch.setattr(store, "write_sealed_state", require_cancelled)
    completed = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        max_elapsed_seconds=60,
    )
    assert completed["status"] != "deferred"
    state = store.read_sealed(state_path)
    assert state["run_id"] == completed["run_id"]


def test_timeout_reports_payload_free_durable_workset_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall.recall_runtime import RecallWallClockTimeout

    runtime_dir = store.distillation_dir(tmp_path)
    for filename, progress in (
        (
            "ox-workset.sqlite3",
            {
                "cursor": {"candidate_count": 0, "label_count": 1, "revision_epoch": 0},
                "ledger_heads": {"candidate": "", "labels": "a" * 64},
                "provenance": {
                    "profile": distill.OX_SINGLE_PROFILE,
                    "profile_contract_id": "b" * 64,
                    "probe_revision": distill.OX_PROBE_REVISION,
                    "split_plan_id": "",
                },
                "progress_kind": "ox-workset-v2",
            },
        ),
        (
            "local-workset.sqlite3",
            {
                "cursor": {"candidate_count": 0, "label_count": 2},
                "ledger_heads": {"candidate": "", "labels": "c" * 64},
                "provenance": {
                    "assignment_revision": distill.ASSIGNMENT_REVISION,
                    "probe_revision": distill.PROBE_REVISION,
                    "split_plan_id": "",
                },
                "progress_kind": "local-workset-v2",
            },
        ),
    ):
        workset.DistillationWorkset(runtime_dir / filename).advance(
            [], {"source": filename}, progress=progress
        )

    def timeout(**_kwargs: object) -> dict[str, object]:
        raise RecallWallClockTimeout("test timeout")

    monkeypatch.setattr(distill, "_run_distillation_chunk_impl", timeout)
    result = distill.run_distillation_chunk(root=tmp_path, max_elapsed_seconds=60)

    assert result["reason"] == "wall_clock_timeout"
    for name in ("ox_workset", "local_workset"):
        status = result[name]
        assert status["observation"] == "available"
        assert status["last_durable_progress"]
        assert status["last_durable_receipt"]["generation"] > 0
        assert "payload_ref" not in json.dumps(status)

    def timeout_status(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RecallWallClockTimeout("second timeout")

    monkeypatch.setattr(workset.DistillationWorkset, "status", timeout_status)
    result = distill.run_distillation_chunk(root=tmp_path, max_elapsed_seconds=60)
    assert result["reason"] == "wall_clock_timeout"
    assert result["ox_workset"] == {"observation": "unavailable"}
    assert result["local_workset"] == {"observation": "unavailable"}


def test_chunk_accepts_bounded_production_sized_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        distill,
        "_run_distillation_chunk_impl",
        lambda **kwargs: {
            "status": "ok",
            "max_elapsed_seconds": kwargs["max_elapsed_seconds"],
        },
    )

    result = distill.run_distillation_chunk(root=tmp_path, max_elapsed_seconds=1_800)

    assert result == {"status": "ok", "max_elapsed_seconds": 1_800}
    with pytest.raises(distill.DistillationError, match="elapsed limit"):
        distill.run_distillation_chunk(root=tmp_path, max_elapsed_seconds=1_801)


def test_timeout_workset_observation_marks_missing_and_corrupt_queue(
    tmp_path: Path,
) -> None:
    runtime_dir = store.distillation_dir(tmp_path)
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "ox-workset.sqlite3").write_bytes(b"not sqlite")

    statuses = distill._timeout_workset_statuses(tmp_path)

    assert statuses["ox_workset"] == {"observation": "unavailable"}
    assert statuses["local_workset"] == {"observation": "missing"}


def test_chunk_commits_ox_ramp_with_the_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(
        tmp_path,
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        ox_free_only=False,
        ox_expires_at="2099-01-01T00:00:00Z",
        teacher_claim_limit=1,
    )

    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise AssertionError("teacher batch is intercepted")

    monkeypatch.setattr(
        distill,
        "_run_teacher_batch",
        lambda **_kwargs: distill._TeacherBatchResult(
            ramp_cap=5,
            ramp_valid_receipts=7,
            ramp_provider_attempts=9,
        ),
    )
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
        max_elapsed_seconds=60,
    )

    state = store.read_sealed(store.distillation_dir(tmp_path) / store.STATE_FILE)
    run = json.loads(
        (
            store.distillation_dir(tmp_path) / "runs" / f"{result['run_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert result["ox_ramp_cap"] == 5
    assert result["ox_ramp_valid_receipts"] == 7
    assert result["ox_ramp_provider_attempts"] == 9
    assert result["ox_ramp_request_revision"] == distill.OX_RAMP_REQUEST_REVISION
    assert state["ox_ramp_cap"] == 5
    assert state["ox_ramp_valid_receipts"] == 7
    assert state["ox_ramp_provider_attempts"] == 9
    assert state["ox_ramp_request_revision"] == distill.OX_RAMP_REQUEST_REVISION
    assert run["ox_ramp_cap"] == 5
    assert run["ox_ramp_valid_receipts"] == 7
    assert run["ox_ramp_provider_attempts"] == 9
    assert run["ox_ramp_request_revision"] == distill.OX_RAMP_REQUEST_REVISION


def test_chunk_parses_committed_raw_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    original_events = distill._events
    calls = 0

    def counted_events(path: Path) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return original_events(path)

    monkeypatch.setattr(distill, "_events", counted_events)
    result = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    assert result["processed"] == 3
    assert calls == 1


def test_preflight_automatically_aggregates_safe_runtime_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    commit = "a" * 40
    monkeypatch.setattr(
        distill.runtime_config,
        "runtime_identity",
        lambda: {
            "commit_id": commit,
            "expected_commit": commit,
            "drift": False,
        },
    )
    log_path = tmp_path / "recall" / "recall-log.jsonl"
    log_path.parent.mkdir(parents=True)
    rows = [
        {
            "decision_id": "one",
            "stage": "injected",
            "decision": "read",
            "latency_ms": 100,
            "status": "ok",
            "prompt_preview": "must never enter baseline",
        },
        {
            "decision_id": "two",
            "stage": "decision",
            "decision": "none",
            "latency_ms": 300,
            "status": "timeout",
            "session_id": "private-session",
        },
    ]
    log_path.write_text(
        json.dumps(rows[0]) + "\n{malformed\n" + json.dumps(rows[1]) + "\n",
        encoding="utf-8",
    )
    baseline = distill.preflight(raw_dir=raw_dir, root=tmp_path, config_path=config)
    metrics = baseline["metrics"]
    assert metrics["archive_commit"] == commit
    assert metrics["expected_commit"] == commit
    assert metrics["drift"] is False
    assert metrics["coverage_rate"] == 0.5
    assert metrics["abstain_rate"] == 0.5
    assert metrics["latency_p50_ms"] == 100
    assert metrics["latency_p95_ms"] == 300
    assert metrics["timeout_rate"] == 0.5
    assert metrics["wrong_domain_rate"] is None
    assert metrics["exact_outcome_links"] == 0
    assert not baseline["hard_floor"]["p5_allowed"]
    assert "teacher_labels_below_floor" in baseline["hard_floor"]["reasons"]
    assert "counterfactual_pairs_below_floor" in baseline["hard_floor"]["reasons"]
    serialized = json.dumps(baseline)
    assert "must never enter baseline" not in serialized
    assert "private-session" not in serialized

    overridden = distill.preflight(
        raw_dir=raw_dir,
        root=tmp_path,
        config_path=config,
        aggregate_metrics={"coverage_rate": 0.75},
    )
    assert overridden["metrics"]["coverage_rate"] == 0.75
    assert overridden["metrics"]["latency_p95_ms"] == 300


def test_p1_to_p4_teacher_backfill_runs_while_p5_is_held(tmp_path: Path) -> None:
    class FakeTeacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "rationale": "bounded fake",
                    }
                    for candidate in payload["candidates"]
                ]
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    teachers = {role: FakeTeacher(role) for role in distill.TEACHER_ROLES}
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers=teachers,
    )
    assert result["status"] == "capture_only"
    assert result["candidate_snapshots"] == 3
    assert result["labels_written"] > 0
    labels = store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl")
    assert all(row["authority"] in {"teacher-only", "verified"} for row in labels)
    assert all("reason" not in row and "rationale" not in row for row in labels)


def test_counterfactual_turn_without_real_work_falls_back_to_teacher(
    tmp_path: Path,
) -> None:
    class Teacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": item["candidate_id"],
                        "verdict": "uncertain",
                        "rationale": "bounded",
                    }
                    for item in payload["candidates"]
                ]
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "capture_only",
            "rollout_percent": 0,
            "teacher_model_calls": 3,
            "counterfactual_model_calls": 0,
        },
    )
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={role: Teacher(role) for role in distill.TEACHER_ROLES},
    )
    assert result["labels_written"] > 0


def test_deferred_counterfactual_turn_falls_back_to_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Teacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": item["candidate_id"],
                        "verdict": "uncertain",
                        "rationale": "bounded",
                    }
                    for item in payload["candidates"]
                ]
            }

    counterfactual_calls: list[bool] = []

    def counterfactual_block(*, execute: bool, **_kwargs: object) -> object:
        counterfactual_calls.append(execute)
        return distill._CounterfactualBlockResult(
            pending=True, deferred=execute
        )

    monkeypatch.setattr(distill, "_run_counterfactual_block", counterfactual_block)
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "capture_only",
            "rollout_percent": 0,
            "teacher_model_calls": 3,
            "counterfactual_model_calls": 0,
        },
    )
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={role: Teacher(role) for role in distill.TEACHER_ROLES},
    )

    assert result["labels_written"] > 0
    assert counterfactual_calls == [False, True]


def test_teacher_routes_make_progress_across_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Teacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": item["candidate_id"],
                        "verdict": "uncertain",
                        "rationale": "bounded",
                    }
                    for item in payload["candidates"]
                ]
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)

    def assigned(rally_id: str, candidate_id: str) -> dict[str, object]:
        del rally_id, candidate_id
        return {
            "revision": distill.ASSIGNMENT_REVISION,
            "owner": distill.TEACHER_ROLES[0],
            "probe_revision": distill.PROBE_REVISION,
            "probe": True,
            "routes": list(distill.TEACHER_ROLES),
        }

    monkeypatch.setattr(distill, "teacher_assignment", assigned)
    teachers = {role: Teacher(role) for role in distill.TEACHER_ROLES}
    for _ in range(6):
        distill.run_distillation_chunk(
            root=tmp_path,
            raw_dir=raw_dir,
            config_path=config,
            teachers=teachers,
        )
    labels = store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl")
    counts = {
        role: sum(row.get("route") == role for row in labels)
        for role in distill.TEACHER_ROLES
    }
    assert all(count > 0 for count in counts.values())
    assert max(counts.values()) - min(counts.values()) <= 16


def test_ox_single_teacher_materialization_binds_temporal_quality_evidence(
    tmp_path: Path,
) -> None:
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    rallies = [
        {
            "rally_id": f"rally-{index}",
            "session_cluster_id": f"session-{index}",
            "as_of": f"2026-01-0{index + 1}T00:00:00Z",
        }
        for index in range(3)
    ]
    plan = distill._ensure_split_plan(
        tmp_path,
        rallies,
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    snapshots = {
        str(rally["rally_id"]): {
            "as_of": rally["as_of"],
            "snapshot_sha256": "c" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "candidates": [
                {
                    "candidate_id": f"candidate-{index}",
                    "features": features,
                }
            ],
        }
        for index, rally in enumerate(rallies)
    }
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        hard_floor_teacher_labels=1,
        hard_floor_teacher_per_class=1,
        hard_floor_probe_pairs=1,
    )
    profile_contract_id = distill._ensure_ox_profile_contract(tmp_path, config)[
        "artifact_id"
    ]
    for index, rally in enumerate(rallies):
        payload_source = {
            "rally_id": rally["rally_id"],
            "candidate_id": f"candidate-{index}",
            "snapshot_sha256": "c" * 64,
            "query_sha256": "",
            "candidate_text_sha256": "",
            "context_sha256": [],
        }
        payload_digest = canonical_json.canonical_json_sha256_strict(payload_source)
        store.append_chain(
            label_path,
            {
                "kind": "teacher-label",
                "status": "completed",
                "work_id": canonical_json.canonical_json_sha256_strict(
                    {
                        "kind": "ox-teacher-label-v1",
                        "profile": distill.OX_SINGLE_PROFILE,
                        "cohort": distill.OX_SINGLE_COHORT,
                        "route": "opencode-go/deepseek-v4-flash",
                        "profile_contract_id": profile_contract_id,
                        "payload_digest": payload_digest,
                    }
                ),
                "payload_digest": payload_digest,
                "payload_source": payload_source,
                "expires_at": config.ox_expires_at,
                "request_revision": distill.OX_RAMP_REQUEST_REVISION,
                "rally_id": rally["rally_id"],
                "candidate_id": f"candidate-{index}",
                "route": "opencode-go/deepseek-v4-flash",
                "model_digest": "d" * 64,
                "prompt_sha256": "e" * 64,
                "schema_sha256": "f" * 64,
                "profile": distill.OX_SINGLE_PROFILE,
                "cohort": distill.OX_SINGLE_COHORT,
                "profile_contract_id": profile_contract_id,
                "source_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_ox_identity_sha256": "c" * 64,
                "route_identity": {
                    "provider": "opencode-go",
                    "model": "opencode-go/deepseek-v4-flash",
                    "location": "remote",
                },
                "as_of": rally["as_of"],
                "group_id": rally["session_cluster_id"],
                "split_plan_id": plan["artifact_id"],
                "assignment": {"probe": False},
                "dimension": "relevance",
                "verdict": "relevant",
                "authority": "teacher-only",
                "features": features,
            },
        )
    artifact = distill.materialize_training_rows(
        tmp_path,
        _rallies=rallies,
        _snapshots=snapshots,
        _label_rows=store.read_chain(label_path),
    )
    rows = artifact["rows"]
    test_row = next(row for row in rows if row["split"] == "test")
    assert all(row["feature_parity"] is True for row in rows)
    assert all(row["future_leakage"] is False for row in rows)
    assert all(row["fixed_split_plan"] is True for row in rows)
    assert test_row["locked_test_read_only"] is True
    assert test_row["locked_test_evidence_ref"] == f"split-plan:{plan['artifact_id']}"
    gate = distill._offline_training_gate(
        rows,
        config,
        root=tmp_path,
    )
    assert gate["schema"] == "chronovisor.recall-single-teacher-gate.v1"
    assert "blind_repeat_pairs_below_floor" in gate["reasons"]
    assert "teacher_models_not_distinct" not in gate["reasons"]
    assert gate["identity"]["profile_contract_id"] == profile_contract_id
    tampered_labels = [dict(row) for row in store.read_chain(label_path)]
    tampered_labels[0]["request_revision"] = (
        "json-schema-core-label-abstain-16k-240s-v5"
    )
    tampered_labels[0]["record_sha256"] = canonical_json.canonical_json_sha256_strict(
        {
            key: value
            for key, value in tampered_labels[0].items()
            if key != "record_sha256"
        }
    )
    tampered_rows = distill.materialize_training_rows(
        tmp_path,
        _rallies=rallies,
        _snapshots=snapshots,
        _label_rows=tampered_labels,
    )["rows"]
    assert all(
        row["request_revision"] == distill.OX_RAMP_REQUEST_REVISION
        for row in tampered_rows
    )
    assert len(tampered_rows) < len(rows)
    _, ox_model_cohort = distill._active_training_cohort(
        rows,
        teacher_profile=distill.OX_SINGLE_PROFILE,
        profile_contract_id=profile_contract_id,
    )

    extended_plan_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": "1" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "model_cohort_sha256": ox_model_cohort["cohort_sha256"],
            "split_revision": "grouped-rolling-v1",
            "assignments": {**plan["assignments"], "future-rally": "embargo"},
        },
        schema=distill.SPLIT_PLAN_SCHEMA,
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": extended_plan_id},
    )
    extended = distill.materialize_training_rows(
        tmp_path,
        _rallies=rallies,
        _snapshots=snapshots,
        _label_rows=store.read_chain(label_path),
    )["rows"]
    assert all(row["fixed_split_plan"] is True for row in extended)
    assert (
        next(row for row in extended if row["split"] == "test")[
            "locked_test_evidence_ref"
        ]
        == f"split-plan:{extended_plan_id}"
    )
    assert (
        "fixed_split_plan_missing"
        not in distill._offline_training_gate(extended, config, root=tmp_path)[
            "reasons"
        ]
    )
    assert (
        "split_plan_cohort_mismatch"
        not in distill._offline_training_gate(extended, config, root=tmp_path)[
            "reasons"
        ]
    )

    changed_assignments = dict(plan["assignments"])
    changed_assignments[test_row["rally_id"]] = "train"
    changed_plan_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": "2" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "model_cohort_sha256": ox_model_cohort["cohort_sha256"],
            "split_revision": "grouped-rolling-v1",
            "assignments": changed_assignments,
        },
        schema=distill.SPLIT_PLAN_SCHEMA,
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": changed_plan_id},
    )
    incompatible = distill.materialize_training_rows(
        tmp_path,
        _rallies=rallies,
        _snapshots=snapshots,
        _label_rows=store.read_chain(label_path),
    )["rows"]
    assert (
        next(row for row in incompatible if row["rally_id"] == test_row["rally_id"])[
            "fixed_split_plan"
        ]
        is False
    )

    next_cohort_plan_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": "3" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "model_cohort_sha256": "3" * 64,
            "split_revision": "grouped-rolling-v1",
            "assignments": plan["assignments"],
        },
        schema=distill.SPLIT_PLAN_SCHEMA,
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": next_cohort_plan_id},
    )
    next_cohort = distill.materialize_training_rows(
        tmp_path,
        _rallies=rallies,
        _snapshots=snapshots,
        _label_rows=store.read_chain(label_path),
    )["rows"]
    assert all(row["fixed_split_plan"] is True for row in next_cohort)
    assert (
        "split_plan_cohort_mismatch"
        in distill._offline_training_gate(next_cohort, config, root=tmp_path)["reasons"]
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": extended_plan_id},
    )

    mismatched = [{**row, "profile_contract_id": "0" * 64} for row in rows]
    rejected = distill._offline_training_gate(mismatched, config, root=tmp_path)
    assert "profile_contract_mismatch" in rejected["reasons"]

    disabled = distill._offline_training_gate(
        rows, replace(config, ox_enabled=False), root=tmp_path
    )
    assert "ox_profile_disabled" in disabled["reasons"]

    invalid = distill._offline_training_gate(
        cast(list[dict[str, object]], [None]), config, root=tmp_path
    )
    assert "input_row_invalid" in invalid["reasons"]

    unsafe_labels = [
        {
            **row,
            "error_class": "invalid_teacher_output",
            "negative_veto_conflict": True,
        }
        for row in store.read_chain(label_path)
    ]
    unsafe = distill.materialize_training_rows(
        tmp_path,
        _rallies=rallies,
        _snapshots=snapshots,
        _label_rows=unsafe_labels,
    )
    unsafe_gate = distill._offline_training_gate(unsafe["rows"], config, root=tmp_path)
    assert unsafe_gate["labels"]["eligible"] == 0
    assert "teacher_labels_below_floor" in unsafe_gate["reasons"]


def test_authoritative_row_binding_rejects_recomputed_cross_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def receipt_binding(self) -> dict[str, str]:
            return {
                "source_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_ox_identity_sha256": "c" * 64,
            }

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in payload["candidates"]
                ],
                **_ox_metadata(payload),
            }

    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    rallies = [
        {
            "rally_id": f"r{index}",
            "session_cluster_id": f"s{index}",
            "as_of": f"2026-01-{index + 1:02}T00:00:00Z",
            "query_sha256": f"q{index}",
            "context_refs": [],
        }
        for index in range(10)
    ]
    plan = distill._ensure_split_plan(
        tmp_path,
        rallies,
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    snapshots = {
        rally["rally_id"]: {
            "as_of": rally["as_of"],
            "snapshot_sha256": f"{index + 1:064x}",
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "candidates": [
                {
                    "candidate_id": f"c{index}",
                    "text_sha256": f"text-{index}",
                    "features": features,
                }
            ],
        }
        for index, rally in enumerate(rallies)
    }
    monkeypatch.setattr(
        distill,
        "_materialization_rallies",
        lambda _root, _supplied: {str(rally["rally_id"]): rally for rally in rallies},
    )
    monkeypatch.setattr(
        distill, "_materialization_snapshots", lambda _root, _supplied: snapshots
    )
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        teacher_claim_limit=1,
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    results = [
        distill._run_teacher_batch(
            root=tmp_path,
            config=config,
            teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
            snapshots=snapshots,
            rally_by_id={str(rally["rally_id"]): rally for rally in rallies},
            texts={
                **{f"q{index}": f"question {index}" for index in range(10)},
                **{f"text-{index}": f"evidence {index}" for index in range(10)},
            },
            label_path=label_path,
            label_rows=store.read_chain(label_path),
            structural_verifier=lambda *_args: None,
        )
        for _ in range(10)
    ]
    assert sum(result.labels_written for result in results) == 10
    rows = distill.materialize_training_rows(tmp_path)["rows"]
    assert rows and all(
        distill._materialized_row_integrity(row, root=tmp_path, split_plan=plan)
        for row in rows
    )
    evil_route = {
        **rows[0],
        "route_identity": {
            "provider": "evil-provider",
            "model": "evil-model",
            "location": "remote",
        },
        "route_identity_exact": False,
    }
    assert distill._materialized_row_integrity(evil_route) is False
    original, unrelated = rows[0], rows[-1]
    source = dict(unrelated["payload_source"])
    payload_digest = canonical_json.canonical_json_sha256_strict(source)
    forged = {
        **original,
        "rally_id": unrelated["rally_id"],
        "candidate_id": unrelated["candidate_id"],
        "session_cluster_id": unrelated["session_cluster_id"],
        "as_of": unrelated["as_of"],
        "features": unrelated["features"],
        "payload_source": source,
        "payload_digest": payload_digest,
    }
    forged["work_id"] = canonical_json.canonical_json_sha256_strict(
        {
            "kind": "ox-teacher-label-v1",
            "profile": distill.OX_SINGLE_PROFILE,
            "cohort": distill.OX_SINGLE_COHORT,
            "route": "opencode-go/deepseek-v4-flash",
            "profile_contract_id": forged["profile_contract_id"],
            "payload_digest": payload_digest,
        }
    )
    forged["request_sha256"] = distill.expected_ox_request_sha256(
        profile_contract_id=str(forged["profile_contract_id"]),
        payload_digest=payload_digest,
    )
    forged["provider_request_sha256"] = distill.expected_ox_provider_request_sha256(
        profile_contract_id=str(forged["profile_contract_id"]),
        payload_digest=payload_digest,
        work_id=str(forged["work_id"]),
        expires_at=str(forged["expires_at"]),
    )
    forged["provider_receipt_sha256"] = "e" * 64
    assert distill._materialized_row_integrity(forged) is False
    assert (
        distill._materialized_row_integrity(forged, root=tmp_path, split_plan=plan)
        is False
    )


def test_configured_local_route_binding_rejects_evil_canonical_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    roles = (
        *distill.TEACHER_ROLES,
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
    )
    routes = tuple(
        SimpleNamespace(
            role=role,
            provider=f"provider-{index}",
            model=f"model-{index}",
            location="local",
        )
        for index, role in enumerate(roles)
    )
    digests = {role: f"{index + 1:064x}" for index, role in enumerate(roles)}
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama, "runtime_generation_route_fingerprints", lambda _routes: digests
    )
    identities = {
        route.role: {
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location,
        }
        for route in routes
    }
    teacher = {
        "source": "teacher-label",
        "profile": distill.LOCAL_TRIAD_PROFILE,
        "cohort": distill.LOCAL_TRIAD_PROFILE,
        "profile_contract_id": "",
        "expires_at": "",
        "identity_revision": "local-teacher-v1",
        "request_revision": "local-teacher-v1",
        "assignment_revision": distill.ASSIGNMENT_REVISION,
        "route": distill.TEACHER_ROLES[0],
        "route_identity": identities[distill.TEACHER_ROLES[0]],
        "model_digest": digests[distill.TEACHER_ROLES[0]],
    }
    counterfactual = {
        "source": "counterfactual-label",
        "profile": distill.LOCAL_TRIAD_PROFILE,
        "cohort": distill.LOCAL_TRIAD_PROFILE,
        "profile_contract_id": "",
        "expires_at": "",
        "identity_revision": "local-blind-counterfactual-v1",
        "request_revision": "local-blind-counterfactual-v1",
        "assignment_revision": distill.ASSIGNMENT_REVISION,
        "counterfactual_producer": "chronovisor-local-blind-v1",
        "counterfactual_revision": "two-order-locked-v1",
        "blind_orders": ["a0_first", "a1_first"],
        "generator_route_identity": identities[roles[-2]],
        "judge_route_identity": identities[roles[-1]],
        "generator_model_digest": digests[roles[-2]],
        "judge_model_digest": digests[roles[-1]],
    }
    assert distill._configured_local_route_binding(teacher) is True
    assert distill._configured_local_route_binding(counterfactual) is True
    assert (
        distill._configured_local_route_binding(
            {
                **teacher,
                "route_identity": {**teacher["route_identity"], "model": "evil"},
            }
        )
        is False
    )
    assert (
        distill._configured_local_route_binding(
            {**teacher, "identity_revision": "evil-revision"}
        )
        is False
    )
    assert (
        distill._configured_local_route_binding(
            {**counterfactual, "profile": "evil-profile", "cohort": "evil-cohort"}
        )
        is False
    )
    assert (
        distill._configured_local_route_binding(
            {**counterfactual, "identity_revision": "evil-revision"}
        )
        is False
    )
    assert (
        distill._configured_local_route_binding(
            {
                **counterfactual,
                "judge_model_digest": "f" * 64,
            }
        )
        is False
    )


def test_ox_locked_blind_repeats_are_reversed_and_resume_without_duplicates(
    tmp_path: Path,
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.requests: list[list[str]] = []

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            self.requests.append([str(row["candidate_id"]) for row in candidates])
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": (
                            "relevant"
                            if candidate["candidate_id"] == "candidate-a"
                            else "irrelevant"
                        ),
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in candidates
                ],
                **_ox_metadata(payload),
            }

    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    rally = {
        "rally_id": "rally-test",
        "session_cluster_id": "session-test",
        "as_of": "2026-01-03T00:00:00Z",
        "query_sha256": "query",
        "context_refs": [],
    }
    distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    snapshot = {
        "as_of": rally["as_of"],
        "snapshot_sha256": "c" * 64,
        "feature_revision": distill.TEXT_FEATURE_REVISION,
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "text_sha256": "candidate-a",
                "features": features,
            },
            {
                "candidate_id": "candidate-b",
                "text_sha256": "candidate-b",
                "features": features,
            },
        ],
    }
    teacher = RemoteTeacher()
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        teacher_claim_limit=1,
        hard_floor_teacher_labels=2,
        hard_floor_teacher_per_class=1,
        hard_floor_probe_pairs=2,
    )
    results = [
        distill._run_teacher_batch(
            root=tmp_path,
            config=config,
            teachers={distill.OX_TEACHER_ROLE: teacher},
            snapshots={"rally-test": snapshot},
            rally_by_id={"rally-test": rally},
            texts={
                "query": "what proves the claim",
                "candidate-a": "first bounded fact",
                "candidate-b": "second bounded fact",
            },
            label_path=label_path,
            label_rows=store.read_chain(label_path),
            structural_verifier=lambda *_args: None,
        )
        for _ in range(12)
    ]
    assert sum(result.labels_written for result in results) == 6
    assert ["candidate-a", "candidate-b"] in teacher.requests
    assert ["candidate-b", "candidate-a"] in teacher.requests
    labels = store.read_chain(label_path)
    probes = [row for row in labels if row["assignment"]["probe"] is True]
    assert len(probes) == 4
    assert {row["assignment"]["blind_order"] for row in probes} == {
        "a_first",
        "b_first",
    }
    assert all(row["assignment"]["fixed_repeat"] is True for row in probes)
    training = distill.materialize_training_rows(
        tmp_path,
        _rallies=[rally],
        _snapshots={"rally-test": snapshot},
        _label_rows=labels,
    )
    gate = distill._offline_training_gate(training["rows"], config, root=tmp_path)
    assert gate["blind_repeat"]["complete_pairs"] == 2
    before = len(teacher.requests)
    resumed = distill._run_teacher_batch(
        root=tmp_path,
        config=config,
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={"rally-test": snapshot},
        rally_by_id={"rally-test": rally},
        texts={
            "query": "what proves the claim",
            "candidate-a": "first bounded fact",
            "candidate-b": "second bounded fact",
        },
        label_path=label_path,
        label_rows=labels,
        structural_verifier=lambda *_args: None,
    )
    assert resumed.labels_written == 0
    assert len(teacher.requests) == before
    assert len(store.read_chain(label_path)) == len(labels)


def test_ox_incomplete_locked_repeat_is_not_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_distillation_workset as workset_module

    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, _payload: object) -> dict[str, object]:
            self.calls += 1
            raise AssertionError("incomplete repeat must not reach the provider")

    rally = {
        "rally_id": "rally-test",
        "session_cluster_id": "session-test",
        "as_of": "2026-01-03T00:00:00Z",
        "query_sha256": "query",
        "context_refs": [],
    }
    distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    original_claim = workset_module.DistillationWorkset.claim
    claim_calls = 0

    def claim_one(
        self: object, kind: str, _limit: int, owner: str, lease: float
    ) -> object:
        nonlocal claim_calls
        claim_calls += 1
        if claim_calls > 1:
            return ()
        return original_claim(self, kind, 1, owner, lease)  # type: ignore[arg-type]

    monkeypatch.setattr(workset_module.DistillationWorkset, "claim", claim_one)
    teacher = RemoteTeacher()
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            "rally-test": {
                "candidates": [
                    {"candidate_id": "candidate-a", "text_sha256": "candidate-a"},
                    {"candidate_id": "candidate-b", "text_sha256": "candidate-b"},
                ]
            }
        },
        rally_by_id={"rally-test": rally},
        texts={
            "query": "what proves the claim",
            "candidate-a": "first bounded fact",
            "candidate-b": "second bounded fact",
        },
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )
    assert result.labels_written == 0
    assert result.workset_status["quarantined"] == 1  # type: ignore[index]
    assert teacher.calls == 0


def test_ox_probe_revision_reissues_terminal_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.accept = False
            self.calls = 0

        def accepts_egress_payload(self, _payload: object) -> bool:
            return self.accept

        def evaluate(self, _payload: object) -> dict[str, object]:
            self.calls += 1
            return {"_failure": {"class": "invalid_response"}}

    rally = {
        "rally_id": "rally-test",
        "session_cluster_id": "session-test",
        "as_of": "2026-01-03T00:00:00Z",
        "query_sha256": "query",
        "context_refs": [],
    }
    distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        teacher_claim_limit=1,
        hard_floor_probe_pairs=1,
    )
    snapshots = {
        "rally-test": {
            "snapshot_sha256": "c" * 64,
            "candidates": [
                {"candidate_id": "candidate-a", "text_sha256": "candidate-a"},
                {"candidate_id": "candidate-b", "text_sha256": "candidate-b"},
            ],
        }
    }
    texts = {
        "query": "what proves the claim",
        "candidate-a": "first bounded fact",
        "candidate-b": "second bounded fact",
    }
    teacher = RemoteTeacher()
    monkeypatch.setattr(
        distill, "OX_PROBE_REVISION", "single-teacher-repeat-v1", raising=False
    )
    distill._run_teacher_batch(
        root=tmp_path,
        config=config,
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots=snapshots,
        rally_by_id={"rally-test": rally},
        texts=texts,
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )
    teacher.accept = True
    monkeypatch.setattr(distill, "OX_PROBE_REVISION", "deepseek-single-teacher-repeat-v1")

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=config,
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots=snapshots,
        rally_by_id={"rally-test": rally},
        texts=texts,
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert teacher.calls == 1
    assert result.model_calls == 1
    assert result.workset_status["quarantined"] == 6  # type: ignore[index]
    assert result.workset_status["ready"] == 4  # type: ignore[index]


def test_ox_ramp_counts_provider_receipts_not_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_distillation_dispatcher as dispatcher

    captured: dict[str, int] = {}
    fields = {
        "candidate_id",
        "verdict",
        "confidence",
        "rationale",
    }

    def dispatch(_batches: object, _evaluate: object, **kwargs: object) -> list[object]:
        callback = kwargs["valid_result_count"]
        assert callable(callback)
        response = {
            "labels": [
                {
                    "candidate_id": str(index),
                    "verdict": "relevant",
                    "confidence": 1.0,
                    "rationale": "bounded",
                }
                for index in range(16)
            ]
        }
        assert all(set(label) == fields for label in response["labels"])
        captured["receipt_count"] = callback(response)
        captured["max_inflight"] = cast(int, kwargs["max_inflight"])
        return []

    monkeypatch.setattr(dispatcher, "dispatch_claimed_work", dispatch)

    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise AssertionError("intercepted dispatcher must not evaluate")

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE, ox_enabled=True
        ),
        teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
        snapshots={
            "rally": {
                "candidates": [
                    {"candidate_id": "candidate", "text_sha256": "candidate"}
                ]
            }
        },
        rally_by_id={
            "rally": {"rally_id": "rally", "query_sha256": "query", "context_refs": []}
        },
        texts={"query": "what proves the claim", "candidate": "bounded fact"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )
    assert result.labels_written == 0
    assert captured == {"receipt_count": 0, "max_inflight": 1}

    active = 0
    peak = 0
    active_lock = threading.Lock()

    def one_receipt(_work: int) -> dict[str, object]:
        nonlocal active, peak
        with active_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with active_lock:
            active -= 1
        return {"labels": []}

    dispatcher.SingleTeacherDispatcher(
        one_receipt,
        max_inflight=10,
        min_valid_results_per_cap=20,
        valid_result_count=lambda _response: 1,
    ).dispatch(list(range(19)))
    assert peak == 1


@pytest.mark.parametrize(
    (
        "state_kind",
        "matching_contract",
        "request_revision",
        "provider_attempts",
        "expected_initial",
    ),
    [
        ("worker-state", True, distill.OX_RAMP_REQUEST_REVISION, 20, (2, 19, 20)),
        ("worker-state", True, distill.OX_RAMP_REQUEST_REVISION, None, (1, 0, 0)),
        ("worker-state", True, None, 20, (1, 0, 0)),
        (
            "worker-state",
            True,
            "json-schema-core-label-abstain-16k-240s-v5",
            20,
            (1, 0, 0),
        ),
        ("worker-state", False, distill.OX_RAMP_REQUEST_REVISION, 20, (1, 0, 0)),
        ("forged", True, distill.OX_RAMP_REQUEST_REVISION, 20, (1, 0, 0)),
    ],
)
def test_ox_ramp_resumes_only_for_the_same_profile_contract_and_request_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_kind: str,
    matching_contract: bool,
    request_revision: str | None,
    provider_attempts: int | None,
    expected_initial: tuple[int, int, int],
) -> None:
    from chronovisor.recall import recall_distillation_dispatcher as dispatcher

    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
    )
    profile_contract_id = distill._ensure_ox_profile_contract(tmp_path, config)[
        "artifact_id"
    ]
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": state_kind,
            "status": "capture_only",
            "ox_profile_stopped": not matching_contract,
            "ox_profile_contract_id": (
                profile_contract_id if matching_contract else "0" * 64
            ),
            "ox_ramp_cap": 2,
            "ox_ramp_valid_receipts": 19,
            **(
                {"ox_ramp_request_revision": request_revision}
                if request_revision is not None
                else {}
            ),
            **(
                {"ox_ramp_provider_attempts": provider_attempts}
                if provider_attempts is not None
                else {}
            ),
        },
    )
    captured: dict[str, int] = {}

    def dispatch(_batches: object, _evaluate: object, **kwargs: object) -> list[object]:
        captured["initial_cap"] = cast(int, kwargs["initial_cap"])
        captured["initial_valid_results"] = cast(int, kwargs["initial_valid_results"])
        captured["max_inflight"] = cast(int, kwargs["max_inflight"])
        return []

    monkeypatch.setattr(dispatcher, "dispatch_claimed_work", dispatch)

    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise AssertionError("intercepted dispatcher must not evaluate")

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=config,
        teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
        snapshots={
            "rally": {
                "candidates": [
                    {"candidate_id": "candidate", "text_sha256": "candidate"}
                ]
            }
        },
        rally_by_id={
            "rally": {
                "rally_id": "rally",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        texts={"query": "what proves the claim", "candidate": "bounded fact"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert captured == {
        "initial_cap": expected_initial[0],
        "initial_valid_results": 0,
        "max_inflight": expected_initial[0],
    }
    assert result.ramp_cap == expected_initial[0]
    assert result.ramp_valid_receipts == expected_initial[1]
    assert result.ramp_provider_attempts == expected_initial[2]


def test_ox_bulk_resume_does_not_reseal_completed_ramp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_distillation_dispatcher as dispatcher

    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        teacher_max_inflight=10,
    )
    contract = distill._ensure_ox_profile_contract(tmp_path, config)
    first_label = store.append_chain(
        store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )
    distill._append_ox_event(
        tmp_path,
        "ox-ramp-receipts.jsonl",
        {
            "event_version": 2,
            "kind": "ox-ramp-stage",
            "profile_contract_id": contract["artifact_id"],
            "source_commit": contract["source_commit"],
            "source_tree_sha256": contract["source_tree_sha256"],
            "source_ox_identity_sha256": contract["source_ox_identity_sha256"],
            "request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "expires_at": contract["expires_at"],
            "cap": 10,
            "next_cap": 10,
            "valid_receipts": 20,
            "attempts": 20,
            "work_ids": [f"{index + 1:064x}" for index in range(20)],
            "label_count": 1,
            "label_head_sha256": first_label["record_sha256"],
            "failure_record_count": 0,
            "failure_head_sha256": "",
            "captured_at": "2026-08-25T00:00:00Z",
        },
    )
    store.append_chain(
        store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        {
            "kind": "teacher-label",
            "status": "completed",
            "profile": distill.LOCAL_TRIAD_PROFILE,
        },
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "capture_only",
            "ox_profile_stopped": False,
            "ox_profile_contract_id": contract["artifact_id"],
            "ox_ramp_cap": 10,
            "ox_ramp_valid_receipts": 20,
            "ox_ramp_request_revision": distill.OX_RAMP_REQUEST_REVISION,
            "ox_ramp_provider_attempts": 20,
        },
    )
    monkeypatch.setattr(
        dispatcher, "dispatch_claimed_work", lambda *_args, **_kwargs: []
    )

    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise AssertionError("intercepted dispatcher must not evaluate")

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=config,
        teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
        snapshots={
            "rally": {
                "candidates": [
                    {"candidate_id": "candidate", "text_sha256": "candidate"}
                ]
            }
        },
        rally_by_id={
            "rally": {
                "rally_id": "rally",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        texts={"query": "what proves the claim", "candidate": "bounded fact"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.ramp_cap == 10
    assert len(
        store.read_chain(store.distillation_dir(tmp_path) / "ox-ramp-receipts.jsonl")
    ) == 1


def test_ox_claim_limit_one_keeps_one_batch_per_ramp_slot() -> None:
    claims = [SimpleNamespace(work_id=f"work-{index}") for index in range(3)]
    tasks = {
        claim.work_id: {
            "probe_batch_id": "",
            "candidate": {"candidate_id": f"candidate-{index}"},
            "assignment": {},
            "input": {"candidate_id": f"candidate-{index}"},
        }
        for index, claim in enumerate(claims)
    }
    released: list[object] = []
    queue = SimpleNamespace(release_unattempted=released.extend)

    batches, rescanned = distill._ox_prepare_batches(
        claims=claims,
        tasks=tasks,
        workset=queue,
        claim_limit=1,
        ramp_cap=2,
        preflight=None,
        payload_scan_remaining=None,
    )

    assert [[claim.work_id for claim in batch] for batch in batches] == [
        ["work-0"],
        ["work-1"],
    ]
    assert released == [claims[2]]
    assert rescanned == 0


def test_ox_ramp_request_revision_tracks_core_label_schema() -> None:
    assert (
        distill.OX_RAMP_REQUEST_REVISION == "json-schema-core-label-abstain-16k-600s-v8"
    )


def test_ox_ramp_requires_a_95_percent_provider_success_rate_to_advance() -> None:
    held = distill._advance_ox_ramp(
        cap=1,
        valid_receipts=0,
        provider_attempts=0,
        valid_results=20,
        actual_attempts=22,
        rate_limited=False,
        stopped=False,
        max_inflight=10,
    )
    assert held == (1, 20, 22)
    assert distill._advance_ox_ramp(
        cap=held[0],
        valid_receipts=held[1],
        provider_attempts=held[2],
        valid_results=18,
        actual_attempts=18,
        rate_limited=False,
        stopped=False,
        max_inflight=10,
    ) == (2, 0, 0)


def test_ox_ramp_low_rate_state_survives_each_chunk_normalization() -> None:
    state = (1, 20, 22)
    for receipts in range(21, 38):
        state = distill._advance_ox_ramp(
            cap=state[0],
            valid_receipts=state[1],
            provider_attempts=state[2],
            valid_results=1,
            actual_attempts=1,
            rate_limited=False,
            stopped=False,
            max_inflight=10,
        )
        assert state == (1, receipts, receipts + 2)
        assert (
            distill._ox_ramp_state(
                {
                    "ox_ramp_cap": state[0],
                    "ox_ramp_valid_receipts": state[1],
                    "ox_ramp_provider_attempts": state[2],
                    "ox_ramp_request_revision": distill.OX_RAMP_REQUEST_REVISION,
                },
                10,
            )
            == state
        )
    assert distill._advance_ox_ramp(
        cap=state[0],
        valid_receipts=state[1],
        provider_attempts=state[2],
        valid_results=1,
        actual_attempts=1,
        rate_limited=False,
        stopped=False,
        max_inflight=10,
    ) == (2, 0, 0)


def test_ox_ramp_final_cap_low_rate_state_recovers_then_freezes() -> None:
    state = (10, 20, 22)
    for receipts in range(21, 39):
        state = distill._advance_ox_ramp(
            cap=state[0],
            valid_receipts=state[1],
            provider_attempts=state[2],
            valid_results=1,
            actual_attempts=1,
            rate_limited=False,
            stopped=False,
            max_inflight=10,
        )
        assert state == (10, receipts, receipts + 2)
        assert (
            distill._ox_ramp_state(
                {
                    "ox_ramp_cap": state[0],
                    "ox_ramp_valid_receipts": state[1],
                    "ox_ramp_provider_attempts": state[2],
                    "ox_ramp_request_revision": distill.OX_RAMP_REQUEST_REVISION,
                },
                10,
            )
            == state
        )
    assert (
        distill._advance_ox_ramp(
            cap=state[0],
            valid_receipts=state[1],
            provider_attempts=state[2],
            valid_results=1,
            actual_attempts=1,
            rate_limited=False,
            stopped=False,
            max_inflight=10,
        )
        == state
    )


def test_ox_ramp_counts_invalid_output_and_retries_in_provider_denominator() -> None:
    assert distill._advance_ox_ramp(
        cap=1,
        valid_receipts=19,
        provider_attempts=19,
        valid_results=0,
        actual_attempts=1,
        rate_limited=False,
        stopped=False,
        max_inflight=10,
    ) == (1, 19, 20)
    assert distill._advance_ox_ramp(
        cap=1,
        valid_receipts=19,
        provider_attempts=19,
        valid_results=1,
        actual_attempts=3,
        rate_limited=False,
        stopped=False,
        max_inflight=10,
    ) == (1, 20, 22)


def test_ox_ramp_429_halves_and_resets_but_final_acceptance_is_stable() -> None:
    assert distill._advance_ox_ramp(
        cap=5,
        valid_receipts=12,
        provider_attempts=12,
        valid_results=1,
        actual_attempts=1,
        rate_limited=True,
        stopped=False,
        max_inflight=10,
    ) == (2, 0, 0)
    assert distill._advance_ox_ramp(
        cap=6,
        valid_receipts=12,
        provider_attempts=12,
        valid_results=1,
        actual_attempts=1,
        rate_limited=True,
        stopped=False,
        max_inflight=6,
    ) == (2, 0, 0)
    assert distill._advance_ox_ramp(
        cap=2,
        valid_receipts=19,
        provider_attempts=19,
        valid_results=1,
        actual_attempts=1,
        rate_limited=False,
        stopped=True,
        max_inflight=10,
    ) == (2, 19, 20)
    accepted = (10, 20, 21)
    assert (
        distill._advance_ox_ramp(
            cap=accepted[0],
            valid_receipts=accepted[1],
            provider_attempts=accepted[2],
            valid_results=10,
            actual_attempts=10,
            rate_limited=False,
            stopped=False,
            max_inflight=10,
        )
        == accepted
    )


def test_ox_terminal_429_reopens_the_ramp() -> None:
    assert distill._advance_ox_ramp(
        cap=10,
        valid_receipts=20,
        provider_attempts=20,
        valid_results=0,
        actual_attempts=1,
        rate_limited=True,
        stopped=False,
        max_inflight=10,
    ) == (5, 0, 0)


def test_ox_contract_rotation_changes_work_split_and_cohort_without_reusing_old_rows(
    tmp_path: Path,
) -> None:
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
    )
    first_contract = distill._ensure_ox_profile_contract(tmp_path, config)[
        "artifact_id"
    ]
    rotated_config = replace(config, max_input_bytes=config.max_input_bytes - 1)
    second_contract = distill._ensure_ox_profile_contract(tmp_path, rotated_config)[
        "artifact_id"
    ]
    assert first_contract != second_contract
    rows = [
        {
            "profile": distill.OX_SINGLE_PROFILE,
            "cohort": distill.OX_SINGLE_COHORT,
            "profile_contract_id": contract_id,
            "model_digest": "a" * 64,
        }
        for contract_id in (first_contract, second_contract)
    ]
    _, first_cohort = distill._active_training_cohort(
        rows,
        teacher_profile=distill.OX_SINGLE_PROFILE,
        profile_contract_id=first_contract,
    )
    _, second_cohort = distill._active_training_cohort(
        rows,
        teacher_profile=distill.OX_SINGLE_PROFILE,
        profile_contract_id=second_contract,
    )
    assert first_cohort["cohort_sha256"] != second_cohort["cohort_sha256"]
    rally = {
        "rally_id": "rally",
        "query_sha256": "query",
        "session_cluster_id": "session",
        "as_of": "2026-01-01T00:00:00Z",
        "context_refs": [],
    }
    first_plan = distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="b" * 64,
        model_cohort_sha256=first_cohort["cohort_sha256"],
    )
    second_plan = distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="b" * 64,
        model_cohort_sha256=second_cohort["cohort_sha256"],
    )
    assert first_plan["artifact_id"] != second_plan["artifact_id"]
    payload = {
        "rally": {
            "snapshot_sha256": "snapshot",
            "candidates": [{"candidate_id": "candidate", "text_sha256": "text"}],
        }
    }
    first = distill._ox_prepare_tasks(
        config=config,
        snapshots=payload,
        rally_by_id={"rally": rally},
        assignments=first_plan["assignments"],
        split_plan_id=first_plan["artifact_id"],
        profile_contract_id=first_contract,
        candidate_indexed=False,
        candidate_state={},
    )
    second = distill._ox_prepare_tasks(
        config=rotated_config,
        snapshots=payload,
        rally_by_id={"rally": rally},
        assignments=second_plan["assignments"],
        split_plan_id=second_plan["artifact_id"],
        profile_contract_id=second_contract,
        candidate_indexed=False,
        candidate_state={},
    )
    assert set(first["tasks"]).isdisjoint(second["tasks"])
    audit = distill.materialize_training_rows(
        tmp_path,
        _label_rows=[
            {
                "profile": distill.OX_SINGLE_PROFILE,
                "profile_contract_id": first_contract,
            }
        ],
    )
    assert audit["excluded_prior_contract_rows"] == 1


def test_ox_ramp_only_counts_deep_valid_provider_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_distillation_dispatcher as dispatcher

    def dispatch(batches: object, _evaluate: object, **_kwargs: object) -> list[object]:
        batch = cast(list[list[object]], batches)[0]
        return [
            dispatcher.DispatchResult(
                batch,
                "ok",
                value={
                    "labels": [
                        {
                            "candidate_id": "wrong-candidate",
                            "verdict": "relevant",
                            "confidence": 1.0,
                            "rationale": "bounded",
                        }
                    ],
                    "_route_identity": {
                        "provider": "not-opencode-go",
                        "model": "opencode-go/deepseek-v4-flash",
                        "location": "remote",
                    },
                    "_route_digest": "a" * 64,
                        "_model_digest": "b" * 64,
                        "_prompt_digest": "c" * 64,
                        "_schema_digest": "d" * 64,
                        "_provider_receipt_sha256": "e" * 64,
                    },
                attempts=1,
            )
        ]

    monkeypatch.setattr(dispatcher, "dispatch_claimed_work", dispatch)

    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise AssertionError("intercepted dispatcher must not evaluate")

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE, ox_enabled=True
        ),
        teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
        snapshots={
            "rally": {
                "candidates": [
                    {"candidate_id": "candidate", "text_sha256": "candidate"}
                ]
            }
        },
        rally_by_id={
            "rally": {"rally_id": "rally", "query_sha256": "query", "context_refs": []}
        },
        texts={"query": "what proves the claim", "candidate": "bounded fact"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.labels_written == 0
    assert result.profile_stopped is True
    assert (
        store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl") == []
    )
    assert (
        result.ramp_cap,
        result.ramp_valid_receipts,
        result.ramp_provider_attempts,
    ) == (
        1,
        0,
        1,
    )
    failures = store.read_chain(
        store.distillation_dir(tmp_path) / "ox-failure-receipts.jsonl"
    )
    assert len(failures) == 1
    assert failures[0]["category"] == "model_drift"
    assert failures[0]["status"] == "hard_stop"


def test_ox_single_teacher_batch_dispatches_in_order_and_writes_only_valid_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def receipt_binding(self) -> dict[str, str]:
            return {
                "source_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_ox_identity_sha256": "c" * 64,
            }

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in payload["candidates"]
                ],
                **_ox_metadata(payload),
            }

    rally = {
        "rally_id": "rally-1",
        "query_sha256": "query",
        "context_refs": [],
    }
    candidates = [
        {"candidate_id": "candidate-1", "text_sha256": "candidate-1"},
        {"candidate_id": "candidate-2", "text_sha256": "candidate-2"},
    ]
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    teacher = RemoteTeacher()
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        max_input_bytes=4_096,
        teacher_max_inflight=10,
        teacher_claim_limit=1,
    )
    results = [
        distill._run_teacher_batch(
            root=tmp_path,
            config=config,
            teachers={distill.OX_TEACHER_ROLE: teacher},
            snapshots={"rally-1": {"candidates": candidates}},
            rally_by_id={"rally-1": rally},
            texts={
                "query": "what proves the claim",
                "candidate-1": "first bounded fact",
                "candidate-2": "second bounded fact",
            },
            label_path=label_path,
            label_rows=store.read_chain(label_path),
            structural_verifier=lambda *_args: None,
        )
        for _ in range(2)
    ]
    result = results[-1]
    labels = store.read_chain(label_path)

    assert sum(current.labels_written for current in results) == 2
    assert sum(current.model_calls for current in results) == 2
    assert {
        key: value
        for key, value in result.workset_status.items()  # type: ignore[union-attr]
        if not key.startswith("last_durable_")
    } == {
        "ready": 0,
        "leased": 0,
        "completed": 2,
        "quarantined": 0,
        "backlog": 0,
        "total": 2,
    }
    assert result.workset_status["last_durable_progress"]["cursor"]["label_count"] == 2  # type: ignore[index]
    assert [row["candidate_id"] for row in labels] == ["candidate-1", "candidate-2"]
    assert all(row["route"] == "opencode-go/deepseek-v4-flash" for row in labels)
    assert all(row["teacher_role"] == distill.OX_TEACHER_ROLE for row in labels)
    assert all(row["status"] == "completed" for row in labels)
    assert all(row["profile"] == distill.OX_SINGLE_PROFILE for row in labels)
    assert all(row["cohort"] == distill.OX_SINGLE_COHORT for row in labels)
    assert all(len(row["prompt_sha256"]) == 64 for row in labels)
    assert all(row["assignment"]["probe"] is False for row in labels)
    assert all(row["source_commit"] == "a" * 40 for row in labels)
    assert all(row["source_tree_sha256"] == "b" * 64 for row in labels)
    assert all(row["source_ox_identity_sha256"] == "c" * 64 for row in labels)
    assert all(row["ramp_cap"] == 1 and row["attempt_count"] == 1 for row in labels)

    with sqlite3.connect(store.distillation_dir(tmp_path) / "ox-workset.sqlite3") as db:
        db.execute(
            """
            UPDATE work_items
            SET state = 'leased', lease_id = 'crashed', lease_owner = 'crashed',
                lease_expires_at = 0
            """
        )
    original_recent = workset.DistillationWorkset.recent_transition_receipts

    def recent_with_legacy(
        queue: workset.DistillationWorkset, limit: int = 2
    ) -> tuple[dict[str, Any], ...]:
        return (
            *original_recent(queue, limit),
            {
                "generation": 1,
                "receipt_sha256": "f" * 64,
                "operation": "claim_reclaim",
                "details": {"kind": "ox", "count": 1, "selection_sha256": "e" * 64},
            },
        )

    monkeypatch.setattr(
        workset.DistillationWorkset,
        "recent_transition_receipts",
        recent_with_legacy,
    )
    calls = 0

    def should_not_call(_payload: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("crash recovery must reconcile the existing label")

    teacher.evaluate = should_not_call  # type: ignore[method-assign]
    recovered = [
        distill._run_teacher_batch(
            root=tmp_path,
            config=distill.DistillationConfig(
                teacher_profile=distill.OX_SINGLE_PROFILE,
                ox_enabled=True,
                max_input_bytes=4_096,
                teacher_claim_limit=1,
            ),
            teachers={distill.OX_TEACHER_ROLE: teacher},
            snapshots={"rally-1": {"candidates": candidates}},
            rally_by_id={"rally-1": rally},
            texts={
                "query": "what proves the claim",
                "candidate-1": "first bounded fact",
                "candidate-2": "second bounded fact",
            },
            label_path=label_path,
            label_rows=labels,
            structural_verifier=lambda *_args: None,
        )
        for _ in range(2)
    ][-1]

    assert calls == 0
    assert recovered.labels_written == 0
    assert recovered.workset_status["completed"] == 2  # type: ignore[index]
    assert len(store.read_chain(label_path)) == 2
    progress = recovered.workset_status["last_durable_progress"]  # type: ignore[index]
    assert progress["cursor"]["label_count"] == 2
    assert (
        progress["ledger_heads"]["labels"]
        == store.chain_head(label_path)["head_sha256"]
    )
    recovery = store.read_chain(
        store.distillation_dir(tmp_path) / "ox-lease-recovery-receipts.jsonl"
    )
    assert len(recovery) == 1
    assert recovery[0]["reclaimed"] == 2
    assert recovery[0]["leased_after"] == 1


def test_ox_metadata_drift_stops_before_the_next_wave_and_releases_all_claims(
    tmp_path: Path,
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            self.calls += 1
            metadata = _ox_metadata(payload)
            metadata["_request_digest"] = "0" * 64
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in payload["candidates"]
                ],
                **metadata,
            }

    teacher = RemoteTeacher()
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            "rally": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"},
                    {"candidate_id": "candidate-2", "text_sha256": "candidate-2"},
                ]
            }
        },
        rally_by_id={
            "rally": {"rally_id": "rally", "query_sha256": "query", "context_refs": []}
        },
        texts={
            "query": "what proves the claim",
            "candidate-1": "first bounded fact",
            "candidate-2": "second bounded fact",
        },
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert teacher.calls == 1
    assert result.labels_written == 0
    assert result.profile_stopped is True
    assert result.workset_status["ready"] == 2  # type: ignore[index]
    assert (
        store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl") == []
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("confidence", float("nan")), ("rationale", "untrusted free text")],
)
def test_ox_invalid_label_body_retries_without_profile_stop(
    tmp_path: Path, field: str, value: object
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            label = {
                "candidate_id": "candidate",
                "verdict": "relevant",
                "confidence": 0.9,
                "rationale": "direct_match",
            }
            label[field] = value
            return {"labels": [label], **_ox_metadata(payload)}

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
        snapshots={
            "rally": {
                "candidates": [
                    {"candidate_id": "candidate", "text_sha256": "candidate"}
                ]
            }
        },
        rally_by_id={
            "rally": {"rally_id": "rally", "query_sha256": "query", "context_refs": []}
        },
        texts={"query": "what proves the claim", "candidate": "bounded fact"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.labels_written == 0
    assert result.profile_stopped is False
    assert result.workset_status["ready"] == 1  # type: ignore[index]


def test_ox_indexed_workset_reads_only_delta_then_claimed_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.valid = True
            self.calls = 0

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            self.calls += 1
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant" if self.valid else "invalid",
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in payload["candidates"]
                ],
                **_ox_metadata(payload),
            }

    def snapshot(rally_id: str, candidate_id: str, text_hash: str) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": catalog.CANDIDATE_SNAPSHOT_SCHEMA,
            "rally_id": rally_id,
            "as_of": "2026-01-03T00:00:00Z",
            "retriever_revision": "historical-fts-v1",
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "query_feature_text_sha256": "e" * 64,
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "rank": 1,
                    "text_sha256": text_hash,
                    "candidate_feature_text_sha256": "f" * 64,
                }
            ],
        }
        value["snapshot_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value

    query_one = "1" * 64
    text_one = "2" * 64
    query_two = "3" * 64
    text_two = "4" * 64
    rallies = {
        "rally-1": {
            "rally_id": "rally-1",
            "session_cluster_id": "session-1",
            "as_of": "2026-01-03T00:00:00Z",
            "query_sha256": query_one,
            "context_refs": [],
        },
        "rally-2": {
            "rally_id": "rally-2",
            "session_cluster_id": "session-2",
            "as_of": "2026-01-03T00:00:00Z",
            "query_sha256": query_two,
            "context_refs": [],
        },
        "rally-3": {
            "rally_id": "rally-3",
            "session_cluster_id": "session-3",
            "as_of": "2026-01-03T00:00:00Z",
            "query_sha256": "5" * 64,
            "context_refs": [],
        },
    }
    catalog.advance(_raw(tmp_path), tmp_path, 4096)
    workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    )
    plan = distill._ensure_split_plan(
        tmp_path,
        list(rallies.values()),
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    candidate_path = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    store.append_chain(
        candidate_path,
        {
            "kind": "candidate-snapshot",
            "rally_id": "rally-1",
            "snapshot": snapshot("rally-1", "candidate-1", text_one),
        },
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        teacher_claim_limit=1,
    )
    teacher = RemoteTeacher()
    calls: list[set[str]] = []
    original_read = catalog.read_candidate_snapshots

    def tracked_read(
        root: Path, path: Path, rally_ids: Iterable[str]
    ) -> dict[str, dict[str, object]]:
        values = set(rally_ids)
        if values:
            calls.append(values)
        return original_read(root, path, values)

    monkeypatch.setattr(catalog, "read_candidate_snapshots", tracked_read)

    def run(labels: list[dict[str, object]]) -> distill._TeacherBatchResult:
        return distill._run_teacher_batch(
            root=tmp_path,
            config=config,
            teachers={distill.OX_TEACHER_ROLE: teacher},
            snapshots={},
            rally_by_id=rallies,
            texts={
                query_one: "what proves the first claim",
                text_one: "first bounded fact",
                query_two: "what proves the second claim",
                text_two: "second bounded fact",
                "5" * 64: "what proves the third claim",
                "6" * 64: "third bounded fact",
                "7" * 64: "what proves the future claim",
                "8" * 64: "future bounded fact",
            },
            label_path=label_path,
            label_rows=labels,
            candidate_indexed=True,
            structural_verifier=lambda *_args: None,
        )

    first = run([])
    assert first.labels_written == 1
    assert calls == [{"rally-1"}]
    profile_contract_id = distill._ensure_ox_profile_contract(tmp_path, config)[
        "artifact_id"
    ]
    assert store.read_chain(label_path)[0]["profile_contract_id"] == profile_contract_id
    with sqlite3.connect(store.distillation_dir(tmp_path) / "ox-workset.sqlite3") as db:
        provenance = json.loads(
            db.execute("SELECT provenance_json FROM work_items").fetchone()[0]
        )
    assert provenance["profile_contract_id"] == profile_contract_id

    calls.clear()
    second = run(store.read_chain(label_path))
    assert second.labels_written == 0
    assert calls == []

    workset_path = store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    with sqlite3.connect(workset_path) as db:
        completed_before = db.execute(
            "SELECT temporal_split_json,completion_ref,completion_digest,"
            "attempt_count,updated_at FROM work_items"
        ).fetchone()
    extended_plan_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": "c" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "model_cohort_sha256": "b" * 64,
            "split_revision": "grouped-rolling-v1",
            "assignments": {**plan["assignments"], "future-rally": "embargo"},
        },
        schema=distill.SPLIT_PLAN_SCHEMA,
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": extended_plan_id},
    )
    before_calls = teacher.calls
    rotated = run(store.read_chain(label_path))
    assert rotated.labels_written == 0
    assert teacher.calls == before_calls
    assert len(store.read_chain(label_path)) == 1
    with sqlite3.connect(workset_path) as db:
        assert (
            db.execute(
                "SELECT temporal_split_json,completion_ref,completion_digest,"
                "attempt_count,updated_at FROM work_items"
            ).fetchone()
            == completed_before
        )
        assert (
            json.loads(
                db.execute(
                    "SELECT value_json FROM workset_state WHERE key='watermark'"
                ).fetchone()[0]
            )["split_plan_id"]
            == extended_plan_id
        )

    changed_assignments = dict(plan["assignments"])
    changed_assignments["rally-1"] = (
        "test" if changed_assignments["rally-1"] != "test" else "train"
    )
    changed_plan_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": "d" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "model_cohort_sha256": "b" * 64,
            "split_revision": "grouped-rolling-v1",
            "assignments": {**changed_assignments, "future-rally": "embargo"},
        },
        schema=distill.SPLIT_PLAN_SCHEMA,
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": changed_plan_id},
    )
    with pytest.raises(workset.DistillationWorksetError, match="identity conflict"):
        run(store.read_chain(label_path))
    assert teacher.calls == before_calls
    assert len(store.read_chain(label_path)) == 1
    with sqlite3.connect(workset_path) as db:
        assert (
            db.execute(
                "SELECT temporal_split_json,completion_ref,completion_digest,"
                "attempt_count,updated_at FROM work_items"
            ).fetchone()
            == completed_before
        )
        assert (
            json.loads(
                db.execute(
                    "SELECT value_json FROM workset_state WHERE key='watermark'"
                ).fetchone()[0]
            )["split_plan_id"]
            == extended_plan_id
        )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": extended_plan_id},
    )

    rallies["future-rally"] = {
        "rally_id": "future-rally",
        "session_cluster_id": "future-session",
        "as_of": "2026-01-03T00:00:00Z",
        "query_sha256": "7" * 64,
        "context_refs": [],
    }
    store.append_chain(
        candidate_path,
        {
            "kind": "candidate-snapshot",
            "rally_id": "future-rally",
            "snapshot": snapshot("future-rally", "future-candidate", "8" * 64),
        },
    )
    calls.clear()
    assert run(store.read_chain(label_path)).labels_written == 0
    assert teacher.calls == before_calls
    assert calls == [{"future-rally"}]
    with sqlite3.connect(workset_path) as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM work_items "
                "WHERE payload_ref='candidate-snapshot:future-rally:future-candidate'"
            ).fetchone()[0]
            == 0
        )

    store.append_chain(
        candidate_path,
        {
            "kind": "candidate-snapshot",
            "rally_id": "rally-2",
            "snapshot": snapshot("rally-2", "candidate-2", text_two),
        },
    )
    calls.clear()
    third = run(store.read_chain(label_path))
    assert third.labels_written == 1
    assert calls == [{"rally-2"}]

    teacher.valid = False
    store.append_chain(
        candidate_path,
        {
            "kind": "candidate-snapshot",
            "rally_id": "rally-3",
            "snapshot": snapshot("rally-3", "candidate-3", "6" * 64),
        },
    )
    calls.clear()
    assert run(store.read_chain(label_path)).labels_written == 0
    assert calls == [{"rally-3"}]
    teacher.valid = True
    calls.clear()
    with sqlite3.connect(workset_path) as db:
        work_id, temporal_json = db.execute(
            "SELECT work_id,temporal_split_json FROM work_items "
            "WHERE payload_ref='candidate-snapshot:rally-3:candidate-3'"
        ).fetchone()
        tampered = json.loads(temporal_json)
        tampered["split"] = "tampered"
        db.execute(
            "UPDATE work_items SET temporal_split_json=? WHERE work_id=?",
            (json.dumps(tampered, sort_keys=True, separators=(",", ":")), work_id),
        )
    before_calls = teacher.calls
    assert run(store.read_chain(label_path)).labels_written == 0
    assert teacher.calls == before_calls
    with sqlite3.connect(workset_path) as db:
        db.execute(
            "UPDATE work_items SET temporal_split_json=? WHERE work_id=?",
            (temporal_json, work_id),
        )
    assert run(store.read_chain(label_path)).labels_written == 1
    assert calls == [{"rally-3"}, {"rally-3"}]


def test_ox_single_teacher_uncertain_output_completes_as_non_training_abstention(
    tmp_path: Path,
) -> None:
    class UncertainTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "uncertain",
                        "confidence": 0.5,
                        "rationale": "insufficient_evidence",
                    }
                    for candidate in payload["candidates"]
                ],
                **_ox_metadata(payload),
            }

    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    rally = {
        "rally_id": "rally-1",
        "session_cluster_id": "session-1",
        "as_of": "2026-01-03T00:00:00Z",
        "query_sha256": "query",
        "context_refs": [],
    }
    distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    snapshots = {
        "rally-1": {
            "as_of": rally["as_of"],
            "snapshot_sha256": "c" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "text_sha256": "candidate-1",
                    "features": features,
                }
            ],
        }
    }
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"

    def run() -> distill._TeacherBatchResult:
        return distill._run_teacher_batch(
            root=tmp_path,
            config=distill.DistillationConfig(
                teacher_profile=distill.OX_SINGLE_PROFILE,
                ox_enabled=True,
                max_input_bytes=4_096,
            ),
            teachers={distill.OX_TEACHER_ROLE: UncertainTeacher()},
            snapshots=snapshots,
            rally_by_id={"rally-1": rally},
            texts={"query": "what proves the claim", "candidate-1": "bounded fact"},
            label_path=label_path,
            label_rows=[],
            structural_verifier=lambda *_args: None,
        )

    result = run()
    labels = store.read_chain(label_path)
    training = distill.materialize_training_rows(
        tmp_path,
        _rallies=[rally],
        _snapshots=snapshots,
        _label_rows=labels,
    )

    assert result.labels_written == 1
    assert result.deferred is False
    assert result.workset_status["completed"] == 1  # type: ignore[index]
    assert result.ramp_valid_receipts == 1
    assert result.ramp_provider_attempts == 1
    assert labels[0]["verdict"] == "uncertain"
    assert labels[0]["authority"] == "uncertain"
    assert training["rows"] == []
    assert distill.train_tiny_policy(training["rows"])["training_rows"] == 0


def test_ox_resolves_text_only_for_claimed_work_and_uses_long_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_distillation_workset as workset_module

    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in payload["candidates"]
                ],
                **_ox_metadata(payload),
            }

    class GuardedTexts(dict[str, str]):
        def get(self, key: str, default: object = None) -> object:
            if key not in {"query-1", "candidate-1"}:
                raise AssertionError(f"unclaimed text resolved: {key}")
            return super().get(key, default)

    leases: list[float] = []
    original_claim = workset_module.DistillationWorkset.claim

    def claim(
        self: object, kind: str, limit: int, owner: str, lease_seconds: float
    ) -> object:
        leases.append(lease_seconds)
        return original_claim(self, kind, limit, owner, lease_seconds)  # type: ignore[arg-type]

    monkeypatch.setattr(workset_module.DistillationWorkset, "claim", claim)
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
        snapshots={
            "rally-1": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"},
                    {"candidate_id": "candidate-2", "text_sha256": "candidate-2"},
                ]
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query-1",
                "context_refs": [],
            }
        },
        texts=GuardedTexts(
            {"query-1": "what proves the claim", "candidate-1": "bounded fact"}
        ),
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.labels_written == 1
    assert leases == [7200]


def test_ox_missing_payload_quarantines_without_remote_call(
    tmp_path: Path,
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise AssertionError("missing payload must not reach remote teacher")

    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"

    def run() -> distill._TeacherBatchResult:
        return distill._run_teacher_batch(
            root=tmp_path,
            config=distill.DistillationConfig(
                teacher_profile=distill.OX_SINGLE_PROFILE,
                ox_enabled=True,
            ),
            teachers={distill.OX_TEACHER_ROLE: RemoteTeacher()},
            snapshots={
                "rally-1": {
                    "candidates": [
                        {"candidate_id": "candidate-1", "text_sha256": "missing"}
                    ]
                }
            },
            rally_by_id={
                "rally-1": {
                    "rally_id": "rally-1",
                    "query_sha256": "query",
                    "context_refs": [],
                }
            },
            texts={"query": "what proves the claim"},
            label_path=label_path,
            label_rows=[],
            structural_verifier=lambda *_args: None,
        )

    result = run()
    assert result.workset_status["quarantined"] == 1  # type: ignore[index]
    assert result.workset_status["ready"] == 0  # type: ignore[index]
    assert store.read_chain(label_path) == []


def test_ox_canary_skips_payload_rejected_probe_before_one_safe_request(
    tmp_path: Path,
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.requests: list[list[str]] = []

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            self.requests.append(
                [str(candidate["candidate_id"]) for candidate in candidates]
            )
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in candidates
                ],
                **_ox_metadata(payload),
            }

    rally = {
        "rally_id": "rally-test",
        "session_cluster_id": "session-test",
        "as_of": "2026-01-03T00:00:00Z",
        "query_sha256": "query",
        "context_refs": [],
    }
    distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    teacher = RemoteTeacher()
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
            hard_floor_probe_pairs=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            "rally-test": {
                "snapshot_sha256": "c" * 64,
                "candidates": [
                    {"candidate_id": "candidate-a", "text_sha256": "missing-a"},
                    {"candidate_id": "candidate-b", "text_sha256": "missing-b"},
                    {"candidate_id": "candidate-c", "text_sha256": "candidate-c"},
                ],
            }
        },
        rally_by_id={"rally-test": rally},
        texts={"query": "what proves the claim", "candidate-c": "bounded fact"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.labels_written == 1
    assert teacher.requests == [["candidate-c"]]
    with sqlite3.connect(store.distillation_dir(tmp_path) / "ox-workset.sqlite3") as db:
        counts = dict(
            db.execute("SELECT state, COUNT(*) FROM work_items GROUP BY state")
        )
        attempts = db.execute("SELECT MAX(attempt_count) FROM work_items").fetchone()[0]
    assert counts == {"completed": 1, "quarantined": 6}
    assert attempts == 1


def test_ox_canary_skips_oversize_probe_before_one_request(
    tmp_path: Path,
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.requests: list[list[str]] = []

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            self.requests.append(
                [str(candidate["candidate_id"]) for candidate in candidates]
            )
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in candidates
                ],
                **_ox_metadata(payload),
            }

    rally = {
        "rally_id": "rally-test",
        "session_cluster_id": "session-test",
        "as_of": "2026-01-03T00:00:00Z",
        "query_sha256": "query",
        "context_refs": [],
    }
    distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    teacher = RemoteTeacher()
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
            hard_floor_probe_pairs=1,
            max_input_bytes=20_000,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            "rally-test": {
                "snapshot_sha256": "c" * 64,
                "candidates": [
                    {"candidate_id": "candidate-a", "text_sha256": "candidate-a"},
                    {"candidate_id": "candidate-b", "text_sha256": "candidate-b"},
                    {"candidate_id": "candidate-c", "text_sha256": "candidate-c"},
                ],
            }
        },
        rally_by_id={"rally-test": rally},
        texts={
            "query": "q",
            "candidate-a": "a" * 6_000,
            "candidate-b": "b" * 6_000,
            "candidate-c": "bounded fact",
        },
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert teacher.requests == [["candidate-a"]]
    assert result.labels_written == 1
    assert result.workset_status["completed"] == 1  # type: ignore[index]
    assert result.workset_status["quarantined"] == 4  # type: ignore[index]
    assert result.workset_status["ready"] == 2  # type: ignore[index]


def test_ox_profile_stop_returns_claims_to_ready(tmp_path: Path) -> None:
    from chronovisor.recall.recall_distillation_dispatcher import DispatchFailure

    class StoppedTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise DispatchFailure("http_402")

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
        ),
        teachers={distill.OX_TEACHER_ROLE: StoppedTeacher()},
        snapshots={
            "rally-1": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"},
                    {"candidate_id": "candidate-2", "text_sha256": "candidate-2"},
                ]
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        texts={
            "query": "what proves the claim",
            "candidate-1": "bounded fact",
            "candidate-2": "another bounded fact",
        },
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.profile_stopped is True
    assert result.workset_status["ready"] == 2  # type: ignore[index]
    assert result.labels_written == 0


def test_ox_dispatch_guard_denial_releases_without_attempt_or_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, _payload: object) -> dict[str, object]:
            self.calls += 1
            raise AssertionError("eligibility denial must precede evaluation")

    teacher = RemoteTeacher()
    monkeypatch.setattr(distill, "_current_ox_profile_contract_id", lambda _root: "")
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            "rally-1": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"}
                ]
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        texts={"query": "what proves the claim", "candidate-1": "bounded fact"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert teacher.calls == 0
    assert result.model_calls == 0
    assert result.labels_written == 0
    assert result.profile_stopped is True
    assert result.workset_status["ready"] == 1  # type: ignore[index]
    assert store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl") == []
    assert store.read_chain(store.distillation_dir(tmp_path) / "ox-ramp-receipts.jsonl") == []
    assert store.read_chain(store.distillation_dir(tmp_path) / "ox-failure-receipts.jsonl") == []


def test_ox_egress_guard_does_not_rescan_source_per_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        teacher_claim_limit=2,
    )
    contract = distill._ensure_ox_profile_contract(
        tmp_path, config, source_binding=source
    )
    scans = 0

    def scan() -> dict[str, str]:
        nonlocal scans
        scans += 1
        return dict(source)

    monkeypatch.setattr(distill, "ox_alpha_source_binding", scan)
    for _ in range(3):
        distill._ox_eligibility_guard(
            root=tmp_path,
            config=config,
            teacher=object(),
            profile_contract_id=str(contract["artifact_id"]),
            source_binding=source,
        )
    assert scans == 0


def test_ox_post_http_source_drift_consumes_then_retries_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            self.calls += 1
            return {
                "labels": [
                    {
                        "candidate_id": payload["candidates"][0]["candidate_id"],
                        "verdict": "irrelevant",
                        "confidence": 0.8,
                        "rationale": "direct_match",
                    }
                ],
                **_ox_metadata(payload),
            }

    teacher = RemoteTeacher()
    monkeypatch.setattr(distill, "_ox_source_binding_matches", lambda *_args: False)
    kwargs = {
        "root": tmp_path,
        "config": distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE, ox_enabled=True
        ),
        "teachers": {distill.OX_TEACHER_ROLE: teacher},
        "snapshots": {
            "rally-1": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"}
                ]
            }
        },
        "rally_by_id": {
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        "texts": {"query": "what proves the claim", "candidate-1": "bounded fact"},
        "label_path": store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        "label_rows": [],
        "structural_verifier": lambda *_args: None,
    }
    first = distill._run_teacher_batch(**kwargs)
    assert first.model_calls == 1
    assert first.labels_written == 0
    workset_path = store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    with sqlite3.connect(workset_path) as db:
        assert db.execute(
            "SELECT attempt_count,last_error_class,state FROM work_items"
        ).fetchone() == (1, "source_binding_drift", "ready")
        db.execute("UPDATE work_items SET next_attempt_at = 0")
    monkeypatch.setattr(distill, "_ox_source_binding_matches", lambda *_args: True)
    second = distill._run_teacher_batch(**kwargs)
    assert second.labels_written == 1
    assert teacher.calls == 2
    with sqlite3.connect(workset_path) as db:
        assert db.execute("SELECT attempt_count,state FROM work_items").fetchone() == (
            2,
            "completed",
        )


def test_ox_post_append_expiry_releases_mixed_claims_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_distillation_dispatcher as dispatcher

    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    config = distill.DistillationConfig(
        teacher_profile=distill.OX_SINGLE_PROFILE,
        ox_enabled=True,
        teacher_max_inflight=2,
        teacher_claim_limit=2,
    )
    contract = distill._ensure_ox_profile_contract(
        tmp_path, config, source_binding=source
    )
    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "ox-workset.sqlite3"
    )
    tasks: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    for index in range(2):
        payload_digest = hashlib.sha256(f"payload-{index}".encode()).hexdigest()
        work_id = hashlib.sha256(f"work-{index}".encode()).hexdigest()
        tasks[work_id] = {
            "rally": {"rally_id": f"rally-{index}"},
            "candidate": {"candidate_id": f"candidate-{index}"},
            "input": {
                "candidate_id": f"candidate-{index}",
                "rally_id": f"rally-{index}",
                "query": "question",
                "context": [],
                "evidence": "bounded evidence",
            },
            "payload_source": {"candidate_id": f"candidate-{index}"},
            "assignment": {"revision": "single-teacher-v1", "probe": False},
            "temporal": {
                "as_of": "2026-01-01T00:00:00Z",
                "group_id": f"group-{index}",
                "split": "train",
                "split_plan_id": "",
            },
        }
        items.append(
            {
                "work_id": work_id,
                "kind": "ox",
                "payload_ref": f"candidate-snapshot:rally-{index}:candidate-{index}",
                "payload_digest": payload_digest,
                "temporal_split": tasks[work_id]["temporal"],
                "provenance": {"route": "opencode-go/deepseek-v4-flash"},
            }
        )
    queue.advance(items, {"candidate_records": 2})
    claims = list(queue.claim("ox", 2, distill.OX_TEACHER_ROLE, 60))
    response = {
        "labels": [
            {
                "candidate_id": tasks[claims[0].work_id]["candidate"]["candidate_id"],
                "verdict": "relevant",
                "confidence": 0.9,
                "rationale": "direct_match",
            }
        ],
        **_ox_metadata(distill._ox_batch_payload(tasks, [claims[0]])),
    }
    results = [
        SimpleNamespace(
            work=[claims[0]], status="ok", value=response, attempts=1,
            rate_limited=False, category="", error=None,
        ),
        SimpleNamespace(
            work=[claims[1]], status="deferred", value=None, attempts=0,
            rate_limited=False, category="shutdown", error=None,
        ),
    ]
    monkeypatch.setattr(
        dispatcher, "dispatch_claimed_work", lambda *_args, **_kwargs: results
    )
    expired = False
    real_append = store.append_chain_batch
    real_expiry = distill._ox_expiry

    def append_then_expire(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        nonlocal expired
        appended = real_append(*args, **kwargs)
        expired = True
        return appended

    def expiry(value: object) -> str:
        if expired:
            raise distill.DistillationError("expired after append")
        return real_expiry(value)

    monkeypatch.setattr(store, "append_chain_batch", append_then_expire)
    monkeypatch.setattr(distill, "_ox_expiry", expiry)
    result = distill._ox_dispatch_and_commit(
        root=tmp_path,
        claims=claims,
        batches=[[claims[0]], [claims[1]]],
        ramp_cap=2,
        ramp_valid_receipts=0,
        ramp_provider_attempts=0,
        teacher=SimpleNamespace(),
        tasks=tasks,
        config=config,
        workset=queue,
        profile_contract_id=str(contract["artifact_id"]),
        source_binding=source,
        split_plan_id="",
        candidate_state={"record_count": 2, "head_sha256": "d" * 64},
        structural_verifier=lambda *_args: None,
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
    )

    assert result.deferred is True
    assert result.labels_written == 1
    assert queue.status("ox")["ready"] == 2
    assert queue.status("ox")["leased"] == 0


@pytest.mark.parametrize(
    ("category", "terminal_state", "terminal_count"),
    [("remote_payload_rejected", "quarantined", 1), ("http_429", "ready", 2)],
)
def test_ox_canary_failure_is_single_attempt(
    tmp_path: Path, category: str, terminal_state: str, terminal_count: int
) -> None:
    class GuardedTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, _payload: object) -> dict[str, object]:
            self.calls += 1
            return {
                "_failure": {
                    "class": category,
                    "retryable": category == "http_429",
                    "labelable": False,
                }
            }

    teacher = GuardedTeacher()
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            "rally-1": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"},
                    {"candidate_id": "candidate-2", "text_sha256": "candidate-2"},
                ]
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        texts={
            "query": "what proves the claim",
            "candidate-1": "bounded fact",
            "candidate-2": "another bounded fact",
        },
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert teacher.calls == 1
    assert result.model_calls == 1
    assert result.workset_status[terminal_state] == terminal_count  # type: ignore[index]


def test_ox_unreceipted_provider_attempt_stops_the_profile(tmp_path: Path) -> None:
    class TimeoutTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            return {
                "_failure": {
                    "class": "timeout",
                    "retryable": True,
                    "labelable": False,
                }
            }

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: TimeoutTeacher()},
        snapshots={
            "rally-1": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"}
                ]
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        texts={"query": "what proves the claim", "candidate-1": "bounded fact"},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.profile_stopped is True
    assert result.ramp_provider_attempts == 1
    assert result.workset_status["ready"] == 1  # type: ignore[index]
    assert result.workset_status["leased"] == 0  # type: ignore[index]
    assert store.read_chain(
        store.distillation_dir(tmp_path) / "ox-ramp-receipts.jsonl"
    ) == []
    assert store.read_chain(
        store.distillation_dir(tmp_path) / "ox-failure-receipts.jsonl"
    ) == []


def test_ox_failure_stage_is_durable_without_changing_retry_policy(
    tmp_path: Path,
) -> None:
    class GuardedTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def evaluate(self, _payload: object) -> dict[str, object]:
            return {
                "_failure": {
                    "class": "invalid_response",
                    "stage": "teacher_json_parse",
                    "request_id": "ox_req_1",
                    "labelable": False,
                }
            }

    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: GuardedTeacher()},
        snapshots={
            "rally-1": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"},
                    {"candidate_id": "candidate-2", "text_sha256": "candidate-2"},
                ]
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        texts={
            "query": "what proves the claim",
            "candidate-1": "bounded fact",
            "candidate-2": "another bounded fact",
        },
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.workset_status["ready"] == 2  # type: ignore[index]
    with sqlite3.connect(store.distillation_dir(tmp_path) / "ox-workset.sqlite3") as db:
        rows = db.execute(
            "SELECT DISTINCT last_error_class FROM work_items "
            "WHERE last_error_class != ''"
        ).fetchall()
    assert rows == [("invalid_response.teacher_json_parse",)]


@pytest.mark.parametrize(
    ("claim_limit", "expected_requests"),
    [(1, [["candidate-2"]])],
)
def test_ox_scans_adapter_preflight_reject_without_losing_safe_work(
    tmp_path: Path, claim_limit: int, expected_requests: list[list[str]]
) -> None:
    class GuardedTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.requests: list[list[str]] = []

        def accepts_egress_payload(self, payload: object) -> bool:
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            return all(
                candidate["candidate_id"] != "candidate-1" for candidate in candidates
            )

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            self.requests.append(
                [str(candidate["candidate_id"]) for candidate in candidates]
            )
            return {"_failure": {"class": "invalid_response"}}

    teacher = GuardedTeacher()
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=claim_limit,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            "rally-1": {
                "candidates": [
                    {"candidate_id": "candidate-1", "text_sha256": "candidate-1"},
                    {"candidate_id": "candidate-2", "text_sha256": "candidate-2"},
                    {"candidate_id": "candidate-3", "text_sha256": "candidate-3"},
                ]
            }
        },
        rally_by_id={
            "rally-1": {
                "rally_id": "rally-1",
                "query_sha256": "query",
                "context_refs": [],
            }
        },
        texts={
            "query": "what proves the claim",
            "candidate-1": "blocked before egress",
            "candidate-2": "safe request",
            "candidate-3": "another safe request",
        },
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert teacher.requests == expected_requests
    assert result.model_calls == len(expected_requests)
    assert result.workset_status["quarantined"] == 1  # type: ignore[index]
    assert result.workset_status["ready"] == 2  # type: ignore[index]


def test_ox_canary_preflights_bounded_windows_before_one_request(tmp_path: Path) -> None:
    target_id = "candidate-19-3"

    class PrefetchingTexts(dict[str, str]):
        def __init__(self, values: dict[str, str]) -> None:
            super().__init__(values)
            self.prefetches: list[set[str]] = []

        def prefetch(self, hashes: Iterable[str]) -> None:
            self.prefetches.append(set(hashes))

    class GuardedTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.requests: list[str] = []
            self.preflight_calls = 0

        def accepts_egress_payload(self, payload: object) -> bool:
            self.preflight_calls += 1
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            return candidates[0]["candidate_id"] == target_id

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            self.requests.append(str(candidates[0]["candidate_id"]))
            return {"_failure": {"class": "invalid_response"}}

    rally_ids = [f"rally-{index:02d}" for index in range(20)]
    candidate_ids = {
        rally_id: [f"candidate-{index:02d}-{position}" for position in range(4)]
        for index, rally_id in enumerate(rally_ids)
    }
    teacher = GuardedTeacher()
    texts = PrefetchingTexts(
        {
            "query": "what proves the claim",
            **{
                candidate_id: "bounded fact"
                for ids in candidate_ids.values()
                for candidate_id in ids
            },
        }
    )
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            rally_id: {
                "candidates": [
                    {"candidate_id": candidate_id, "text_sha256": candidate_id}
                    for candidate_id in candidate_ids[rally_id]
                ]
            }
            for rally_id in rally_ids
        },
        rally_by_id={
            rally_id: {
                "rally_id": rally_id,
                "query_sha256": "query",
                "context_refs": [],
            }
            for rally_id in rally_ids
        },
        texts=texts,
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert teacher.requests == [target_id]
    assert teacher.preflight_calls == 80
    assert len(texts.prefetches) == 2
    assert max(map(len, texts.prefetches)) == 65
    assert set().union(*texts.prefetches) == {
        "query",
        *set().union(*candidate_ids.values()),
    }
    assert result.model_calls == 1
    assert result.workset_status["quarantined"] == 79  # type: ignore[index]
    assert result.workset_status["ready"] == 1  # type: ignore[index]


def test_teacher_payload_does_not_resolve_context_older_than_bounded_suffix() -> None:
    class TrackingTexts(dict[str, str]):
        def __init__(self) -> None:
            super().__init__(
                query="question",
                candidate="answer",
                oldest="oldest context",
                old="old context",
                new="new context",
            )
            self.reads: list[str] = []

        def get(self, key: str, default: str = "") -> str:
            self.reads.append(key)
            return super().get(key, default)

    texts = TrackingTexts()
    rally = {
        "rally_id": "rally-1",
        "query_sha256": "query",
        "context_refs": [
            {"semantic_sha256": "oldest"},
            {"semantic_sha256": "old"},
            {"semantic_sha256": "new"},
        ],
    }
    candidate = {"candidate_id": "candidate-1", "text_sha256": "candidate"}
    limit = len(
        distill.canonical_json.canonical_json_bytes_strict(
            {
                "schema": "chronovisor.recall-distill-teacher-input.v1",
                "rally_id": "rally-1",
                "candidate_id": "candidate-1",
                "query": "question",
                "context": ["new context"],
                "candidate": "answer",
            }
        )
    )

    payload = distill._teacher_payload(
        rally,
        candidate,
        texts,
        max_input_bytes=limit,
    )

    assert payload is not None
    assert payload["context"] == ["new context"]
    assert texts.reads == ["query", "candidate", "new", "old"]


def test_remote_teacher_payload_omits_context_and_binds_local_context_hashes(
    tmp_path: Path,
) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            self.requests.extend(candidates)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "relevant",
                        "confidence": 0.9,
                        "rationale": "direct_match",
                    }
                    for candidate in candidates
                ],
                **_ox_metadata(payload),
            }

    teacher = RemoteTeacher()
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    rally = {
        "rally_id": "rally-1",
        "session_cluster_id": "session-1",
        "as_of": "2026-01-03T00:00:00Z",
        "query_sha256": "query",
        "context_refs": [{"semantic_sha256": "private-context"}],
    }
    distill._ensure_split_plan(
        tmp_path,
        [rally],
        raw_watermark="a" * 64,
        model_cohort_sha256="b" * 64,
    )
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=1,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots={
            "rally-1": {
                "as_of": rally["as_of"],
                "snapshot_sha256": "a" * 64,
                "feature_revision": distill.TEXT_FEATURE_REVISION,
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "text_sha256": "candidate",
                        "features": features,
                    }
                ],
            }
        },
        rally_by_id={"rally-1": rally},
        texts={
            "query": "what proves the claim",
            "candidate": "bounded fact",
            "private-context": "private context stays local",
        },
        label_path=label_path,
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.labels_written == 1
    assert [request["context"] for request in teacher.requests] == [[]]
    labels = store.read_chain(label_path)
    assert labels[0]["payload_source"]["context_sha256"] == ["private-context"]
    assert distill._materialization_payload_source_matches(
        labels[0]["payload_source"],
        rally_id="rally-1",
        candidate_id="candidate-1",
        rally=rally,
        snapshot_sha256="a" * 64,
        candidate_text_sha256="candidate",
    )


def test_remote_probe_selection_uses_provider_safe_candidate_pair() -> None:
    class PrefetchingTexts(dict[str, str]):
        def __init__(self) -> None:
            super().__init__(
                query="what proves the claim",
                blocked_a="blocked a",
                blocked_b="blocked b",
                safe_a="short safe a",
                safe_b="short safe b",
                private_context="must stay local",
            )
            self.prefetches: list[set[str]] = []

        def prefetch(self, hashes: Iterable[str]) -> None:
            self.prefetches.append(set(hashes))

    texts = PrefetchingTexts()
    eligible = [
        (
            "rally-1",
            {"snapshot_sha256": "a" * 64},
            {
                "rally_id": "rally-1",
                "as_of": "2026-01-01T00:00:00Z",
                "query_sha256": "query",
                "context_refs": [{"semantic_sha256": "private_context"}],
            },
            [
                {"candidate_id": "a", "text_sha256": "blocked_a"},
                {"candidate_id": "b", "text_sha256": "blocked_b"},
                {"candidate_id": "c", "text_sha256": "safe_a"},
                {"candidate_id": "d", "text_sha256": "safe_b"},
            ],
        )
    ]

    selected = distill._ox_select_remote_probe_rallies(
        config=distill.DistillationConfig(hard_floor_probe_pairs=1),
        eligible=eligible,
        texts=texts,
        preflight=lambda payload: all(
            candidate["candidate_id"] not in {"a", "b"}
            and candidate["context"] == []
            for candidate in payload["candidates"]
        ),
    )

    assert [candidate["candidate_id"] for candidate in selected[0][3]] == ["c", "d"]
    assert texts.prefetches == [{"query", "blocked_a", "blocked_b", "safe_a", "safe_b"}]


def test_ox_bulk_claim_config_seals_and_batches_provider_dispatch(tmp_path: Path) -> None:
    class RemoteTeacher:
        local = False
        role = distill.OX_TEACHER_ROLE

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            candidates = payload["candidates"]
            assert isinstance(candidates, list)
            self.batch_sizes.append(len(candidates))
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "irrelevant",
                        "confidence": 0.8,
                        "rationale": "direct_match",
                    }
                    for candidate in candidates
                ],
                **_ox_metadata(payload),
                "_provider_receipt_sha256": hashlib.sha256(
                    canonical_json.canonical_json_bytes_strict(payload)
                ).hexdigest(),
            }

    snapshots: dict[str, dict[str, object]] = {}
    rallies: dict[str, dict[str, object]] = {}
    texts: dict[str, str] = {}
    for index in range(125):
        rally_id = f"rally-{index}"
        query = f"query-{index}"
        rallies[rally_id] = {
            "rally_id": rally_id,
            "query_sha256": query,
            "context_refs": [],
        }
        texts[query] = "what proves the claim"
        candidates = []
        for candidate_index in range(4):
            candidate_id = f"candidate-{index}-{candidate_index}"
            candidates.append(
                {"candidate_id": candidate_id, "text_sha256": candidate_id}
            )
            texts[candidate_id] = "bounded fact"
        snapshots[rally_id] = {
            "snapshot_sha256": f"snapshot-{index}",
            "candidates": candidates,
        }
    teacher = RemoteTeacher()
    result = distill._run_teacher_batch(
        root=tmp_path,
        config=distill.DistillationConfig(
            teacher_profile=distill.OX_SINGLE_PROFILE,
            ox_enabled=True,
            teacher_claim_limit=500,
        ),
        teachers={distill.OX_TEACHER_ROLE: teacher},
        snapshots=snapshots,
        rally_by_id=rallies,
        texts=texts,
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result.labels_written == 500
    assert teacher.batch_sizes
    assert result.model_calls == 32


def test_transient_teacher_defer_writes_no_label_and_retries(tmp_path: Path) -> None:
    class DeferredTeacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise distill.DistillationDeferred("foreground")

    class WorkingTeacher(DeferredTeacher):
        def evaluate(self, payload: object) -> dict[str, object]:
            assert isinstance(payload, dict)
            return {
                "labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "verdict": "uncertain",
                        "rationale": "retry",
                    }
                    for candidate in payload["candidates"]
                ]
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    deferred = {role: DeferredTeacher(role) for role in distill.TEACHER_ROLES}
    first = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers=deferred
    )
    assert first["status"] == "deferred"
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    assert store.read_chain(label_path) == []
    working = {role: WorkingTeacher(role) for role in distill.TEACHER_ROLES}
    second = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers=working
    )
    assert second["labels_written"] > 0


def test_timeout_teacher_is_deferred_without_advancing_label_cursor(
    tmp_path: Path,
) -> None:
    class TimeoutTeacher:
        local = True

        def __init__(self, role: str) -> None:
            self.role = role

        def evaluate(self, _payload: object) -> dict[str, object]:
            raise TimeoutError("temporary backend timeout")

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    teachers = {role: TimeoutTeacher(role) for role in distill.TEACHER_ROLES}
    result = distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers=teachers
    )
    assert result["status"] == "deferred"
    assert (
        store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl") == []
    )


def test_chunk_preserves_rollout_traffic_state(tmp_path: Path) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    state_path = store.distillation_dir(tmp_path) / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "kind": "worker-state",
            "status": "canary",
            "rollout_percent": 25,
            "stage_started_at": "2026-08-01T00:00:00Z",
        },
    )
    distill.run_distillation_chunk(
        root=tmp_path, raw_dir=raw_dir, config_path=config, teachers={}
    )
    state = store.read_sealed(state_path)
    assert (state["status"], state["rollout_percent"]) == ("canary", 25)
    assert state["worker_status"] == "capture_only"


def test_automatic_rollout_holds_and_ignores_caller_supplied_metrics(
    tmp_path: Path,
) -> None:
    _config(tmp_path)
    policy = distill.train_tiny_policy([])

    def write_policy(name: str) -> tuple[str, dict[str, object]]:
        return_value = store.write_immutable(
            store.distillation_dir(tmp_path) / "policies",
            {
                "kind": "tiny-logistic-policy",
                **policy,
                "lineage": {
                    "locked_replay_id": hashlib.sha256(name.encode()).hexdigest(),
                    "model_cohort_sha256": "e" * 64,
                },
            },
            schema=distill.POLICY_SCHEMA,
        )
        return return_value[0], return_value[2]

    incumbent_id, _ = write_policy("incumbent")
    candidate_id, _ = write_policy("candidate")
    store.write_pointer(tmp_path, "active", incumbent_id)
    store.write_pointer(tmp_path, "lkg", incumbent_id)
    store.write_pointer(tmp_path, "candidate", candidate_id)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "replay",
            "rollout_percent": 0,
            "stage_started_at": "2026-08-01T00:00:00Z",
            "stage_run_id": "d" * 64,
        },
    )
    _, _, baseline = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {
            "kind": "test-baseline",
            "raw_watermark": "a" * 64,
            "hard_floor": {"p5_allowed": True, "reasons": []},
            "offline_training_gate": {"passed": True, "revision": "test-v2"},
        },
        schema=distill.BASELINE_SCHEMA,
    )
    first = distill._automatic_rollout_evaluation(
        tmp_path, baseline, {"status": "candidate", "policy_id": candidate_id}
    )
    assert first == {"status": "held", "reason": "rollout_baseline_mismatch"}
    assert not list(
        (store.distillation_dir(tmp_path) / "evaluations").glob("*.json")
    )


def test_automatic_replay_materializes_actual_paired_shadow_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.recall import recall_distillation_rollout as rollout

    _config(tmp_path)
    _, _, baseline = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {
            "kind": "test-baseline",
            "raw_watermark": "a" * 64,
            "hard_floor": {"p5_allowed": True, "reasons": []},
            "offline_training_gate": {"passed": True, "revision": "test-v2"},
        },
        schema=distill.BASELINE_SCHEMA,
    )
    bootstrap = distill._ensure_bootstrap_policy(tmp_path, baseline)
    run_id = "a" * 64
    cohort = "paired-test-cohort"
    candidate = _fixture_candidate(
        tmp_path,
        baseline_artifact_id=str(baseline["artifact_id"]),
        model_cohort_sha256=cohort,
    )
    candidate_id = str(candidate["artifact_id"])
    incumbent_id = str(bootstrap["artifact_id"])
    context = {
        "stage": "shadow",
        "stage_started_at": "2026-08-01T00:00:00Z",
        "qualified_run_id": run_id,
        "cohort": cohort,
        "baseline_artifact_id": str(baseline["artifact_id"]),
        "served_policy_id": incumbent_id,
        "candidate_policy_id": candidate_id,
        "incumbent_policy_id": incumbent_id,
        "served_policy": bootstrap,
        "candidate_policy": candidate,
        "incumbent_policy": bootstrap,
    }
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "shadow",
            "rollout_percent": 0,
            "stage_started_at": context["stage_started_at"],
            "stage_run_id": run_id,
        },
    )
    monkeypatch.setattr(
        distill,
        "load_policy_observation_context",
        lambda _session_id, _root=None: context,
    )
    observations = (
        ("00" + "0" + "a" * 61, "2026-08-01T00:00:00Z"),
        ("b5" + "1" + "b" * 61, "2026-08-04T00:00:00Z"),
        ("e0" + "2" + "c" * 61, "2026-08-08T00:00:00Z"),
    )
    for index, (query_hash, observed_at) in enumerate(observations):
        candidate_key = f"candidate-{index}"
        unselected_key = f"aaa-{index}"
        features = distill.build_fast_features(
            query_chargram_coverage=0.5,
            candidate_chargram_precision=0.5,
        )
        page_sha = hashlib.sha256(f"page-{index}".encode()).hexdigest()
        pool = [
            {
                "candidate_id": unselected_key,
                "selected": False,
                "page_id": f"page-unselected-{index}",
                "page_content_sha256": hashlib.sha256(
                    f"page-unselected-{index}".encode()
                ).hexdigest(),
                "rendered_context_sha256": hashlib.sha256(
                    f"context-unselected-{index}".encode()
                ).hexdigest(),
            },
            {
                "candidate_id": candidate_key,
                "selected": True,
                "page_id": f"page-{index}",
                "page_content_sha256": page_sha,
                "rendered_context_sha256": hashlib.sha256(
                    f"context-{index}".encode()
                ).hexdigest(),
            }
        ]
        evidence = distill.ShadowOperationalEvidence(
            candidate_quality=True,
            baseline_quality=True,
            candidate_covered=True,
            baseline_covered=True,
            candidate_anchor_retained=True,
            baseline_anchor_retained=True,
            candidate_abstained=False,
            baseline_abstained=False,
            candidate_score_ms=10,
            live_latency_ms=10,
            resource_ok=True,
            integrity_ok=True,
            negative_veto=False,
            deadline_ms=1_200,
            stage="shadow",
            run_id=run_id,
            cohort=cohort,
            host="codex",
            feature_parity=True,
        )
        distill.record_shadow_observation(
            decision_id=f"decision-{index}",
            host="codex",
            session_id=f"session-{index}",
            query_semantic_sha256=query_hash,
            policy_id=candidate_id,
            incumbent_policy_id=incumbent_id,
            served_policy_id=incumbent_id,
            selected_candidate_ids=[candidate_key],
            incumbent_selected_candidate_ids=[candidate_key],
            paired_eligible=True,
            candidate_feature_snapshot=[
                {"candidate_id": candidate_key, "features": features},
                {"candidate_id": unselected_key, "features": features},
            ],
            candidate_pool_refs=pool,
            baseline_feature_snapshot=[
                {"candidate_id": candidate_key, "features": features},
                {"candidate_id": unselected_key, "features": features},
            ],
            baseline_pool_refs=pool,
            observed_at=observed_at,
            decision_latency_ms=10,
            timed_out=False,
            operational_evidence=evidence,
            root=tmp_path,
        )
    result = distill._automatic_rollout_evaluation(
        tmp_path,
        baseline,
        {"status": "candidate", "policy_id": candidate_id},
    )
    assert result["status"] == "shadow"
    evaluation = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{result['evaluation_artifact_id']}.json"
    )
    observation = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "rollout-observations"
        / f"{result['replay_observation_artifact_id']}.json"
    )
    split_id = evaluation["split_sha256"]
    assert split_id and len(split_id) == 64
    assert observation["pair_count"] == 3
    assert len(observation["pairs"]) == 3
    assert all(set(pair) == rollout._REPLAY_PAIR_KEYS for pair in observation["pairs"])
    split = store.read_sealed(
        store.distillation_dir(tmp_path) / "locked-replays" / f"{split_id}.json",
        schema="chronovisor.recall-distill-locked-replay.v1",
    )
    assert all(set(row) == rollout._LOCKED_REPLAY_ROW_KEYS for row in split["training_rows"])
    assert {
        row["split"] for row in split["training_rows"]
    } == {"train", "validation", "test"}
    assert {
        row["candidate_id"] for row in split["training_rows"]
    } == {f"candidate-{index}" for index in range(3)}


def test_shadow_replay_source_identity_uses_selected_pool_row() -> None:
    pool = [
        {"candidate_id": "aaa", "selected": False},
        {"candidate_id": "zzz", "selected": True},
    ]
    fields = distill._shadow_replay_source_fields(
        decision_id="decision",
        query_semantic_sha256="a" * 64,
        observed_at="2026-08-01T00:00:00Z",
        pool_rows=pool,
        selected_candidate_ids=["zzz"],
        baseline_pool_rows=pool,
        baseline_selected_candidate_ids=["zzz"],
        paired_eligible=True,
    )
    assert fields["candidate_id"] == "zzz"
    with pytest.raises(distill.DistillationError, match="not selected"):
        distill._shadow_replay_source_fields(
            decision_id="decision",
            query_semantic_sha256="a" * 64,
            observed_at="2026-08-01T00:00:00Z",
            pool_rows=pool,
            selected_candidate_ids=["aaa"],
            baseline_pool_rows=pool,
            baseline_selected_candidate_ids=["zzz"],
            paired_eligible=True,
        )
    with pytest.raises(distill.DistillationError, match="ambiguous"):
        distill._shadow_replay_source_fields(
            decision_id="decision",
            query_semantic_sha256="a" * 64,
            observed_at="2026-08-01T00:00:00Z",
            pool_rows=pool,
            selected_candidate_ids=["aaa", "zzz"],
            baseline_pool_rows=pool,
            baseline_selected_candidate_ids=["zzz"],
            paired_eligible=True,
        )


def test_qualified_shadow_policy_records_private_operational_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config(tmp_path)
    _, _, baseline = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {
            "kind": "test-baseline",
            "raw_watermark": "a" * 64,
            "hard_floor": {"p5_allowed": True, "reasons": []},
            "offline_training_gate": {"passed": True, "revision": "test-v2"},
        },
        schema=distill.BASELINE_SCHEMA,
    )
    bootstrap = distill._ensure_bootstrap_policy(tmp_path, baseline)
    candidate = _fixture_candidate(
        tmp_path, baseline_artifact_id=str(baseline["artifact_id"])
    )
    run_id = "c" * 64
    context = {
        "stage": "shadow",
        "stage_started_at": "2026-08-14T00:00:00Z",
        "qualified_run_id": run_id,
        "cohort": "rollout-test-cohort",
        "baseline_artifact_id": str(baseline["artifact_id"]),
        "served_policy_id": str(bootstrap["artifact_id"]),
        "candidate_policy_id": str(candidate["artifact_id"]),
        "incumbent_policy_id": str(bootstrap["artifact_id"]),
        "served_policy": bootstrap,
        "candidate_policy": candidate,
        "incumbent_policy": bootstrap,
    }
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "shadow",
            "rollout_percent": 0,
            "stage_started_at": "2026-08-14T00:00:00Z",
            "stage_run_id": run_id,
        },
    )
    monkeypatch.setattr(
        distill,
        "load_policy_observation_context",
        lambda _session_id, _root=None: context,
    )
    assert distill.load_shadow_policy(tmp_path)["artifact_id"] == candidate["artifact_id"]

    rendered = "private bounded snippet"
    features = distill.build_fast_features(query_chargram_coverage=1)
    shadow_ledger = (
        store.distillation_dir(tmp_path) / "shadow-observation-receipts.jsonl"
    )
    lock = store.acquire_nonblocking_lock(shadow_ledger.with_suffix(".jsonl.lock"))
    assert lock is not None
    try:
        deferred = distill.record_shadow_observation(
            decision_id="shadow-busy",
            host="codex",
            session_id="private-session",
            query_semantic_sha256="e" * 64,
            policy_id=candidate["artifact_id"],
            incumbent_policy_id=bootstrap["artifact_id"],
            served_policy_id=bootstrap["artifact_id"],
            selected_candidate_ids=["page-v1"],
            incumbent_selected_candidate_ids=["page-v1"],
            paired_eligible=True,
            candidate_feature_snapshot=[
                {"candidate_id": "page-v1", "features": features}
            ],
            candidate_pool_refs=[
                {
                    "candidate_id": "page-v1",
                    "selected": True,
                    "page_id": "page",
                    "page_content_sha256": "f" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            observed_at="2026-08-14T00:00:01Z",
            decision_latency_ms=42,
            timed_out=False,
            nonblocking=True,
            root=tmp_path,
        )
    finally:
        store.release_lock(lock)
    assert deferred == {"status": "deferred", "reason": "receipt_ledger_busy"}
    assert not shadow_ledger.exists()
    assert not (store.distillation_dir(tmp_path) / "shadow-observations").exists()
    evidence = distill.ShadowOperationalEvidence(
        candidate_quality=True,
        baseline_quality=True,
        candidate_covered=True,
        baseline_covered=True,
        candidate_anchor_retained=True,
        baseline_anchor_retained=True,
        candidate_abstained=False,
        baseline_abstained=False,
        candidate_score_ms=42,
        live_latency_ms=42,
        resource_ok=True,
        integrity_ok=True,
        negative_veto=False,
        deadline_ms=1_200,
        stage="shadow",
        run_id=run_id,
        cohort="rollout-test-cohort",
        host="codex",
        feature_parity=True,
    )
    receipt = distill.record_shadow_observation(
        decision_id="shadow-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=candidate["artifact_id"],
        incumbent_policy_id=bootstrap["artifact_id"],
        served_policy_id=bootstrap["artifact_id"],
        selected_candidate_ids=["page-v1"],
        incumbent_selected_candidate_ids=["page-v1"],
        paired_eligible=True,
        candidate_feature_snapshot=[{"candidate_id": "page-v1", "features": features}],
        candidate_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": True,
                "page_id": "page",
                "page_content_sha256": "f" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        observed_at="2026-08-14T00:00:01Z",
        decision_latency_ms=42,
        timed_out=False,
        operational_evidence=evidence,
        root=tmp_path,
    )
    retry = distill.record_shadow_observation(
        decision_id="shadow-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=candidate["artifact_id"],
        incumbent_policy_id=bootstrap["artifact_id"],
        served_policy_id=bootstrap["artifact_id"],
        selected_candidate_ids=["page-v1"],
        incumbent_selected_candidate_ids=["page-v1"],
        paired_eligible=True,
        candidate_feature_snapshot=[{"candidate_id": "page-v1", "features": features}],
        candidate_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": True,
                "page_id": "page",
                "page_content_sha256": "f" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        observed_at="2026-08-14T00:00:02Z",
        decision_latency_ms=42,
        timed_out=False,
        operational_evidence=evidence,
        root=tmp_path,
    )
    assert retry["record_sha256"] == receipt["record_sha256"]
    distill.record_shadow_observation(
        decision_id="shadow-unpaired",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="e" * 64,
        policy_id=candidate["artifact_id"],
        incumbent_policy_id=bootstrap["artifact_id"],
        served_policy_id=bootstrap["artifact_id"],
        selected_candidate_ids=["page-v1"],
        incumbent_selected_candidate_ids=[],
        paired_eligible=False,
        candidate_feature_snapshot=[{"candidate_id": "page-v1", "features": features}],
        candidate_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": True,
                "page_id": "page",
                "page_content_sha256": "f" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        baseline_pool_refs=[
            {
                "candidate_id": "page-v1",
                "selected": False,
                "page_id": "page",
                "page_content_sha256": "f" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        observed_at="2026-08-14T00:00:03Z",
        decision_latency_ms=41,
        timed_out=False,
        root=tmp_path,
    )
    artifact_path = (
        store.distillation_dir(tmp_path)
        / "shadow-observations"
        / f"{receipt['shadow_observation_artifact_id']}.json"
    )
    artifact = store.read_sealed(
        artifact_path, schema=distill.SHADOW_OBSERVATION_SCHEMA
    )
    assert "rendered_context" not in artifact["candidate_pool_refs"][0]
    operational = distill._operational_rollout_metrics(
        tmp_path,
        candidate["artifact_id"],
        bootstrap["artifact_id"],
        baseline_artifact_id=str(baseline["artifact_id"]),
        cohort="rollout-test-cohort",
    )
    assert operational["coverage_abstain"]["denominator"] == 0
    assert operational["latency_timeout"]["denominator"] == 0
    artifact_path.write_text("{}\n", encoding="utf-8")
    tampered = distill._operational_rollout_metrics(
        tmp_path,
        candidate["artifact_id"],
        bootstrap["artifact_id"],
        baseline_artifact_id=str(baseline["artifact_id"]),
        cohort="rollout-test-cohort",
    )
    assert tampered["coverage_abstain"]["denominator"] == 0


def test_unpaired_exact_receipts_do_not_qualify_rollout_metrics(
    tmp_path: Path,
) -> None:
    candidate_id = _baseline_identity(tmp_path)
    _, _, incumbent = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {"kind": "second-incumbent"},
        schema=distill.BASELINE_SCHEMA,
    )
    rendered = "bounded card"
    features = distill.build_fast_features(query_chargram_coverage=1)
    for index, policy_id in enumerate((candidate_id, incumbent["artifact_id"])):
        distill.record_exact_exposure(
            decision_id=f"metric-{index}",
            host="codex",
            session_id=f"session-{index}",
            query_semantic_sha256=hashlib.sha256(f"query-{index}".encode()).hexdigest(),
            policy_id=policy_id,
            candidate_refs=[
                {
                    "candidate_id": f"page-{index}",
                    "page_id": f"page-{index}",
                    "page_content_sha256": "9" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            candidate_feature_snapshot=[
                {"candidate_id": f"page-{index}", "features": features}
            ],
            candidate_pool_refs=[
                {
                    "candidate_id": f"page-{index}",
                    "selected": True,
                    "page_id": f"page-{index}",
                    "page_content_sha256": "9" * 64,
                    "rendered_context": rendered,
                    "rendered_context_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                }
            ],
            render_sha256="8" * 64,
            candidate_snapshot_sha256="7" * 64,
            observed_at=f"2026-08-14T00:00:0{index}Z",
            decision_latency_ms=100 + index,
            timed_out=False,
            root=tmp_path,
        )
    metrics = distill._operational_rollout_metrics(
        tmp_path,
        candidate_id,
        incumbent["artifact_id"],
        baseline_artifact_id=str(incumbent["artifact_id"]),
        cohort="rollout-test-cohort",
    )
    assert metrics["coverage_abstain"]["denominator"] == 0
    assert metrics["latency_timeout"]["denominator"] == 0
    assert metrics["feature_parity"]["denominator"] == 0
    automatic = distill._automatic_baseline_metrics(tmp_path)
    assert automatic["coverage_rate"] == 1.0
    assert automatic["latency_p95_ms"] == 101


def test_page_fallback_counts_only_coverage_and_runtime_denominators(
    tmp_path: Path,
) -> None:
    candidate_id = _baseline_identity(tmp_path)
    incumbent_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "baselines",
        {"kind": "page-fallback-incumbent"},
        schema=distill.BASELINE_SCHEMA,
    )
    for index, policy_id in enumerate((candidate_id, incumbent_id)):
        receipt = distill.record_exposure(
            decision_id=f"capture-error-{index}",
            host="codex",
            session_id=f"private-{index}",
            prompt_hash=f"prompt-{index}",
            policy_id=policy_id,
            candidate_ids=[f"selected-{index}"],
            candidate_snapshot_sha256=hashlib.sha256(
                f"snapshot-{index}".encode()
            ).hexdigest(),
            observed_at=f"2026-08-14T00:00:0{index}Z",
            decision_latency_ms=125 + index,
            timed_out=False,
            error_code="exact_capture_error",
            root=tmp_path,
        )
        assert receipt["runtime_observation"]["error_code"] == "exact_capture_error"
    metrics = distill._operational_rollout_metrics(
        tmp_path,
        candidate_id,
        incumbent_id,
        baseline_artifact_id=incumbent_id,
        cohort="rollout-test-cohort",
    )
    assert metrics["coverage_abstain"]["denominator"] == 0
    assert metrics["latency_timeout"]["denominator"] == 0
    assert metrics["feature_parity"]["denominator"] == 0
    automatic = distill._automatic_baseline_metrics(tmp_path)
    assert automatic["coverage_rate"] == 1.0
    assert automatic["latency_p95_ms"] == 126
    assert "candidate_recall" not in automatic


def test_page_operational_receipt_is_deduped_when_exact_decision_exists(
    tmp_path: Path,
) -> None:
    policy_id = _baseline_identity(tmp_path)
    distill.record_exact_exposure(
        decision_id="same-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="a" * 64,
        policy_id=policy_id,
        candidate_refs=[],
        render_sha256="b" * 64,
        candidate_snapshot_sha256="c" * 64,
        observed_at="2026-08-14T00:00:00Z",
        decision_latency_ms=50,
        timed_out=False,
        root=tmp_path,
    )
    observation = {
        "decision": "none",
        "selected_count": 0,
        "latency_ms": 50.0,
        "timed_out": False,
        "error_code": "exact_capture_error",
    }
    binding = {
        "decision_id": "same-decision",
        "host": "codex",
        "session_id_sha256": hashlib.sha256(b"private-session").hexdigest(),
        "prompt_hash": "prompt",
        "policy_id": policy_id,
        "candidate_ids": [],
        "candidate_snapshot_sha256": "c" * 64,
        "runtime_observation_sha256": distill.canonical_json.canonical_json_sha256_strict(
            observation
        ),
        "observed_at": "2026-08-14T00:00:00Z",
    }
    store.append_chain(
        store.distillation_dir(tmp_path) / "exposure-receipts.jsonl",
        {
            "kind": "prospective-page-exposure",
            **binding,
            "runtime_observation": observation,
            "binding_sha256": distill.canonical_json.canonical_json_sha256_strict(
                binding
            ),
            "idempotency_sha256": "d" * 64,
        },
    )
    metrics = distill._operational_rollout_metrics(
        tmp_path,
        policy_id,
        "f" * 64,
        baseline_artifact_id="f" * 64,
        cohort="rollout-test-cohort",
    )
    assert metrics["latency_timeout"]["denominator"] == 0


def test_counterfactual_uses_exact_arms_and_copies_live_features(
    tmp_path: Path,
) -> None:
    class Counterfactual:
        local = True

        def __init__(self, outcome_receipt_id: str, *, fail_transient: bool) -> None:
            self.outcome_receipt_id = outcome_receipt_id
            self.fail_transient = fail_transient

        def compare(self, payload: object) -> dict[str, object]:
            if self.fail_transient:
                raise OSError("temporary local worker failure")
            assert isinstance(payload, dict)
            assert payload["mode"] == "remove"
            assert payload["a0_evidence"] != payload["a1_evidence"]
            return {
                "verdict": "helpful",
                "reason": "matched",
                "a0_sha256": "1" * 64,
                "a1_sha256": "2" * 64,
                "blind_orders": ["a0_first", "a1_first"],
                "order_agreement": True,
                "generator_route_identity": {
                    "role": "generator",
                    "provider": "test",
                    "model": "generator",
                    "location": "local",
                },
                "generator_model_digest": "3" * 64,
                "judge_route_identity": {
                    "role": "judge",
                    "provider": "test",
                    "model": "judge",
                    "location": "local",
                },
                "judge_model_digest": "4" * 64,
                "closed_outcome_receipt_id": self.outcome_receipt_id,
            }

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    rally = distill.extract_rallies(raw_dir, root=tmp_path)[1]
    rendered = "page context"
    features = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    identity = _baseline_identity(tmp_path)
    exposure = distill.record_exact_exposure(
        decision_id="counterfactual",
        host="codex",
        session_id="session-one",
        query_semantic_sha256=rally["query_sha256"],
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "page-v1",
                "page_id": "page",
                "page_content_sha256": "9" * 64,
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        candidate_feature_snapshot=[{"candidate_id": "page-v1", "features": features}],
        render_sha256="5" * 64,
        candidate_snapshot_sha256="6" * 64,
        observed_at="2026-08-02T00:00:00.500000Z",
        root=tmp_path,
    )
    outcome = distill.record_closed_outcome(
        outcome_id="test-outcome",
        exposure_artifact_id=exposure["exposure_artifact_id"],
        candidate_id="page-v1",
        candidate_version_sha256="9" * 64,
        kind="test",
        status="passed",
        evidence_sha256="7" * 64,
        observed_at="2026-08-02T00:00:00.750000Z",
        root=tmp_path,
    )
    with pytest.raises(distill.DistillationError, match="not exposed"):
        distill.record_closed_outcome(
            outcome_id="pool-only",
            exposure_artifact_id=exposure["exposure_artifact_id"],
            candidate_id="unselected",
            candidate_version_sha256="8" * 64,
            kind="test",
            status="passed",
            evidence_sha256="7" * 64,
            observed_at="2026-08-02T00:00:00.800000Z",
            root=tmp_path,
        )
    failed = distill.record_closed_outcome(
        outcome_id="failed-outcome",
        exposure_artifact_id=exposure["exposure_artifact_id"],
        candidate_id="page-v1",
        candidate_version_sha256="9" * 64,
        kind="test",
        status="failed",
        evidence_sha256="6" * 64,
        observed_at="2026-08-02T00:00:00.900000Z",
        root=tmp_path,
    )
    failed_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "outcomes"
        / f"{failed['outcome_artifact_id']}.json",
        schema=distill.OUTCOME_SCHEMA,
    )
    assert failed_artifact["authority"] == "capture-only"
    counterfactual = Counterfactual("f" * 64, fail_transient=True)
    deferred = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        counterfactual=counterfactual,
        structural_verifier=lambda *_args: "exact_test_outcome",
    )
    assert deferred["status"] == "deferred"
    assert (
        store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl") == []
    )
    counterfactual.fail_transient = False
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        counterfactual=counterfactual,
        structural_verifier=lambda *_args: "exact_test_outcome",
    )
    assert result["counterfactuals_written"] == 1
    row = store.read_chain(store.distillation_dir(tmp_path) / "label-ledger.jsonl")[0]
    assert row["authority"] == "teacher-only"
    assert row["features"] == features
    assert distill.train_tiny_policy([row])["training_rows"] == 1
    training = distill.materialize_training_rows(tmp_path)
    assert training["rows"][0]["features"] == features
    assert (
        distill._resolve_closed_outcome(
            tmp_path,
            outcome["record_sha256"],
            exposure_artifact_id=exposure["exposure_artifact_id"],
            candidate_id="page-v1",
        )
        is None
    )
    outcome_path = (
        store.distillation_dir(tmp_path)
        / "outcomes"
        / f"{outcome['outcome_artifact_id']}.json"
    )
    outcome_path.write_text("{}\n")
    assert (
        distill._resolve_closed_outcome(
            tmp_path,
            outcome["record_sha256"],
            exposure_artifact_id=exposure["exposure_artifact_id"],
            candidate_id="page-v1",
        )
        is None
    )


def test_snapshot_distinguishes_missing_from_tampering(tmp_path: Path) -> None:
    assert store.snapshot(tmp_path)["error_code"] == "missing_state"
    path = store.distillation_dir(tmp_path) / store.STATE_FILE
    path.parent.mkdir(parents=True)
    path.write_text('{"schema":"broken"}\n')
    snapshot = store.snapshot(tmp_path)
    assert snapshot["status"] == "tampered"
    assert snapshot["error_code"] == "invalid_state"


def test_text_feature_v2_is_identical_for_historical_and_live_jp_en() -> None:
    english = distill.build_text_features(
        "Recall rollout status", "Recall rollout status and latency"
    )
    full_width = distill.build_text_features(
        "Ｒｅｃａｌｌ rollout status", "recall ROLLOUT status and latency"
    )
    japanese = distill.build_text_features(
        "クロノバイザーの検索精度", "検索精度を改善するクロノバイザー"
    )
    unrelated = distill.build_text_features("クロノバイザー", "転職と給与の相談")
    assert english == full_width
    assert japanese["query_chargram_coverage"] > 0
    assert japanese["query_chargram_coverage"] > unrelated["query_chargram_coverage"]
    assert set(english) == set(distill.FAST_FEATURE_KEYS)


def test_cwd_and_three_model_votes_never_become_verified() -> None:
    structural = distill._structural_tokens({"cwd": "/private/project/src/app.py"})
    assert structural["path"] == []
    labels = [
        distill._teacher_label(
            {"verdict": "relevant"}, verified_predicate="exact_path_overlap"
        )
        for _route in distill.TEACHER_ROLES
    ]
    assert {row["authority"] for row in labels} == {"teacher-only"}


def test_historical_teacher_row_materializes_without_live_exposure(
    tmp_path: Path,
) -> None:
    rally_id = "historical-rally"
    candidate_id = "historical-candidate"
    features = distill.build_text_features("検索精度", "検索精度を改善")
    store.append_chain(
        store.distillation_dir(tmp_path) / "rally-manifest.jsonl",
        {
            "kind": "rally-manifest",
            "manifest": {
                "rally_id": rally_id,
                "session_cluster_id": "session-cluster",
                "as_of": "2026-01-01T00:00:00Z",
            },
        },
    )
    store.append_chain(
        store.distillation_dir(tmp_path) / "candidate-ledger.jsonl",
        {
            "kind": "candidate-snapshot",
            "rally_id": rally_id,
            "snapshot": {
                "feature_revision": distill.TEXT_FEATURE_REVISION,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "feature_revision": distill.TEXT_FEATURE_REVISION,
                        "features": features,
                    }
                ],
            },
        },
    )
    store.append_chain(
        store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        {
            "kind": "teacher-label",
            "rally_id": rally_id,
            "candidate_id": candidate_id,
            "route": distill.TEACHER_ROLES[0],
            "model_digest": "a" * 64,
            "assignment": {"probe": False},
            "dimension": "relevance",
            "verdict": "relevant",
            "authority": "teacher-only",
        },
    )
    manifest = store.read_chain(
        store.distillation_dir(tmp_path) / "rally-manifest.jsonl"
    )[0]["manifest"]
    split_plan = distill._ensure_split_plan(
        tmp_path,
        [manifest],
        raw_watermark="b" * 64,
        model_cohort_sha256="c" * 64,
    )
    artifact = distill.materialize_training_rows(tmp_path)
    assert artifact["rows"][0]["features"] == features
    assert artifact["rows"][0]["split_plan_id"] == split_plan["artifact_id"]
    assert (
        distill.train_tiny_policy([{**artifact["rows"][0], "split": "train"}])[
            "training_rows"
        ]
        == 1
    )


def test_probe_and_locked_test_rows_cannot_change_policy_bytes() -> None:
    positive = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    negative = distill.build_fast_features()
    base = [
        {
            "rally_id": "train-positive",
            "candidate_id": "positive",
            "dimension": "relevance",
            "verdict": "relevant",
            "authority": "teacher-only",
            "features": positive,
            "split": "train",
        },
        {
            "rally_id": "train-negative",
            "candidate_id": "negative",
            "dimension": "relevance",
            "verdict": "irrelevant",
            "authority": "teacher-only",
            "features": negative,
            "split": "train",
        },
    ]
    locked = distill.train_tiny_policy(base)
    changed_holdouts = [
        *base,
        {
            **base[0],
            "rally_id": "probe",
            "candidate_id": "probe",
            "probe": True,
            "verdict": "irrelevant",
        },
        {
            **base[1],
            "rally_id": "locked-test",
            "candidate_id": "locked-test",
            "split": "test",
            "verdict": "relevant",
        },
    ]
    assert distill.train_tiny_policy(changed_holdouts) == locked


def test_offline_gate_uses_route_stability_and_agreed_counterfactuals(
    tmp_path: Path,
) -> None:
    positive = distill.build_fast_features(
        query_chargram_coverage=1, candidate_chargram_precision=1
    )
    negative = distill.build_fast_features()
    digests = {
        role: f"{index + 1}" * 64 for index, role in enumerate(distill.TEACHER_ROLES)
    }
    rows: list[dict[str, object]] = []
    for index in range(680):
        route = distill.TEACHER_ROLES[index % 3]
        verdict = "relevant" if index % 2 == 0 else "irrelevant"
        rows.append(
            {
                "rally_id": f"owner-{index}",
                "candidate_id": f"owner-candidate-{index}",
                "session_cluster_id": f"owner-session-{index}",
                "as_of": f"{index:06}",
                "dimension": "relevance",
                "verdict": verdict,
                "authority": "teacher-only",
                "features": positive if verdict == "relevant" else negative,
                "route": route,
                "model_digest": digests[route],
                "probe": False,
                "source": "teacher-label",
            }
        )
    for index in range(61):
        verdict = "relevant" if index < 31 else "irrelevant"
        for route in distill.TEACHER_ROLES:
            rows.append(
                {
                    "rally_id": f"probe-{index}",
                    "candidate_id": f"probe-candidate-{index}",
                    "session_cluster_id": f"probe-session-{index}",
                    "as_of": f"{680 + index:06}",
                    "dimension": "relevance",
                    "verdict": verdict,
                    "authority": "teacher-only",
                    "features": positive if verdict == "relevant" else negative,
                    "route": route,
                    "model_digest": digests[route],
                    "probe": True,
                    "source": "teacher-label",
                }
            )
    for index in range(60):
        verdict = "helpful" if index < 30 else "harmful"
        rows.append(
            {
                "rally_id": f"cf-{index}",
                "candidate_id": f"cf-candidate-{index}",
                "session_cluster_id": f"cf-session-{index}",
                "as_of": f"{741 + index:06}",
                "dimension": "answer_utility",
                "verdict": verdict,
                "authority": "teacher-only",
                "features": positive if verdict == "helpful" else negative,
                "route": "counterfactual",
                "model_digest": "f" * 64,
                "generator_model_digest": "4" * 64,
                "judge_model_digest": "5" * 64,
                "probe": False,
                "source": "counterfactual-label",
                "order_agreement": True,
            }
        )
    for index, row in enumerate(rows):
        row.update(
            {
                "label_record_sha256": f"{index:064x}",
                "future_leakage": False,
                "feature_parity": True,
                "negative_veto_conflict": False,
            }
        )
        if row["source"] == "counterfactual-label":
            row.update(
                {
                    "counterfactual_ref": "a" * 64,
                    "a0_sha256": "b" * 64,
                    "a1_sha256": "c" * 64,
                    "counterfactual_producer": "chronovisor-local-blind-v1",
                    "counterfactual_revision": "two-order-locked-v1",
                    "blind_orders": ["a0_first", "a1_first"],
                    "profile": distill.LOCAL_TRIAD_PROFILE,
                    "cohort": distill.LOCAL_TRIAD_PROFILE,
                    "profile_contract_id": "",
                    "expires_at": "",
                    "identity_revision": "local-blind-counterfactual-v1",
                    "request_revision": "local-blind-counterfactual-v1",
                    "assignment_revision": distill.ASSIGNMENT_REVISION,
                    "assignment_authority": distill._training_assignment_authority(
                        {
                            "revision": distill.ASSIGNMENT_REVISION,
                            "kind": "counterfactual",
                        }
                    ),
                    "generator_route_identity": {
                        "role": "generator",
                        "provider": "test",
                        "model": "generator",
                        "location": "local",
                    },
                    "judge_route_identity": {
                        "role": "judge",
                        "provider": "test",
                        "model": "judge",
                        "location": "local",
                    },
                }
            )
    config = distill.DistillationConfig(
        hard_floor_teacher_labels=1,
        hard_floor_teacher_per_class=1,
        hard_floor_probe_pairs=1,
        hard_floor_counterfactual_pairs=1,
    )

    def bind_split(values: list[dict[str, object]]) -> list[dict[str, object]]:
        active, cohort = distill._active_training_cohort(values)
        plan = distill._ensure_split_plan(
            tmp_path,
            active,
            raw_watermark="0" * 64,
            model_cohort_sha256=cohort["cohort_sha256"],
        )
        bound: list[dict[str, object]] = []
        for row in values:
            split = plan["assignments"].get(str(row["rally_id"]))
            fields: dict[str, object] = {}
            if split is not None:
                fields = {
                    "split": split,
                    "split_plan_id": plan["artifact_id"],
                    "locked_test_read_only": split == "test",
                    "locked_test_evidence_ref": (
                        f"split-plan:{plan['artifact_id']}" if split == "test" else ""
                    ),
                }
            bound.append({**row, **fields})
        return bound

    missing = distill._offline_training_gate(rows, config, root=tmp_path)
    assert "fixed_split_plan_missing" in missing["reasons"]
    rows = bind_split(rows)
    gate = distill._offline_training_gate(rows, config, root=tmp_path)
    assert all(value["passed"] for value in gate["route_folds"].values()), gate[
        "route_folds"
    ]
    assert gate["reasons"] == []
    assert gate["passed"] is True
    assert gate["truth_authority"] == "teacher_only_not_verified"
    unstable = [dict(row) for row in rows]
    for row in unstable:
        if row.get("probe") is True and row.get("route") == distill.TEACHER_ROLES[0]:
            row["verdict"] = (
                "irrelevant" if row["verdict"] == "relevant" else "relevant"
            )
    rejected = distill._offline_training_gate(unstable, config, root=tmp_path)
    assert rejected["passed"] is False
    assert "probe_route_stability_below_gate" in rejected["reasons"]

    mixed = [{**rows[0], "split_plan_id": "f" * 64}, *rows[1:]]
    assert (
        "fixed_split_plan_missing"
        in distill._offline_training_gate(mixed, config, root=tmp_path)["reasons"]
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": "e" * 64},
    )
    assert (
        "fixed_split_plan_missing"
        in distill._offline_training_gate(rows, config, root=tmp_path)["reasons"]
    )

    updated = bind_split([*rows, {**rows[0], "model_digest": "a" * 64}])
    partial_update = distill._offline_training_gate(updated, config, root=tmp_path)
    assert partial_update["passed"] is False
    assert "probe_pairs_below_floor" in partial_update["reasons"]
    next_digests = {
        role: f"{index + 6}" * 64 for index, role in enumerate(distill.TEACHER_ROLES)
    }
    complete_update = bind_split(
        [
            *updated,
            *[
                {**row, "model_digest": next_digests[str(row["route"])]}
                for row in rows
                if row.get("source") == "teacher-label"
            ],
        ]
    )
    recovered = distill._offline_training_gate(complete_update, config, root=tmp_path)
    assert recovered["passed"] is True
    assert set(recovered["model_cohort"]["teacher_model_digests"].values()) == set(
        next_digests.values()
    )


def test_authenticated_correction_is_negative_veto_only(tmp_path: Path) -> None:
    identity = _baseline_identity(tmp_path)
    preimage = b"stale page bytes"
    postimage = b"corrected page bytes"
    rendered = "stale page"
    exposure = distill.record_exact_exposure(
        decision_id="veto-decision",
        host="codex",
        session_id="private-session",
        query_semantic_sha256="a" * 64,
        policy_id=identity,
        candidate_refs=[
            {
                "candidate_id": "page-v1",
                "page_id": "page",
                "page_content_sha256": hashlib.sha256(preimage).hexdigest(),
                "rendered_context": rendered,
                "rendered_context_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
            }
        ],
        render_sha256="b" * 64,
        candidate_snapshot_sha256="c" * 64,
        observed_at="2026-08-14T00:00:00Z",
        root=tmp_path,
    )
    receipt = distill.record_authenticated_exact_correction_veto(
        decision_id="veto-decision",
        correction_id="correction-one",
        candidate_id="page-v1",
        page_id="page",
        preimage_bytes=preimage,
        postimage_bytes=postimage,
        readback_bytes=postimage,
        cas_status="applied",
        observed_at="2026-08-14T00:01:00Z",
        root=tmp_path,
    )
    assert receipt["policy_id"] == identity
    assert distill._authenticated_negative_vetoes(tmp_path, identity) == 1
    with pytest.raises(distill.DistillationError, match="readback"):
        distill.record_authenticated_exact_correction_veto(
            decision_id="veto-decision",
            correction_id="forged-correction",
            candidate_id="page-v1",
            page_id="page",
            preimage_bytes=preimage,
            postimage_bytes=postimage,
            readback_bytes=b"different bytes",
            cas_status="applied",
            observed_at="2026-08-14T00:02:00Z",
            root=tmp_path,
        )
    assert exposure["exposure_artifact_id"]


def test_v1_policy_artifact_fails_closed_to_legacy(tmp_path: Path) -> None:
    _config(tmp_path)
    policy_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "policies",
        {
            "kind": "tiny-logistic-policy",
            **distill.train_tiny_policy([]),
        },
        schema="chronovisor.recall-distill-policy.v1",
    )
    store.write_pointer(tmp_path, "active", policy_id)
    store.write_pointer(tmp_path, "lkg", policy_id)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {"kind": "worker-state", "status": "active", "rollout_percent": 100},
    )
    assert distill.load_active_policy(tmp_path) == {}
    assert distill.load_policy_for_session("private-session", tmp_path) == {}


def test_counterfactual_blinding_rejects_same_generator_and_judge_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    answers = iter(("without", "with"))

    def fake_worker_call(
        operation: str, _role: str, payload: object, **_kwargs: object
    ) -> dict[str, object]:
        assert isinstance(payload, dict)
        assert "candidate_arm" not in payload
        calls.append((operation, payload))
        if operation == "answer":
            return {
                "answer": next(answers),
                "_model_digest": "a" * 64,
                "_route_identity": {},
            }
        return {
            "blind_choice": "b" if len(calls) == 3 else "a",
            "_model_digest": "a" * 64,
            "_route_identity": {},
        }

    monkeypatch.setattr(distill, "_worker_call", fake_worker_call)
    worker = distill._WorkerCounterfactual(
        12_000,
        {
            "recall.distill.answer_generator": {},
            "recall.distill.utility_judge": {},
        },
        {
            "recall.distill.answer_generator": "a" * 64,
            "recall.distill.utility_judge": "a" * 64,
        },
    )
    result = worker.compare(
        {
            "rally_id": "rally",
            "candidate_id": "candidate",
            "query": "query",
            "context": [],
            "a0_evidence": [],
            "a1_evidence": ["candidate"],
            "actual_answer": "answer",
        }
    )
    assert result["verdict"] == "uncertain"
    assert result["order_agreement"] is False


def test_cold_start_api_uses_fixed_split_and_nonblocking_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    original_events = distill._events
    monkeypatch.setattr(
        distill,
        "_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Raw scan")),
    )
    assert distill.cold_start_due(tmp_path) is True
    monkeypatch.setattr(distill, "_events", original_events)
    first = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        cold_start=True,
        max_elapsed_seconds=300,
    )
    assert first["cold_start_pending"] is True
    assert first["manifest_backlog"] == 0
    assert first["candidate_backlog"] == 0
    plan = distill._read_split_plan(tmp_path)
    assert plan["artifact_id"] == first["split_plan_id"]
    assert plan["raw_watermark"] == distill.committed_raw_watermark(raw_dir)
    assert plan["feature_revision"] == distill.TEXT_FEATURE_REVISION
    second = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        cold_start=True,
        max_elapsed_seconds=300,
    )
    assert second["split_plan_id"] == first["split_plan_id"]
    normal = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
    )
    assert normal["cold_start_pending"] is True
    assert distill.cold_start_due(tmp_path) is True
    rallies = distill.extract_rallies(raw_dir, root=tmp_path)
    updated_plan = distill._ensure_split_plan(
        tmp_path,
        rallies,
        raw_watermark=distill.committed_raw_watermark(raw_dir),
        model_cohort_sha256="f" * 64,
    )
    assert updated_plan["artifact_id"] != first["split_plan_id"]
    assert updated_plan["assignments"] == plan["assignments"]
    lock = store.acquire_nonblocking_lock(
        store.distillation_dir(tmp_path) / "distillation-worker.lock"
    )
    assert lock is not None
    try:
        busy = distill.run_distillation_chunk(
            root=tmp_path,
            raw_dir=raw_dir,
            config_path=config,
            teachers={},
            cold_start=True,
        )
    finally:
        store.release_lock(lock)
    assert busy == {"status": "deferred", "processed": 0, "reason": "worker_busy"}


def test_chain_batch_keeps_standard_hash_chain(tmp_path: Path) -> None:
    path = store.distillation_dir(tmp_path) / "batch-ledger.jsonl"
    rows = store.append_chain_batch(path, ({"index": index} for index in range(500)))
    assert len(rows) == 500
    assert store.verify_chain(path)["records"] == 500
    assert store.read_chain(path)[-1]["index"] == 499


def test_chain_batch_replace_failure_preserves_old_head_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / "batch-ledger.jsonl"
    store.append_chain(path, {"index": 0})
    before = path.read_bytes()
    head = store.verify_chain(path)
    replace = store.os.replace

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        store.append_chain_batch(path, ({"index": 1}, {"index": 2}))
    assert path.read_bytes() == before
    assert store.verify_chain(path) == head
    monkeypatch.setattr(store.os, "replace", replace)
    store.append_chain_batch(path, ({"index": 1}, {"index": 2}))
    assert [row["index"] for row in store.read_chain(path)] == [0, 1, 2]


def _committed_local_r4_entries(
    root: Path, count: int = 2
) -> tuple[workset.DistillationWorkset, list[dict[str, Any]]]:
    distill.bootstrap_r4_distillation_root_authority(root)
    queue = workset.DistillationWorkset(
        store.distillation_dir(root) / "local-workset.sqlite3"
    )
    prepared: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    work_items: list[dict[str, Any]] = []
    for index in range(count):
        rally_id = f"rally-{index}"
        candidate_id = f"candidate-{index}"
        assignment = distill.teacher_assignment(rally_id, candidate_id)
        route = str(assignment["owner"])
        payload_digest = canonical_json.canonical_json_sha256_strict(
            {"rally_id": rally_id, "candidate_id": candidate_id, "route": route}
        )
        work_id = distill._local_work_id(payload_digest, route)
        task = {
            "rally": {"rally_id": rally_id},
            "candidate": {"candidate_id": candidate_id},
            "assignment": assignment,
            "route": route,
        }
        route_identity = {
            "role": route,
            "provider": "ollama",
            "model": f"local-model-{index}",
            "location": "local",
        }
        prepared.append((task, route_identity, {"work_id": work_id}))
        work_items.append(
            {
                "work_id": work_id,
                "kind": f"local-teacher:{route}",
                "payload_ref": f"candidate-snapshot:{rally_id}:{candidate_id}",
                "payload_digest": payload_digest,
                "priority": 0,
                "temporal_split": {"split": "train"},
                "provenance": {"route": route},
            }
        )
    queue.advance(work_items, {"source": 1})
    claims = queue.claim(None, count, "local-r4-test", 60)
    prepared_by_work = {item[2]["work_id"]: item for item in prepared}
    labels: dict[str, dict[str, Any]] = {}
    for claim in claims:
        task, route_identity, _ = prepared_by_work[claim.work_id]
        labels[claim.work_id] = store.append_chain(
            store.distillation_dir(root) / "label-ledger.jsonl",
            {
                "kind": "teacher-label",
                "status": "completed",
                "teacher_profile": distill.LOCAL_TRIAD_PROFILE,
                "work_id": claim.work_id,
                "attempt": claim.attempt,
                "rally_id": task["rally"]["rally_id"],
                "candidate_id": task["candidate"]["candidate_id"],
                "route": task["route"],
                "captured_at": "2026-08-25T00:00:00Z",
                "assignment": task["assignment"],
                "route_identity": route_identity,
                "source_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_ox_identity_sha256": "c" * 64,
            },
        )
    queue.commit(
        claims,
        [
            {
                "status": "completed",
                "completion_ref": f"label-ledger:{labels[claim.work_id]['record_sha256']}",
                "completion_digest": labels[claim.work_id]["record_sha256"],
            }
            for claim in claims
        ],
    )
    entries = []
    for claim in claims:
        task, route_identity, _ = prepared_by_work[claim.work_id]
        entry = distill._local_r4_receipt_entry(
            work_id=claim.work_id,
            attempt=claim.attempt,
            task=task,
            label=labels[claim.work_id],
            route_identity=route_identity,
        )
        assert entry is not None
        entries.append(entry)
    return queue, entries


def test_local_r4_receipts_are_batched_private_and_crash_repairable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    source_calls = 0

    def source_binding() -> dict[str, str]:
        nonlocal source_calls
        source_calls += 1
        return dict(source)

    monkeypatch.setattr(distill, "ox_alpha_source_binding", source_binding)
    queue, entries = _committed_local_r4_entries(tmp_path)
    config = distill.DistillationConfig(teacher_max_inflight=10)

    assert distill._write_r4_local_receipts(
        root=tmp_path, config=config, workset=queue, entries=entries
    )
    assert source_calls == 1
    directory = (
        store.distillation_dir(tmp_path)
        / "r4-receipts"
        / "local"
        / source["source_commit"]
    )
    paths = sorted(directory.glob("*.json"))
    assert len(paths) == 2
    before = {path.name: path.read_bytes() for path in paths}
    receipt = store.read_sealed(paths[0], schema=distill.R4_RECEIPT_SCHEMA)
    assert receipt["artifact_id"] == receipt["receipt_id"]
    assert receipt["lane"] == {
        "mode": "sleep",
        "purpose": "sleep",
        "admitted": True,
        "inflight": 1,
    }
    assert receipt["live_recall"] == {"model_calls": 0, "remote_egress": 0}
    rendered = canonical_json.canonical_json_bytes_strict(receipt).decode()
    assert not any(
        marker in rendered.lower()
        for marker in ("prompt", "response", "rationale", "raw", "span", "path")
    )

    queue.advance(
        [
            {
                "work_id": "later",
                "kind": "local-teacher:test",
                "payload_ref": "candidate-ledger:later",
                "payload_digest": "d" * 64,
                "priority": 0,
                "temporal_split": {"split": "train"},
                "provenance": {"route": "test"},
            }
        ],
        {"source": 2},
    )
    assert distill._write_r4_local_receipts(
        root=tmp_path, config=config, workset=queue, entries=entries
    )
    assert {path.name: path.read_bytes() for path in paths} == before

    distill._ensure_r4_local_receipts(
        root=tmp_path, config=config, workset=queue, entries=entries
    )
    assert source_calls == 3


def test_local_r4_receipt_source_drift_removes_only_new_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, entries = _committed_local_r4_entries(tmp_path, count=1)
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        distill,
        "ox_alpha_source_binding",
        lambda: {**source, "source_tree_sha256": "d" * 64},
    )

    assert not distill._write_r4_local_receipts(
        root=tmp_path,
        config=distill.DistillationConfig(),
        workset=queue,
        entries=entries,
    )
    directory = (
        store.distillation_dir(tmp_path)
        / "r4-receipts"
        / "local"
        / source["source_commit"]
    )
    assert not list(directory.glob("*.json"))


def test_authentic_local_teacher_rechecks_source_before_label_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rally = {
        "rally_id": "rally-source-drift",
        "query_sha256": "query",
        "context_refs": [],
    }
    candidate = {"candidate_id": "candidate-source-drift", "text_sha256": "text"}
    assignment = distill.teacher_assignment(
        str(rally["rally_id"]), str(candidate["candidate_id"])
    )
    route = str(assignment["owner"])
    payload_digest = canonical_json.canonical_json_sha256_strict(
        {"rally_id": rally["rally_id"], "candidate_id": candidate["candidate_id"]}
    )
    work_id = distill._local_work_id(payload_digest, route)
    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "local-workset.sqlite3"
    )
    queue.advance(
        [
            {
                "work_id": work_id,
                "kind": f"local-teacher:{route}",
                "payload_ref": "candidate-snapshot:source-drift",
                "payload_digest": payload_digest,
                "priority": 0,
                "temporal_split": {"split": "train"},
                "provenance": {"route": route},
            }
        ],
        {"source": 1},
    )
    route_identity = {
        "role": route,
        "provider": "ollama",
        "model": "local-model",
        "location": "local",
    }
    model_digest = "d" * 64
    teacher = distill._WorkerTeacher(
        route, 4_096, route_identity, model_digest
    )
    teacher_calls = 0

    def evaluate(_payload: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal teacher_calls
        teacher_calls += 1
        return {
            "labels": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "verdict": "relevant",
                }
            ],
            "_route_identity": route_identity,
            "_model_digest": model_digest,
        }

    monkeypatch.setattr(teacher, "evaluate", evaluate)
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    source_calls = 0

    def source_binding() -> dict[str, str]:
        nonlocal source_calls
        source_calls += 1
        return (
            dict(source)
            if source_calls == 1
            else {**source, "source_tree_sha256": "e" * 64}
        )

    monkeypatch.setattr(distill, "ox_alpha_source_binding", source_binding)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"

    result = distill._run_local_teacher_route(
        workset=queue,
        route=route,
        config=distill.DistillationConfig(teacher_claim_limit=1),
        teachers={route: teacher},
        tasks={
            work_id: {
                "rally": rally,
                "candidate": candidate,
                "assignment": assignment,
                "route": route,
            }
        },
        texts={"query": "question", "text": "evidence"},
        root=tmp_path,
        raw_dir=tmp_path / "raw",
        label_path=label_path,
        snapshots={},
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result is not None and result.deferred is True
    assert teacher_calls == 1
    assert source_calls == 2
    assert not label_path.exists()
    status = queue.status(include_timing=True)
    assert status["ready"] == 1
    assert status["leased"] == status["completed"] == 0
    assert status["retry_wait"] == 1


@pytest.mark.parametrize(
    ("category", "operation"),
    (
        ("capacity", "commit"),
        ("timeout", "commit"),
        ("preemption", "commit"),
        ("schema", "commit"),
        ("coverage", "commit"),
        ("route_model_mismatch", "commit"),
    ),
)
def test_authentic_local_r4_owned_failure_receipts_are_provider_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category: str,
    operation: str,
) -> None:
    rally = {"rally_id": "r4-owned-failure", "query_sha256": "query", "context_refs": []}
    candidate = {"candidate_id": "r4-owned-candidate", "text_sha256": "text"}
    assignment = distill.teacher_assignment(
        str(rally["rally_id"]), str(candidate["candidate_id"])
    )
    route = str(assignment["owner"])
    payload_digest = canonical_json.canonical_json_sha256_strict(
        {"rally_id": rally["rally_id"], "candidate_id": candidate["candidate_id"]}
    )
    work_id = distill._local_work_id(payload_digest, route)
    clock = [100.0]
    distill.bootstrap_r4_distillation_root_authority(tmp_path)
    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "local-workset.sqlite3",
        clock=lambda: clock[0],
    )
    queue.advance(
        [
            {
                "work_id": work_id,
                "kind": f"local-teacher:{route}",
                "payload_ref": "candidate-snapshot:r4-owned-failure",
                "payload_digest": payload_digest,
                "priority": 0,
                "temporal_split": {"split": "train"},
                "provenance": {"route": route},
            }
        ],
        {"source": 1},
    )
    route_identity = {
        "role": route,
        "provider": "ollama",
        "model": "local-r4-owned-model",
        "location": "local",
    }
    teacher = distill._WorkerTeacher(route, 4_096, route_identity, "d" * 64)
    provider_calls = 0

    def evaluate(_payload: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("owned diagnostic must not call a provider")

    monkeypatch.setattr(teacher, "evaluate", evaluate)
    monkeypatch.setattr(
        distill,
        "_local_r4_owned_failure_injection",
        lambda **_kwargs: category,
    )
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"

    result = distill._run_local_teacher_route(
        workset=queue,
        route=route,
        config=distill.DistillationConfig(teacher_claim_limit=1),
        teachers={route: teacher},
        tasks={
            work_id: {
                "rally": rally,
                "candidate": candidate,
                "assignment": assignment,
                "route": route,
            }
        },
        texts={"query": "question", "text": "evidence"},
        root=tmp_path,
        raw_dir=tmp_path / "raw",
        label_path=label_path,
        snapshots={},
        label_rows=[],
        structural_verifier=lambda *_args: None,
    )

    assert result is not None and result.deferred is True and result.model_calls == 0
    assert provider_calls == 0
    assert not label_path.exists()
    assert queue.completion_identities([work_id]) == {}
    transition = queue.recent_transition_receipts(limit=1)[0]
    assert transition["operation"] == operation
    directory = (
        store.distillation_dir(tmp_path)
        / "r4-receipts"
        / "local"
        / ("a" * 40)
    )
    receipts = list(directory.glob("*.json"))
    assert len(receipts) == 1
    receipt = store.read_sealed(receipts[0], schema=distill.R4_RECEIPT_SCHEMA)
    assert receipt["failure_injection"] is True
    assert receipt["outcome"] == {
        "class": "deferred" if category in {"capacity", "timeout", "preemption"} else "invalid",
        "reason": category,
    }
    assert receipt["diagnostic"] == {"provider_calls": 0, "network_egress": 0}
    assert receipt["workset_receipt"] == {
        "generation": transition["generation"],
        "head_sha256": transition["receipt_sha256"],
        "operation": transition["operation"],
        "selection_sha256": transition["details"]["selection_sha256"],
        "work_ids_sha256": transition["details"]["work_ids_sha256"],
        "context_sha256": transition["details"]["context_sha256"],
    }
    assert "label_record_sha256" not in receipt
    assert isinstance(receipt["attempt_record_sha256"], str)
    if category == "capacity":
        before = receipts[0].read_bytes()
        claim_transition = queue.transition_receipt_binding(
            int(transition["generation"]) - 1
        )
        settle_transition = queue.transition_receipt_binding(
            int(transition["generation"])
        )
        assert claim_transition is not None and settle_transition is not None
        entry = distill._local_r4_failure_entry(
            work_id=work_id,
            attempt=1,
            task={
                "rally": rally,
                "candidate": candidate,
                "assignment": assignment,
                "route": route,
            },
            route_identity=route_identity,
            source={
                "source_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_ox_identity_sha256": "c" * 64,
            },
            category=category,
            owned_diagnostic=True,
        )
        assert entry is not None
        forged_timestamp = dict(receipt)
        forged_timestamp["captured_at"] = "2099-01-01T00:00:00Z"
        forged_timestamp["receipt_sha256"] = (
            canonical_json.canonical_json_sha256_strict(
                {
                    key: value
                    for key, value in forged_timestamp.items()
                    if key not in {"receipt_sha256", "seal_sha256"}
                }
            )
        )
        forged_timestamp["seal_sha256"] = (
            canonical_json.canonical_json_sha256_strict(
                {
                    key: value
                    for key, value in forged_timestamp.items()
                    if key != "seal_sha256"
                }
            )
        )
        receipts[0].write_bytes(
            canonical_json.canonical_json_bytes_strict(forged_timestamp) + b"\n"
        )
        assert not distill._write_r4_local_failure_receipts(
            root=tmp_path,
            config=distill.DistillationConfig(teacher_claim_limit=1),
            workset=queue,
            entries=[entry],
            label_head=store.chain_head(label_path),
            settle_transition=settle_transition,
            captured_at=receipt["captured_at"],
            claim_transition=claim_transition,
        )
        receipts[0].write_bytes(before)

        def forge_workset_binding(binding: Mapping[str, Any]) -> None:
            forged = dict(receipt)
            forged["workset_receipt"] = {
                "generation": binding["generation"],
                "head_sha256": binding["receipt_sha256"],
                "operation": binding["operation"],
                "selection_sha256": binding["selection_sha256"],
                "work_ids_sha256": binding["work_ids_sha256"],
            }
            forged["receipt_sha256"] = canonical_json.canonical_json_sha256_strict(
                {
                    key: value
                    for key, value in forged.items()
                    if key not in {"receipt_sha256", "seal_sha256"}
                }
            )
            forged["seal_sha256"] = canonical_json.canonical_json_sha256_strict(
                {key: value for key, value in forged.items() if key != "seal_sha256"}
            )
            receipts[0].write_bytes(
                canonical_json.canonical_json_bytes_strict(forged) + b"\n"
            )

        forge_workset_binding(claim_transition)
        assert not distill._write_r4_local_failure_receipts(
            root=tmp_path,
            config=distill.DistillationConfig(teacher_claim_limit=1),
            workset=queue,
            entries=[entry],
            label_head=store.chain_head(label_path),
            settle_transition=settle_transition,
            captured_at=receipt["captured_at"],
            claim_transition=claim_transition,
        )
        receipts[0].write_bytes(before)
        replacement = tmp_path / "r4-receipts-replacement"
        displaced = tmp_path / "r4-receipts-displaced"
        replacement.mkdir()
        (replacement / "sentinel").write_text("must-survive")
        directory.rename(displaced)
        replacement.rename(directory)
        assert not distill._write_r4_local_failure_receipts(
            root=tmp_path,
            config=distill.DistillationConfig(teacher_claim_limit=1),
            workset=queue,
            entries=[entry],
            label_head=store.chain_head(label_path),
            settle_transition=settle_transition,
            captured_at=receipt["captured_at"],
            claim_transition=claim_transition,
        )
        assert {path.name for path in directory.iterdir()} == {"sentinel"}
        directory.rename(replacement)
        displaced.rename(directory)
        clock[0] += 61
        later_claim = queue.claim(
            f"local-teacher:{route}", 1, "later-worker", 60
        )[0]
        assert later_claim.work_id == work_id and later_claim.attempt == 2
        queue.release_unattempted([later_claim])
        later_release = queue.transition_receipt_binding(
            int(queue.recent_transition_receipts(limit=1)[0]["generation"])
        )
        assert later_release is not None
        forge_workset_binding(later_release)
        assert not distill._write_r4_local_failure_receipts(
            root=tmp_path,
            config=distill.DistillationConfig(teacher_claim_limit=1),
            workset=queue,
            entries=[entry],
            label_head=store.chain_head(label_path),
            settle_transition=settle_transition,
            captured_at=receipt["captured_at"],
            claim_transition=claim_transition,
        )
        receipts[0].write_bytes(before)
        unrelated_digest = canonical_json.canonical_json_sha256_strict(
            {"rally_id": "r4-owned-unrelated", "candidate_id": "unrelated"}
        )
        unrelated_work_id = distill._local_work_id(unrelated_digest, route)
        queue.advance(
            [
                {
                    "work_id": unrelated_work_id,
                    "kind": f"local-teacher:{route}",
                    "payload_ref": "candidate-snapshot:r4-owned-unrelated",
                    "payload_digest": unrelated_digest,
                    "priority": 1,
                    "temporal_split": {"split": "train"},
                    "provenance": {"route": route},
                }
            ],
            {"source": 2},
        )
        unrelated_claim = queue.claim(
            f"local-teacher:{route}", 1, "unrelated-worker", 60
        )[0]
        assert unrelated_claim.work_id == unrelated_work_id
        queue.release_unattempted([unrelated_claim])
        unrelated_release = queue.transition_receipt_binding(
            int(queue.recent_transition_receipts(limit=1)[0]["generation"])
        )
        assert unrelated_release is not None
        forge_workset_binding(unrelated_release)
        assert not distill._write_r4_local_failure_receipts(
            root=tmp_path,
            config=distill.DistillationConfig(teacher_claim_limit=1),
            workset=queue,
            entries=[entry],
            label_head=store.chain_head(label_path),
            settle_transition=settle_transition,
            captured_at=receipt["captured_at"],
            claim_transition=claim_transition,
        )
        receipts[0].write_bytes(before)
        cleanup_claim = queue.claim(
            f"local-teacher:{route}", 1, "unrelated-worker", 60
        )[0]
        assert cleanup_claim.work_id == unrelated_work_id
        queue.commit(
            [cleanup_claim],
            [{"status": "quarantined", "error_class": "test_cleanup"}],
        )
        repeated = distill._run_local_teacher_route(
            workset=queue,
            route=route,
            config=distill.DistillationConfig(teacher_claim_limit=1),
            teachers={route: teacher},
            tasks={
                work_id: {
                    "rally": rally,
                    "candidate": candidate,
                    "assignment": assignment,
                    "route": route,
                }
            },
            texts={"query": "question", "text": "evidence"},
            root=tmp_path,
            raw_dir=tmp_path / "raw",
            label_path=label_path,
            snapshots={},
            label_rows=[],
            structural_verifier=lambda *_args: None,
        )
        assert repeated is not None and repeated.model_calls == 0
        # A later lease has a distinct sealed claim binding, so it must not
        # overwrite or be mistaken for the original provider-free failure.
        assert receipts[0].read_bytes() == before
        assert len(list(directory.glob("*.json"))) == 2


def test_local_r4_directory_authority_refuses_tofu_for_an_existing_directory(
    tmp_path: Path,
) -> None:
    source_commit = "a" * 40
    directory = (
        store.distillation_dir(tmp_path)
        / "r4-receipts"
        / "local"
        / source_commit
    )
    distill.bootstrap_r4_distillation_root_authority(tmp_path)
    directory.mkdir(parents=True)
    (directory / "sentinel").write_text("untrusted")

    with pytest.raises(distill.DistillationError, match="bootstrap is untrusted"):
        distill._r4_directory_authority(
            root=tmp_path,
            directory=directory,
            role="receipts",
            source_commit=source_commit,
            register=True,
        )

    assert {path.name for path in directory.iterdir()} == {"sentinel"}


def test_local_r4_directory_authority_rejects_a_distillation_root_replacement(
    tmp_path: Path,
) -> None:
    source_commit = "a" * 40
    distillation_root = store.distillation_dir(tmp_path)
    directory = distillation_root / "r4-receipts" / "local" / source_commit
    distill.bootstrap_r4_distillation_root_authority(tmp_path)
    distill._r4_directory_authority(
        root=tmp_path,
        directory=directory,
        role="receipts",
        source_commit=source_commit,
        register=True,
    )
    displaced = tmp_path / "displaced-distillation-root"
    replacement = tmp_path / "replacement-distillation-root"
    replacement.mkdir()
    (replacement / "sentinel").write_text("must-survive")
    distillation_root.rename(displaced)
    replacement.rename(distillation_root)

    with pytest.raises(distill.DistillationError, match="root authority changed"):
        distill._r4_directory_authority(
            root=tmp_path,
            directory=directory,
            role="receipts",
            source_commit=source_commit,
            register=True,
        )

    assert {path.name for path in distillation_root.iterdir()} == {"sentinel"}


def test_local_r4_root_authority_never_tofus_a_replaced_root_after_deletion(
    tmp_path: Path,
) -> None:
    original = distill.bootstrap_r4_distillation_root_authority(tmp_path)
    assert distill._r4_distillation_root_authority(tmp_path, register=False) == original
    authority_id = canonical_json.canonical_json_sha256_strict(
        {"kind": "local-r4-distillation-root-authority"}
    )
    (tmp_path / f"{authority_id}.json").unlink()
    distillation_root = store.distillation_dir(tmp_path)
    displaced = tmp_path / "displaced-distillation-root"
    replacement = tmp_path / "replacement-distillation-root"
    replacement.mkdir()
    (replacement / "sentinel").write_text("must-survive")
    distillation_root.rename(displaced)
    replacement.rename(distillation_root)

    with pytest.raises(distill.DistillationError, match="bootstrap is untrusted"):
        distill.bootstrap_r4_distillation_root_authority(tmp_path)
    with pytest.raises(distill.DistillationError, match="root authority is missing"):
        distill._r4_distillation_root_authority(tmp_path, register=False)

    assert {path.name for path in distillation_root.iterdir()} == {"sentinel"}


def _legacy_r4_root_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    from chronovisor.core.durable_state import okf_writer_lock

    root = tmp_path / "legacy-root"
    with okf_writer_lock(root):
        pass
    distillation = store.distillation_dir(root)
    distillation.mkdir()
    workset_path = distillation / "ox-workset.sqlite3"
    with sqlite3.connect(workset_path) as connection:
        connection.execute(
            "CREATE TABLE work_items (state TEXT, lease_id TEXT, lease_owner TEXT, lease_expires_at REAL)"
        )
    workset_path.with_name(f"{workset_path.name}-wal").touch()
    (distillation / "candidate-ledger.jsonl").write_text("candidate\n")
    (distillation / "candidate-ledger.jsonl.head.json").write_text("checkpoint\n")
    (distillation / "distillation-worker.lock").touch()
    (root / "config.toml").write_text("[recall_distillation]\nenabled = true\n")
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    return root, tmp_path / "offline-bootstrap.json", source


def _legacy_r4_file_state(path: Path) -> dict[str, int | str] | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    observed = path.stat()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": observed.st_size,
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "st_mtime_ns": observed.st_mtime_ns,
        "st_ctime_ns": observed.st_ctime_ns,
    }


def _write_legacy_r4_evidence(
    root: Path, path: Path, source: Mapping[str, str], *, captured_at: int = 1
) -> None:
    distillation = store.distillation_dir(root)
    after = {
        "candidate": _legacy_r4_file_state(distillation / "candidate-ledger.jsonl"),
        "candidate_anchor": _legacy_r4_file_state(
            distillation / distill.R4_CANDIDATE_ANCHOR_FILE
        ),
        "candidate_checkpoint": _legacy_r4_file_state(
            distillation / "candidate-ledger.jsonl.head.json"
        ),
        "config": _legacy_r4_file_state(root / "config.toml"),
        "distillation_lock": _legacy_r4_file_state(
            distillation / "distillation-worker.lock"
        ),
        "workset": _legacy_r4_file_state(distillation / "ox-workset.sqlite3"),
        "workset_journal": _legacy_r4_file_state(
            distillation / "ox-workset.sqlite3-journal"
        ),
        "workset_shm": _legacy_r4_file_state(distillation / "ox-workset.sqlite3-shm"),
        "workset_wal": _legacy_r4_file_state(distillation / "ox-workset.sqlite3-wal"),
    }
    unsigned = {
        "captured_at_unix": captured_at,
        "kind": "r4-offline-bootstrap-receipt",
        "namespace": "recall-distillation",
        "production": {
            "root": str(root),
            "unchanged": True,
            "before": after,
            "after": after,
        },
        "schema": distill.R4_OFFLINE_BOOTSTRAP_SCHEMA,
        "scope": {
            "provider_calls": 0,
            "ox_enabled": False,
            "owned_clone_only": True,
            "production_certification": False,
            "r4_checkbox_complete": False,
        },
        "source": {"binding": dict(source), "commit": source["source_commit"]},
        "verdict": "passed",
    }
    artifact_id = canonical_json.canonical_json_sha256_strict(unsigned)
    payload = {"artifact_id": artifact_id, **unsigned}
    payload["seal_sha256"] = canonical_json.canonical_json_sha256_strict(payload)
    path.write_bytes(canonical_json.canonical_json_bytes_strict(payload) + b"\n")


def _migrate_legacy_r4_root(root: Path, evidence: Path, source: Mapping[str, str]) -> tuple[int, int]:
    return distill.migrate_r4_legacy_distillation_root_authority(
        root=root,
        offline_bootstrap_evidence=evidence,
        expected_source_binding=source,
    )


def test_legacy_r4_root_migration_is_explicit_and_verifiable(tmp_path: Path) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)

    with pytest.raises(distill.DistillationError, match="bootstrap is untrusted"):
        distill.bootstrap_r4_distillation_root_authority(root)
    observed = _migrate_legacy_r4_root(root, evidence, source)

    assert distill._r4_distillation_root_authority(root, register=False) == observed
    assert _migrate_legacy_r4_root(root, evidence, source) == observed


def test_legacy_r4_root_migration_accepts_only_shm_time_drift(tmp_path: Path) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    shm = store.distillation_dir(root) / "ox-workset.sqlite3-shm"
    shm.write_bytes(b"same")
    _write_legacy_r4_evidence(root, evidence, source)
    now = time.time_ns() + 1_000_000
    os.utime(shm, ns=(now, now))
    assert _migrate_legacy_r4_root(root, evidence, source)

    other_root, other_evidence, other_source = _legacy_r4_root_fixture(tmp_path / "other")
    other_shm = store.distillation_dir(other_root) / "ox-workset.sqlite3-shm"
    other_shm.write_bytes(b"same")
    _write_legacy_r4_evidence(other_root, other_evidence, other_source)
    other_shm.write_bytes(b"different")
    with pytest.raises(distill.DistillationError, match="content drift"):
        _migrate_legacy_r4_root(other_root, other_evidence, other_source)


@pytest.mark.parametrize("fault", ["unsealed", "source", "root", "lease"])
def test_legacy_r4_root_migration_rejects_untrusted_preflight(
    tmp_path: Path, fault: str
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    if fault == "unsealed":
        payload = json.loads(evidence.read_bytes())
        payload["seal_sha256"] = "0" * 64
        evidence.write_text(json.dumps(payload))
    elif fault == "source":
        source = {**source, "source_tree_sha256": "d" * 64}
    elif fault == "root":
        payload = json.loads(evidence.read_bytes())
        payload["production"]["root"] = str(tmp_path / "wrong")
        payload["seal_sha256"] = canonical_json.canonical_json_sha256_strict(
            {key: value for key, value in payload.items() if key != "seal_sha256"}
        )
        evidence.write_bytes(canonical_json.canonical_json_bytes_strict(payload))
    else:
        with sqlite3.connect(store.distillation_dir(root) / "ox-workset.sqlite3") as connection:
            connection.execute(
                "INSERT INTO work_items VALUES ('leased', 'lease', 'owner', 1.0)"
            )
        _write_legacy_r4_evidence(root, evidence, source)

    with pytest.raises(distill.DistillationError):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_migration_rechecks_before_immutable_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    original = distill._r4_legacy_offline_bootstrap
    calls = 0

    def drift(*args: object, **kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            (store.distillation_dir(root) / "candidate-ledger.jsonl").write_text("drift\n")
        return result

    monkeypatch.setattr(distill, "_r4_legacy_offline_bootstrap", drift)
    with pytest.raises(distill.DistillationError):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_migration_rejects_evidence_replacement_after_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    original = distill._r4_legacy_offline_bootstrap
    calls = 0

    def replace(*args: object, **kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            _write_legacy_r4_evidence(root, evidence, source, captured_at=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(distill, "_r4_legacy_offline_bootstrap", replace)
    with pytest.raises(distill.DistillationError, match="evidence changed"):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_migration_rejects_directory_replacement_after_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    original = distill._r4_legacy_offline_bootstrap
    calls = 0

    def replace(*args: object, **kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            directory = store.distillation_dir(root)
            directory.rename(root / "displaced-distillation")
            directory.mkdir()
        return result

    monkeypatch.setattr(distill, "_r4_legacy_offline_bootstrap", replace)
    with pytest.raises(distill.DistillationError, match="content drift"):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_migration_cleans_authority_after_write_boundary_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    original = store.write_immutable_pinned
    displaced = root.with_name("displaced-root")

    def replace(
        directory: Path, payload: Mapping[str, Any], **kwargs: Any
    ) -> tuple[str, Path, dict[str, Any]]:
        before_persist = kwargs.pop("before_persist")

        def race() -> None:
            before_persist()
            root.rename(displaced)
            root.mkdir()

        return original(directory, payload, before_persist=race, **kwargs)

    monkeypatch.setattr(distill.store, "write_immutable_pinned", replace)
    with pytest.raises(distill.DistillationError):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))
    assert not list(displaced.glob("*.json"))


def test_legacy_r4_root_migration_rechecks_config_after_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    original = store.write_immutable_pinned

    def replace(
        directory: Path, payload: Mapping[str, Any], **kwargs: Any
    ) -> tuple[str, Path, dict[str, Any]]:
        before_persist = kwargs.pop("before_persist")

        def race() -> None:
            before_persist()
            (root / "config.toml").write_text("[recall.distillation]\nox_enabled = true\n")

        return original(directory, payload, before_persist=race, **kwargs)

    monkeypatch.setattr(distill.store, "write_immutable_pinned", replace)
    with pytest.raises(distill.DistillationError):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


@pytest.mark.parametrize("config", ["[recall.distillation]\nox_enabled = true\n", "not toml = ["])
def test_legacy_r4_root_migration_rechecks_actual_ox_config(
    tmp_path: Path, config: str
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    (root / "config.toml").write_text(config)
    _write_legacy_r4_evidence(root, evidence, source)

    with pytest.raises(distill.DistillationError):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_migration_bounds_offline_evidence_read(tmp_path: Path) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    evidence.write_bytes(b"[" + b" " * distill.R4_OFFLINE_BOOTSTRAP_MAX_BYTES)

    with pytest.raises(distill.DistillationError, match="evidence is invalid"):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_migration_rejects_busy_worker_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    monkeypatch.setattr(distill.store, "acquire_nonblocking_lock", lambda _path: None)

    with pytest.raises(distill.DistillationError, match="is busy"):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_migration_rejects_different_evidence_after_adoption(
    tmp_path: Path,
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    _migrate_legacy_r4_root(root, evidence, source)
    _write_legacy_r4_evidence(root, evidence, source, captured_at=2)

    with pytest.raises(distill.DistillationError, match="is invalid"):
        _migrate_legacy_r4_root(root, evidence, source)


def test_legacy_r4_root_migration_remeasures_current_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    monkeypatch.setattr(
        distill,
        "ox_alpha_source_binding",
        lambda: {**source, "source_tree_sha256": "d" * 64},
    )

    with pytest.raises(distill.DistillationError, match="evidence is invalid"):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_migration_normalizes_source_probe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    monkeypatch.setattr(
        distill,
        "ox_alpha_source_binding",
        lambda: (_ for _ in ()).throw(ValueError("unavailable")),
    )

    with pytest.raises(distill.DistillationError, match="source is invalid"):
        _migrate_legacy_r4_root(root, evidence, source)
    assert not list(root.glob("*.json"))


def test_legacy_r4_root_verifier_rejects_two_authorities(tmp_path: Path) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    _migrate_legacy_r4_root(root, evidence, source)
    normal_id = canonical_json.canonical_json_sha256_strict(
        {"kind": "local-r4-distillation-root-authority"}
    )
    identity = store.pinned_directory_identity(root, create=False)
    directory = store.pinned_directory_identity(store.distillation_dir(root), create=False)
    store.write_immutable_pinned(
        root,
        {
            "kind": "local-r4-distillation-root-authority",
            "directory_identity": {"device": directory[0], "inode": directory[1]},
        },
        schema=distill.R4_DIRECTORY_AUTHORITY_SCHEMA,
        artifact_id=normal_id,
        expected_directory_identity=identity,
    )

    with pytest.raises(distill.DistillationError, match="conflicts"):
        distill._r4_distillation_root_authority(root, register=False)


def test_normal_r4_root_bootstrap_rejects_existing_legacy_authority(
    tmp_path: Path,
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    _migrate_legacy_r4_root(root, evidence, source)
    previous = store.distillation_dir(root)
    previous.rename(root / "displaced-distillation")

    with pytest.raises(distill.DistillationError, match="legacy migration exists"):
        distill.bootstrap_r4_distillation_root_authority(root)
    assert not store.distillation_dir(root).exists()


def test_normal_and_legacy_r4_root_bootstrap_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    root, evidence, source = _legacy_r4_root_fixture(tmp_path)
    _write_legacy_r4_evidence(root, evidence, source)
    barrier = threading.Barrier(2)
    results: list[object] = []

    def migrate() -> None:
        barrier.wait()
        try:
            results.append(_migrate_legacy_r4_root(root, evidence, source))
        except distill.DistillationError as exc:
            results.append(exc)

    def bootstrap() -> None:
        barrier.wait()
        try:
            results.append(distill.bootstrap_r4_distillation_root_authority(root))
        except distill.DistillationError as exc:
            results.append(exc)

    first = threading.Thread(target=migrate)
    second = threading.Thread(target=bootstrap)
    first.start()
    second.start()
    first.join()
    second.join()

    assert sum(isinstance(result, tuple) for result in results) == 1
    assert sum(isinstance(result, distill.DistillationError) for result in results) == 1
    legacy_id = distill._r4_legacy_root_authority_id()
    normal_id = canonical_json.canonical_json_sha256_strict(
        {"kind": "local-r4-distillation-root-authority"}
    )
    names = {path.stem for path in root.glob("*.json")}
    assert names in ({legacy_id}, {normal_id})
    assert distill._r4_distillation_root_authority(root, register=False)


def test_r4_candidate_anchor_rejects_self_sealed_fixed_id_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    evidence = tmp_path / "r0.json"
    unsigned = {
        "schema": "chronovisor.recall-r0.v1",
        "namespace": "recall-distillation",
        "production": {
            "ledgers": {"candidate-ledger.jsonl": {"records": 0, "head_sha256": "x"}}
        },
    }
    forged = {"artifact_id": distill.R4_R0_EVIDENCE_ID, **unsigned}
    forged["seal_sha256"] = canonical_json.canonical_json_sha256_strict(forged)
    evidence.write_bytes(canonical_json.canonical_json_bytes_strict(forged))

    with pytest.raises(distill.DistillationError, match="preflight failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def test_r4_candidate_anchor_rejects_r0_evidence_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    target = tmp_path / "r0-target.json"
    target.write_text("{}")
    evidence = tmp_path / "r0-link.json"
    evidence.symlink_to(target)

    with pytest.raises(distill.DistillationError, match="preflight failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def test_r4_candidate_anchor_rejects_symlinked_distillation_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "distillation-target"
    target.mkdir()
    directory = store.distillation_dir(root)
    directory.parent.mkdir()
    directory.symlink_to(target)
    evidence = tmp_path / "r0.json"
    evidence.write_text("{}")

    with pytest.raises(distill.DistillationError, match="preflight failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not list(target.iterdir())


def test_r4_candidate_anchor_normalizes_recursive_json(tmp_path: Path) -> None:
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    evidence = tmp_path / "r0.json"
    evidence.write_bytes(b"[" * 100_000 + b"0" + b"]" * 100_000)

    with pytest.raises(distill.DistillationError, match="preflight failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


@pytest.mark.parametrize(
    "name", ["candidate-ledger.jsonl", "candidate-ledger.jsonl.head.json"]
)
def test_r4_candidate_anchor_rejects_symlinked_candidate_inputs(
    tmp_path: Path, name: str
) -> None:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "_handoff/evidence/2026-08-23-recall-distillation-recovery/r0-measured-baseline-4de2cfe3.json"
    )
    r0 = json.loads(evidence.read_bytes())
    ledger = r0["production"]["ledgers"]["candidate-ledger.jsonl"]
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    candidate = directory / "candidate-ledger.jsonl"
    with candidate.open("wb") as handle:
        handle.truncate(ledger["bytes"])
    store.write_sealed_state(
        directory / "candidate-ledger.jsonl.head.json",
        {
            "head_sha256": ledger["head_sha256"],
            "records": ledger["records"],
            "file_state": ledger["file_state"],
        },
    )
    target = directory / f"{name}.target"
    (directory / name).replace(target)
    (directory / name).symlink_to(target)

    with pytest.raises(distill.DistillationError, match="preflight failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


@pytest.mark.parametrize("input_name", ["candidate", "checkpoint"])
def test_r4_candidate_anchor_rejects_input_drift_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_name: str
) -> None:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "_handoff/evidence/2026-08-23-recall-distillation-recovery/r0-measured-baseline-4de2cfe3.json"
    )
    r0 = json.loads(evidence.read_bytes())
    ledger = r0["production"]["ledgers"]["candidate-ledger.jsonl"]
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    candidate = directory / "candidate-ledger.jsonl"
    with candidate.open("wb") as handle:
        handle.truncate(ledger["bytes"])
    checkpoint = directory / "candidate-ledger.jsonl.head.json"
    store.write_sealed_state(
        checkpoint,
        {
            "head_sha256": ledger["head_sha256"],
            "records": ledger["records"],
            "file_state": ledger["file_state"],
        },
    )

    def drift() -> dict[str, str]:
        if input_name == "candidate":
            with candidate.open("r+b") as handle:
                handle.write(b"x")
        else:
            store.write_sealed_state(
                checkpoint,
                {
                    "head_sha256": ledger["head_sha256"],
                    "records": ledger["records"] + 1,
                    "file_state": ledger["file_state"],
                },
            )
        return {"x": "y"}

    monkeypatch.setattr(distill, "_r4_critical_module_sha256", drift)
    with pytest.raises(distill.DistillationError, match="preflight failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def test_r4_candidate_anchor_rejects_busy_candidate_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "_handoff/evidence/2026-08-23-recall-distillation-recovery/r0-measured-baseline-4de2cfe3.json"
    )
    r0 = json.loads(evidence.read_bytes())
    ledger = r0["production"]["ledgers"]["candidate-ledger.jsonl"]
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    candidate = directory / "candidate-ledger.jsonl"
    with candidate.open("wb") as handle:
        handle.truncate(ledger["bytes"])
    store.write_sealed_state(
        directory / "candidate-ledger.jsonl.head.json",
        {
            "head_sha256": ledger["head_sha256"],
            "records": ledger["records"],
            "file_state": ledger["file_state"],
        },
    )
    monkeypatch.setattr(distill, "_r4_critical_module_sha256", lambda: {"x": "y"})
    candidate_lock = store.acquire_nonblocking_lock(
        candidate.with_suffix(candidate.suffix + ".lock")
    )
    assert candidate_lock is not None
    try:
        with pytest.raises(distill.DistillationError, match="bootstrap is busy"):
            distill.bootstrap_r4_candidate_anchor(
                root=root,
                tracked_r0_evidence=evidence,
                source_binding={"source_commit": "a" * 40},
            )
    finally:
        store.release_lock(candidate_lock)
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def test_r4_candidate_anchor_accepts_the_canonical_fixed_r0_evidence() -> None:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "_handoff/evidence/2026-08-23-recall-distillation-recovery/r0-measured-baseline-4de2cfe3.json"
    )
    payload = json.loads(evidence.read_bytes())
    store.verify_seal(payload, schema="chronovisor.recall-r0.v1")

    distill._r4_require_canonical_artifact_id(
        payload, expected_artifact_id=distill.R4_R0_EVIDENCE_ID
    )


def test_r4_candidate_anchor_cleans_failed_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "_handoff/evidence/2026-08-23-recall-distillation-recovery/r0-measured-baseline-4de2cfe3.json"
    )
    r0 = json.loads(evidence.read_bytes())
    ledger = r0["production"]["ledgers"]["candidate-ledger.jsonl"]
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    candidate = directory / "candidate-ledger.jsonl"
    with candidate.open("wb") as handle:
        handle.truncate(ledger["bytes"])
    store.write_sealed_state(
        directory / "candidate-ledger.jsonl.head.json",
        {
            "head_sha256": ledger["head_sha256"],
            "records": ledger["records"],
            "file_state": ledger["file_state"],
        },
    )
    monkeypatch.setattr(distill, "_r4_critical_module_sha256", lambda: {"x": "y"})

    original_write = distill.atomic_write_bytes_at

    def forged_write(directory_fd: int, name: str, raw: bytes) -> None:
        payload = json.loads(raw)
        payload["artifact_id"] = "0" * 64
        payload["seal_sha256"] = canonical_json.canonical_json_sha256_strict(
            {key: value for key, value in payload.items() if key != "seal_sha256"}
        )
        original_write(
            directory_fd,
            name,
            canonical_json.canonical_json_bytes_strict(payload) + b"\n",
        )

    monkeypatch.setattr(distill, "atomic_write_bytes_at", forged_write)

    with pytest.raises(distill.DistillationError, match="read-back failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def test_r4_candidate_anchor_cleans_pinned_directory_after_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "_handoff/evidence/2026-08-23-recall-distillation-recovery/r0-measured-baseline-4de2cfe3.json"
    )
    r0 = json.loads(evidence.read_bytes())
    ledger = r0["production"]["ledgers"]["candidate-ledger.jsonl"]
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    candidate = directory / "candidate-ledger.jsonl"
    with candidate.open("wb") as handle:
        handle.truncate(ledger["bytes"])
    store.write_sealed_state(
        directory / "candidate-ledger.jsonl.head.json",
        {
            "head_sha256": ledger["head_sha256"],
            "records": ledger["records"],
            "file_state": ledger["file_state"],
        },
    )
    monkeypatch.setattr(distill, "_r4_critical_module_sha256", lambda: {"x": "y"})
    original_write = distill.atomic_write_bytes_at
    displaced = root / "displaced-distillation"

    def replace(directory_fd: int, name: str, raw: bytes) -> None:
        original_write(directory_fd, name, raw)
        directory.rename(displaced)
        directory.mkdir()

    monkeypatch.setattr(distill, "atomic_write_bytes_at", replace)
    with pytest.raises(distill.DistillationError, match="read-back failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()
    assert not (displaced / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def _r4_candidate_anchor_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Mapping[str, Any]]:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "_handoff/evidence/2026-08-23-recall-distillation-recovery/r0-measured-baseline-4de2cfe3.json"
    )
    r0 = json.loads(evidence.read_bytes())
    ledger = r0["production"]["ledgers"]["candidate-ledger.jsonl"]
    root = tmp_path / "root"
    directory = store.distillation_dir(root)
    directory.mkdir(parents=True)
    candidate = directory / "candidate-ledger.jsonl"
    with candidate.open("wb") as handle:
        handle.truncate(ledger["bytes"])
    checkpoint = directory / "candidate-ledger.jsonl.head.json"
    store.write_sealed_state(
        checkpoint,
        {
            "head_sha256": ledger["head_sha256"],
            "records": ledger["records"],
            "file_state": ledger["file_state"],
        },
    )
    return root, directory, evidence, candidate, checkpoint, ledger


def test_r4_candidate_anchor_survives_apfs_device_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _directory, evidence, _candidate, checkpoint, ledger = (
        _r4_candidate_anchor_fixture(tmp_path)
    )
    rebound = {**ledger["file_state"], "st_dev": ledger["file_state"]["st_dev"] + 1}
    store.write_sealed_state(
        checkpoint,
        {
            "head_sha256": ledger["head_sha256"],
            "records": ledger["records"],
            "file_state": rebound,
        },
    )
    monkeypatch.setattr(distill, "_r4_critical_module_sha256", lambda: {"x": "y"})

    anchor = distill.bootstrap_r4_candidate_anchor(
        root=root,
        tracked_r0_evidence=evidence,
        source_binding={"source_commit": "a" * 40},
    )

    assert anchor["candidate_checkpoint"]["file_state"] == {
        key: value for key, value in rebound.items() if key != "st_dev"
    }


def test_r4_candidate_anchor_rejects_directory_replacement_before_input_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory, evidence, candidate, checkpoint, _ledger = (
        _r4_candidate_anchor_fixture(tmp_path)
    )
    displaced = root / "displaced-distillation"
    original_loads = distill.json.loads
    calls = 0

    def replace(raw: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        result = original_loads(raw, *args, **kwargs)
        calls += 1
        if calls == 1:
            directory.rename(displaced)
            directory.mkdir()
            os.link(displaced / candidate.name, candidate)
            os.link(displaced / checkpoint.name, checkpoint)
        return result

    monkeypatch.setattr(distill.json, "loads", replace)
    monkeypatch.setattr(distill, "_r4_critical_module_sha256", lambda: {"x": "y"})
    with pytest.raises(distill.DistillationError, match="preflight failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()
    assert not (displaced / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def test_r4_candidate_anchor_cleans_recursive_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory, evidence, _candidate, _checkpoint, _ledger = (
        _r4_candidate_anchor_fixture(tmp_path)
    )
    monkeypatch.setattr(distill, "_r4_critical_module_sha256", lambda: {"x": "y"})
    original_write = distill.atomic_write_bytes_at

    def recursive_write(directory_fd: int, name: str, _raw: bytes) -> None:
        original_write(
            directory_fd,
            name,
            b"[" * 100_000 + b"0" + b"]" * 100_000,
        )

    monkeypatch.setattr(distill, "atomic_write_bytes_at", recursive_write)
    with pytest.raises(distill.DistillationError, match="read-back failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def test_r4_candidate_anchor_cleans_critical_module_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory, evidence, _candidate, _checkpoint, _ledger = (
        _r4_candidate_anchor_fixture(tmp_path)
    )
    calls = 0

    def critical_modules() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"x": "y" if calls == 1 else "z"}

    monkeypatch.setattr(distill, "_r4_critical_module_sha256", critical_modules)
    with pytest.raises(distill.DistillationError, match="read-back failed"):
        distill.bootstrap_r4_candidate_anchor(
            root=root,
            tracked_r0_evidence=evidence,
            source_binding={"source_commit": "a" * 40},
        )
    assert not (directory / distill.R4_CANDIDATE_ANCHOR_FILE).exists()


def test_authentic_local_r4_owned_failure_source_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    source_calls = 0

    def source_binding() -> dict[str, str]:
        nonlocal source_calls
        source_calls += 1
        return dict(source) if source_calls < 3 else {**source, "source_tree_sha256": "d" * 64}

    monkeypatch.setattr(distill, "ox_alpha_source_binding", source_binding)
    monkeypatch.setattr(
        distill, "_local_r4_owned_failure_injection", lambda **_kwargs: "capacity"
    )
    rally = {"rally_id": "r4-drift", "query_sha256": "query", "context_refs": []}
    candidate = {"candidate_id": "r4-drift-candidate", "text_sha256": "text"}
    assignment = distill.teacher_assignment("r4-drift", "r4-drift-candidate")
    route = str(assignment["owner"])
    digest = canonical_json.canonical_json_sha256_strict({"rally": "r4-drift"})
    work_id = distill._local_work_id(digest, route)
    distill.bootstrap_r4_distillation_root_authority(tmp_path)
    queue = workset.DistillationWorkset(store.distillation_dir(tmp_path) / "local-workset.sqlite3")
    queue.advance(
        [{"work_id": work_id, "kind": f"local-teacher:{route}", "payload_ref": "candidate-snapshot:r4-drift", "payload_digest": digest, "priority": 0, "temporal_split": {"split": "train"}, "provenance": {"route": route}}],
        {"source": 1},
    )
    identity = {"role": route, "provider": "ollama", "model": "local-r4-drift", "location": "local"}
    teacher = distill._WorkerTeacher(route, 4_096, identity, "d" * 64)
    monkeypatch.setattr(teacher, "evaluate", lambda _payload: pytest.fail("provider called"))

    with pytest.raises(distill.DistillationError, match="failure receipt"):
        distill._run_local_teacher_route(
            workset=queue, route=route, config=distill.DistillationConfig(teacher_claim_limit=1),
            teachers={route: teacher}, tasks={work_id: {"rally": rally, "candidate": candidate, "assignment": assignment, "route": route}},
            texts={"query": "question", "text": "evidence"}, root=tmp_path, raw_dir=tmp_path / "raw",
            label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl", snapshots={}, label_rows=[], structural_verifier=lambda *_args: None,
        )
    assert queue.status(include_timing=True)["ready"] == 1
    assert not list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))


@pytest.mark.parametrize("category", ("capacity", "preemption"))
def test_local_r4_attempted_capacity_and_preemption_commit(
    tmp_path: Path, category: str
) -> None:
    queue = workset.DistillationWorkset(store.distillation_dir(tmp_path) / "workset.sqlite3")
    queue.advance(
        [{"work_id": "local-teacher-" + "a" * 64, "kind": "local-teacher:test", "payload_ref": "candidate-snapshot:attempted", "payload_digest": "b" * 64, "priority": 0, "temporal_split": {"split": "train"}, "provenance": {"route": "test"}}],
        {"source": 1},
    )
    claim = queue.claim("local-teacher:test", 1, "test", 60)[0]
    result = distill._settle_local_r4_failure(
        root=tmp_path,
        config=distill.DistillationConfig(),
        workset=queue,
        batch=[{"claim": claim}],
        category=category,
        source=None,
        route_identity={},
        label_head={"records": 0, "head_sha256": ""},
        owned_diagnostic=False,
        model_calls=1,
        attempted=True,
    )
    assert result.model_calls == 1
    assert queue.status(include_timing=True)["retry_wait"] == 1


def test_local_r4_owned_failure_crash_repairs_before_next_route_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rally = {"rally_id": "r4-crash", "query_sha256": "query", "context_refs": []}
    candidate = {"candidate_id": "r4-crash-candidate", "text_sha256": "text"}
    assignment = distill.teacher_assignment("r4-crash", "r4-crash-candidate")
    route = str(assignment["owner"])
    digest = canonical_json.canonical_json_sha256_strict({"rally": "r4-crash"})
    work_id = distill._local_work_id(digest, route)
    distill.bootstrap_r4_distillation_root_authority(tmp_path)
    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "local-workset.sqlite3"
    )
    queue.advance(
        [{"work_id": work_id, "kind": f"local-teacher:{route}", "payload_ref": "candidate-snapshot:r4-crash", "payload_digest": digest, "priority": 0, "temporal_split": {"split": "train"}, "provenance": {"route": route}}],
        {"source": 1},
    )
    identity = {"role": route, "provider": "ollama", "model": "local-r4-crash", "location": "local"}
    teacher = distill._WorkerTeacher(route, 4_096, identity, "d" * 64)
    monkeypatch.setattr(teacher, "evaluate", lambda _payload: pytest.fail("provider called"))
    monkeypatch.setattr(distill, "_local_r4_owned_failure_injection", lambda **_kwargs: "timeout")
    monkeypatch.setattr(distill, "_local_r4_owned_failure_crash_after_settle", lambda _marker: True)
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"

    with pytest.raises(distill.DistillationError, match="injected crash"):
        distill._run_local_teacher_route(
            workset=queue, route=route, config=distill.DistillationConfig(teacher_claim_limit=1),
            teachers={route: teacher}, tasks={work_id: {"rally": rally, "candidate": candidate, "assignment": assignment, "route": route}},
            texts={"query": "question", "text": "evidence"}, root=tmp_path, raw_dir=tmp_path / "raw",
            label_path=label_path, snapshots={}, label_rows=[], structural_verifier=lambda *_args: None,
        )
    assert queue.status(include_timing=True)["retry_wait"] == 1
    assert not list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))
    assert list((store.distillation_dir(tmp_path) / "r4-failure-pending").rglob("*.json"))

    monkeypatch.setattr(distill, "_local_r4_owned_failure_injection", lambda **_kwargs: None)
    monkeypatch.setattr(distill, "_local_r4_owned_failure_crash_after_settle", lambda _marker: False)
    monkeypatch.setattr(distill, "_prepare_local_teacher_work", lambda **_kwargs: ({}, []))
    monkeypatch.setattr(distill, "_advance_local_workset", lambda *_args, **_kwargs: None)
    result = distill._run_local_teacher_batch(
        root=tmp_path, config=distill.DistillationConfig(teacher_claim_limit=1), teachers={},
        snapshots={}, rally_by_id={}, texts={}, label_path=label_path, label_rows=[], structural_verifier=lambda *_args: None,
    )
    assert result.labels_written == result.model_calls == 0
    receipts = list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))
    assert len(receipts) == 1
    receipt = store.read_sealed(receipts[0], schema=distill.R4_RECEIPT_SCHEMA)
    assert receipt["outcome"] == {"class": "deferred", "reason": "timeout"}
    assert receipt["diagnostic"] == {"provider_calls": 0, "network_egress": 0}
    assert queue.completion_identities([work_id]) == {}
    assert not label_path.exists()


def test_local_r4_repair_handles_aggregate_reclaim_without_cross_deleting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1_000.0]
    route = distill.TEACHER_ROLES[0]
    distill.bootstrap_r4_distillation_root_authority(tmp_path)
    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "local-workset.sqlite3",
        clock=lambda: now[0],
    )
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    route_identity = {
        "role": route,
        "provider": "ollama",
        "model": "local-r4-reclaim",
        "location": "local",
    }
    tasks: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    for index in range(100):
        rally_id = f"r4-reclaim-rally-{index}"
        candidate_id = f"r4-reclaim-candidate-{index}"
        assignment = distill.teacher_assignment(rally_id, candidate_id)
        if assignment["owner"] != route:
            continue
        digest = canonical_json.canonical_json_sha256_strict({"index": index})
        work_id = distill._local_work_id(digest, route)
        tasks[work_id] = {
            "rally": {"rally_id": rally_id},
            "candidate": {"candidate_id": candidate_id},
            "assignment": assignment,
            "route": route,
        }
        items.append(
            {
                "work_id": work_id,
                "kind": f"local-teacher:{route}",
                "payload_ref": f"candidate-snapshot:r4-reclaim-{index}",
                "payload_digest": digest,
                "priority": 0,
                "temporal_split": {"split": "train"},
                "provenance": {"route": route},
            }
        )
        if len(items) == 2:
            break
    assert len(items) == 2
    queue.advance(items, {"source": 1})
    label_head = {"records": 0, "head_sha256": ""}
    markers: list[Mapping[str, Any]] = []
    claims: list[Any] = []
    for _index in range(2):
        claim = queue.claim(f"local-teacher:{route}", 1, route, 1)[0]
        claims.append(claim)
        transition = queue.transition_receipt_binding(
            queue.recent_transition_receipts(limit=1)[0]["generation"]
        )
        assert transition is not None
        markers.append(
            distill._write_local_r4_pending_failure_marker(
                root=tmp_path,
                config=distill.DistillationConfig(teacher_max_inflight=1),
                item={"claim": claim, "task": tasks[claim.work_id]},
                category="timeout",
                source=source,
                route_identity=route_identity,
                label_head=label_head,
                claim_transition=transition,
            )
        )

    now[0] += 2
    reclaimed = queue.claim(f"local-teacher:{route}", 1, route, 60)[0]
    assert reclaimed.work_id == claims[0].work_id
    assert reclaimed.attempt == claims[0].attempt + 1
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    real_chain_head = store.chain_head
    monkeypatch.setattr(
        store,
        "chain_head",
        lambda path: (
            {"records": 1, "head_sha256": "f" * 64}
            if path == label_path
            else real_chain_head(path)
        ),
    )
    with pytest.raises(distill.DistillationError, match="evidence drifted"):
        distill._repair_local_r4_pending_failure_markers(
            root=tmp_path,
            config=distill.DistillationConfig(teacher_max_inflight=1),
            workset=queue,
            label_path=label_path,
            reclaimed_claim=reclaimed,
        )
    assert len(
        list(
            (store.distillation_dir(tmp_path) / "r4-failure-pending").rglob(
                "*.json"
            )
        )
    ) == 2
    monkeypatch.setattr(store, "chain_head", real_chain_head)
    real_completions = queue.completion_identities
    monkeypatch.setattr(
        queue,
        "completion_identities",
        lambda work_ids: {
            work_id: {
                "work_id": work_id,
                "attempt": 1,
                "completion_ref": "label-ledger:" + "e" * 64,
                "completion_digest": "e" * 64,
            }
            for work_id in work_ids
        },
    )
    with pytest.raises(distill.DistillationError, match="evidence drifted"):
        distill._repair_local_r4_pending_failure_markers(
            root=tmp_path,
            config=distill.DistillationConfig(teacher_max_inflight=1),
            workset=queue,
            label_path=label_path,
            reclaimed_claim=reclaimed,
        )
    assert len(
        list(
            (store.distillation_dir(tmp_path) / "r4-failure-pending").rglob(
                "*.json"
            )
        )
    ) == 2
    monkeypatch.setattr(queue, "completion_identities", real_completions)
    distill._repair_local_r4_pending_failure_markers(
        root=tmp_path,
        config=distill.DistillationConfig(teacher_max_inflight=1),
        workset=queue,
        label_path=label_path,
        reclaimed_claim=reclaimed,
    )
    pending = list(
        (store.distillation_dir(tmp_path) / "r4-failure-pending").rglob("*.json")
    )
    assert [path.stem for path in pending] == [markers[1]["artifact_id"]]
    queue.commit(
        [reclaimed],
        [{"status": "quarantined", "error_class": "test_reclaim_cleanup"}],
    )
    second = queue.claim(f"local-teacher:{route}", 1, route, 60)[0]
    assert second.work_id == claims[1].work_id
    assert second.attempt == claims[1].attempt + 1
    distill._repair_local_r4_pending_failure_markers(
        root=tmp_path,
        config=distill.DistillationConfig(teacher_max_inflight=1),
        workset=queue,
        label_path=label_path,
        reclaimed_claim=second,
    )
    assert not list(
        (store.distillation_dir(tmp_path) / "r4-failure-pending").rglob("*.json")
    )
    assert queue.release_unattempted([second]) == 1


def test_local_r4_receipt_crash_repairs_under_stored_cap_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_a = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    route = distill.TEACHER_ROLES[0]
    rally = {"rally_id": "r4-receipt-crash"}
    candidate = {"candidate_id": "r4-receipt-crash-candidate"}
    assignment = distill.teacher_assignment(
        str(rally["rally_id"]), str(candidate["candidate_id"])
    )
    route = str(assignment["owner"])
    digest = canonical_json.canonical_json_sha256_strict(
        {"rally": rally["rally_id"], "candidate": candidate["candidate_id"]}
    )
    work_id = distill._local_work_id(digest, route)
    distill.bootstrap_r4_distillation_root_authority(tmp_path)
    queue = workset.DistillationWorkset(
        store.distillation_dir(tmp_path) / "local-workset.sqlite3"
    )
    queue.advance(
        [
            {
                "work_id": work_id,
                "kind": f"local-teacher:{route}",
                "payload_ref": "candidate-snapshot:r4-receipt-crash",
                "payload_digest": digest,
                "priority": 0,
                "temporal_split": {"split": "train"},
                "provenance": {"route": route},
            }
        ],
        {"source": 1},
    )
    claim = queue.claim(f"local-teacher:{route}", 1, route, 60)[0]
    claim_transition = queue.transition_receipt_binding(
        queue.recent_transition_receipts(limit=1)[0]["generation"]
    )
    assert claim_transition is not None
    task = {
        "rally": rally,
        "candidate": candidate,
        "assignment": assignment,
        "route": route,
    }
    route_identity = {
        "role": route,
        "provider": "ollama",
        "model": "local-r4-receipt-crash",
        "location": "local",
    }
    monkeypatch.setattr(
        distill, "_local_r4_owned_failure_crash_after_receipt", lambda _marker: True
    )
    with pytest.raises(distill.DistillationError, match="receipt crash"):
        distill._settle_local_r4_failure(
            root=tmp_path,
            config=distill.DistillationConfig(teacher_max_inflight=7),
            workset=queue,
            batch=[{"claim": claim, "task": task}],
            category="capacity",
            source=source_a,
            route_identity=route_identity,
            label_head={"records": 0, "head_sha256": ""},
            owned_diagnostic=True,
            model_calls=0,
            attempted=False,
            claim_transition=claim_transition,
        )
    pending_root = store.distillation_dir(tmp_path) / "r4-failure-pending"
    receipt_root = store.distillation_dir(tmp_path) / "r4-receipts" / "local"
    assert len(list(pending_root.rglob("*.json"))) == 1
    receipts = list(receipt_root.rglob("*.json"))
    assert len(receipts) == 1
    monkeypatch.setattr(
        distill,
        "ox_alpha_source_binding",
        lambda: {
            "source_commit": "d" * 40,
            "source_tree_sha256": "e" * 64,
            "source_ox_identity_sha256": "f" * 64,
        },
    )
    distill._repair_local_r4_pending_failure_markers(
        root=tmp_path,
        config=distill.DistillationConfig(teacher_max_inflight=3),
        workset=queue,
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
    )
    assert not list(pending_root.rglob("*.json"))
    assert list(receipt_root.rglob("*.json")) == receipts
    receipt = store.read_sealed(receipts[0], schema=distill.R4_RECEIPT_SCHEMA)
    assert receipt["configured_max_inflight"] == 7
    assert receipt["source_commit"] == source_a["source_commit"]
    assert not (receipt_root / ("d" * 40)).exists()


def test_local_r4_owned_failure_injection_forces_one_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = distill.TEACHER_ROLES[0]
    distill.bootstrap_r4_distillation_root_authority(tmp_path)
    queue = workset.DistillationWorkset(store.distillation_dir(tmp_path) / "local-workset.sqlite3")
    tasks: dict[str, dict[str, object]] = {}
    items: list[dict[str, object]] = []
    for index in range(100):
        rally_id = f"r4-one-claim-rally-{index}"
        candidate_id = f"r4-one-claim-candidate-{index}"
        assignment = distill.teacher_assignment(rally_id, candidate_id)
        if assignment["owner"] != route:
            continue
        digest = canonical_json.canonical_json_sha256_strict({"index": index})
        work_id = distill._local_work_id(digest, route)
        tasks[work_id] = {"rally": {"rally_id": rally_id, "query_sha256": f"query-{index}", "context_refs": []}, "candidate": {"candidate_id": candidate_id, "text_sha256": f"text-{index}"}, "assignment": assignment, "route": route}
        items.append({"work_id": work_id, "kind": f"local-teacher:{route}", "payload_ref": f"candidate-snapshot:r4-one-claim-{index}", "payload_digest": digest, "priority": 0, "temporal_split": {"split": "train"}, "provenance": {"route": route}})
        if len(items) == 2:
            break
    assert len(items) == 2
    queue.advance(items, {"source": 1})
    identity = {"role": route, "provider": "ollama", "model": "local-r4-one-claim", "location": "local"}
    teacher = distill._WorkerTeacher(route, 4_096, identity, "d" * 64)
    monkeypatch.setattr(teacher, "evaluate", lambda _payload: pytest.fail("provider called"))
    monkeypatch.setattr(distill, "_local_r4_owned_failure_injection", lambda **_kwargs: "capacity")

    result = distill._run_local_teacher_route(
        workset=queue, route=route, config=distill.DistillationConfig(teacher_claim_limit=2),
        teachers={route: teacher}, tasks=tasks,
        texts={**{f"query-{index}": "question" for index in range(100)}, **{f"text-{index}": "evidence" for index in range(100)}}, root=tmp_path, raw_dir=tmp_path / "raw",
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl", snapshots={}, label_rows=[], structural_verifier=lambda *_args: None,
    )
    assert result is not None and result.model_calls == 0
    assert queue.status(include_timing=True)["ready"] == 2
    assert len(list((store.distillation_dir(tmp_path) / "r4-receipts").rglob("*.json"))) == 1


def test_local_r4_existing_receipt_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    }
    monkeypatch.setattr(distill, "ox_alpha_source_binding", lambda: dict(source))
    queue, entries = _committed_local_r4_entries(tmp_path, count=1)
    config = distill.DistillationConfig()
    assert distill._write_r4_local_receipts(
        root=tmp_path, config=config, workset=queue, entries=entries
    )
    directory = (
        store.distillation_dir(tmp_path)
        / "r4-receipts"
        / "local"
        / source["source_commit"]
    )
    path = next(directory.glob("*.json"))
    target = tmp_path / "receipt-target.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    assert not distill._write_r4_local_receipts(
        root=tmp_path, config=config, workset=queue, entries=entries
    )


def test_local_r4_repair_cursor_covers_backlog_beyond_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"

    def payload(index: int) -> dict[str, Any]:
        rally_id = f"repair-rally-{index}"
        candidate_id = f"repair-candidate-{index}"
        assignment = distill.teacher_assignment(rally_id, candidate_id)
        route = str(assignment["owner"])
        return {
            "kind": "teacher-label",
            "status": "completed",
            "teacher_profile": distill.LOCAL_TRIAD_PROFILE,
            "work_id": f"local-teacher-{index:064x}",
            "attempt": 1,
            "rally_id": rally_id,
            "candidate_id": candidate_id,
            "route": route,
            "captured_at": "2026-08-25T00:00:00Z",
            "assignment": assignment,
            "route_identity": {
                "role": route,
                "provider": "ollama",
                "model": "repair-model",
                "location": "local",
            },
            "source_commit": commit,
            "source_tree_sha256": "b" * 64,
            "source_ox_identity_sha256": "c" * 64,
        }

    rows = [
        *store.append_chain_batch(label_path, [payload(index) for index in range(500)]),
        *store.append_chain_batch(
            label_path, [payload(index) for index in range(500, 505)]
        ),
    ]

    class Queue:
        def completion_identities(
            self, work_ids: Iterable[str]
        ) -> dict[str, dict[str, Any]]:
            selected = set(work_ids)
            return {
                str(row["work_id"]): {
                    "work_id": row["work_id"],
                    "attempt": 1,
                    "completion_ref": f"label-ledger:{row['record_sha256']}",
                    "completion_digest": row["record_sha256"],
                }
                for row in rows
                if row["work_id"] in selected
            }

    monkeypatch.setattr(
        distill.runtime_config, "runtime_identity", lambda: {"commit_id": commit}
    )
    batches: list[list[str]] = []
    monkeypatch.setattr(
        distill,
        "_ensure_r4_local_receipts",
        lambda **kwargs: batches.append(
            [str(entry["work_id"]) for entry in kwargs["entries"]]
        ),
    )
    arguments = {
        "root": tmp_path,
        "config": distill.DistillationConfig(),
        "workset": Queue(),
        "tasks": {},
        "label_path": label_path,
    }

    distill._repair_r4_local_receipts(**arguments, label_rows=rows)
    assert len(batches[-1]) == 505

    added = store.append_chain_batch(
        label_path, [payload(index) for index in range(505, 508)]
    )
    rows.extend(added)
    distill._repair_r4_local_receipts(**arguments, label_rows=rows)
    assert len(batches[-1]) == 508

    incomplete = store.append_chain_batch(label_path, [payload(508)])
    distill._repair_r4_local_receipts(
        **arguments, label_rows=[*rows, *incomplete]
    )
    assert len(batches[-1]) == 508
    rows.extend(incomplete)
    distill._repair_r4_local_receipts(**arguments, label_rows=rows)
    assert len(batches[-1]) == 509


def test_local_r4_repair_never_touches_a_symlinked_legacy_state_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    external = tmp_path / "external"
    external.mkdir()
    state_directory = store.distillation_dir(tmp_path) / "r4-local-repair"
    state_directory.parent.mkdir(parents=True, exist_ok=True)
    state_directory.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        distill.runtime_config, "runtime_identity", lambda: {"commit_id": commit}
    )

    distill._repair_r4_local_receipts(
        root=tmp_path,
        config=distill.DistillationConfig(),
        workset=object(),
        tasks={},
        label_path=store.distillation_dir(tmp_path) / "label-ledger.jsonl",
        label_rows=[],
    )

    assert not list(external.iterdir())


def test_local_r4_repair_skips_other_teacher_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    rows = [
        store.append_chain(
            label_path,
            {
                "kind": "teacher-label",
                "teacher_profile": distill.OX_SINGLE_PROFILE,
                "source_commit": commit,
            },
        )
    ]
    monkeypatch.setattr(
        distill.runtime_config, "runtime_identity", lambda: {"commit_id": commit}
    )

    distill._repair_r4_local_receipts(
        root=tmp_path,
        config=distill.DistillationConfig(),
        workset=object(),
        tasks={},
        label_path=label_path,
        label_rows=rows,
    )



def test_local_teacher_batch_republishes_the_post_commit_crash_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rally_id = "repair-rally"
    candidate_id = "repair-candidate"
    assignment = distill.teacher_assignment(rally_id, candidate_id)
    route = str(assignment["owner"])
    work_id = "local-teacher-" + "a" * 64
    task = {
        "rally": {"rally_id": rally_id},
        "candidate": {"candidate_id": candidate_id},
        "assignment": assignment,
        "route": route,
    }
    route_identity = {
        "role": route,
        "provider": "ollama",
        "model": "repair-model",
        "location": "local",
    }
    label_path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    label = store.append_chain(label_path, {
        "kind": "teacher-label",
        "status": "completed",
        "teacher_profile": distill.LOCAL_TRIAD_PROFILE,
        "work_id": work_id,
        "attempt": 1,
        "rally_id": rally_id,
        "candidate_id": candidate_id,
        "route": route,
        "assignment": assignment,
        "route_identity": route_identity,
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_ox_identity_sha256": "c" * 64,
    })

    class Queue:
        def status(self, **_kwargs: object) -> dict[str, int]:
            return {"completed": 1}

        def completion_identities(
            self, work_ids: Iterable[str]
        ) -> dict[str, dict[str, Any]]:
            assert list(work_ids) == [work_id]
            return {
                work_id: {
                    "work_id": work_id,
                    "attempt": 1,
                    "completion_ref": f"label-ledger:{label['record_sha256']}",
                    "completion_digest": label["record_sha256"],
                }
            }

    monkeypatch.setattr(workset, "DistillationWorkset", lambda _path: Queue())
    monkeypatch.setattr(distill, "_advance_local_workset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        distill,
        "_prepare_local_teacher_work",
        lambda **_kwargs: ({work_id: task}, []),
    )
    monkeypatch.setattr(
        distill.runtime_config,
        "runtime_identity",
        lambda: {"commit_id": "a" * 40},
    )
    captured: list[Mapping[str, Any]] = []
    monkeypatch.setattr(
        distill,
        "_ensure_r4_local_receipts",
        lambda **kwargs: captured.extend(kwargs["entries"]),
    )

    result = distill._run_local_teacher_batch(
        root=tmp_path,
        raw_dir=tmp_path / "raw",
        config=distill.DistillationConfig(),
        teachers={},
        snapshots={},
        rally_by_id={},
        texts={},
        label_path=label_path,
        label_rows=[label],
        structural_verifier=lambda *_args: None,
    )

    assert result.workset_status == {"completed": 1}
    assert [(entry["work_id"], entry["attempt"]) for entry in captured] == [
        (work_id, 1)
    ]


def test_cold_start_does_not_begin_counterfactual_without_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Counterfactual:
        local = True

        def compare(self, _payload: object) -> dict[str, object]:
            raise AssertionError("counterfactual must not start")

    raw_dir = _raw(tmp_path)
    config = _config(tmp_path)
    monkeypatch.setattr(distill.time, "monotonic", lambda: 0.0)
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=raw_dir,
        config_path=config,
        teachers={},
        counterfactual=Counterfactual(),
        cold_start=True,
        max_elapsed_seconds=60,
    )
    assert result["status"] == "deferred"
    assert result["counterfactuals_written"] == 0


def test_shadow_hashes_serialize_candidate_and_baseline_arms_independently() -> None:
    rendered = "{\"page_id\":\"page\"}"
    source = {
        "candidate_id": "page",
        "selected": True,
        "page_id": "page",
        "page_content_sha256": "a" * 64,
        "rendered_context": rendered,
        "rendered_context_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }
    features = distill.build_fast_features(
        query_chargram_coverage=1.0, candidate_chargram_precision=0.5
    )
    equal = distill.shadow_observation_hashes(
        [{"candidate_id": "page", "features": features}],
        [{"candidate_id": "page", "features": features}],
        [source],
        [source],
        selected_candidate_ids=["page"],
        baseline_selected_candidate_ids=["page"],
    )
    assert equal["feature_parity"] is True
    changed = distill.shadow_observation_hashes(
        [{"candidate_id": "page", "features": features}],
        [
            {
                "candidate_id": "page",
                "features": distill.build_fast_features(
                    query_chargram_coverage=0.0, candidate_chargram_precision=0.5
                ),
            }
        ],
        [source],
        [source],
        selected_candidate_ids=["page"],
        baseline_selected_candidate_ids=["page"],
    )
    assert changed["feature_parity"] is False
    assert changed["candidate_feature_bytes_sha256"] != changed[
        "baseline_feature_bytes_sha256"
    ]
    assert changed["pair_id"] != equal["pair_id"]


def test_shadow_operational_evidence_rejects_arbitrary_mapping() -> None:
    hashes = {
        "candidate_decision_sha256": "a" * 64,
        "baseline_decision_sha256": "b" * 64,
        "candidate_pool_sha256": "c" * 64,
        "baseline_pool_sha256": "d" * 64,
        "candidate_feature_snapshot_sha256": "e" * 64,
        "baseline_feature_snapshot_sha256": "f" * 64,
        "candidate_feature_bytes_sha256": "0" * 64,
        "baseline_feature_bytes_sha256": "1" * 64,
        "feature_snapshot_sha256": "2" * 64,
        "feature_parity": True,
        "pair_id": "3" * 64,
    }
    with pytest.raises(distill.DistillationError, match="must be typed"):
        distill._shadow_evidence_with_hashes(
            {"candidate_quality": True},  # type: ignore[arg-type]
            hashes,
            stage="shadow",
            run_id="4" * 64,
            cohort="5" * 64,
            host="codex",
        )
