from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.ops import autonomy, health
from tests.semantic_hold_support import semantic_authority


def _enable_native_luna_health(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"native-luna-single-v1\"\n"
        "native_luna_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        health,
        "recall_distillation_kpi",
        lambda *_args: {
            "status": "ready",
            "worker_status": "idle",
            "hold_reason": "",
            "alert": False,
        },
    )
    monkeypatch.setattr(
        health,
        "_distillation_contract_expiry",
        lambda *_args: ("", True),
    )


def _write_native_luna_supervisor_status(
    root: Path, *, status: str = "running", **overrides: object
) -> None:
    payload: dict[str, object] = {
        "schema": health.NATIVE_LUNA_SUPERVISOR_SCHEMA,
        "status": status,
        "pid": 123,
        "child_pid": 456,
        "updated_at": datetime.now(UTC).isoformat(),
        "last_progress_at": datetime.now(UTC).isoformat(),
        "completed_total": 3,
        "pending_total": 2,
        "last_exit_code": 0,
        "last_error_code": "",
        "bundles": {"bundle-A": {"completed": 3, "pending": 2}},
        "reason": "private detail",
    }
    payload.update(overrides)
    path = root / "runtime" / "recall-luna-bulk" / "supervisor-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def _asia_tokyo_timezone(monkeypatch: pytest.MonkeyPatch):
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "Asia/Tokyo")
            time.tzset()
            yield
    finally:
        time.tzset()


def test_recall_distillation_kpi_uses_public_snapshot_without_raw_content(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    snapshot_module = SimpleNamespace(
        snapshot=lambda _root: {
            "schema": 1,
            "status": "running",
            "state": "backfill",
            "rollout": 5.0,
            "state_sha256": "a" * 64,
            "active_policy_id": "active-policy",
            "candidate_policy_id": "candidate-policy",
            "lkg_policy_id": "lkg-policy",
            "feature_revision": "recall-distill-text-v2",
            "teacher_only": 12,
            "verified_truth": 3,
            "probe_not_truth": 4,
            "paired_denominator": 7,
            "hold_reason": "verified_truth_below_floor",
        }
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setitem(
        sys.modules,
        "chronovisor.recall.recall_distillation_store",
        snapshot_module,
    )

    payload = health.recall_distillation_kpi()

    assert payload["worker_status"] == "backfill"
    assert payload["rollout_percent"] == 5.0
    assert payload["active_policy_id"] == "active-policy"
    assert payload["candidate_policy_id"] == "candidate-policy"
    assert payload["lkg_policy_id"] == "lkg-policy"
    assert payload["teacher_only"] == 12
    assert payload["verified_truth"] == 3
    assert payload["probe_not_truth"] == 4
    assert payload["paired_denominator"] == 7
    assert payload["hold_reason"] == "verified_truth_below_floor"
    assert payload["feature_revision"] == "recall-distill-text-v2"
    assert payload["alert"] is False


def test_recall_distillation_kpi_surfaces_tamper_without_alerting_core_recall(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot_module = SimpleNamespace(
        snapshot=lambda _root: {"status": "tampered", "state": "halted", "rollout": 0}
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setitem(
        sys.modules,
        "chronovisor.recall.recall_distillation_store",
        snapshot_module,
    )

    payload = health.recall_distillation_kpi()

    assert payload["status"] == "tampered"
    assert payload["worker_status"] == "halted"
    assert payload["alert"] is True


def test_recall_distillation_kpi_reads_real_sealed_state_and_pointers(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.recall import recall_distillation_store as store

    chronovisor_root = tmp_path / "wiki"
    state_path = store.distillation_dir(chronovisor_root) / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "status": "ready",
            "worker_status": "idle",
            "rollout_percent": 5,
            "last_success_at": "2026-08-14T20:00:00Z",
            "hold_reason": "verified_truth_below_floor",
        },
    )
    store.append_chain(
        store.distillation_dir(chronovisor_root) / "label-ledger.jsonl",
        {"authority": "teacher-only", "assignment": {"probe": True}},
    )
    store.append_chain(
        store.distillation_dir(chronovisor_root) / "label-ledger.jsonl",
        {"authority": "verified", "assignment": {"probe": False}},
    )
    store.append_chain(
        store.distillation_dir(chronovisor_root) / "shadow-observation-receipts.jsonl",
        {"kind": "shadow-policy-observation"},
    )
    policy_ids = []
    for kind, marker in (("active", "a"), ("candidate", "b"), ("lkg", "c")):
        policy_id, _, _ = store.write_immutable(
            store.distillation_dir(chronovisor_root) / "policies",
            {"kind": "test-policy", "marker": marker},
            schema="chronovisor.recall-distill-policy.v2",
        )
        policy_ids.append(policy_id)
        store.write_pointer(chronovisor_root, kind, policy_id)
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", tmp_path / "unrelated-root")

    payload = health.recall_distillation_kpi(chronovisor_root)

    assert payload["status"] == "idle"
    assert payload["worker_status"] == "idle"
    assert payload["rollout_status"] == "ready"
    assert payload["rollout_percent"] == 5.0
    assert payload["active_policy_id"] == policy_ids[0][:12]
    assert payload["candidate_policy_id"] == policy_ids[1][:12]
    assert payload["lkg_policy_id"] == policy_ids[2][:12]
    assert payload["teacher_only"] == 1
    assert payload["verified_truth"] == 1
    # Probe labels stay visible as non-truth telemetry; the unbound observation
    # receipt still does not enter the stage-bound paired denominator.
    assert payload["probe_not_truth"] == 1
    assert payload["paired_denominator"] == 0
    assert payload["hold_reason"] == "verified_truth_below_floor"
    assert payload["feature_revision"] == "recall-distill-text-v2"
    assert payload["alert"] is False


def test_recall_distillation_kpi_marks_tampered_real_state_visible(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.recall import recall_distillation_store as store

    chronovisor_root = tmp_path / "wiki"
    state_path = store.distillation_dir(chronovisor_root) / store.STATE_FILE
    store.write_sealed_state(state_path, {"status": "capture_only"})
    tampered = json.loads(state_path.read_text(encoding="utf-8"))
    tampered["status"] = "active"
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.recall_distillation_kpi()

    assert payload["status"] == "tampered"
    assert payload["worker_status"] == "tampered"
    assert payload["rollout_status"] == "tampered"
    assert payload["alert"] is True


def test_recall_distillation_health_explains_disabled_capture_as_human_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "config.toml").write_text(
        "[recall.distillation]\nenabled = false\ncapture_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "recall_distillation_kpi", lambda *_args: {
        "status": "capture_only",
        "worker_status": "capture_only",
        "hold_reason": "blind_repeat_pairs_below_floor",
        "alert": False,
    })

    payload = health.recall_distillation_health(root)

    assert payload["alert"] is True
    assert payload["stop_reason"] == {
        "category": "human_wait",
        "reason": "enabled_false",
        "message": "人の判断待ち: 蒸留が無効です (enabled=false)",
    }


def test_recall_distillation_health_does_not_treat_legacy_success_as_label_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"local-triad-v1\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "recall_distillation_kpi", lambda *_args: {
        "status": "capture_only",
        "worker_status": "capture_only",
        "hold_reason": "counterfactual_pairs_below_floor",
        "last_success_at": "2020-01-01T00:00:00Z",
        "alert": False,
    })

    payload = health.recall_distillation_health(root)
    reasons = {item["reason"] for item in payload["alerts"]}

    assert payload["progress_stale"] is False
    assert payload["progress_source"] == "missing"
    assert "counterfactual_pairs_below_floor" in reasons
    assert "durable_progress_missing" in reasons
    assert all(item["category"] == "data_wait" for item in payload["alerts"])


def test_recall_distillation_health_reports_missing_and_expired_teacher_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"native-luna-single-v1\"\n"
        "native_luna_enabled = true\n"
        "ox_enabled = true\n"
        "ox_expires_at = \"2020-01-01T00:00:00Z\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "recall_distillation_kpi", lambda *_args: {
        "status": "deferred",
        "worker_status": "deferred",
        "hold_reason": "",
        "alert": False,
    })

    payload = health.recall_distillation_health(root)
    reasons = {item["reason"] for item in payload["alerts"]}

    assert "teacher_contract_missing" in reasons
    assert "teacher_contract_expired" in reasons
    assert all(
        item["category"] == "human_wait"
        for item in payload["alerts"]
        if item["reason"] in {"teacher_contract_missing", "teacher_contract_expired"}
    )


def test_recall_distillation_health_uses_only_the_native_contract_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_distillation_store as store

    root = tmp_path / "wiki"
    root.mkdir()
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"native-luna-single-v1\"\n"
        "native_luna_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "recall_distillation_kpi", lambda *_args: {
        "status": "ready",
        "worker_status": "idle",
        "hold_reason": "",
        "alert": False,
    })
    old_contract = root / "runtime" / "recall-distillation" / "ox-profile-contract.json"
    store.write_sealed_state(old_contract, {"expires_at": ""})

    missing = health.recall_distillation_health(root)
    assert "teacher_contract_missing" in missing["reason_codes"]

    native_contract = (
        root / "runtime" / "recall-distillation" / "native-luna-profile-contract.json"
    )
    store.write_sealed_state(native_contract, {"expires_at": ""})
    tampered = health.recall_distillation_health(root)
    assert "teacher_contract_missing" in tampered["reason_codes"]
    from chronovisor.recall import recall_distillation_native as native

    monkeypatch.setattr(native, "read_contract", lambda _root: {"expires_at": ""})
    present = health.recall_distillation_health(root)
    assert "teacher_contract_missing" not in present["reason_codes"]
    assert "teacher_contract_expiry_warning" not in present["reason_codes"]
    assert present["teacher_contract_expires_at"] is None


def test_recall_distillation_health_rejects_partial_native_profile_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"native-luna-single-v1-canary\"\n"
        "native_luna_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "recall_distillation_kpi", lambda *_args: {
        "status": "ready",
        "worker_status": "idle",
        "hold_reason": "",
        "alert": False,
    })

    payload = health.recall_distillation_health(root)
    reasons = set(payload["reason_codes"])

    assert "native_luna_profile_mismatch" in reasons
    assert "teacher_contract_missing" in reasons


def test_recall_distillation_health_distinguishes_teacher_stop_from_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"local-triad-v1\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "recall_distillation_kpi", lambda *_args: {
        "status": "deferred",
        "worker_status": "deferred",
        "error_code": "ox_teacher_stopped",
        "hold_reason": "",
        "alert": False,
    })

    payload = health.recall_distillation_health(root)

    assert "teacher_stopped" in payload["reason_codes"]
    assert "teacher_missing" not in payload["reason_codes"]


def test_recall_distillation_health_surfaces_native_teacher_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"native-luna-single-v1\"\n"
        "native_luna_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "recall_distillation_kpi", lambda *_args: {
        "status": "deferred",
        "worker_status": "deferred",
        "capture_only_reasons": ["native_luna_teacher_unavailable"],
        "hold_reason": "",
        "alert": False,
    })

    payload = health.recall_distillation_health(root)

    assert "teacher_missing" in payload["reason_codes"]


def test_recall_distillation_health_prefers_label_progress_over_last_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"local-triad-v1\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "recall_distillation_kpi", lambda *_args: {
        "status": "capture_only",
        "worker_status": "capture_only",
        "last_success_at": datetime.now(UTC).isoformat(),
        "last_label_at": "2020-01-01T00:00:00Z",
        "hold_reason": "counterfactual_pairs_below_floor",
        "alert": False,
    })

    payload = health.recall_distillation_health(root)

    assert payload["last_label_at"] == "2020-01-01T00:00:00+00:00"
    assert payload["progress_stale"] is True
    assert "durable_progress_stale" in payload["reason_codes"]


def test_native_luna_supervisor_absence_preserves_existing_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)

    payload = health.recall_distillation_health(root)

    assert "native_luna_supervisor" not in payload
    assert "native_luna_executor_error" not in payload["reason_codes"]
    assert "native_luna_executor_stalled" not in payload["reason_codes"]


def test_native_luna_supervisor_retrying_error_is_human_wait_and_redacts_private_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC).isoformat()
    _write_native_luna_supervisor_status(
        root,
        status="retrying",
        updated_at=now,
        last_progress_at=now,
        last_error_code="child_failed",
        path="/private/raw-path",
        reason="private reason",
    )

    payload = health.recall_distillation_health(root)
    supervisor = payload["native_luna_supervisor"]

    assert "native_luna_executor_error" in payload["reason_codes"]
    assert payload["alerts"][0]["category"] == "human_wait"
    assert supervisor == {
        "status": "retrying",
        "completed_total": 3,
        "pending_total": 2,
        "bundles": {"bundle-A": {"completed": 3, "pending": 2}},
        "last_progress_at": now,
    }
    assert "pid" not in supervisor
    assert "child_pid" not in supervisor
    assert "last_error_code" not in supervisor
    assert "reason" not in supervisor
    assert "path" not in supervisor


def test_repeated_mainline_failure_is_human_wait_without_replacing_supervisor_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC).isoformat()
    _write_native_luna_supervisor_status(
        root,
        status="retrying",
        updated_at=now,
        last_progress_at=now,
        last_error_code="child_failed",
    )
    monkeypatch.setattr(
        health,
        "_distillation_state",
        lambda _root: {
            "worker_failure_count": 2,
            "error_code": "distillation_run_failed",
            "worker_error_type": "RuntimeError",
            "worker_error_sha256": "e" * 64,
            "last_failure_at": now,
            "private_error": "/private/raw-error",
        },
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_executor_error" in payload["reason_codes"]
    assert "distillation_run_repeated_failure" in payload["reason_codes"]
    repeated = payload["distillation_mainline_failure"]
    assert repeated == {
        "worker_failure_count": 2,
        "error_code": "distillation_run_failed",
        "worker_error_stage": "mainline",
        "worker_error_type": "RuntimeError",
        "worker_error_sha256": "e" * 64,
        "last_failure_at": now,
    }
    assert "/private/raw-error" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("worker_error_stage", "expected_stage"),
    [("counterfactual", "counterfactual"), ("/private/raw-stage", "mainline"), (["private-stage"], "mainline")],
)
def test_repeated_mainline_failure_stage_is_allowlisted_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_error_stage: object,
    expected_stage: str,
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC).isoformat()
    _write_native_luna_supervisor_status(root, status="running")
    monkeypatch.setattr(
        health,
        "_distillation_state",
        lambda _root: {
            "worker_failure_count": 2,
            "error_code": "distillation_run_failed",
            "worker_error_stage": worker_error_stage,
            "worker_error_type": "RuntimeError",
            "worker_error_sha256": "e" * 64,
            "last_failure_at": now,
        },
    )

    payload = health.recall_distillation_health(root)

    repeated = payload["distillation_mainline_failure"]
    assert repeated["worker_error_stage"] == expected_stage
    alert = next(
        item
        for item in payload["alerts"]
        if item["reason"] == "distillation_run_repeated_failure"
    )
    assert f"蒸留の {expected_stage} 実行" in alert["message"]
    if expected_stage == "mainline":
        assert "private" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("worker_error_kind", "worker_error_type", "expects_human_wait"),
    [
        ("counterfactual_data_reject", "CounterfactualInvalid", False),
        ("counterfactual_data_reject", "CounterfactualTimeout", True),
        ("counterfactual_system_error", "CounterfactualTimeout", True),
        ("counterfactual_system_error", "CounterfactualBindingError", True),
        ("counterfactual_system_error", "CounterfactualCrash", True),
        ("", "CounterfactualInvalid", True),
        ("unknown", "CounterfactualInvalid", True),
    ],
)
def test_counterfactual_rejection_kind_controls_repeated_failure_alert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_error_kind: str,
    worker_error_type: str,
    expects_human_wait: bool,
) -> None:
    root_name = worker_error_kind.replace("/", "_") or "empty"
    root = tmp_path / root_name
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC).isoformat()
    _write_native_luna_supervisor_status(root, status="running")
    monkeypatch.setattr(
        health,
        "_distillation_state",
        lambda _root: {
            "worker_failure_count": 2,
            "error_code": "distillation_run_failed",
            "worker_error_stage": "counterfactual",
            "worker_error_type": worker_error_type,
            "worker_error_kind": worker_error_kind,
            "worker_error_sha256": "e" * 64,
            "last_failure_at": now,
        },
    )

    payload = health.recall_distillation_health(root)

    failure = payload["distillation_mainline_failure"]
    assert failure["worker_failure_count"] == 2
    if worker_error_kind in {"counterfactual_data_reject", "counterfactual_system_error"}:
        assert failure["worker_error_kind"] == worker_error_kind
    else:
        assert "worker_error_kind" not in failure
    has_repeated_failure = "distillation_run_repeated_failure" in payload["reason_codes"]
    assert has_repeated_failure is expects_human_wait
    if not expects_human_wait:
        assert all(item["category"] != "human_wait" for item in payload["alerts"])


def test_native_luna_supervisor_disabled_is_human_wait_without_stall_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    _write_native_luna_supervisor_status(
        root,
        status="disabled",
        updated_at=old,
        last_progress_at=old,
        pending_total=7,
    )

    payload = health.recall_distillation_health(root)

    assert "executor_disabled" in payload["reason_codes"]
    assert "native_luna_executor_stalled" not in payload["reason_codes"]


def test_native_luna_supervisor_stale_progress_is_data_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    old_heartbeat = (datetime.now(UTC) - timedelta(minutes=16)).isoformat()
    old_progress = (datetime.now(UTC) - timedelta(minutes=31)).isoformat()
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=old_heartbeat,
        last_progress_at=old_progress,
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_executor_stalled" in payload["reason_codes"]
    stalled = next(
        item
        for item in payload["alerts"]
        if item["reason"] == "native_luna_executor_stalled"
    )
    assert stalled["category"] == "data_wait"


@pytest.mark.parametrize("status", ["running", "idle"])
def test_native_luna_supervisor_fresh_heartbeat_has_no_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC).isoformat()
    _write_native_luna_supervisor_status(
        root,
        status=status,
        updated_at=now,
        last_progress_at=now,
        pending_total=0,
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_executor_error" not in payload["reason_codes"]
    assert "native_luna_executor_stalled" not in payload["reason_codes"]


def test_native_luna_supervisor_invalid_state_is_human_wait_without_raw_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    _write_native_luna_supervisor_status(
        root,
        pending_total="not-a-count",
        error="private error text",
        path="/private/raw-path",
    )

    payload = health.recall_distillation_health(root)

    assert "executor_state_invalid" in payload["reason_codes"]
    assert payload["native_luna_supervisor"] == {
        "status": "invalid",
        "completed_total": 0,
        "pending_total": 0,
        "bundles": {},
        "last_progress_at": None,
    }
    assert "private error text" not in json.dumps(payload)
    assert "/private/raw-path" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("operation_name", "operation_reason", "expected_reason"),
    [
        ("import", "daily_import_failed", "native_luna_import_failed"),
        ("run", "daily_run_failed", "native_luna_daily_run_failed"),
        (
            "capture",
            "daily_capture_receipt_stale",
            "native_luna_daily_capture_stalled",
        ),
        ("capture", "daily_capture_failed", "native_luna_daily_capture_failed"),
    ],
)
def test_native_luna_operation_failure_is_specific_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
    operation_reason: str,
    expected_reason: str,
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    _write_native_luna_supervisor_status(
        root,
        status="retrying",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        last_error_code="private-top-level-error",
        operation={
            "name": operation_name,
            "state": "backoff",
            "status": "failed",
            "bundle": "bundle-A",
            "source_snapshot_sha256": "a" * 64,
            "failure_count": 3,
            "last_error_code": operation_reason,
            "last_error_message_sha256": "b" * 64,
            "next_retry_at": (now + timedelta(minutes=5)).isoformat(),
            "started_at": (now - timedelta(minutes=1)).isoformat(),
            "updated_at": now.isoformat(),
            "reason": operation_reason,
            "path": "/private/raw-path",
            "error": "private raw error",
        },
    )

    payload = health.recall_distillation_health(root)

    assert expected_reason in payload["reason_codes"]
    alert = next(item for item in payload["alerts"] if item["reason"] == expected_reason)
    assert alert["category"] == "human_wait"
    operation = payload["native_luna_supervisor"]["operation"]
    assert operation["name"] == operation_name
    assert operation["last_error_message_sha256"] == "b" * 64
    assert "path" not in operation
    assert "error" not in operation
    assert "private raw error" not in json.dumps(payload)
    assert "/private/raw-path" not in json.dumps(payload)


def test_native_luna_running_operation_stale_is_visible_with_fresh_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation={
            "name": "run",
            "state": "running",
            "status": "running",
            "bundle": "2026-10-14",
            "failure_count": 0,
            "started_at": (now - timedelta(minutes=31)).isoformat(),
            "updated_at": (now - timedelta(minutes=31)).isoformat(),
        },
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_daily_run_failed" in payload["reason_codes"]
    assert payload["alerts"][0]["category"] == "human_wait"


def test_native_luna_lock_wait_timeout_is_visible_with_fresh_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    waiting = {
        "name": "run",
        "state": "waiting",
        "status": "waiting",
        "bundle": "2026-10-14",
        "failure_count": 0,
        "reason": "distillation_worker_lock_busy",
        "started_at": (now - timedelta(minutes=31)).isoformat(),
        "updated_at": now.isoformat(),
    }
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=waiting,
        operations={"run:2026-10-14:": waiting},
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_daily_run_failed" in payload["reason_codes"]
    alert = next(
        item
        for item in payload["alerts"]
        if item["reason"] == "native_luna_daily_run_failed"
    )
    assert alert["category"] == "human_wait"


def test_native_luna_history_keeps_bundle_a_failure_when_bundle_b_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation={
            "name": "import",
            "state": "succeeded",
            "status": "succeeded",
            "bundle": "bundle-B",
            "failure_count": 0,
            "updated_at": now.isoformat(),
        },
        operations={
            "import:bundle-A:": {
                "name": "import",
                "state": "backoff",
                "status": "failed",
                "bundle": "bundle-A",
                "failure_count": 2,
                "last_error_code": "NativeDistillationError",
                "last_error_message_sha256": "c" * 64,
                "next_retry_at": (now + timedelta(minutes=5)).isoformat(),
                "updated_at": now.isoformat(),
            },
            "import:bundle-B:": {
                "name": "import",
                "state": "succeeded",
                "status": "succeeded",
                "bundle": "bundle-B",
                "failure_count": 0,
                "updated_at": now.isoformat(),
            },
        },
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_import_failed" in payload["reason_codes"]
    history = payload["native_luna_supervisor"]["operations"]
    assert {entry["bundle"] for entry in history} == {"bundle-A", "bundle-B"}


def test_native_luna_daily_run_history_uses_latest_real_success_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    old_failure_at = now - timedelta(hours=3)
    empty_identity_running_at = now - timedelta(hours=2)
    success_at = now - timedelta(hours=1)
    old_failure = {
        "name": "run",
        "state": "backoff",
        "status": "failed",
        "bundle": "2026-10-14-bundle",
        "source_snapshot_sha256": "a" * 64,
        "failure_count": 2,
        "last_error_code": "daily_prepare",
        "last_error_message_sha256": "b" * 64,
        "next_retry_at": (old_failure_at + timedelta(minutes=5)).isoformat(),
        "updated_at": old_failure_at.isoformat(),
    }
    empty_identity_running = {
        "name": "run",
        "state": "running",
        "status": "running",
        "bundle": "2026-10-15-bundle",
        "source_snapshot_sha256": "",
        "failure_count": 0,
        "started_at": empty_identity_running_at.isoformat(),
        "updated_at": empty_identity_running_at.isoformat(),
    }
    running_after_success = {
        "name": "run",
        "state": "running",
        "status": "running",
        "bundle": "2026-10-15-bundle",
        "source_snapshot_sha256": "c" * 64,
        "failure_count": 0,
        "last_success_at": success_at.isoformat(),
        "last_success_status": "succeeded",
        "last_success_reason": "daily_prepare",
        "started_at": (now - timedelta(minutes=1)).isoformat(),
        "updated_at": (now - timedelta(minutes=1)).isoformat(),
    }
    operations = {
        "run:2026-10-14-bundle:a": old_failure,
        "run:2026-10-15-bundle:": empty_identity_running,
        "run:2026-10-15-bundle:c": running_after_success,
    }
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=running_after_success,
        operations=operations,
    )

    after_success = health.recall_distillation_health(root)

    assert "native_luna_daily_run_failed" not in after_success["reason_codes"]
    assert len(after_success["native_luna_supervisor"]["operations"]) == 3

    new_failure_at = now
    new_failure = {
        **running_after_success,
        "state": "backoff",
        "status": "failed",
        "bundle": "2026-10-15-bundle",
        "source_snapshot_sha256": "d" * 64,
        "failure_count": 1,
        "last_error_code": "daily_prepare",
        "last_error_message_sha256": "b" * 64,
        "updated_at": new_failure_at.isoformat(),
        "next_retry_at": (new_failure_at + timedelta(minutes=5)).isoformat(),
    }
    operations["run:2026-10-15-bundle:c"] = new_failure
    _write_native_luna_supervisor_status(
        root,
        status="retrying",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=new_failure,
        operations=operations,
    )

    after_new_failure = health.recall_distillation_health(root)

    assert "native_luna_daily_run_failed" in after_new_failure["reason_codes"]
    assert len(after_new_failure["native_luna_supervisor"]["operations"]) == 3


def test_native_luna_daily_capture_history_uses_latest_real_success_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    old_failure_at = now - timedelta(hours=3)
    success_at = now - timedelta(hours=1)
    old_failure = {
        "name": "capture",
        "state": "backoff",
        "status": "failed",
        "bundle": "2026-10-16",
        "failure_count": 1,
        "last_error_code": "daily_capture_receipt_stale",
        "last_error_message_sha256": "a" * 64,
        "next_retry_at": (old_failure_at + timedelta(minutes=5)).isoformat(),
        "updated_at": old_failure_at.isoformat(),
    }
    success = {
        "name": "capture",
        "state": "succeeded",
        "status": "succeeded",
        "bundle": "2026-10-17",
        "failure_count": 0,
        "last_success_at": success_at.isoformat(),
        "reason": "capture_complete",
        "updated_at": success_at.isoformat(),
    }
    operations = {
        "capture:2026-10-16:": old_failure,
        "capture:2026-10-17:": success,
    }
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=success,
        operations=operations,
    )

    after_success = health.recall_distillation_health(root)

    assert "native_luna_daily_capture_stalled" not in after_success["reason_codes"]
    assert len(after_success["native_luna_supervisor"]["operations"]) == 2

    new_failure_at = datetime.now(UTC) - timedelta(seconds=1)
    new_failure = {
        **success,
        "state": "backoff",
        "status": "failed",
        "failure_count": 1,
        "last_error_code": "daily_capture_receipt_stale",
        "last_error_message_sha256": "b" * 64,
        "next_retry_at": (new_failure_at + timedelta(minutes=5)).isoformat(),
        "updated_at": new_failure_at.isoformat(),
    }
    operations["capture:2026-10-17:"] = new_failure
    _write_native_luna_supervisor_status(
        root,
        status="retrying",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=new_failure,
        operations=operations,
    )

    after_new_failure = health.recall_distillation_health(root)

    assert "native_luna_daily_capture_stalled" in after_new_failure["reason_codes"]
    assert len(after_new_failure["native_luna_supervisor"]["operations"]) == 2


@pytest.mark.parametrize(
    ("label", "marker"),
    [
        (
            "malformed-success",
            {
                "name": "capture",
                "state": "succeeded",
                "status": "succeeded",
                "bundle": "2026-10-17",
                "failure_count": 0,
                "last_success_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                "last_success_status": "succeeded",
                "last_success_reason": "capture_started",
                "reason": "capture_complete",
                "updated_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            },
        ),
        (
            "future-success",
            {
                "name": "capture",
                "state": "succeeded",
                "status": "succeeded",
                "bundle": "2026-10-17",
                "failure_count": 0,
                "last_success_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "last_success_status": "succeeded",
                "last_success_reason": "capture_complete",
                "reason": "capture_complete",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        ),
        (
            "post-entry-success",
            {
                "name": "capture",
                "state": "succeeded",
                "status": "succeeded",
                "bundle": "2026-10-17",
                "failure_count": 0,
                "last_success_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                "last_success_status": "succeeded",
                "last_success_reason": "capture_complete",
                "reason": "capture_complete",
                "updated_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            },
        ),
    ],
)
def test_native_luna_capture_malformed_or_future_success_does_not_clear_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    marker: dict[str, object],
) -> None:
    root = tmp_path / label
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    old_failure_at = now - timedelta(hours=3)
    old_failure = {
        "name": "capture",
        "state": "backoff",
        "status": "failed",
        "bundle": "2026-10-16",
        "failure_count": 1,
        "last_error_code": "daily_capture_receipt_stale",
        "last_error_message_sha256": "c" * 64,
        "next_retry_at": (old_failure_at + timedelta(minutes=5)).isoformat(),
        "updated_at": old_failure_at.isoformat(),
    }
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=marker,
        operations={
            "capture:2026-10-16:": old_failure,
            "capture:2026-10-17:": marker,
        },
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_daily_capture_stalled" in payload["reason_codes"]


@pytest.mark.parametrize(
    "marker",
    [
        {
            "name": "capture",
            "state": "succeeded",
            "status": "succeeded",
            "bundle": "2026-10-17",
            "failure_count": 0,
            "reason": "capture_started",
            "updated_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        },
        {
            "name": "import",
            "state": "succeeded",
            "status": "succeeded",
            "bundle": "2026-10-17",
            "failure_count": 0,
            "reason": "imported",
            "updated_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        },
        {
            "name": "capture",
            "state": "running",
            "status": "running",
            "bundle": "2026-10-17",
            "failure_count": 0,
            "reason": "daily_capture_in_progress",
            "updated_at": datetime.now(UTC).isoformat(),
        },
    ],
)
def test_native_luna_capture_unqualified_progress_does_not_clear_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: dict[str, object],
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    old_failure_at = now - timedelta(hours=3)
    old_failure = {
        "name": "capture",
        "state": "backoff",
        "status": "failed",
        "bundle": "2026-10-16",
        "failure_count": 1,
        "last_error_code": "daily_capture_receipt_stale",
        "last_error_message_sha256": "d" * 64,
        "next_retry_at": (old_failure_at + timedelta(minutes=5)).isoformat(),
        "updated_at": old_failure_at.isoformat(),
    }
    marker_key = f"{marker['name']}:{marker['bundle']}:"
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=marker,
        operations={
            "capture:2026-10-16:": old_failure,
            marker_key: marker,
        },
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_daily_capture_stalled" in payload["reason_codes"]


def test_native_luna_capture_handoff_does_not_clear_run_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    old_failure = {
        "name": "run",
        "state": "backoff",
        "status": "failed",
        "bundle": "2026-10-14-bundle",
        "failure_count": 1,
        "last_error_code": "daily_prepare",
        "last_error_message_sha256": "a" * 64,
        "next_retry_at": (now + timedelta(minutes=5)).isoformat(),
        "updated_at": (now - timedelta(minutes=5)).isoformat(),
    }
    capture_handoff = {
        "name": "run",
        "state": "succeeded",
        "status": "succeeded",
        "bundle": "2026-10-15-bundle",
        "source_snapshot_sha256": "b" * 64,
        "failure_count": 0,
        "reason": "capture_started",
        "updated_at": now.isoformat(),
    }
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=capture_handoff,
        operations={
            "run:2026-10-14-bundle:": old_failure,
            "run:2026-10-15-bundle:b": capture_handoff,
        },
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_daily_run_failed" in payload["reason_codes"]


def test_native_luna_run_success_watermark_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    old_failure = {
        "name": "run",
        "state": "backoff",
        "status": "failed",
        "bundle": "2026-10-14-bundle",
        "failure_count": 1,
        "last_error_code": "daily_prepare",
        "last_error_message_sha256": "a" * 64,
        "next_retry_at": (now + timedelta(minutes=5)).isoformat(),
        "updated_at": (now - timedelta(hours=3)).isoformat(),
    }
    cases = (
        (
            "import",
            {
                "name": "import",
                "state": "succeeded",
                "status": "succeeded",
                "bundle": "2026-10-15-bundle",
                "failure_count": 0,
                "last_success_at": (now + timedelta(hours=1)).isoformat(),
                "last_success_status": "succeeded",
                "last_success_reason": "daily_prepare",
                "updated_at": now.isoformat(),
            },
        ),
        (
            "run-future-success",
            {
                "name": "run",
                "state": "backoff",
                "status": "failed",
                "bundle": "2026-10-15-bundle",
                "source_snapshot_sha256": "b" * 64,
                "failure_count": 1,
                "last_error_code": "daily_prepare",
                "last_error_message_sha256": "c" * 64,
                "last_success_at": (now + timedelta(hours=1)).isoformat(),
                "last_success_status": "succeeded",
                "last_success_reason": "daily_prepare",
                "next_retry_at": (now + timedelta(minutes=5)).isoformat(),
                "updated_at": now.isoformat(),
            },
        ),
        (
            "run-success-after-entry",
            {
                "name": "run",
                "state": "backoff",
                "status": "failed",
                "bundle": "2026-10-15-bundle",
                "source_snapshot_sha256": "d" * 64,
                "failure_count": 1,
                "last_error_code": "daily_prepare",
                "last_error_message_sha256": "e" * 64,
                "last_success_at": (now - timedelta(hours=1)).isoformat(),
                "last_success_status": "succeeded",
                "last_success_reason": "daily_prepare",
                "next_retry_at": (now + timedelta(minutes=5)).isoformat(),
                "updated_at": (now - timedelta(hours=2)).isoformat(),
            },
        ),
    )

    for index, (label, marker) in enumerate(cases):
        root = tmp_path / label
        _enable_native_luna_health(root, monkeypatch)
        marker_key = (
            f"{marker['name']}:2026-10-15-bundle:"
            f"{marker.get('source_snapshot_sha256', '')}"
        )
        _write_native_luna_supervisor_status(
            root,
            status="retrying",
            updated_at=now.isoformat(),
            last_progress_at=now.isoformat(),
            operation=marker,
            operations={
                "run:2026-10-14-bundle:": old_failure,
                marker_key: marker,
            },
        )

        payload = health.recall_distillation_health(root)

        assert "native_luna_daily_run_failed" in payload["reason_codes"], index


def test_native_luna_running_retry_keeps_prior_failure_visible_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    entry = {
        "name": "import",
        "state": "running",
        "status": "running",
        "bundle": "bundle-A",
        "failure_count": 1,
        "last_error_code": "NativeDistillationError",
        "last_error_message_sha256": "d" * 64,
        "started_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        operation=entry,
        operations={"import:bundle-A:": entry},
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_import_failed" in payload["reason_codes"]


def test_native_luna_malformed_operation_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC).isoformat()
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now,
        last_progress_at=now,
        operations={
            "import:bundle-A:": {
                "name": "import",
                "state": "backoff",
                "status": "failed",
                "bundle": "bundle-A",
                "failure_count": "2",
                "updated_at": now,
            }
        },
    )

    payload = health.recall_distillation_health(root)

    assert payload["native_luna_supervisor"]["status"] == "invalid"
    assert "executor_state_invalid" in payload["reason_codes"]


def test_native_luna_import_waiting_stale_is_human_wait_with_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    oldest = now - timedelta(minutes=31)
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        pending_total=0,
        import_waiting={
            "bundles": 2,
            "labels": 61884,
            "oldest_at": oldest.isoformat(),
            "oldest_waiting_at": oldest.isoformat(),
            "bundle_names": ["/private/bundle-A", "bundle-B"],
        },
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_import_pending" in payload["reason_codes"]
    assert payload["native_luna_supervisor"]["import_waiting"] == {
        "bundles": 2,
        "labels": 61884,
        "oldest_at": oldest.isoformat(),
    }
    assert "bundle_names" not in payload["native_luna_supervisor"]["import_waiting"]


def test_native_luna_import_waiting_fresh_does_not_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        pending_total=0,
        import_waiting={"bundles": 1, "labels": 10, "oldest_at": now.isoformat()},
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_import_pending" not in payload["reason_codes"]


def test_native_luna_import_label_watermark_is_progress_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    imported_at = now - timedelta(minutes=2)
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        pending_total=0,
        last_imported_label_at=imported_at.isoformat(),
    )

    payload = health.recall_distillation_health(root)

    assert payload["last_label_at"] == imported_at.isoformat()
    assert payload["progress_source"] == "last_label_at"
    assert payload["progress_stale"] is False


def test_native_luna_legacy_import_waiting_is_visible_after_stale_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    old = datetime.now(UTC) - timedelta(minutes=31)
    _write_native_luna_supervisor_status(
        root,
        status="idle",
        updated_at=old.isoformat(),
        last_progress_at=old.isoformat(),
        pending_total=0,
        reason="daily_import",
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_import_pending" in payload["reason_codes"]
    assert "native_luna_executor_stalled" not in payload["reason_codes"]


def test_native_luna_legacy_import_waiting_uses_old_progress_with_fresh_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    old_progress = now - timedelta(minutes=31)
    _write_native_luna_supervisor_status(
        root,
        status="running",
        updated_at=now.isoformat(),
        last_progress_at=old_progress.isoformat(),
        pending_total=0,
        reason="daily_import",
    )

    payload = health.recall_distillation_health(root)

    assert "native_luna_import_pending" in payload["reason_codes"]


def test_native_luna_idle_schedule_stall_is_human_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC)
    _write_native_luna_supervisor_status(
        root,
        status="idle",
        updated_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        pending_total=0,
        reason="waiting_for_incremental_capture",
        next_capture_after_unix=now.timestamp() - 31 * 60,
    )

    payload = health.recall_distillation_health(root)

    alert = next(
        item
        for item in payload["alerts"]
        if item["reason"] == "native_luna_daily_schedule_stalled"
    )
    assert alert["category"] == "human_wait"
    assert "予定時刻を過ぎても開始されていません" in alert["message"]


def test_native_luna_operation_malformed_state_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    _enable_native_luna_health(root, monkeypatch)
    now = datetime.now(UTC).isoformat()
    _write_native_luna_supervisor_status(
        root,
        status="retrying",
        updated_at=now,
        last_progress_at=now,
        operation={
            "name": "import",
            "state": "backoff",
            "status": "failed",
            "bundle": "/private/raw-bundle",
            "failure_count": "3",
            "updated_at": now,
            "error": "private raw error",
        },
    )

    payload = health.recall_distillation_health(root)

    assert payload["native_luna_supervisor"]["status"] == "invalid"
    assert "executor_state_invalid" in payload["reason_codes"]
    assert "private raw error" not in json.dumps(payload)
    assert "/private/raw-bundle" not in json.dumps(payload)


def test_semantic_index_kpi_is_inactive_when_rollout_is_off(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chronovisor.core.runtime_config.load_search_embedding_config",
        lambda: SearchEmbeddingConfig(rollout_mode="off"),
    )

    assert health.semantic_index_kpi()["status"] == "inactive"


def test_semantic_index_kpi_requires_matching_live_service(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime = chronovisor_root / "runtime"
    runtime.mkdir(parents=True)
    socket_path = runtime / "semantic.sock"
    (runtime / "semantic-service-status.json").write_text(
        json.dumps(
            {
                "ready": True,
                "pid": os.getpid(),
                    "generation_id": "generation-a",
                    "observed_at_epoch": time.time(),
                    "routes": {
                        "search.semantic.foreground": {
                            "role": "search.semantic.foreground",
                            "provider": "remote",
                            "model": "embed-model",
                            "location": "remote",
                        }
                    },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        Path,
        "is_socket",
        lambda path: path == socket_path,
    )
    monkeypatch.setattr(
        "chronovisor.core.runtime_config.load_search_embedding_config",
        lambda: SearchEmbeddingConfig(
            enabled=True,
            rollout_mode="on",
            socket=str(socket_path),
        ),
    )
    monkeypatch.setattr(
        "chronovisor.core.semantic_index.semantic_index_status",
        lambda **_kwargs: {
            "status": "ok",
            "coverage": 1.0,
            "generation_id": "generation-a",
        },
    )
    monkeypatch.setattr(
        "chronovisor.core.semantic_jobs.job_status",
        lambda: {"status": "ok", "counts": {}},
    )
    payload = health.semantic_index_kpi()

    assert payload["status"] == "ok"
    assert payload["service_process_alive"] is True
    assert payload["generation_matches"] is True


def test_lint_queue_kpi_counts_only_unresolved_issue_keys(tmp_path: Path) -> None:
    queue = tmp_path / "lint-repair-queue.jsonl"
    convergence = tmp_path / "state.json"
    rows = [
        {"issue_key": "handled", "page": "a"},
        {"issue_key": "active", "page": "b"},
        {"issue_key": "new", "page": "c"},
        {"page": "missing-key"},
    ]
    queue.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    convergence.write_text(
        json.dumps(
            {
                "items": {
                    "handled": {
                        "lane": "lint_repair",
                        "status": "rejected",
                        "metadata": {"issue_key": "handled"},
                    },
                    "active": {
                        "lane": "lint_repair",
                        "status": "pending_local",
                        "metadata": {"issue_key": "active"},
                    },
                    "other-lane": {
                        "lane": "orphan_link",
                        "status": "pending_local",
                        "metadata": {"issue_key": "new"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = health._lint_queue_kpi(queue, convergence)

    assert payload == {
        "total": 4,
        "unique": 4,
        "actionable": 3,
        "active": 1,
        "untracked": 2,
        "handled": 1,
    }


def test_queue_health_alerts_only_stale_active_lanes_and_keeps_quarantine_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    label_path = root / "recall" / "search-label-queue.jsonl"
    lint_state = root / "runtime" / "convergence" / "state.json"
    replay_path = root / "review" / "raw-replay-queue.jsonl"
    label_path.parent.mkdir(parents=True)
    lint_state.parent.mkdir(parents=True)
    replay_path.parent.mkdir(parents=True)
    label_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "queue_status": "pending_review",
                    "preregistered_at": "2020-01-01T00:00:00Z",
                },
                {
                    "queue_status": "pending_frontier_review",
                    "preregistered_at": "2020-01-01T00:00:00Z",
                },
                {"queue_status": "completed", "ts": "2020-01-01T00:00:00Z"},
            )
        ),
        encoding="utf-8",
    )
    recent = datetime.now(UTC).isoformat()
    lint_state.write_text(
        json.dumps(
            {
                "items": {
                    "local": {
                        "lane": "lint_repair",
                        "status": "pending_local",
                        "created_at": "2020-01-01T00:00:00Z",
                        "updated_at": recent,
                        "last_attempt_at": recent,
                    },
                    "local_route": {
                        "lane": "lint_repair",
                        "status": "local_route_provenance_invalid",
                        "created_at": "2020-01-01T00:00:00Z",
                    },
                    "frontier": {
                        "lane": "lint_repair",
                        "status": "pending_frontier",
                        "created_at": "2020-01-01T00:00:00Z",
                        "updated_at": "2020-01-01T00:00:00Z",
                        "human_required": True,
                    },
                    "quarantine": {
                        "lane": "lint_repair",
                        "status": "quarantined",
                        "created_at": "2020-01-01T00:00:00Z",
                        "updated_at": "2020-01-01T00:00:00Z",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps(
            {"status": "pending", "created_at": "2020-01-01T00:00:00Z"}
        )
        + "\n",
        encoding="utf-8",
    )

    payload = health.queue_health_snapshot(root)
    reasons = {item["reason"] for item in payload["alerts"]}

    assert "search_labels_late_lane" in reasons
    assert "search_labels_authority_wait" in reasons
    assert "lint_repair_authority_wait" in reasons
    assert "raw_replay_late_lane" in reasons
    assert payload["lint_repair"]["quarantined"] == 1
    assert all(item["queue"] != "quarantine" for item in payload["alerts"])


def test_queue_progress_uses_completed_same_lane_attempts_without_changing_counters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    label_path = root / "recall" / "search-label-queue.jsonl"
    lint_state = root / "runtime" / "convergence" / "state.json"
    replay_path = root / "review" / "raw-replay-queue.jsonl"
    label_path.parent.mkdir(parents=True)
    lint_state.parent.mkdir(parents=True)
    replay_path.parent.mkdir(parents=True)
    old = "2020-01-01T00:00:00Z"
    recent = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    label_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"queue_status": "pending_review", "preregistered_at": old},
                {
                    "queue_status": "completed",
                    "ts": old,
                    "last_attempt_lane": "review",
                    "last_attempt_at": recent,
                },
                {"queue_status": "pending_frontier_review", "preregistered_at": old},
                {
                    "queue_status": "completed",
                    "ts": old,
                    "last_attempt_lane": "frontier",
                    "last_attempt_at": recent,
                },
            )
        ),
        encoding="utf-8",
    )
    lint_state.write_text(
        json.dumps(
            {
                "items": {
                    "local-active": {
                        "lane": "lint_repair",
                        "status": "pending_local",
                        "created_at": old,
                    },
                    "local-done": {
                        "lane": "lint_repair",
                        "status": "completed",
                        "created_at": old,
                        "last_local_attempt_at": recent,
                    },
                    "frontier-active": {
                        "lane": "lint_repair",
                        "status": "pending_frontier",
                        "created_at": old,
                    },
                    "frontier-done": {
                        "lane": "lint_repair",
                        "status": "completed",
                        "created_at": old,
                        "last_frontier_attempt_at": recent,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text("", encoding="utf-8")

    payload = health.queue_health_snapshot(root)
    reasons = {item["reason"] for item in payload["alerts"]}

    assert "search_labels_late_lane" not in reasons
    assert "lint_repair_late_lane" not in reasons
    assert payload["search_labels"]["active"] == 2
    assert payload["search_labels"]["status_counts"]["completed"] == 2
    assert payload["search_labels"]["lanes"]["review"]["oldest_at"].startswith(
        "2020-01-01"
    )
    assert payload["search_labels"]["lanes"]["review"]["last_progress_at"] == recent
    assert payload["lint_repair"]["active"] == 2
    assert payload["lint_repair"]["status_counts"]["completed"] == 2
    assert payload["lint_repair"]["lanes"]["local"]["last_progress_at"] == recent


def test_search_queue_progress_normalizes_local_naive_terminal_attempts(
    tmp_path: Path,
    _asia_tokyo_timezone,
) -> None:
    root = tmp_path / "wiki"
    label_path = root / "recall" / "search-label-queue.jsonl"
    lint_state = root / "runtime" / "convergence" / "state.json"
    replay_path = root / "review" / "raw-replay-queue.jsonl"
    label_path.parent.mkdir(parents=True)
    lint_state.parent.mkdir(parents=True)
    replay_path.parent.mkdir(parents=True)
    old = "2020-01-01T00:00:00Z"
    recent_local = datetime.now().replace(microsecond=0).isoformat()
    label_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "queue_status": "pending_frontier_review",
                    "preregistered_at": old,
                },
                {
                    "queue_status": "frontier_rejected",
                    "preregistered_at": old,
                    "last_attempt_lane": "frontier",
                    "last_attempt_at": recent_local,
                },
                {
                    "queue_status": "source_stale",
                    "preregistered_at": old,
                    "reviewed_at": recent_local,
                },
            )
        ),
        encoding="utf-8",
    )
    lint_state.write_text("{}", encoding="utf-8")
    replay_path.write_text("", encoding="utf-8")

    payload = health.queue_health_snapshot(root)
    frontier = payload["search_labels"]["lanes"]["frontier"]
    expected = datetime.fromisoformat(recent_local).astimezone(UTC).isoformat()

    assert frontier["last_progress_at"] == expected
    assert "search_labels_authority_wait" not in {
        item["reason"] for item in payload["alerts"]
    }
    assert list(
        health._search_lane_progress_at([{"reviewed_at": recent_local}], "frontier")
    ) == []


def test_queue_progress_rejects_other_lane_and_invalid_attempt_timestamps(
    tmp_path: Path,
) -> None:
    old = "2020-01-01T00:00:00Z"
    recent = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    future = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    rows = [{"queue_status": "pending_review", "preregistered_at": old}]

    diagnostic = health._queue_group_progress(
        rows,
        status_field="queue_status",
        statuses={"pending_review"},
        created_fields=("preregistered_at",),
        progress_fields=(),
        additional_progress_at=(future, "not-a-timestamp"),
    )

    assert diagnostic["active"] == 1
    assert diagnostic["oldest_at"].startswith("2020-01-01")
    assert diagnostic["last_progress_at"] is None
    assert diagnostic["stale"] is True

    aged_progress = health._queue_group_progress(
        rows,
        status_field="queue_status",
        statuses={"pending_review"},
        created_fields=("preregistered_at",),
        progress_fields=(),
        additional_progress_at=(old,),
    )
    assert aged_progress["last_progress_at"].startswith("2020-01-01")
    assert aged_progress["stale"] is True

    root = tmp_path / "wiki"
    label_path = root / "recall" / "search-label-queue.jsonl"
    lint_state = root / "runtime" / "convergence" / "state.json"
    replay_path = root / "review" / "raw-replay-queue.jsonl"
    label_path.parent.mkdir(parents=True)
    lint_state.parent.mkdir(parents=True)
    replay_path.parent.mkdir(parents=True)
    label_path.write_text(
        json.dumps(
            {
                "queue_status": "pending_review",
                "preregistered_at": old,
                "last_attempt_lane": "frontier",
                "last_attempt_at": recent,
                "reviewed_at": recent,
            }
        )
        + "\n"
        + json.dumps(
            {
                "queue_status": "completed",
                "ts": old,
                "last_attempt_lane": "frontier",
                "last_attempt_at": future,
            }
        )
        + "\n"
        + json.dumps(
            {
                "queue_status": "completed",
                "ts": old,
                "reviewed_at": recent,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lint_state.write_text(
        json.dumps(
            {
                "items": {
                    "local-active": {
                        "lane": "lint_repair",
                        "status": "pending_local",
                        "created_at": old,
                        "last_frontier_attempt_at": recent,
                    },
                    "frontier-done": {
                        "lane": "lint_repair",
                        "status": "completed",
                        "created_at": old,
                        "last_frontier_attempt_at": future,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text("", encoding="utf-8")

    payload = health.queue_health_snapshot(root)
    reasons = {item["reason"] for item in payload["alerts"]}

    assert "search_labels_late_lane" in reasons
    assert "lint_repair_late_lane" in reasons


def test_capture_kpi_counts_raw_claim_coverage(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    claims_dir = chronovisor_root / "claims"
    raw_dir.mkdir(parents=True)
    claims_dir.mkdir(parents=True)
    (raw_dir / "20260706-codex-a.md").write_text("a", encoding="utf-8")
    (raw_dir / "20260706-claude-b.md").write_text("b", encoding="utf-8")
    (claims_dir / "claims.jsonl").write_text(
        json.dumps({"source_raw": "20260706-codex-a.md"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(health, "RAW_DIR", raw_dir)

    payload = health.capture_kpi()

    assert payload["raw_files"] == 2
    assert payload["claimed_raw_files"] == 1
    assert payload["claim_coverage"] == 0.5
    assert payload["raw_by_host"] == {"codex": 1, "claude": 1}


def test_latest_memory_integrity_reads_summary(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    eval_dir = chronovisor_root / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "memory-integrity-latest.json").write_text(
        json.dumps(
            {"status": "ok", "total": 2, "passed": 1, "missed": 1, "capture_rate": 0.5}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.latest_memory_integrity()

    assert payload["status"] == "ok"
    assert payload["capture_rate"] == 0.5


@pytest.mark.parametrize(
    "runtime_status",
    ["waiting_for_ingest_runtime", "waiting_for_ollama"],
)
def test_ingest_liveness_kpi_alerts_when_runtime_blocks_pending_raws(
    tmp_path: Path, monkeypatch, runtime_status: str
) -> None:
    chronovisor_root = tmp_path / "wiki"
    state_path = chronovisor_root / "runtime" / "ingest-liveness.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": runtime_status,
                "pending_raws": 7,
                "observed_at": "2026-07-17T12:00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.ingest_liveness_kpi()

    assert payload["status"] == "alert"
    assert payload["runtime_status"] == runtime_status
    assert payload["pending_raws"] == 7


def test_ingest_liveness_kpi_alerts_on_invalid_authority_without_pending_raws(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    state_path = chronovisor_root / "runtime" / "ingest-liveness.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "blocked_by_decision_authority",
                "pending_raws": 0,
                "alert": True,
                "error": "adoption artifact policy version mismatch",
                "observed_at": "2026-07-28T20:42:42",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.ingest_liveness_kpi()

    assert payload["status"] == "alert"
    assert payload["runtime_status"] == "blocked_by_decision_authority"
    assert payload["pending_raws"] == 0
    assert payload["alert"] is True


def test_ingest_liveness_kpi_alerts_on_current_mutation_authority_block(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    state_path = chronovisor_root / "runtime" / "ingest-liveness.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "runtime_state": "ready",
                "mutation_ready": False,
                "mutation_authority": {"ok": False, "error": "authority changed"},
                "pending_raws": 0,
                "observed_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.ingest_liveness_kpi()

    assert payload["status"] == "alert"
    assert payload["mutation_ready"] is False
    assert payload["active_failure"]["kind"] == "authority"
    assert payload["stale"] is False


def test_ingest_liveness_kpi_alerts_when_observation_is_stale(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    state_path = chronovisor_root / "runtime" / "ingest-liveness.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "authority_available": True,
                "pending_raws": 0,
                "observed_at": (
                    datetime.now(UTC)
                    - timedelta(seconds=health.INGEST_LIVENESS_TTL_SECONDS + 1)
                ).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.ingest_liveness_kpi()

    assert payload["status"] == "alert"
    assert payload["liveness"]["fresh"] is False
    assert payload["liveness"]["stale"] is True
    assert payload["stale"] is True


def test_ingest_liveness_kpi_uses_legacy_values_when_canonical_fields_are_null(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    state_path = chronovisor_root / "runtime" / "ingest-liveness.json"
    state_path.parent.mkdir(parents=True)
    active_failure = {"kind": "authority", "timestamp": datetime.now(UTC).isoformat()}
    state_path.write_text(
        json.dumps(
            {
                "status": "blocked_by_decision_authority",
                "runtime_state": None,
                "authority_preflight": {"ok": False},
                "mutation_authority": None,
                "active_failure": active_failure,
                "last_failure": None,
                "observed_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.ingest_liveness_kpi()

    assert payload["runtime_state"] == "blocked_by_decision_authority"
    assert payload["mutation_authority"] == {"ok": False}
    assert payload["mutation_ready"] is False
    assert payload["last_failure"] == active_failure


def test_health_snapshot_alerts_on_current_runtime_authority_block(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.core import runtime_config, runtime_status
    from chronovisor.recall import librarian_status

    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    runtime_status.write_status(
        {
            "state": "blocked",
            "stage": "decision-authority",
            "authority_preflight": {"ok": False, "error": "authority changed"},
        }
    )
    for name in (
        "summary_coverage",
        "capture_kpi",
        "latest_memory_integrity",
        "cofire_kpi",
        "prefetch_kpi",
        "derived_memory_kpi",
        "read_back_kpi",
        "autonomy_hardening_kpi",
        "recall_feedback_kpi",
        "recall_distillation_kpi",
        "convergence_kpi",
        "capture_pipeline_kpi",
        "research_kpi",
    ):
        monkeypatch.setattr(health, name, dict)
    monkeypatch.setattr(health, "_queue_status_counts", lambda *_args: {})
    monkeypatch.setattr(
        health,
        "_lint_queue_kpi",
        lambda *_args: {"actionable": 0, "total": 0, "active": 0, "untracked": 0, "handled": 0},
    )
    monkeypatch.setattr(health, "ingest_liveness_kpi", lambda: {"alert": False})
    monkeypatch.setattr(health, "semantic_index_kpi", lambda: {"status": "ok"})
    monkeypatch.setattr(runtime_config, "runtime_identity", dict)
    monkeypatch.setattr(librarian_status, "build_librarian_status", lambda _root: {})

    payload = health.health_snapshot()

    assert payload["status"] == "alert"
    assert payload["runtime_status"]["mutation_ready"] is False
    assert payload["runtime_status"]["authority_blocked"] is True


def test_cofire_kpi_reads_graph_summary(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    recall_dir = chronovisor_root / "recall"
    recall_dir.mkdir(parents=True)
    (recall_dir / "cofire.json").write_text(
        json.dumps({"status": "ok", "nodes": 3, "edges": 4}),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.cofire_kpi()

    assert payload["nodes"] == 3
    assert payload["edges"] == 4


def test_derived_memory_kpi_counts_generated_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    (chronovisor_root / "claims").mkdir(parents=True)
    (chronovisor_root / "recall").mkdir(parents=True)
    (chronovisor_root / "distill").mkdir(parents=True)
    (chronovisor_root / "pages" / "hubs").mkdir(parents=True)
    (chronovisor_root / "claims" / "claims-index.jsonl").write_text(
        "{}\n{}\n", encoding="utf-8"
    )
    (chronovisor_root / "recall" / "search-golden.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (chronovisor_root / "recall" / "retention.json").write_text(
        json.dumps({"counts": {"pages": 5, "deprecation_candidates": 1}}),
        encoding="utf-8",
    )
    (chronovisor_root / "distill" / "wiki-qa.jsonl").write_text(
        "{}\n{}\n{}\n", encoding="utf-8"
    )
    (chronovisor_root / "pages" / "hubs" / "ai-hub.md").write_text(
        "---\ntitle: AI hub\nstatus: stable\ntype: knowledge\n---\nhub",
        encoding="utf-8",
    )
    (chronovisor_root / "pages" / "hubs" / "draft.md").write_text(
        "---\ntitle: Draft\nstatus: draft\ntype: knowledge\n---\ndraft",
        encoding="utf-8",
    )
    (chronovisor_root / "pages" / "hubs" / "invalid.md").write_text(
        "---\ntitle: Invalid\nstatus: stable\n---\ninvalid",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.derived_memory_kpi()

    assert payload["claims"] == 2
    assert payload["golden"] == 1
    assert payload["retention_pages"] == 5
    assert payload["distill_rows"] == 3
    assert payload["hubs"] == 1


def test_convergence_kpi_splits_semantic_defer_from_operational_quarantine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime = chronovisor_root / "runtime" / "convergence"
    runtime.mkdir(parents=True)
    (runtime / "state.json").write_text(
        json.dumps(
            {
                "items": {
                    "semantic": {
                        "status": "quarantined",
                        "lane": "content_correction",
                        "input_hash": "a" * 64,
                        "last_failure_class": "local_semantic_no_quorum",
                        "quarantine_reason": (
                            "semantic_no_quorum:content_correction_review"
                        ),
                        "result": {
                            "terminal_reason": "semantic_no_quorum",
                            "semantic_hold": {
                                "kind": "content_correction_semantic_no_quorum",
                                "decision_lane": "content_correction_review",
                                "input_hash": "a" * 64,
                                "proposal_sha256": "b" * 64,
                                "page_evidence_hashes": {"page": "c" * 64},
                                "authority": {"source": "adopted_local_consensus"},
                            },
                        },
                    },
                    "legacy-no-quorum-retry-exhausted": {
                        "status": "quarantined",
                        "lane": "autonomy_duplicate_resolution",
                        "last_failure_class": "local_semantic_no_quorum",
                        "quarantine_reason": "retry_exhausted:frontier",
                    },
                    "raw-semantic-class": {
                        "status": "quarantined",
                        "lane": "content_correction",
                        "last_failure_class": "ingest.semantic_no_quorum",
                        "quarantine_reason": "semantic_no_quorum:ingest",
                        "result": {
                            "terminal_reason": "semantic_no_quorum",
                            "semantic_hold": {"kind": "ingest_semantic_no_quorum"},
                        },
                    },
                    "operational": {
                        "status": "quarantined",
                        "lane": "content_correction",
                        "last_failure_class": "review_artifact_invalid",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.convergence_kpi()

    assert payload["items"] == 4
    assert payload["semantic_deferred"] == 1
    assert payload["quarantined"] == 3
    assert payload["by_status"] == {"quarantined": 3, "semantic_deferred": 1}
    assert payload["by_lane"]["autonomy_duplicate_resolution"] == {"quarantined": 1}
    assert payload["by_lane"]["content_correction"] == {
        "quarantined": 2,
        "semantic_deferred": 1,
    }


def test_health_accepts_only_self_hashed_legacy_semantic_migration() -> None:
    lane = autonomy.DUPLICATE_FRONTIER_LANE
    authority = semantic_authority(lane)
    original = {
        "quarantine_reason": "retry_exhausted:frontier",
        "frontier_attempts": 3,
        "last_error": "three-way split",
    }
    marker = autonomy._legacy_semantic_hold(
        lane=lane,
        epoch={"input_hash": "a" * 64},
        authority=authority,
        item=original,
    )
    item = {
        "status": "quarantined",
        "lane": lane,
        "last_failure_class": "local_semantic_no_quorum",
        "quarantine_reason": f"semantic_no_quorum_legacy:{lane}",
        "result": {
            "terminal_reason": "semantic_no_quorum",
            "legacy_semantic_hold": marker,
        },
    }

    assert health._is_terminal_semantic_defer(item) is True
    marker["epoch"]["input_hash"] = "b" * 64
    assert health._is_terminal_semantic_defer(item) is False
