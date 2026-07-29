from __future__ import annotations

import inspect
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chronovisor.core import store as chronovisor_store
from chronovisor.ops import convergence_drain
from chronovisor.ops.convergence import ConvergenceStore


def _store(tmp_path: Path) -> ConvergenceStore:
    root = tmp_path / "convergence"
    return ConvergenceStore(
        state_file=root / "state.json",
        events_file=root / "events.jsonl",
        lock_file=root / "state.lock",
    )


def _add(
    store: ConvergenceStore,
    *,
    lane: str = "content_correction",
    source: str,
) -> dict:
    return store.merge_item(
        lane=lane,
        source_id=source,
        input_data={"source": source},
        resolver_version="test-v1",
    )["item"]


def _inventory(items: list[dict], *, absent: set[str] | None = None):
    absent = absent or set()
    by_source: dict[str, dict[str, set[str]]] = {}
    for item in items:
        key = str(item["key"])
        if key in absent:
            continue
        by_source.setdefault(str(item["lane"]), {}).setdefault(
            str(item["source_id"]), set()
        ).add(key)
    return convergence_drain.Inventory(
        keys_by_source=by_source,
        payloads={},
        indeterminate_sources=set(),
        derived_items=[],
    )


def _bootstrap_adoption_fingerprint() -> dict:
    lanes = {
        name: {
            "kind": "validated_local"
            if name == "exact_user_correction"
            else "consensus",
            "schema_name": None if name == "exact_user_correction" else name,
            "mode": "enabled" if name == "exact_user_correction" else "shadow",
            "error": None,
        }
        for name in convergence_drain.DECISION_POLICY_LANES
    }
    config = {
        "primary_model": "primary:test",
        "challenger_model": "challenger:test",
        "tie_break_model": "tie:test",
        "adoption_artifact": "",
    }
    audit = {
        "source": "bootstrap_current_policy",
        "artifact_sha256": None,
        "error": None,
        "models": ["primary:test", "challenger:test", "tie:test"],
    }
    return {
        "path": None,
        "status": "not_nominated",
        "sha256": None,
        "bytes": 0,
        "decision_policies": {
            "lanes": lanes,
            "sha256": convergence_drain._sha256_value(lanes),
        },
        "configured_router": config,
        "configured_router_sha256": convergence_drain._sha256_value(config),
        "configured_router_error": None,
        "resolved_router_policy": {
            "status": "ok",
            "source": "bootstrap_current_policy",
            "artifact_path": None,
            "artifact_sha256": None,
            "error": None,
            "config_error": None,
            "config": config,
            "config_sha256": convergence_drain._sha256_value(config),
            "audit": audit,
        },
    }


@pytest.fixture
def isolated_drain(tmp_path, monkeypatch):
    monkeypatch.setattr(chronovisor_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(convergence_drain, "_runtime_commit", lambda: "abc123")
    monkeypatch.setattr(
        convergence_drain,
        "_adoption_artifact_fingerprint",
        _bootstrap_adoption_fingerprint,
    )
    return convergence_drain._frontier_fingerprint()


def test_frontier_fingerprint_tracks_only_frontier_events_and_active_markers(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "wiki"
    monkeypatch.setattr(chronovisor_store, "CHRONOVISOR_ROOT", root)
    baseline = convergence_drain._frontier_fingerprint()
    assert baseline["frontier_events"]["count"] == 0
    assert baseline["frontier_active"]["count"] == 0

    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    events = runtime / "events.jsonl"
    events.write_text(
        json.dumps({"source": "ingest", "message": "ordinary"}) + "\n",
        encoding="utf-8",
    )
    assert convergence_drain._frontier_fingerprint() == baseline

    with events.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "source": "frontier",
                    "message": "frontier | review started",
                    "review_id": "review-1",
                }
            )
            + "\n"
        )
    with_frontier = convergence_drain._frontier_fingerprint()
    assert with_frontier != baseline
    assert with_frontier["frontier_events"]["count"] == 1

    active = runtime / "frontier-reviews" / "active"
    active.mkdir(parents=True)
    (active / "review-1.json").write_text('{"active":true}\n', encoding="utf-8")
    with_active = convergence_drain._frontier_fingerprint()
    assert with_active["frontier_active"]["count"] == 1
    assert with_active != with_frontier


def test_dry_run_writes_neither_manifest_nor_convergence(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY", raising=False)
    store = _store(tmp_path)
    item = _add(store, source="one")
    state_before = store.state_file.read_bytes()
    events_before = store.events_file.read_bytes()

    def read_only_inventory(items):
        assert os.environ["CHRONOVISOR_READ_ONLY"] == "1"
        return _inventory(list(items))

    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        read_only_inventory,
    )

    result = convergence_drain.start(store=store, dry_run=True)

    assert result["manifest_written"] is False
    assert result["active_keys"] == 1
    assert result["items"][0]["key"] == item["key"]
    assert store.state_file.read_bytes() == state_before
    assert store.events_file.read_bytes() == events_before
    assert not (
        chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "convergence" / "drains"
    ).exists()
    assert "CHRONOVISOR_READ_ONLY" not in os.environ


def test_plan_with_real_content_inventory_is_byte_for_byte_read_only(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY", raising=False)
    store = _store(tmp_path)
    _add(store, source="malformed-content-event")
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    result = convergence_drain.plan(store=store)

    after = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert result["dry_run"] is True
    assert result["active_keys"] == 1
    assert result["counts"] == {"indeterminate": 1}
    assert after == before
    assert "CHRONOVISOR_READ_ONLY" not in os.environ


def test_start_persists_allowlist_before_any_claim(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    state_before = store.state_file.read_bytes()
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )

    result = convergence_drain.start(store=store, run_once=False)

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert [row["key"] for row in manifest["items"]] == [item["key"]]
    assert manifest["status"] == "created"
    assert store.state_file.read_bytes() == state_before
    assert store.get(str(item["key"]))["local_attempts"] == 0


def test_new_active_key_is_out_of_scope_and_never_terminalized(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    target = _add(store, source="target")

    def inventory(items):
        rows = list(items)
        return _inventory(rows, absent={str(target["key"])})

    monkeypatch.setattr(convergence_drain, "_build_inventory", inventory)
    started = convergence_drain.start(store=store, run_once=False)
    outside = _add(store, source="arrived-after-snapshot")
    monkeypatch.setattr(
        convergence_drain,
        "_run_lanes",
        lambda **_kwargs: {"lanes": {}, "budget": {}},
    )

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert store.get(str(target["key"]))["status"] == "rejected"
    assert store.get(str(target["key"]))["result"] == {
        "reason": "targeted_drain_source_absent"
    }
    assert store.get(str(outside["key"]))["status"] == "pending_local"
    assert result["status"] == "attention"
    assert [row["key"] for row in result["out_of_scope_active"]] == [outside["key"]]


def test_resume_is_crash_safe_and_idempotent(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    first = _add(store, source="first")
    second = _add(store, source="second")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    # Simulate a process dying after its convergence transition was fsynced
    # but before the manifest's last_run projection was replaced.
    store.complete_many([str(first["key"])], "applied", result={"test": True})
    assert store.get(str(first["key"]))["status"] == "applied"
    assert store.get(str(second["key"]))["status"] == "pending_local"

    def finish(*, store, eligible_keys, **_kwargs):
        store.complete_many(eligible_keys, "applied", result={"test": True})
        return {"lanes": {}, "budget": {}}

    monkeypatch.setattr(convergence_drain, "_run_lanes", finish)
    completed = convergence_drain.resume(run_id=started["run_id"], store=store)
    repeated = convergence_drain.resume(run_id=started["run_id"], store=store)
    assert completed["status"] == "completed"
    assert repeated["status"] == "completed"
    assert completed["target_terminal"] == 2


def test_backoff_is_reported_without_spinning(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="backoff")
    owner = store.claim_attempt(str(item["key"]), "local")["owner"]
    store.fail_attempt(
        str(item["key"]),
        "local",
        error="retry later",
        owner=owner,
        now=datetime.now(UTC),
    )
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    monkeypatch.setattr(
        convergence_drain,
        "_run_lanes",
        lambda **_kwargs: {"lanes": {}, "budget": {}},
    )

    result = convergence_drain.start(store=store)

    assert result["status"] == "running"
    assert result["target_active"] == 1
    assert result["next_retry_at"] == store.get(str(item["key"]))["next_attempt_at"]


def test_frontier_hash_change_fails_postcondition(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)

    def mutate_frontier(**_kwargs):
        runtime = chronovisor_store.CHRONOVISOR_ROOT / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "events.jsonl").write_text(
            json.dumps(
                {
                    "source": "frontier",
                    "message": "frontier | review started",
                    "review_id": "during-drain",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"lanes": {}, "budget": {}}

    monkeypatch.setattr(convergence_drain, "_run_lanes", mutate_frontier)

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed_frontier_activity"
    assert result["frontier_repair_postcondition"] == "changed"
    assert result["last_run"]["subscription_frontier_starts"] == 1


def test_lane_exception_is_durably_failed_after_postcondition_check(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    monkeypatch.setattr(
        convergence_drain,
        "_run_lanes",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("lane exploded")),
    )

    result = convergence_drain.resume(run_id=started["run_id"], store=store)
    persisted = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert result["failure_reason"] == "lane_exception"
    assert persisted["frontier_repair_postcondition"] == "unchanged"
    assert persisted["last_run"]["lane_exception"] == "RuntimeError: lane exploded"
    assert persisted["last_run"]["subscription_frontier_starts"] == 0


def test_adoption_drift_during_lane_is_durably_sealed_after_effect(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    baseline = _bootstrap_adoption_fingerprint()
    live = {"value": baseline}
    monkeypatch.setattr(
        convergence_drain,
        "_adoption_artifact_fingerprint",
        lambda: live["value"],
    )
    started = convergence_drain.start(store=store, run_once=False)
    changed = _bootstrap_adoption_fingerprint()
    changed["decision_policies"]["lanes"]["orphan_link"]["mode"] = "off"
    changed["decision_policies"]["sha256"] = convergence_drain._sha256_value(
        changed["decision_policies"]["lanes"]
    )

    def drift_after_effect(*, store, **_kwargs):
        store.complete_many([str(item["key"])], "applied", result={"effect": True})
        live["value"] = changed
        return {"lanes": {"content_correction": {"applied": 1}}}

    monkeypatch.setattr(convergence_drain, "_run_lanes", drift_after_effect)
    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "runtime_or_adoption_changed_during_lane"
    assert store.get(str(item["key"]))["status"] == "applied"
    persisted = convergence_drain._read_manifest(started["run_id"])
    assert persisted["status"] == "failed"
    assert (
        persisted["last_run"]["runtime_adoption_postcondition"]["status"] == "changed"
    )
    assert persisted["last_run"]["authority_sealed_effects"] is False

    live["value"] = baseline
    restored = convergence_drain.status(run_id=started["run_id"], store=store)

    assert restored["status"] == "failed"
    assert restored["failure_reason"] == "runtime_or_adoption_changed_during_lane"


def test_elapsed_budget_is_recomputed_after_authority_lock(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(
        store=store,
        max_elapsed_seconds=1.0,
        run_once=False,
    )
    monotonic_values = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(
        convergence_drain.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        convergence_drain,
        "_run_lanes",
        lambda **_kwargs: pytest.fail("expired post-lock budget must not run lanes"),
    )

    result = convergence_drain.resume(run_id=started["run_id"], store=store)
    persisted = convergence_drain._read_manifest(started["run_id"])

    assert result["status"] == "running"
    assert store.get(str(item["key"]))["status"] == "pending_local"
    assert persisted["last_run"]["lane_result"] == {
        "status": "budget_exhausted",
        "max_elapsed_seconds": 1.0,
    }
    assert persisted["last_run"]["elapsed_seconds"] == 2.0


def test_runtime_or_adoption_drift_stops_before_claim(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    monkeypatch.setattr(convergence_drain, "_runtime_commit", lambda: "new-commit")

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "runtime_commit_changed"
    assert store.get(str(item["key"]))["local_attempts"] == 0


def test_policy_mode_only_drift_stops_before_claim(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    changed = _bootstrap_adoption_fingerprint()
    changed["decision_policies"]["lanes"]["orphan_link"]["mode"] = "off"
    changed["decision_policies"]["sha256"] = convergence_drain._sha256_value(
        changed["decision_policies"]["lanes"]
    )
    monkeypatch.setattr(
        convergence_drain,
        "_adoption_artifact_fingerprint",
        lambda: changed,
    )

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "adoption_artifact_changed"
    assert store.get(str(item["key"]))["local_attempts"] == 0


def test_adoption_fingerprint_binds_resolved_policy_mode(monkeypatch) -> None:
    from chronovisor.core import runtime_config
    from chronovisor.decision import decision_policy

    monkeypatch.setattr(
        runtime_config,
        "load_decision_router_config",
        runtime_config.DecisionRouterConfig,
    )
    # This test verifies the registered default versus an environment override.
    # Do not let the operator's live Chronovisor config decide the baseline.
    monkeypatch.setattr(decision_policy, "load_toml_file", lambda *_args, **_kwargs: {})
    for name in convergence_drain.DECISION_POLICY_LANES:
        env_name = "CHRONOVISOR_DECISION_POLICY_" + name.upper()
        monkeypatch.delenv(env_name, raising=False)

    before = convergence_drain._adoption_artifact_fingerprint()
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_ORPHAN_LINK", "enabled")
    after = convergence_drain._adoption_artifact_fingerprint()

    assert before["resolved_router_policy"]["status"] == "ok"
    assert before["decision_policies"]["lanes"]["orphan_link"]["mode"] == "shadow"
    assert after["decision_policies"]["lanes"]["orphan_link"]["mode"] == "enabled"
    assert after != before


def test_start_rejects_unreadable_nominated_adoption_artifact(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    invalid = _bootstrap_adoption_fingerprint()
    invalid.update(
        {
            "path": str(tmp_path / "missing-adoption.json"),
            "status": "absent",
            "bytes": 0,
        }
    )
    invalid["configured_router"] = {
        **invalid["configured_router"],
        "adoption_artifact": invalid["path"],
    }
    invalid["configured_router_sha256"] = convergence_drain._sha256_value(
        invalid["configured_router"]
    )
    invalid["resolved_router_policy"] = {
        **invalid["resolved_router_policy"],
        "status": "error",
        "artifact_path": invalid["path"],
        "error": "adoption_artifact_invalid:cannot read adoption artifact",
    }
    monkeypatch.setattr(
        convergence_drain,
        "_adoption_artifact_fingerprint",
        lambda: invalid,
    )

    with pytest.raises(
        convergence_drain.DrainError,
        match="nominated_adoption_artifact_unreadable",
    ):
        convergence_drain.start(store=store, run_once=False)

    assert not convergence_drain._drain_dir().exists()


def test_start_rejects_missing_resolved_router_audit(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    invalid = _bootstrap_adoption_fingerprint()
    invalid["resolved_router_policy"] = {
        **invalid["resolved_router_policy"],
        "audit": None,
    }
    monkeypatch.setattr(
        convergence_drain,
        "_adoption_artifact_fingerprint",
        lambda: invalid,
    )

    with pytest.raises(
        convergence_drain.DrainError,
        match="resolved_router_policy_audit_missing",
    ):
        convergence_drain.start(store=store, run_once=False)

    assert not convergence_drain._drain_dir().exists()


def test_start_rejects_unknown_actual_runtime_commit(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    monkeypatch.setattr(convergence_drain, "_runtime_commit", lambda: None)

    with pytest.raises(convergence_drain.DrainError, match="runtime commit identity"):
        convergence_drain.start(store=store, run_once=False)


def test_start_rejects_existing_frontier_activity_before_manifest(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    active = (
        chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "frontier-reviews" / "active"
    )
    active.mkdir(parents=True)
    (active / "already-running.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(convergence_drain.DrainError, match="already_active"):
        convergence_drain.start(store=store, run_once=False)

    assert not (
        chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "convergence" / "drains"
    ).exists()


def test_start_rechecks_frontier_immediately_before_manifest_persistence(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    changed = json.loads(json.dumps(isolated_drain))
    changed["frontier_events"]["count"] = 1
    changed["frontier_events"]["sha256"] = "changed"
    observations = iter([isolated_drain, isolated_drain, changed])
    monkeypatch.setattr(
        convergence_drain,
        "_frontier_fingerprint",
        lambda: next(observations),
    )

    with pytest.raises(convergence_drain.DrainError, match="before manifest"):
        convergence_drain.start(store=store, run_once=False)

    drain_dir = (
        chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "convergence" / "drains"
    )
    assert not list(drain_dir.glob("*.json"))


def test_plan_fails_if_runtime_changes_during_inventory(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    commits = iter(["before", "after"])
    monkeypatch.setattr(convergence_drain, "_runtime_commit", lambda: next(commits))

    with pytest.raises(convergence_drain.DrainError, match="while building"):
        convergence_drain.plan(store=store)


def test_manifest_allowlist_tamper_is_rejected_before_claim(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    path = Path(started["manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["items"][0]["source_id"] = "tampered"
    convergence_drain._atomic_write_json(path, manifest)

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "manifest_allowlist_digest_mismatch"
    assert store.get(str(item["key"]))["local_attempts"] == 0


def test_manifest_non_allowlist_tamper_is_rejected_by_status_and_resume(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    path = Path(started["manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["frontier_repair_baseline"] = {"tampered": True}
    convergence_drain._atomic_write_json(path, manifest)

    observed = convergence_drain.status(run_id=started["run_id"], store=store)
    resumed = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert observed["status"] == "failed"
    assert observed["failure_reason"] == "manifest_integrity_mismatch"
    assert resumed["failure_reason"] == "manifest_integrity_mismatch"
    assert store.get(str(item["key"]))["local_attempts"] == 0


def test_manifest_filename_must_match_sealed_run_id(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    original = Path(started["manifest_path"])
    copied_run_id = "copied-run"
    copied = convergence_drain._manifest_path(copied_run_id)
    copied.write_bytes(original.read_bytes())

    with pytest.raises(convergence_drain.DrainError, match="run_id"):
        convergence_drain.status(run_id=copied_run_id, store=store)
    with pytest.raises(convergence_drain.DrainError, match="run_id"):
        convergence_drain.resume(run_id=copied_run_id, store=store)

    assert store.get(str(item["key"]))["local_attempts"] == 0


def test_status_reports_live_frontier_drift_even_after_prior_failure(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    manifest = convergence_drain._read_manifest(started["run_id"])
    manifest["status"] = "failed"
    manifest["failure_reason"] = "simulated_failure"
    convergence_drain._write_manifest(started["run_id"], manifest)
    events = chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps(
            {
                "source": "frontier",
                "message": "frontier | review started",
                "review_id": "unexpected",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    observed = convergence_drain.status(run_id=started["run_id"], store=store)

    assert observed["status"] == "failed_frontier_activity"
    assert observed["frontier_repair_postcondition"] == "changed_live"


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_start_rejects_unbounded_or_nonpositive_elapsed_budget(
    value, tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )

    with pytest.raises(convergence_drain.DrainError, match="finite positive"):
        convergence_drain.start(
            store=store,
            max_elapsed_seconds=value,
            run_once=False,
        )


def test_state_identity_drift_is_rejected_before_claim(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="one")
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    state = json.loads(store.state_file.read_text(encoding="utf-8"))
    state["items"][str(item["key"])]["resolver_version"] = "tampered"
    convergence_drain._atomic_write_json(store.state_file, state)

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "manifest_state_identity_mismatch"
    assert result["target_active"] == 1
    assert store.get(str(item["key"]))["local_attempts"] == 0


def test_incomplete_inventory_is_indeterminate_and_never_rejected(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, lane="lint_repair", source="tag_missing:page")
    inventory = convergence_drain.Inventory(
        keys_by_source={},
        payloads={},
        indeterminate_sources={("lint_repair", "tag_missing:page")},
        derived_items=[],
    )
    monkeypatch.setattr(convergence_drain, "_build_inventory", lambda _rows: inventory)
    monkeypatch.setattr(
        convergence_drain,
        "_run_lanes",
        lambda **_kwargs: {"lanes": {}, "budget": {}},
    )

    result = convergence_drain.start(store=store)

    assert result["status"] == "running"
    assert store.get(str(item["key"]))["status"] == "pending_local"
    assert result["last_run"]["indeterminate_sources"] == [item["key"]]


def test_missing_lint_inventory_marks_every_target_indeterminate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(chronovisor_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")

    current, path, uncertain = convergence_drain._lint_inventory(
        {"tag_missing:one", "tag_missing:two"}
    )

    assert current == {}
    assert path.name == "lint-repair-queue.jsonl"
    assert uncertain == {
        ("lint_repair", "tag_missing:one"),
        ("lint_repair", "tag_missing:two"),
    }


def test_unreadable_lint_page_is_indeterminate_not_superseded(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(chronovisor_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    queue = chronovisor_store.CHRONOVISOR_ROOT / "review" / "lint-repair-queue.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text(
        json.dumps(
            {
                "issue_type": "tags",
                "page": "memory",
                "detail": "missing tags",
                "severity": "warning",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chronovisor_store,
        "find_page",
        lambda _page: (_ for _ in ()).throw(OSError("read failed")),
    )

    current, _path, uncertain = convergence_drain._lint_inventory({"tags:memory"})

    assert current == {}
    assert uncertain == {("lint_repair", "tags:memory")}


def test_duplicate_inventory_failure_marks_current_and_legacy_indeterminate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chronovisor.recall.duplicate_review.build_duplicate_review_queue",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )

    current, records, derived, uncertain = convergence_drain._duplicate_inventory(
        {"current-left<->current-right"},
        {"legacy-left<->legacy-right"},
    )

    assert current == {}
    assert records == []
    assert derived == []
    assert uncertain == {
        ("autonomy_duplicate_resolution", "current-left<->current-right"),
        ("duplicate_frontier", "legacy-left<->legacy-right"),
    }


def test_truncated_retention_inventory_never_rejects_unlisted_target(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chronovisor.ops.retention.build_retention_scores",
        lambda **_kwargs: {
            "status": "ok",
            "pages": {},
            "archive_candidates": ["some-other-page"],
            "counts": {"archive_candidates": 2},
        },
    )

    current, _payload, derived, uncertain = convergence_drain._retention_inventory(
        {"current-unlisted"},
        {"legacy-unlisted"},
    )

    assert current == {}
    assert derived == []
    assert uncertain == {
        ("autonomy_retention", "current-unlisted"),
        ("retention_frontier", "legacy-unlisted"),
    }


def test_malformed_retention_inventory_marks_every_target_indeterminate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chronovisor.ops.retention.build_retention_scores",
        lambda **_kwargs: {
            "status": "ok",
            "pages": {},
            "archive_candidates": [],
            "counts": {},
        },
    )

    current, payload, derived, uncertain = convergence_drain._retention_inventory(
        {"current"},
        {"legacy"},
    )

    assert current == {}
    assert payload == {}
    assert derived == []
    assert uncertain == {
        ("autonomy_retention", "current"),
        ("retention_frontier", "legacy"),
    }


def test_unreadable_orphan_page_is_indeterminate(monkeypatch) -> None:
    from chronovisor.ops import orphan_link
    from chronovisor.search import index_store

    class Index:
        def refresh(self) -> None:
            return None

        def orphans(self, *, include_system: bool) -> list[str]:
            assert include_system is False
            return ["memory"]

    monkeypatch.setattr(index_store, "get_store", lambda: Index())
    monkeypatch.setattr(orphan_link, "gather_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(orphan_link, "_content_hash", lambda _page: "unreadable")
    monkeypatch.setattr(
        "chronovisor.decision.decision_authority.current_semantic_authority",
        lambda _lane: ({"version": "test"}, None),
    )

    current, uncertain = convergence_drain._orphan_inventory({"orphan:memory"})

    assert current == {}
    assert uncertain == {("orphan_link", "orphan:memory")}


def test_orphan_semantic_outage_is_indeterminate_not_empty_inventory(
    monkeypatch,
) -> None:
    from chronovisor.ops import orphan_link
    from chronovisor.search import index_store, search

    class Index:
        def refresh(self) -> None:
            return None

        def orphans(self, *, include_system: bool) -> list[str]:
            assert include_system is False
            return ["memory"]

    def gather(_orphan, _index, *, semantic_search_fn, **_kwargs):
        return semantic_search_fn("memory", 20)

    def unavailable(_query, _top_n, *, strict=False):
        assert strict is True
        raise RuntimeError("semantic backend unavailable")

    monkeypatch.setattr(index_store, "get_store", lambda: Index())
    monkeypatch.setattr(orphan_link, "gather_candidates", gather)
    monkeypatch.setattr(search, "semantic_search", unavailable)

    current, uncertain = convergence_drain._orphan_inventory({"orphan:memory"})

    assert current == {}
    assert uncertain == {("orphan_link", "orphan:memory")}


def test_content_inventory_classifies_actionable_stale_and_indeterminate(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    actionable = _add(store, source="actionable")
    false_positive = _add(store, source="false-positive")
    malformed = _add(store, source="malformed")

    def tri_state(item):
        return {
            "actionable": (True, "current"),
            "false-positive": (False, "stale"),
            "malformed": (None, "indeterminate"),
        }[item["source_id"]]

    monkeypatch.setattr(
        "chronovisor.recall.content_correction.correction_item_actionability",
        tri_state,
    )

    inventory = convergence_drain._build_inventory(
        [actionable, false_positive, malformed]
    )

    assert convergence_drain._classify_item(actionable, inventory) == "current"
    assert (
        convergence_drain._classify_item(false_positive, inventory) == "non_actionable"
    )
    assert convergence_drain._classify_item(malformed, inventory) == "indeterminate"


def test_content_false_positive_uses_dedicated_migration_without_model_call(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    from chronovisor.recall import content_correction

    store = _store(tmp_path)
    event = {
        "correction_prompt": "いや、それでいい。",
        "candidate_pages": ["memory"],
        "pulled_pages": [],
        "attribution": "ambiguous",
        "signal": {"matched": "いや、それでいい。"},
    }
    item = store.merge_item(
        lane="content_correction",
        source_id="claude:session:1->2",
        input_data={"event": "noise"},
        resolver_version="test-v1",
        metadata=event,
    )["item"]
    monkeypatch.setattr(
        content_correction,
        "run_pending_corrections",
        lambda **_kwargs: pytest.fail("non-actionable item reached a model lane"),
    )

    result = convergence_drain.start(store=store)

    assert result["status"] == "completed"
    terminal = store.get(str(item["key"]))
    assert terminal["status"] == "rejected"
    assert terminal["result"] == {
        "decision": "none",
        "reason": "correction_signal_no_longer_actionable",
        "migration": "retire_non_actionable_correction_v1",
    }
    assert terminal["local_attempts"] == 0
    assert terminal["frontier_attempts"] == 0
    assert result["last_run"]["content_false_positive_migration"]["completed"] == 1


def test_content_predicate_exception_is_indeterminate(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    item = _add(store, source="malformed")
    monkeypatch.setattr(
        "chronovisor.recall.content_correction.correction_item_actionability",
        lambda _item: (_ for _ in ()).throw(ValueError("bad metadata")),
    )

    inventory = convergence_drain._build_inventory([item])

    assert convergence_drain._classify_item(item, inventory) == "indeterminate"


def test_174_like_snapshot_retires_20_and_processes_only_154(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    lane_sizes = {
        "content_correction": 144,
        "duplicate_frontier": 24,
        "lint_repair": 2,
        "orphan_link": 3,
        "retention_frontier": 1,
    }
    items = [
        _add(store, lane=lane, source=f"{lane}-{index}")
        for lane, size in lane_sizes.items()
        for index in range(size)
    ]
    legacy_duplicates = [item for item in items if item["lane"] == "duplicate_frontier"]
    absent_items = [
        *legacy_duplicates[:18],
        next(item for item in items if item["lane"] == "orphan_link"),
        next(item for item in items if item["lane"] == "retention_frontier"),
    ]
    absent = {str(item["key"]) for item in absent_items}
    derived: list[dict] = []
    derived_by_source: dict[str, str] = {}
    for item in legacy_duplicates[18:]:
        source = str(item["source_id"])
        input_data = {"source": source, "migration": "current-v3"}
        key = convergence_drain.stable_item_key(
            "autonomy_duplicate_resolution",
            source,
            input_data,
            resolver_version="fixture-v3",
        )
        derived_by_source[source] = key
        derived.append(
            {
                "key": key,
                "lane": "autonomy_duplicate_resolution",
                "source_id": source,
                "input_hash": convergence_drain.input_fingerprint(input_data),
                "input_data": input_data,
                "resolver_version": "fixture-v3",
                "metadata": {"fixture": True},
                "derived_from_lane": "duplicate_frontier",
            }
        )

    def live_like_inventory(rows):
        by_source: dict[str, dict[str, set[str]]] = {}
        for item in rows:
            key = str(item["key"])
            lane = str(item["lane"])
            source = str(item["source_id"])
            if key in absent:
                continue
            current_key = (
                derived_by_source[source]
                if lane == "duplicate_frontier" and source in derived_by_source
                else key
            )
            by_source.setdefault(lane, {}).setdefault(source, set()).add(current_key)
            if source in derived_by_source:
                by_source.setdefault("autonomy_duplicate_resolution", {}).setdefault(
                    source, set()
                ).add(derived_by_source[source])
        return convergence_drain.Inventory(
            keys_by_source=by_source,
            payloads={},
            indeterminate_sources=set(),
            derived_items=derived,
        )

    monkeypatch.setattr(convergence_drain, "_build_inventory", live_like_inventory)

    processed: list[str] = []

    def finish_current(*, store, eligible_keys, **_kwargs):
        current = [
            key
            for key in eligible_keys
            if (store.get(key) or {}).get("status") == "pending_local"
        ]
        processed.extend(current)
        store.complete_many(current, "applied", result={"fixture": "processed"})
        return {"lanes": {}, "budget": {}}

    monkeypatch.setattr(convergence_drain, "_run_lanes", finish_current)

    result = convergence_drain.start(store=store)

    assert result["status"] == "completed"
    assert result["target_keys"] == 174
    assert result["derived_keys"] == 6
    assert result["allowlist_keys"] == 180
    assert result["target_active"] == 0
    assert result["target_terminal"] == 180
    assert len(processed) == 154
    assert (
        sum(
            (store.get(key) or {}).get("result", {}).get("reason")
            == "targeted_drain_source_absent"
            for key in absent
        )
        == 20
    )
    assert (
        sum(
            (store.get(str(item["key"])) or {}).get("result", {}).get("reason")
            == "targeted_drain_source_superseded"
            for item in legacy_duplicates[18:]
        )
        == 6
    )


def test_derived_merge_refuses_to_supersede_post_snapshot_key(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    legacy = _add(store, lane="duplicate_frontier", source="left<->right")
    input_data = {"pair": ["left", "right"], "version": 3}
    derived_key = convergence_drain.stable_item_key(
        "autonomy_duplicate_resolution",
        "left<->right",
        input_data,
        resolver_version="v3",
    )
    derived = {
        "key": derived_key,
        "lane": "autonomy_duplicate_resolution",
        "source_id": "left<->right",
        "input_hash": convergence_drain.input_fingerprint(input_data),
        "input_data": input_data,
        "resolver_version": "v3",
        "metadata": {},
        "derived_from_lane": "duplicate_frontier",
    }
    inventory = convergence_drain.Inventory(
        keys_by_source={
            "duplicate_frontier": {"left<->right": {derived_key}},
            "autonomy_duplicate_resolution": {"left<->right": {derived_key}},
        },
        payloads={},
        indeterminate_sources=set(),
        derived_items=[derived],
    )
    monkeypatch.setattr(convergence_drain, "_build_inventory", lambda _rows: inventory)
    started = convergence_drain.start(store=store, run_once=False)
    outside = store.merge_item(
        lane="autonomy_duplicate_resolution",
        source_id="left<->right",
        input_data={"pair": ["left", "right"], "version": 4},
        resolver_version="v4",
    )["item"]

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "derived_source_changed_before_merge"
    assert store.get(str(outside["key"]))["status"] == "pending_local"
    assert store.get(str(legacy["key"]))["status"] == "pending_local"
    assert store.get(derived_key) is None


def test_derived_merge_detects_post_snapshot_terminal_source_key(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    legacy = _add(store, lane="duplicate_frontier", source="left<->right")
    input_data = {"pair": ["left", "right"], "version": 3}
    derived_key = convergence_drain.stable_item_key(
        "autonomy_duplicate_resolution",
        "left<->right",
        input_data,
        resolver_version="v3",
    )
    derived = {
        "key": derived_key,
        "lane": "autonomy_duplicate_resolution",
        "source_id": "left<->right",
        "input_hash": convergence_drain.input_fingerprint(input_data),
        "input_data": input_data,
        "resolver_version": "v3",
        "metadata": {},
        "derived_from_lane": "duplicate_frontier",
    }
    inventory = convergence_drain.Inventory(
        keys_by_source={
            "duplicate_frontier": {"left<->right": {derived_key}},
            "autonomy_duplicate_resolution": {"left<->right": {derived_key}},
        },
        payloads={},
        indeterminate_sources=set(),
        derived_items=[derived],
    )
    monkeypatch.setattr(convergence_drain, "_build_inventory", lambda _rows: inventory)
    started = convergence_drain.start(store=store, run_once=False)
    outside = store.merge_item(
        lane="autonomy_duplicate_resolution",
        source_id="left<->right",
        input_data={"pair": ["left", "right"], "version": 4},
        resolver_version="v4",
    )["item"]
    store.complete_many(
        [str(outside["key"])],
        "rejected",
        result={"reason": "completed after snapshot"},
    )
    list_items = store.list_items

    def stale_list_items(**kwargs):
        return [
            item
            for item in list_items(**kwargs)
            if str(item.get("key") or "") != str(outside["key"])
        ]

    # Even if the optimistic pre-scan misses a concurrent terminal key, the
    # store must enforce the history baseline atomically with derived creation.
    monkeypatch.setattr(store, "list_items", stale_list_items)

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "derived_source_changed_before_merge"
    assert store.get(str(outside["key"]))["status"] == "rejected"
    assert store.get(str(legacy["key"]))["status"] == "pending_local"
    assert store.get(derived_key) is None


def test_derived_atomic_batch_blocks_all_creation_on_post_preflight_race(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    sources = ["a<->b", "c<->d"]
    legacy = [
        _add(store, lane="duplicate_frontier", source=source) for source in sources
    ]
    derived: list[dict] = []
    for source in sources:
        input_data = {"pair": source.split("<->"), "version": 3}
        key = convergence_drain.stable_item_key(
            "autonomy_duplicate_resolution",
            source,
            input_data,
            resolver_version="v3",
        )
        derived.append(
            {
                "key": key,
                "lane": "autonomy_duplicate_resolution",
                "source_id": source,
                "input_hash": convergence_drain.input_fingerprint(input_data),
                "input_data": input_data,
                "resolver_version": "v3",
                "metadata": {},
                "derived_from_lane": "duplicate_frontier",
            }
        )
    inventory = convergence_drain.Inventory(
        keys_by_source={
            "duplicate_frontier": {
                source: {row["key"]} for source, row in zip(sources, derived, strict=False)
            },
            "autonomy_duplicate_resolution": {
                source: {row["key"]} for source, row in zip(sources, derived, strict=False)
            },
        },
        payloads={},
        indeterminate_sources=set(),
        derived_items=derived,
    )
    monkeypatch.setattr(convergence_drain, "_build_inventory", lambda _rows: inventory)
    started = convergence_drain.start(store=store, run_once=False)
    atomic_merge = store.merge_items_atomically

    def race_after_preflight(candidates, **kwargs):
        outside = store.merge_item(
            lane="autonomy_duplicate_resolution",
            source_id=sources[1],
            input_data={"pair": ["c", "d"], "version": 4},
            resolver_version="v4",
        )["item"]
        store.complete_many([str(outside["key"])], "rejected", result={"reason": "new"})
        return atomic_merge(candidates, **kwargs)

    monkeypatch.setattr(store, "merge_items_atomically", race_after_preflight)

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "derived_source_changed_before_merge"
    assert all(store.get(str(row["key"])) is None for row in derived)
    assert all(
        store.get(str(item["key"]))["status"] == "pending_local" for item in legacy
    )


def test_derived_baseline_uses_single_state_snapshot_before_inventory(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    source = "left<->right"
    legacy = _add(store, lane="duplicate_frontier", source=source)
    input_data = {"pair": ["left", "right"], "version": 3}
    derived_key = convergence_drain.stable_item_key(
        "autonomy_duplicate_resolution",
        source,
        input_data,
        resolver_version="v3",
    )
    derived = {
        "key": derived_key,
        "lane": "autonomy_duplicate_resolution",
        "source_id": source,
        "input_hash": convergence_drain.input_fingerprint(input_data),
        "input_data": input_data,
        "resolver_version": "v3",
        "metadata": {},
        "derived_from_lane": "duplicate_frontier",
    }
    inventory = convergence_drain.Inventory(
        keys_by_source={
            "duplicate_frontier": {source: {derived_key}},
            "autonomy_duplicate_resolution": {source: {derived_key}},
        },
        payloads={},
        indeterminate_sources=set(),
        derived_items=[derived],
    )
    raced: dict[str, dict] = {}

    def inventory_with_concurrent_terminal(_rows):
        outside = store.merge_item(
            lane="autonomy_duplicate_resolution",
            source_id=source,
            input_data={"pair": ["left", "right"], "version": 4},
            resolver_version="v4",
        )["item"]
        store.complete_many(
            [str(outside["key"])],
            "rejected",
            result={"reason": "completed during inventory"},
        )
        raced["item"] = outside
        return inventory

    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        inventory_with_concurrent_terminal,
    )
    started = convergence_drain.start(store=store, run_once=False)
    manifest = convergence_drain._read_manifest(started["run_id"])

    assert manifest["derived_items"][0]["source_key_baseline"] == []
    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "derived_source_changed_before_merge"
    assert store.get(str(raced["item"]["key"]))["status"] == "rejected"
    assert store.get(str(legacy["key"]))["status"] == "pending_local"
    assert store.get(derived_key) is None


def test_derived_merge_allows_terminal_source_key_present_at_snapshot(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    source = "left<->right"
    legacy = _add(store, lane="duplicate_frontier", source=source)
    prior = store.merge_item(
        lane="autonomy_duplicate_resolution",
        source_id=source,
        input_data={"pair": ["left", "right"], "version": 2},
        resolver_version="v2",
    )["item"]
    store.complete_many(
        [str(prior["key"])],
        "rejected",
        result={"reason": "completed before snapshot"},
    )
    input_data = {"pair": ["left", "right"], "version": 3}
    derived_key = convergence_drain.stable_item_key(
        "autonomy_duplicate_resolution",
        source,
        input_data,
        resolver_version="v3",
    )
    derived = {
        "key": derived_key,
        "lane": "autonomy_duplicate_resolution",
        "source_id": source,
        "input_hash": convergence_drain.input_fingerprint(input_data),
        "input_data": input_data,
        "resolver_version": "v3",
        "metadata": {},
        "derived_from_lane": "duplicate_frontier",
    }
    inventory = convergence_drain.Inventory(
        keys_by_source={
            "duplicate_frontier": {source: {derived_key}},
            "autonomy_duplicate_resolution": {source: {derived_key}},
        },
        payloads={},
        indeterminate_sources=set(),
        derived_items=[derived],
    )
    monkeypatch.setattr(convergence_drain, "_build_inventory", lambda _rows: inventory)

    planned = convergence_drain.plan(store=store)

    assert planned["derived_items"][0]["source_key_baseline"] == [prior["key"]]
    started = convergence_drain.start(store=store, run_once=False)

    def finish_derived(*, store, **_kwargs):
        store.complete_many([derived_key], "applied", result={"reason": "processed"})
        return {"lanes": {}, "budget": {}}

    monkeypatch.setattr(convergence_drain, "_run_lanes", finish_derived)
    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "completed"
    assert store.get(str(prior["key"]))["status"] == "rejected"
    assert store.get(str(legacy["key"]))["status"] == "rejected"
    assert store.get(derived_key)["status"] == "applied"


def test_derived_merge_atomically_requires_snapshotted_source_history(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    source = "left<->right"
    legacy = _add(store, lane="duplicate_frontier", source=source)
    prior = store.merge_item(
        lane="autonomy_duplicate_resolution",
        source_id=source,
        input_data={"pair": ["left", "right"], "version": 2},
        resolver_version="v2",
    )["item"]
    store.complete_many([str(prior["key"])], "rejected", result={"reason": "old"})
    input_data = {"pair": ["left", "right"], "version": 3}
    derived_key = convergence_drain.stable_item_key(
        "autonomy_duplicate_resolution",
        source,
        input_data,
        resolver_version="v3",
    )
    derived = {
        "key": derived_key,
        "lane": "autonomy_duplicate_resolution",
        "source_id": source,
        "input_hash": convergence_drain.input_fingerprint(input_data),
        "input_data": input_data,
        "resolver_version": "v3",
        "metadata": {},
        "derived_from_lane": "duplicate_frontier",
    }
    inventory = convergence_drain.Inventory(
        keys_by_source={
            "duplicate_frontier": {source: {derived_key}},
            "autonomy_duplicate_resolution": {source: {derived_key}},
        },
        payloads={},
        indeterminate_sources=set(),
        derived_items=[derived],
    )
    monkeypatch.setattr(convergence_drain, "_build_inventory", lambda _rows: inventory)
    started = convergence_drain.start(store=store, run_once=False)

    state = store.load()
    state["items"].pop(str(prior["key"]))
    store._save_unlocked(state)
    list_items = store.list_items

    def stale_list_items(**kwargs):
        rows = list_items(**kwargs)
        if kwargs.get("lane") == "autonomy_duplicate_resolution":
            rows.append(prior)
        return rows

    # The optimistic scan sees stale history; the locked merge must still
    # detect that the manifest baseline key disappeared before creation.
    monkeypatch.setattr(store, "list_items", stale_list_items)
    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "derived_source_changed_before_merge"
    assert store.get(str(legacy["key"]))["status"] == "pending_local"
    assert store.get(derived_key) is None


def test_existing_derived_item_still_detects_newer_out_of_scope_source(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    legacy = _add(store, lane="duplicate_frontier", source="left<->right")
    input_data = {"pair": ["left", "right"], "version": 3}
    derived_key = convergence_drain.stable_item_key(
        "autonomy_duplicate_resolution",
        "left<->right",
        input_data,
        resolver_version="v3",
    )
    derived = {
        "key": derived_key,
        "lane": "autonomy_duplicate_resolution",
        "source_id": "left<->right",
        "input_hash": convergence_drain.input_fingerprint(input_data),
        "input_data": input_data,
        "resolver_version": "v3",
        "metadata": {},
        "derived_from_lane": "duplicate_frontier",
    }
    inventory = convergence_drain.Inventory(
        keys_by_source={
            "duplicate_frontier": {"left<->right": {derived_key}},
            "autonomy_duplicate_resolution": {"left<->right": {derived_key}},
        },
        payloads={},
        indeterminate_sources=set(),
        derived_items=[derived],
    )
    monkeypatch.setattr(convergence_drain, "_build_inventory", lambda _rows: inventory)
    started = convergence_drain.start(store=store, run_once=False)
    manifest = convergence_drain._read_manifest(started["run_id"])
    first_merge = convergence_drain._merge_derived_items(manifest, store)
    assert first_merge["created"] == [derived_key]
    outside = store.merge_item(
        lane="autonomy_duplicate_resolution",
        source_id="left<->right",
        input_data={"pair": ["left", "right"], "version": 4},
        resolver_version="v4",
    )["item"]

    result = convergence_drain.resume(run_id=started["run_id"], store=store)

    assert result["status"] == "failed"
    assert result["failure_reason"] == "derived_source_changed_before_merge"
    assert store.get(str(outside["key"]))["status"] == "pending_local"
    assert store.get(derived_key)["status"] == "rejected"
    assert store.get(str(legacy["key"]))["status"] == "pending_local"


def test_source_superseded_reason_is_distinct(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    item = _add(store, source="same-source")
    inventory = convergence_drain.Inventory(
        keys_by_source={"content_correction": {"same-source": {"replacement-key"}}},
        payloads={},
        indeterminate_sources=set(),
        derived_items=[],
    )
    monkeypatch.setattr(convergence_drain, "_build_inventory", lambda _items: inventory)
    monkeypatch.setattr(
        convergence_drain,
        "_run_lanes",
        lambda **_kwargs: {"lanes": {}, "budget": {}},
    )

    result = convergence_drain.start(store=store)

    assert result["status"] == "completed"
    assert store.get(str(item["key"]))["result"] == {
        "reason": "targeted_drain_source_superseded"
    }


def test_expired_lease_reap_is_limited_to_manifest_keys(
    tmp_path, monkeypatch, isolated_drain
) -> None:
    store = _store(tmp_path)
    target = _add(store, source="target")
    store.claim_attempt(str(target["key"]), "local", lease_seconds=0)
    monkeypatch.setattr(
        convergence_drain,
        "_build_inventory",
        lambda items: _inventory(list(items)),
    )
    started = convergence_drain.start(store=store, run_once=False)
    outside = _add(store, source="outside")
    store.claim_attempt(str(outside["key"]), "local", lease_seconds=0)
    monkeypatch.setattr(
        convergence_drain,
        "_run_lanes",
        lambda **_kwargs: {"lanes": {}, "budget": {}},
    )

    convergence_drain.resume(run_id=started["run_id"], store=store)

    assert store.get(str(target["key"]))["status"] == "local_retry"
    assert store.get(str(outside["key"]))["status"] == "local_running"


def test_source_retirement_is_limited_to_explicit_keys(tmp_path) -> None:
    store = _store(tmp_path)
    target = _add(store, lane="orphan_link", source="orphan:target")
    outside = _add(store, lane="orphan_link", source="orphan:outside")

    result = store.retire_absent_sources(
        lane="orphan_link",
        active_source_ids=set(),
        eligible_keys={str(target["key"])},
    )

    assert result["retired"] == [target["key"]]
    assert store.get(str(target["key"]))["status"] == "rejected"
    assert store.get(str(outside["key"]))["status"] == "pending_local"


def test_local_only_environment_is_restored(monkeypatch) -> None:
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_SYSTEM_CODE_REPAIR", "on")
    monkeypatch.delenv("CHRONOVISOR_SELF_HEAL_AUTORUN", raising=False)

    with convergence_drain._local_only_environment():
        assert os.environ["CHRONOVISOR_DECISION_POLICY_SYSTEM_CODE_REPAIR"] == "off"
        assert os.environ["CHRONOVISOR_SELF_HEAL_AUTORUN"] == "0"

    assert os.environ["CHRONOVISOR_DECISION_POLICY_SYSTEM_CODE_REPAIR"] == "on"
    assert "CHRONOVISOR_SELF_HEAL_AUTORUN" not in os.environ


def test_lane_dispatch_passes_exact_per_lane_allowlists(tmp_path, monkeypatch) -> None:
    from chronovisor.ops import autonomy, lint_repair, orphan_link
    from chronovisor.recall import content_correction

    store = _store(tmp_path)
    items = {
        lane: _add(store, lane=lane, source=f"source-{index}")
        for index, lane in enumerate(convergence_drain.PROCESSOR_LANES)
    }
    calls: dict[str, dict] = {}

    def capture(name):
        def run(*_args, **kwargs):
            calls[name] = kwargs
            return {"status": "ok"}

        return run

    monkeypatch.setattr(
        content_correction, "run_pending_corrections", capture("content_correction")
    )
    monkeypatch.setattr(
        autonomy,
        "resolve_deferred_duplicates_with_frontier",
        capture("autonomy_duplicate_resolution"),
    )
    monkeypatch.setattr(lint_repair, "run_lint_repair", capture("lint_repair"))
    monkeypatch.setattr(orphan_link, "run_autonomous", capture("orphan_link"))
    monkeypatch.setattr(
        autonomy, "apply_retention_archives", capture("autonomy_retention")
    )
    inventory = convergence_drain.Inventory(
        keys_by_source={},
        payloads={
            "autonomy_duplicate_resolution": [],
            "lint_repair": tmp_path / "queue.jsonl",
            "autonomy_retention": {},
        },
        indeterminate_sources=set(),
        derived_items=[],
    )

    convergence_drain._run_lanes(
        store=store,
        eligible_keys={str(item["key"]) for item in items.values()},
        inventory=inventory,
        max_elapsed_seconds=30,
    )

    assert set(calls) == set(convergence_drain.PROCESSOR_LANES)
    for lane, item in items.items():
        assert calls[lane]["eligible_keys"] == {item["key"]}
    assert calls["content_correction"]["max_items"] == 6
    assert calls["lint_repair"]["max_items"] == 200
    assert calls["orphan_link"]["orphan_limit"] == 2
    assert calls["autonomy_retention"]["limit"] == 3


def test_cli_exposes_plan_start_resume_and_status(monkeypatch, capsys) -> None:
    from chronovisor.hosts import cli

    calls: list[tuple[str, dict]] = []

    def fake(name, payload):
        def run(**kwargs):
            calls.append((name, kwargs))
            return payload

        return run

    monkeypatch.setattr(
        convergence_drain,
        "plan",
        fake("plan", {"status": "planned", "active_keys": 174}),
    )
    monkeypatch.setattr(
        convergence_drain,
        "start",
        fake("start", {"status": "running", "run_id": "run-1"}),
    )
    monkeypatch.setattr(
        convergence_drain,
        "resume",
        fake("resume", {"status": "completed", "run_id": "run-1"}),
    )
    monkeypatch.setattr(
        convergence_drain,
        "status",
        fake("status", {"status": "completed", "run_id": "run-1"}),
    )

    assert cli.main(["convergence-drain", "plan", "--json"]) == 0
    assert (
        cli.main(
            [
                "convergence-drain",
                "start",
                "--max-elapsed-seconds",
                "60",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    assert cli.main(["convergence-drain", "resume", "--run-id", "run-1", "--json"]) == 0
    assert cli.main(["convergence-drain", "status", "--run-id", "run-1", "--json"]) == 0

    assert calls == [
        ("plan", {}),
        ("start", {"max_elapsed_seconds": 60.0, "dry_run": True}),
        ("resume", {"run_id": "run-1", "dry_run": False}),
        ("status", {"run_id": "run-1"}),
    ]
    output = capsys.readouterr().out
    assert '"planned"' in output
    assert '"running"' in output
    assert output.count('"completed"') == 2


def test_targeted_runner_has_no_broad_or_frontier_entrypoints() -> None:
    source = inspect.getsource(convergence_drain)

    assert "run_sleep_cycle" not in source
    assert "enqueue_due_system_repairs" not in source
    assert "run_pending(" not in source
    assert "run_frontier_review" not in source
    assert "_run_codex" not in source
