from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.decision.semantic_hold import persisted_semantic_no_quorum_hold
from chronovisor.ops.convergence import (
    ConvergenceStateError,
    ConvergenceStore,
    CycleBudget,
    InvalidTransition,
    RetryPolicy,
    exponential_backoff_seconds,
    is_human_required_failure,
    is_human_required_result,
    stable_item_key,
)
from tests.semantic_hold_support import semantic_authority, semantic_review

NOW = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)


def _store(tmp_path: Path, *, policy: RetryPolicy | None = None) -> ConvergenceStore:
    return ConvergenceStore(
        tmp_path / "runtime" / "convergence" / "state.json", policy=policy
    )


def _merge(
    store: ConvergenceStore, *, source_id: str = "item-1", input_data=None
) -> dict:
    result = store.merge_item(
        lane="test",
        source_id=source_id,
        input_data={"value": 1} if input_data is None else input_data,
        resolver_version="v1",
        now=NOW,
    )
    return result["item"]


def test_stable_key_is_order_independent_and_versioned() -> None:
    left = stable_item_key(
        "lint",
        "page-a",
        {"tags": {"t/topic", "d/domain"}, "detail": {"b": 2, "a": 1}},
        resolver_version="v1",
    )
    right = stable_item_key(
        "lint",
        "page-a",
        {"detail": {"a": 1, "b": 2}, "tags": {"d/domain", "t/topic"}},
        resolver_version="v1",
    )

    assert left == right
    assert left != stable_item_key(
        "lint",
        "page-a",
        {"detail": {"a": 1, "b": 3}, "tags": {"d/domain", "t/topic"}},
        resolver_version="v1",
    )
    assert left != stable_item_key(
        "lint",
        "page-a",
        {"detail": {"a": 1, "b": 2}, "tags": {"d/domain", "t/topic"}},
        resolver_version="v2",
    )


def test_merge_preserves_terminal_state_and_changed_input_reopens(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    item = _merge(store)
    key = item["key"]
    store.complete(key, "applied", result={"page": "page-a"}, now=NOW)

    merged_again = store.merge_item(
        lane="test",
        source_id="item-1",
        input_data={"value": 1},
        resolver_version="v1",
        now=NOW + timedelta(days=1),
    )
    changed_input = store.merge_item(
        lane="test",
        source_id="item-1",
        input_data={"value": 2},
        resolver_version="v1",
        now=NOW + timedelta(days=1),
    )

    assert merged_again["created"] is False
    assert merged_again["changed"] is False
    assert merged_again["item"]["status"] == "applied"
    assert merged_again["item"]["result"] == {"page": "page-a"}
    assert changed_input["created"] is True
    assert changed_input["item"]["status"] == "pending_local"
    assert changed_input["item"]["key"] != key


def test_changed_input_retires_obsolete_nonterminal_item(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = _merge(store, source_id="same-source", input_data={"value": 1})

    replacement = store.merge_item(
        lane="test",
        source_id="same-source",
        input_data={"value": 2},
        resolver_version="v1",
        now=NOW + timedelta(minutes=1),
    )

    assert replacement["retired"] == [old["key"]]
    assert store.get(old["key"])["status"] == "rejected"
    assert store.get(old["key"])["result"]["reason"] == "superseded_by_new_input"
    assert replacement["item"]["status"] == "pending_local"


def test_complete_source_inventory_retires_disappeared_items(tmp_path: Path) -> None:
    store = _store(tmp_path)
    keep = _merge(store, source_id="keep")
    gone = _merge(store, source_id="gone")

    result = store.retire_absent_sources(
        lane="test",
        active_source_ids={"keep"},
        now=NOW + timedelta(hours=1),
    )

    assert result["retired"] == [gone["key"]]
    assert store.get(keep["key"])["status"] == "pending_local"
    assert store.get(gone["key"])["status"] == "rejected"


def test_stale_retirement_is_atomic_and_preserves_fresh_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    old = _merge(store, source_id="old")
    current_time = NOW + timedelta(days=8)
    fresh = store.merge_item(
        lane="test",
        source_id="fresh",
        input_data={"value": 1},
        resolver_version="v1",
        now=current_time,
    )["item"]

    def fail_unlocked_snapshot(**_: object) -> list[dict]:
        pytest.fail("retire_stale must not build an unlocked list_items snapshot")

    monkeypatch.setattr(store, "list_items", fail_unlocked_snapshot)
    result = store.retire_stale(
        lane="test",
        max_age_seconds=7 * 24 * 60 * 60,
        now=current_time,
    )

    assert result == {"retired": [old["key"]], "dry_run": False}
    assert store.get(old["key"])["status"] == "rejected"
    assert store.get(old["key"])["result"] == {"reason": "stale_source_ttl"}
    assert store.get(fresh["key"])["status"] == "pending_local"


def test_local_attempts_back_off_then_escalate(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        policy=RetryPolicy(
            max_local_attempts=2,
            max_frontier_attempts=3,
            local_base_delay_seconds=10,
            frontier_base_delay_seconds=20,
            max_delay_seconds=100,
            lease_seconds=30,
        ),
    )
    key = _merge(store)["key"]

    first = store.claim_attempt(key, "local", owner="worker-1", now=NOW)
    failed = store.fail_attempt(
        key,
        "local",
        owner="worker-1",
        error="temporary local error",
        failure_class="local_transient",
        now=NOW,
    )
    early = store.claim_attempt(
        key, "local", owner="worker-2", now=NOW + timedelta(seconds=9)
    )
    second = store.claim_attempt(
        key, "local", owner="worker-2", now=NOW + timedelta(seconds=10)
    )
    escalated = store.fail_attempt(
        key,
        "local",
        owner="worker-2",
        error="still unresolved",
        failure_class="local_uncertain",
        now=NOW + timedelta(seconds=10),
    )

    assert first["claimed"] is True
    assert failed["item"]["status"] == "local_retry"
    assert failed["item"]["next_attempt_at"] == "2026-07-10T12:00:10+00:00"
    assert early["claimed"] is False
    assert early["reason"] == "backoff"
    assert second["claimed"] is True
    assert second["item"]["local_attempts"] == 2
    assert escalated["item"]["status"] == "pending_frontier"
    assert escalated["item"]["next_attempt_at"] is None


def test_foreground_preemption_requeues_without_consuming_attempt(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path,
        policy=RetryPolicy(
            max_local_attempts=1,
            max_frontier_attempts=1,
            local_base_delay_seconds=10,
            lease_seconds=30,
        ),
    )
    key = store.merge_item(
        lane="librarian_classify",
        source_id="batch:0",
        input_data={"sha256": "same"},
        resolver_version="v1",
    )["item"]["key"]
    claim = store.claim_attempt(key, "local", owner="worker", now=NOW)
    assert claim["item"]["local_attempts"] == 1

    preempted = store.fail_attempt(
        key,
        "local",
        owner="worker",
        error="foreground recall",
        failure_class="foreground_preempted",
        allow_frontier=False,
        consume_attempt=False,
        now=NOW,
    )

    assert preempted["item"]["status"] == "pending_local"
    assert preempted["item"]["local_attempts"] == 0
    assert preempted["item"]["next_attempt_at"] is None
    retried = store.claim_attempt(key, "local", owner="worker-2", now=NOW)
    assert retried["claimed"] is True
    assert retried["item"]["local_attempts"] == 1


def test_frontier_attempts_are_bounded_and_terminally_quarantined(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path,
        policy=RetryPolicy(
            max_local_attempts=1,
            max_frontier_attempts=3,
            local_base_delay_seconds=1,
            frontier_base_delay_seconds=10,
            max_delay_seconds=100,
            lease_seconds=30,
        ),
    )
    key = _merge(store)["key"]
    store.escalate(key, reason="local model uncertain", now=NOW)

    at = NOW
    expected_delays = [10, 20]
    for attempt, delay in enumerate(expected_delays, start=1):
        claim = store.claim_attempt(
            key, "frontier", owner=f"frontier-{attempt}", now=at
        )
        failed = store.fail_attempt(
            key,
            "frontier",
            owner=f"frontier-{attempt}",
            error="frontier transient",
            failure_class="network_transient",
            now=at,
        )
        assert claim["claimed"] is True
        assert failed["item"]["status"] == "frontier_retry"
        assert failed["item"]["next_attempt_at"] == (
            at + timedelta(seconds=delay)
        ).isoformat(timespec="seconds")
        at += timedelta(seconds=delay)

    third = store.claim_attempt(key, "frontier", owner="frontier-3", now=at)
    quarantined = store.fail_attempt(
        key,
        "frontier",
        owner="frontier-3",
        error="frontier still unavailable",
        failure_class="network_transient",
        now=at,
    )

    assert third["claimed"] is True
    assert quarantined["item"]["frontier_attempts"] == 3
    assert quarantined["item"]["status"] == "quarantined"
    assert quarantined["item"]["quarantine_reason"] == "retry_exhausted:frontier"
    assert store.claim_attempt(key, "frontier", now=at)["reason"] == "terminal"


def test_only_external_access_failures_cross_human_boundary(tmp_path: Path) -> None:
    assert is_human_required_failure("auth_required") is True
    assert is_human_required_failure("secret_store_permission_required") is True
    assert is_human_required_failure("frontier_tool_unavailable") is False
    assert is_human_required_failure("both_frontiers_unavailable") is False
    assert is_human_required_failure("schema_invalid") is False
    assert is_human_required_failure("model_asked_for_human") is False
    assert is_human_required_result({"human_required": True}) is False
    assert (
        is_human_required_result(
            {
                "human_required": False,
                "frontier_failure": {"failure_class": "keychain_permission_required"},
            }
        )
        is True
    )

    store = _store(tmp_path)
    schema_key = _merge(store, source_id="schema")["key"]
    store.escalate(schema_key, reason="needs structured review", now=NOW)
    store.claim_attempt(schema_key, "frontier", owner="f-schema", now=NOW)
    schema_failure = store.fail_attempt(
        schema_key,
        "frontier",
        owner="f-schema",
        error="invalid schema",
        failure_class="schema_invalid",
        now=NOW,
    )

    auth_key = _merge(store, source_id="auth")["key"]
    store.escalate(auth_key, reason="needs structured review", now=NOW)
    store.claim_attempt(auth_key, "frontier", owner="f-auth", now=NOW)
    auth_failure = store.fail_attempt(
        auth_key,
        "frontier",
        owner="f-auth",
        error="401 unauthorized",
        failure_class="auth_required",
        now=NOW,
    )

    assert schema_failure["item"]["status"] == "frontier_retry"
    assert schema_failure["item"]["human_required"] is False
    assert auth_failure["item"]["status"] == "human_required"
    assert auth_failure["item"]["human_required"] is True
    assert auth_failure["item"]["next_attempt_at"] is None


def test_human_required_resumes_only_with_capability_fingerprint(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    key = _merge(store)["key"]
    store.escalate(key, reason="frontier needed", now=NOW)
    store.claim_attempt(key, "frontier", owner="frontier", now=NOW)
    store.fail_attempt(
        key,
        "frontier",
        owner="frontier",
        error="keychain denied",
        failure_class="keychain_permission_required",
        now=NOW,
    )

    with pytest.raises(ValueError):
        store.resume_human_required(key, capability_fingerprint="", now=NOW)

    resumed = store.resume_human_required(
        key,
        capability_fingerprint="codex-auth-mtime:20260710T121500",
        now=NOW + timedelta(minutes=15),
    )

    assert resumed["item"]["status"] == "pending_frontier"
    assert resumed["item"]["human_required"] is False
    assert resumed["item"]["frontier_attempts"] == 0
    assert resumed["item"]["capability_fingerprint"].startswith("codex-auth-mtime")


def test_due_human_required_is_resumed_after_shared_preflight(tmp_path: Path) -> None:
    store = _store(tmp_path)
    key = _merge(store)["key"]
    store.escalate(key, reason="frontier needed", now=NOW)
    store.claim_attempt(key, "frontier", owner="frontier", now=NOW)
    store.fail_attempt(
        key,
        "frontier",
        owner="frontier",
        error="oauth login required",
        failure_class="oauth_required",
        now=NOW,
    )

    not_ready = store.resume_due_human_required(
        capability_fingerprint="",
        cooldown_seconds=0,
        now=NOW + timedelta(hours=1),
    )
    resumed = store.resume_due_human_required(
        capability_fingerprint="frontier-preflight:ok",
        cooldown_seconds=60,
        now=NOW + timedelta(hours=1),
    )

    assert not_ready["status"] == "preflight_not_ready"
    assert not_ready["resumed"] == 0
    assert resumed["resumed"] == 1
    assert store.get(key)["status"] == "pending_frontier"
    assert store.get(key)["capability_fingerprint"] == "frontier-preflight:ok"


def test_merge_reopens_legacy_non_external_human_required_item(tmp_path: Path) -> None:
    store = _store(tmp_path)
    key = _merge(store)["key"]
    store.escalate(key, reason="frontier needed", now=NOW)
    store.claim_attempt(key, "frontier", owner="frontier", now=NOW)
    store.fail_attempt(
        key,
        "frontier",
        owner="frontier",
        error="auth required",
        failure_class="auth_required",
        now=NOW,
    )
    state = store.load()
    state["items"][key]["last_failure_class"] = "frontier_tool_unavailable"
    store.state_file.write_text(json.dumps(state), encoding="utf-8")

    merged = store.merge_item(
        lane="test",
        source_id="item-1",
        input_data={"value": 1},
        resolver_version="v1",
        now=NOW,
    )

    assert merged["reclassified_human_boundary"] is True
    assert merged["item"]["status"] == "frontier_retry"
    assert merged["item"]["human_required"] is False
    assert merged["item"]["frontier_attempts"] == 1


def test_active_lease_blocks_duplicate_worker_and_expiry_counts_crash(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path,
        policy=RetryPolicy(
            max_local_attempts=2, max_frontier_attempts=1, lease_seconds=30
        ),
    )
    key = _merge(store)["key"]

    first = store.claim_attempt(key, "local", owner="worker-1", now=NOW)
    blocked = store.claim_attempt(
        key, "local", owner="worker-2", now=NOW + timedelta(seconds=29)
    )
    recovered = store.claim_attempt(
        key, "local", owner="worker-2", now=NOW + timedelta(seconds=30)
    )

    assert first["claimed"] is True
    assert blocked["claimed"] is False
    assert blocked["reason"] == "leased"
    assert blocked["item"]["local_attempts"] == 1
    assert recovered["claimed"] is True
    assert recovered["item"]["local_attempts"] == 2
    with pytest.raises(InvalidTransition):
        store.fail_attempt(
            key,
            "local",
            owner="worker-1",
            error="late result from stale lease",
            now=NOW + timedelta(seconds=31),
        )


def test_running_transition_requires_matching_lease_owner(tmp_path: Path) -> None:
    store = _store(tmp_path)
    key = _merge(store)["key"]
    claim = store.claim_attempt(key, "local", owner="worker-1", now=NOW)
    assert claim["claimed"] is True

    with pytest.raises(InvalidTransition):
        store.complete(key, "applied", now=NOW)
    with pytest.raises(InvalidTransition):
        store.complete(key, "applied", owner="worker-2", now=NOW)

    completed = store.complete(key, "applied", owner="worker-1", now=NOW)
    assert completed["item"]["status"] == "applied"


def test_complete_many_rewrites_state_once_and_skips_active_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    first = _merge(store, source_id="first")["key"]
    second = _merge(store, source_id="second")["key"]
    leased = _merge(store, source_id="leased")["key"]
    store.claim_attempt(leased, "local", owner="worker", now=NOW)
    original_save = store._save_unlocked
    saves = 0

    def counted_save(state: dict) -> None:
        nonlocal saves
        saves += 1
        original_save(state)

    monkeypatch.setattr(store, "_save_unlocked", counted_save)

    result = store.complete_many(
        [first, second, leased, "missing"],
        "rejected",
        result={"migration": "test"},
        now=NOW + timedelta(minutes=1),
    )

    assert result == {
        "status": "ok",
        "dry_run": False,
        "requested": 4,
        "completed": 2,
        "skipped": 2,
        "skipped_reasons": {"leased": 1, "missing": 1},
    }
    assert saves == 1
    assert store.get(first)["status"] == "rejected"
    assert store.get(second)["result"] == {"migration": "test"}
    assert store.get(leased)["status"] == "local_running"
    completed_events = [
        row
        for row in (
            json.loads(line)
            for line in store.events_file.read_text(encoding="utf-8").splitlines()
        )
        if row.get("event") == "completed"
    ]
    assert {row["key"] for row in completed_events} == {first, second}


def test_expired_crash_at_limit_routes_to_frontier_without_another_call(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path,
        policy=RetryPolicy(
            max_local_attempts=1, max_frontier_attempts=1, lease_seconds=10
        ),
    )
    key = _merge(store)["key"]
    store.claim_attempt(key, "local", owner="crashed", now=NOW)

    recovered = store.claim_attempt(
        key, "local", owner="new-worker", now=NOW + timedelta(seconds=10)
    )

    assert recovered["claimed"] is False
    assert recovered["reason"] == "retry_exhausted"
    assert recovered["item"]["status"] == "pending_frontier"
    assert recovered["item"]["local_attempts"] == 1


def test_cycle_budget_bounds_calls_mutations_bytes_and_elapsed(tmp_path: Path) -> None:
    ticks = [100.0]
    budget = CycleBudget(
        max_local_calls=1,
        max_frontier_calls=1,
        max_mutations=1,
        max_raw_bytes=100,
        max_elapsed_seconds=10,
        clock=lambda: ticks[0],
    )
    store = _store(tmp_path)
    first_key = _merge(store, source_id="first")["key"]
    second_key = _merge(store, source_id="second")["key"]

    first = store.claim_attempt(first_key, "local", owner="one", budget=budget, now=NOW)
    second = store.claim_attempt(
        second_key, "local", owner="two", budget=budget, now=NOW
    )

    assert first["claimed"] is True
    assert second["claimed"] is False
    assert second["reason"] == "local_budget_exhausted"
    assert budget.consume("mutation") == (True, "ok")
    assert budget.consume("mutation") == (False, "mutation_budget_exhausted")
    assert budget.consume("raw_bytes", 100) == (True, "ok")
    assert budget.consume("raw_bytes", 1) == (False, "raw_bytes_budget_exhausted")
    ticks[0] = 110.0
    assert budget.can_consume("frontier") == (False, "elapsed_budget_exhausted")


def test_budget_slices_prevent_one_lane_from_starving_reserved_peers() -> None:
    parent = CycleBudget(max_frontier_calls=2, max_elapsed_seconds=60)
    first = parent.slice(max_frontier_calls=1)
    second = parent.slice(max_frontier_calls=1)

    assert first.consume("frontier") == (True, "ok")
    assert first.consume("frontier") == (False, "frontier_lane_budget_exhausted")
    assert second.consume("frontier") == (True, "ok")
    assert parent.snapshot()["used"]["frontier"] == 2


def test_dry_run_does_not_create_or_change_any_files(tmp_path: Path) -> None:
    missing_store = _store(tmp_path / "missing")
    preview = missing_store.merge_item(
        lane="raw_replay",
        source_id="raw.md",
        input_data={"sha256": "abc"},
        dry_run=True,
        now=NOW,
    )

    assert preview["created"] is True
    assert not missing_store.state_file.exists()
    assert not missing_store.events_file.exists()
    assert not missing_store.lock_file.exists()

    store = _store(tmp_path / "existing")
    key = _merge(store)["key"]
    state_before = store.state_file.read_bytes()
    events_before = store.events_file.read_bytes()
    budget = CycleBudget(max_local_calls=1)

    claimed = store.claim_attempt(key, "local", budget=budget, dry_run=True, now=NOW)
    completed = store.complete(key, "rejected", dry_run=True, now=NOW)

    assert claimed["claimed"] is True
    assert claimed["item"]["local_attempts"] == 1
    assert completed["item"]["status"] == "rejected"
    assert budget.snapshot()["used"]["local"] == 0
    assert store.state_file.read_bytes() == state_before
    assert store.events_file.read_bytes() == events_before

    live_claim = store.claim_attempt(key, "local", owner="dry-run-failure", now=NOW)
    assert live_claim["claimed"] is True
    state_before_failure = store.state_file.read_bytes()
    events_before_failure = store.events_file.read_bytes()
    failure_preview = store.fail_attempt(
        key,
        "local",
        owner="dry-run-failure",
        error="preview only",
        failure_class="network_transient",
        dry_run=True,
        now=NOW,
    )
    assert failure_preview["item"]["status"] == "local_retry"
    assert store.state_file.read_bytes() == state_before_failure
    assert store.events_file.read_bytes() == events_before_failure


def test_corrupt_state_fails_closed_without_overwrite(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.state_file.parent.mkdir(parents=True)
    store.state_file.write_text("{not-json", encoding="utf-8")
    before = store.state_file.read_bytes()

    with pytest.raises(ConvergenceStateError):
        store.merge_item(
            lane="lint",
            source_id="page-a",
            input_data={"issue": "orphan"},
            now=NOW,
        )

    assert store.state_file.read_bytes() == before


def test_explicit_quarantine_is_terminal_and_completion_is_idempotent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    quarantine_key = _merge(store, source_id="unsafe")["key"]
    quarantined = store.quarantine(
        quarantine_key, reason="CAS verification failed", now=NOW
    )

    assert quarantined["item"]["status"] == "quarantined"
    assert quarantined["item"]["quarantine_reason"] == "CAS verification failed"
    with pytest.raises(InvalidTransition):
        store.complete(quarantine_key, "applied", now=NOW)

    applied_key = _merge(store, source_id="safe")["key"]
    first = store.complete(applied_key, "applied", result={"writes": 1}, now=NOW)
    second = store.complete(applied_key, "applied", result={"writes": 999}, now=NOW)

    assert first["item"]["result"] == {"writes": 1}
    assert second["item"]["result"] == {"writes": 1}
    assert (
        json.loads(store.state_file.read_text())["items"][applied_key]["status"]
        == "applied"
    )


def test_due_nonhuman_quarantine_is_reopened_without_resetting_other_lanes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    retry_key = _merge(store, source_id="retryable")["key"]
    excluded_key = _merge(store, source_id="content")["key"]
    state = store.load()
    state["items"][retry_key].update(
        {
            "lane": "lint_repair",
            "status": "quarantined",
            "quarantine_reason": "retry_exhausted:frontier",
            "last_failure_class": "network_transient",
            "updated_at": NOW.isoformat(),
        }
    )
    state["items"][excluded_key].update(
        {
            "lane": "content_correction",
            "status": "quarantined",
            "quarantine_reason": "retry_exhausted:frontier",
            "updated_at": NOW.isoformat(),
        }
    )
    store.state_file.write_text(json.dumps(state), encoding="utf-8")

    result = store.resume_due_quarantined(
        cooldown_seconds=3600,
        exclude_lanes={"content_correction"},
        now=NOW + timedelta(hours=2),
    )

    assert result["resumed"] == 1
    assert store.get(retry_key)["status"] == "pending_frontier"
    assert store.get(excluded_key)["status"] == "quarantined"


def test_generic_cooldown_never_resamples_semantic_no_quorum(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    semantic_key = _merge(store, source_id="semantic-split")["key"]
    operational_key = _merge(store, source_id="operational-failure")["key"]
    state = store.load()
    state["items"][semantic_key].update(
        {
            "lane": "autonomy_duplicate_resolution",
            "status": "quarantined",
            "quarantine_reason": "retry_exhausted:frontier",
            "last_failure_class": "local_semantic_no_quorum",
            "updated_at": NOW.isoformat(),
        }
    )
    state["items"][operational_key].update(
        {
            "lane": "autonomy_duplicate_resolution",
            "status": "quarantined",
            "quarantine_reason": "retry_exhausted:frontier",
            "last_failure_class": "network_transient",
            "updated_at": NOW.isoformat(),
        }
    )
    store.state_file.write_text(json.dumps(state), encoding="utf-8")

    result = store.resume_due_quarantined(
        cooldown_seconds=0,
        now=NOW + timedelta(days=1),
    )

    assert result["resumed"] == 1
    assert result["semantic_deferred"] == 1
    assert store.get(semantic_key)["status"] == "quarantined"
    assert store.get(operational_key)["status"] == "pending_frontier"


def test_semantic_no_quorum_is_immediately_terminal_and_dry_run_is_read_only(
    tmp_path: Path,
) -> None:
    lane = "recall_auto_apply"
    store = _store(tmp_path)
    key = _merge(store, source_id="semantic-hold")["key"]
    store.escalate(key, reason="semantic review required", now=NOW)
    claim = store.claim_attempt(key, "frontier", owner="worker", now=NOW)
    assert claim["claimed"] is True
    authority = semantic_authority(lane)
    review = semantic_review(authority, lane=lane)
    epoch = {"input_hash": store.get(key)["input_hash"], "prompt_sha256": "f" * 64}
    state_before = store.state_file.read_bytes()
    events_before = store.events_file.read_bytes()

    preview = store.hold_semantic_no_quorum(
        key,
        lane=lane,
        stage="frontier",
        review=review,
        epoch=epoch,
        authority=authority,
        owner="worker",
        now=NOW,
        dry_run=True,
    )

    assert preview["item"]["status"] == "quarantined"
    assert preview["item"]["frontier_attempts"] == 1
    assert store.state_file.read_bytes() == state_before
    assert store.events_file.read_bytes() == events_before

    terminal = store.hold_semantic_no_quorum(
        key,
        lane=lane,
        stage="frontier",
        review=review,
        epoch=epoch,
        authority=authority,
        owner="worker",
        now=NOW,
    )["item"]

    assert terminal["status"] == "quarantined"
    assert terminal["frontier_attempts"] == 1
    assert terminal["last_failure_class"] == "local_semantic_no_quorum"
    assert terminal["quarantine_reason"] == f"semantic_no_quorum:{lane}"
    assert (
        persisted_semantic_no_quorum_hold(
            terminal,
            lane=lane,
            epoch=epoch,
            authority=authority,
        )
        is not None
    )
    events = [json.loads(line) for line in store.events_file.read_text().splitlines()]
    assert events[-1]["event"] == "semantic_no_quorum_held"

    recovery = store.resume_due_quarantined(cooldown_seconds=0, now=NOW)
    assert recovery["resumed"] == 0
    assert recovery["semantic_deferred"] == 1
    assert store.get(key)["status"] == "quarantined"


def test_resumed_semantic_hold_is_restored_before_same_epoch_resampling(
    tmp_path: Path,
) -> None:
    lane = "recall_auto_apply"
    store = _store(tmp_path)
    key = _merge(store, source_id="semantic-aba")["key"]
    store.escalate(key, reason="semantic review required", now=NOW)
    store.claim_attempt(key, "frontier", owner="worker", now=NOW)
    authority_a = semantic_authority(lane)
    authority_b = semantic_authority(lane, artifact_sha256="9" * 64)
    review_a = semantic_review(authority_a, lane=lane)
    review_b = semantic_review(authority_b, lane=lane)
    epoch = {"input_hash": store.get(key)["input_hash"]}
    terminal = store.hold_semantic_no_quorum(
        key,
        lane=lane,
        stage="frontier",
        review=review_a,
        epoch=epoch,
        authority=authority_a,
        owner="worker",
        now=NOW,
    )["item"]
    hold = terminal["result"]["semantic_hold"]
    store.resume_quarantined(
        key,
        stage="frontier",
        reason="semantic hold authority changed",
        resume_context={
            "decision_lane": lane,
            "invalidated_semantic_hold": hold,
            "invalidated_hold_sha256": hold["hold_sha256"],
            "expected_epoch": epoch,
            "expected_epoch_sha256": hold["epoch_sha256"],
            "expected_authority": authority_b,
        },
        now=NOW + timedelta(seconds=1),
    )

    assert (
        store.restore_semantic_no_quorum_hold(
            key,
            lane=lane,
            epoch=epoch,
            authority=authority_b,
            now=NOW + timedelta(seconds=2),
        )
        is None
    )
    claim_b = store.claim_attempt(
        key,
        "frontier",
        owner="worker",
        now=NOW + timedelta(seconds=2),
    )
    assert claim_b["claimed"] is True
    terminal_b = store.hold_semantic_no_quorum(
        key,
        lane=lane,
        stage="frontier",
        review=review_b,
        epoch=epoch,
        authority=authority_b,
        owner="worker",
        now=NOW + timedelta(seconds=2),
    )["item"]
    hold_b = terminal_b["result"]["semantic_hold"]
    assert terminal_b["result"]["semantic_hold_history"] == [hold]
    store.resume_quarantined(
        key,
        stage="frontier",
        reason="semantic hold authority changed",
        resume_context={
            "decision_lane": lane,
            "invalidated_semantic_hold": hold_b,
            "invalidated_hold_sha256": hold_b["hold_sha256"],
            "expected_epoch": epoch,
            "expected_epoch_sha256": hold_b["epoch_sha256"],
            "expected_authority": authority_a,
        },
        now=NOW + timedelta(seconds=3),
    )
    resume_history = store.get(key)["result"]["resume_context"]["semantic_hold_history"]
    assert [entry["hold_sha256"] for entry in resume_history] == [
        hold["hold_sha256"],
        hold_b["hold_sha256"],
    ]
    restored = store.restore_semantic_no_quorum_hold(
        key,
        lane=lane,
        epoch=epoch,
        authority=authority_a,
        now=NOW + timedelta(seconds=4),
    )

    assert restored is not None
    assert restored["item"]["status"] == "quarantined"
    assert restored["item"]["result"]["semantic_hold"] == hold
    assert restored["item"]["result"]["semantic_hold_history"] == [hold_b]


def test_exponential_backoff_is_one_based_and_capped() -> None:
    assert exponential_backoff_seconds(1, base_seconds=5, max_seconds=30) == 5
    assert exponential_backoff_seconds(2, base_seconds=5, max_seconds=30) == 10
    assert exponential_backoff_seconds(10, base_seconds=5, max_seconds=30) == 30
    with pytest.raises(ValueError):
        exponential_backoff_seconds(0, base_seconds=5, max_seconds=30)
