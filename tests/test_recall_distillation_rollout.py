from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_rollout as rollout
from chronovisor.recall import recall_distillation_store as store


def _policy(root: Path, name: str) -> str:
    policy_id = hashlib.sha256(name.encode()).hexdigest()
    store.write_immutable(
        store.distillation_dir(root) / "policies",
        {
            "kind": "tiny-logistic-policy",
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "feature_keys": list(distill.FAST_FEATURE_KEYS),
            "weights": {key: 1.0 for key in distill.FAST_FEATURE_KEYS},
            "bias": 0.0,
            "threshold": 0.5,
            "abstain_margin": 0.0,
            "max_cards": 1,
        },
        schema=rollout.POLICY_SCHEMA,
        artifact_id=policy_id,
    )
    return policy_id


def _baseline(root: Path, *, eligible: bool = True) -> str:
    offline_gate = {"passed": eligible, "revision": "test-offline-gate-v2"}
    baseline_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "baselines",
        {
            "raw_watermark": "b" * 64,
            "hard_floor": {"p5_allowed": eligible},
            "offline_training_gate": offline_gate,
        },
        schema="chronovisor.recall-distill-baseline.v1",
    )
    return baseline_id


def _setup(
    root: Path, status: str = "replay", percent: int = 0
) -> tuple[str, str, str]:
    lkg, candidate = _policy(root, "lkg"), _policy(root, "candidate")
    store.write_pointer(root, "lkg", lkg)
    store.write_pointer(root, "active", lkg)
    store.write_pointer(root, "candidate", candidate)
    baseline = _baseline(root)
    (root / "config.toml").write_text("[recall.distillation]\nenabled = true\n")
    store.write_sealed_state(
        store.distillation_dir(root) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": status,
            "rollout_percent": percent,
            "stage_started_at": "2026-08-01T00:00:00Z",
        },
    )
    return lkg, candidate, baseline


def _gate(*, denominator: int = 500, minimum: int = 500) -> dict[str, object]:
    return {
        "denominator": denominator,
        "min_denominator": minimum,
        "min_days": 7,
        "ci_lower": 0.9,
        "min_ci_lower": 0.8,
    }


def _metrics(*, denominator: int = 500, ci_lower: float = 0.9) -> dict[str, object]:
    return {
        name: {**_gate(denominator=denominator), "ci_lower": ci_lower}
        for name in rollout._METRICS
    }


def _evaluation(
    root: Path,
    candidate: str,
    incumbent: str,
    baseline: str,
    name: str,
    *,
    denominator: int = 500,
    ci_lower: float = 0.9,
    omit_metric: str = "",
    observation_mode: str = "paired",
) -> dict[str, object]:
    run_id = hashlib.sha256(name.encode()).hexdigest()
    baseline_artifact = store.read_sealed(
        store.distillation_dir(root) / "baselines" / f"{baseline}.json",
        schema=distill.BASELINE_SCHEMA,
    )
    replay_metrics = _metrics(denominator=denominator, ci_lower=ci_lower)
    if omit_metric:
        replay_metrics.pop(omit_metric)
    artifact_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "evaluations",
        {
            "kind": "sealed-rollout-evaluation",
            "run_id": run_id,
            "policy_id": candidate,
            "baseline_id": baseline,
            "raw_watermark": "b" * 64,
            "incumbent_policy_id": incumbent,
            "split_sha256": "c" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "feature_parity_sha256": "d" * 64,
            "offline_gate_sha256": distill.canonical_json.canonical_json_sha256_strict(
                baseline_artifact["offline_training_gate"]
            ),
            "observation_mode": observation_mode,
            "replay_metrics": replay_metrics,
            "shadow_metrics": _metrics(denominator=denominator, ci_lower=ci_lower),
            "canary_metrics": _metrics(denominator=denominator, ci_lower=ci_lower),
        },
        schema=rollout.EVALUATION_SCHEMA,
    )
    return {"run_id": run_id, "evaluation_artifact_id": artifact_id}


def test_pass_path_is_replay_then_shadow_then_nested_canaries(tmp_path: Path) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    assert rollout.select_policy_id(tmp_path, "any-session") == lkg
    assert (
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-01T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "replay"),
        )["status"]
        == "shadow"
    )
    assert (
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-08T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "shadow"),
        )["rollout_percent"]
        == 5
    )
    assert (
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-15T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "five"),
        )["rollout_percent"]
        == 25
    )
    assert (
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-22T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "twenty-five"),
        )["rollout_percent"]
        == 100
    )
    assert (
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-29T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "hundred"),
        )["status"]
        == "active"
    )
    assert rollout.select_policy_id(tmp_path, "any-session") == candidate


def test_legacy_incumbent_allows_candidate_only_gate_only_at_100_percent(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.toml").write_text(
        "[recall.distillation]\nenabled = true\n", encoding="utf-8"
    )
    baseline = _baseline(tmp_path)
    baseline_artifact = store.read_sealed(
        store.distillation_dir(tmp_path) / "baselines" / f"{baseline}.json",
        schema=distill.BASELINE_SCHEMA,
    )
    legacy = distill._ensure_bootstrap_policy(tmp_path, baseline_artifact)
    candidate = distill.publish_policy(
        distill.train_tiny_policy([]),
        lineage={"baseline_artifact_id": baseline, "locked_replay_id": "c" * 64},
        root=tmp_path,
    )["artifact_id"]
    for now, name, expected in (
        ("2026-08-01T00:00:00Z", "legacy-replay", 0),
        ("2026-08-08T00:00:00Z", "legacy-shadow", 5),
        ("2026-08-15T00:00:00Z", "legacy-five", 25),
        ("2026-08-22T00:00:00Z", "legacy-twenty-five", 100),
    ):
        result = rollout.evaluate_and_advance(
            tmp_path,
            now,
            _evaluation(tmp_path, candidate, legacy["artifact_id"], baseline, name),
        )
        assert result["rollout_percent"] == expected
    result = rollout.evaluate_and_advance(
        tmp_path,
        "2026-08-29T00:00:00Z",
        _evaluation(
            tmp_path,
            candidate,
            legacy["artifact_id"],
            baseline,
            "legacy-hundred",
            observation_mode="candidate_only_legacy_incumbent",
        ),
    )
    assert result["status"] == "active"


def test_candidate_only_evaluation_is_rejected_before_100_percent(
    tmp_path: Path,
) -> None:
    incumbent, candidate, baseline = _setup(tmp_path)
    with pytest.raises(rollout.RolloutError, match="candidate-only"):
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-01T00:00:00Z",
            _evaluation(
                tmp_path,
                candidate,
                incumbent,
                baseline,
                "early-candidate-only",
                observation_mode="candidate_only_legacy_incumbent",
            ),
        )


def test_hold_is_idempotent_and_never_uses_a_boolean_bypass(tmp_path: Path) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(tmp_path, candidate, lkg, baseline, "hold", denominator=99)
    first = rollout.evaluate_and_advance(tmp_path, "2026-08-01T00:00:00Z", evaluation)
    second = rollout.evaluate_and_advance(tmp_path, "2026-08-01T00:00:00Z", evaluation)
    assert first["status"] == "replay"
    assert first["changed"] is True
    assert second["changed"] is False
    assert rollout.select_policy_id(tmp_path, "session") == lkg
    evaluation["unexpected"] = True
    with pytest.raises(rollout.RolloutError, match="closed"):
        rollout.evaluate_and_advance(tmp_path, "2026-08-01T00:00:00Z", evaluation)


def test_tampered_candidate_fails_closed_to_lkg(tmp_path: Path) -> None:
    lkg, candidate, _baseline_id = _setup(tmp_path, status="canary", percent=25)
    candidate_path = store.distillation_dir(tmp_path) / "policies" / f"{candidate}.json"
    candidate_path.write_text("{}\n", encoding="utf-8")
    assert rollout.select_policy_id(tmp_path, "candidate-session") == lkg


def test_shadow_never_serves_candidate_and_canary_is_nested(tmp_path: Path) -> None:
    lkg, candidate, _baseline_id = _setup(tmp_path, status="shadow")
    assert all(
        rollout.select_policy_id(tmp_path, f"shadow-{index}") == lkg
        for index in range(100)
    )
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {"kind": "worker-state", "status": "canary", "rollout_percent": 5},
    )
    five = {
        session
        for session in (f"session-{index}" for index in range(2_000))
        if rollout.select_policy_id(tmp_path, session) == candidate
    }
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {"kind": "worker-state", "status": "canary", "rollout_percent": 25},
    )
    twenty_five = {
        session
        for session in (f"session-{index}" for index in range(2_000))
        if rollout.select_policy_id(tmp_path, session) == candidate
    }
    assert five
    assert five < twenty_five


def test_rollback_is_state_first_when_active_pointer_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lkg, _candidate, _baseline_id = _setup(tmp_path, status="canary", percent=5)
    original = store.write_pointer

    def fail_active(
        root: Path, kind: str, policy_id: str, **metadata: object
    ) -> dict[str, object]:
        if kind == "active":
            raise OSError("simulated pointer failure")
        return original(root, kind, policy_id, **metadata)

    monkeypatch.setattr(store, "write_pointer", fail_active)
    run_id = hashlib.sha256(b"rollback").hexdigest()
    result = rollout.rollback_to_lkg(tmp_path, run_id, "drill")
    assert result["learning_halted"] is True
    assert rollout.select_policy_id(tmp_path, "still-safe") == lkg
    state = store.read_sealed(store.distillation_dir(tmp_path) / store.STATE_FILE)
    assert state["quarantine_id"]
    assert state["learning_halted"] is True


def test_replay_ci_failure_quarantines_candidate(tmp_path: Path) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(
        tmp_path, candidate, lkg, baseline, "replay-fail", ci_lower=0.1
    )
    result = rollout.evaluate_and_advance(tmp_path, "2026-08-01T00:00:00Z", evaluation)
    assert result["status"] == "rolled_back"
    assert result["learning_halted"] is True
    assert rollout.select_policy_id(tmp_path, "safe") == lkg


def test_observation_days_come_from_sealed_stage_start_not_evaluation(
    tmp_path: Path,
) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    rollout.evaluate_and_advance(
        tmp_path,
        "2026-08-01T00:00:00Z",
        _evaluation(tmp_path, candidate, lkg, baseline, "replay"),
    )
    hold = rollout.evaluate_and_advance(
        tmp_path,
        "2026-08-07T23:59:59Z",
        _evaluation(tmp_path, candidate, lkg, baseline, "too-early"),
    )
    assert hold["rollout_percent"] == 0
    advanced = rollout.evaluate_and_advance(
        tmp_path,
        "2026-08-08T00:00:00Z",
        _evaluation(tmp_path, candidate, lkg, baseline, "seven-days"),
    )
    assert advanced["rollout_percent"] == 5


def test_invalid_target_and_disabled_config_never_select_candidate(
    tmp_path: Path,
) -> None:
    lkg, candidate, baseline = _setup(tmp_path, status="canary", percent=25)
    candidate_path = store.distillation_dir(tmp_path) / "policies" / f"{candidate}.json"
    candidate_path.write_text("{}\n", encoding="utf-8")
    assert rollout.select_policy_id(tmp_path, "safe") == lkg
    (tmp_path / "config.toml").write_text("[recall.distillation]\nenabled = false\n")
    assert rollout.select_policy_id(tmp_path, "safe") == ""
    result = rollout.evaluate_and_advance(
        tmp_path,
        "2026-08-08T00:00:00Z",
        _evaluation(tmp_path, candidate, lkg, baseline, "disabled"),
    )
    assert result["changed"] is False


def test_partial_adoption_write_halts_traffic_on_lkg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lkg, candidate, baseline = _setup(tmp_path, status="canary", percent=100)
    original = store.write_pointer

    def fail_lkg(
        root: Path, kind: str, policy_id: str, **metadata: object
    ) -> dict[str, object]:
        if kind == "lkg":
            raise OSError("simulated lkg promotion failure")
        return original(root, kind, policy_id, **metadata)

    monkeypatch.setattr(store, "write_pointer", fail_lkg)
    result = rollout.evaluate_and_advance(
        tmp_path,
        "2026-08-08T00:00:00Z",
        _evaluation(tmp_path, candidate, lkg, baseline, "adoption"),
    )
    assert result["status"] == "adopting"
    assert result["learning_halted"] is True
    assert rollout.select_policy_id(tmp_path, "safe") == lkg


def test_fake_baseline_and_missing_named_metric_are_rejected(tmp_path: Path) -> None:
    lkg, candidate, _baseline_id = _setup(tmp_path)
    ineligible = _baseline(tmp_path, eligible=False)
    with pytest.raises(rollout.RolloutError, match="P5 eligible"):
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-01T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, ineligible, "fake-baseline"),
        )
    with pytest.raises(rollout.RolloutError, match="named rollout metrics"):
        rollout.evaluate_and_advance(
            tmp_path,
            "2026-08-01T00:00:00Z",
            _evaluation(
                tmp_path,
                candidate,
                lkg,
                _baseline_id,
                "missing-metric",
                omit_metric="feature_parity",
            ),
        )


def test_wrong_policy_feature_keys_or_card_limit_fails_closed(tmp_path: Path) -> None:
    lkg, _candidate, _baseline_id = _setup(tmp_path, status="canary", percent=25)
    bad_id = hashlib.sha256(b"bad-policy").hexdigest()
    store.write_immutable(
        store.distillation_dir(tmp_path) / "policies",
        {
            "kind": "tiny-logistic-policy",
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "feature_keys": list(distill.FAST_FEATURE_KEYS[:-1]),
            "weights": {key: 1.0 for key in distill.FAST_FEATURE_KEYS[:-1]},
            "bias": 0.0,
            "threshold": 0.5,
            "abstain_margin": 0.0,
            "max_cards": 4,
        },
        schema=rollout.POLICY_SCHEMA,
        artifact_id=bad_id,
    )
    store.write_pointer(tmp_path, "candidate", bad_id)
    assert rollout.select_policy_id(tmp_path, "safe") == lkg
