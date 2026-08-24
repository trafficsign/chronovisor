from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_rollout as rollout
from chronovisor.recall import recall_distillation_store as store


def _clocked_advance(
    root: Path, now: str, evaluation: dict[str, object]
) -> dict[str, object]:
    return rollout._evaluate_at(
        root, now, evaluation, _token=rollout._TEST_CLOCK_TOKEN
    )


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


def _baseline(root: Path, *, eligible: bool = True, marker: str = "b") -> str:
    offline_gate = {"passed": eligible, "revision": "test-offline-gate-v2"}
    baseline_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "baselines",
        {
            "raw_watermark": marker * 64,
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


def _replay_observation(
    root: Path,
    *,
    name: str,
    run_id: str,
    candidate: str,
    baseline: str,
    baseline_policy: str,
    count: int,
    first: str | None = "2026-08-01T00:00:00Z",
    last: str | None = "2026-08-08T00:00:00Z",
) -> tuple[str, str]:
    try:
        current_state = store.read_sealed(
            store.distillation_dir(root) / store.STATE_FILE,
            schema=store.DISTILLATION_SCHEMA,
        )
        stage_started_at = str(
            current_state.get("stage_started_at") or "2026-08-01T00:00:00Z"
        )
    except store.DistillationStoreError:
        stage_started_at = "2026-08-01T00:00:00Z"
    first_dt = datetime.fromisoformat(first.replace("Z", "+00:00")) if first else datetime(2026, 8, 1, tzinfo=UTC)
    last_dt = datetime.fromisoformat(last.replace("Z", "+00:00")) if last else first_dt
    total = max(3, count)
    shadow_ids: list[str] = []
    receipt_rows: list[dict[str, object]] = []
    for index in range(total):
        observed_at = first_dt if total == 1 else first_dt + (last_dt - first_dt) * index / (total - 1)
        pool = [
            {
                "candidate_id": f"{name}-candidate-{index}",
                "selected": True,
                "page_id": f"page-{index}",
                "page_content_sha256": hashlib.sha256(f"page:{index}".encode()).hexdigest(),
                "rendered_context_sha256": hashlib.sha256(f"context:{index}".encode()).hexdigest(),
            }
        ]
        features = [
            {
                "candidate_id": f"{name}-candidate-{index}",
                "features": distill.build_fast_features(
                    query_chargram_coverage=(index % 10) / 10,
                    candidate_chargram_precision=((index + 3) % 10) / 10,
                ),
            }
        ]
        hashes = rollout._shadow_hashes(
            features,
            features,
            pool,
            pool,
            selected_candidate_ids=[f"{name}-candidate-{index}"],
            baseline_selected_candidate_ids=[f"{name}-candidate-{index}"],
        )
        evidence = {
            "candidate_quality": True,
            "baseline_quality": True,
            "candidate_covered": True,
            "baseline_covered": True,
            "candidate_anchor_retained": True,
            "baseline_anchor_retained": True,
            "candidate_abstained": False,
            "baseline_abstained": False,
            "candidate_score_ms": 1,
            "live_latency_ms": 1,
            "resource_ok": True,
            "integrity_ok": True,
            "negative_veto": False,
            "deadline_ms": 100,
            "producer": {
                "name": "chronovisor.recall-runtime",
                "version": 1,
                "synthetic_fixture": False,
            },
            "stage": "replay",
            "run_id": run_id,
            "cohort": "rollout-test-cohort",
            "host": "test",
            **hashes,
        }
        runtime_observation = {
            "decision": "read",
            "index": index,
            "latency_ms": 1,
            "timed_out": False,
        }
        decision_id = hashlib.sha256(
            f"decision:{name}:{index}".encode()
        ).hexdigest()
        split_prefix = ("00", "b5", "e0")[index % 3]
        query_semantic_sha256 = split_prefix + hashlib.sha256(
            f"query:{name}:{index}".encode()
        ).hexdigest()[2:]
        observed_iso = observed_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        source_fields = distill._shadow_replay_source_fields(
            decision_id=decision_id,
            query_semantic_sha256=query_semantic_sha256,
            observed_at=observed_iso,
            pool_rows=pool,
            selected_candidate_ids=[f"{name}-candidate-{index}"],
            baseline_pool_rows=pool,
            baseline_selected_candidate_ids=[f"{name}-candidate-{index}"],
            paired_eligible=True,
        )
        runtime_sha = distill.canonical_json.canonical_json_sha256_strict(
            runtime_observation
        )
        evidence_sha = distill.canonical_json.canonical_json_sha256_strict(evidence)
        shadow_id, _, shadow_artifact = store.write_immutable(
            store.distillation_dir(root) / "shadow-observations",
                {
                    "kind": "non-causal-shadow-observation",
                    "decision_id": decision_id,
                    "host": "test",
                    "session_id_sha256": hashlib.sha256(
                        f"session:{name}:{index}".encode()
                    ).hexdigest(),
                    "query_semantic_sha256": query_semantic_sha256,
                    "run_id": run_id,
                    "qualified_run_id": run_id,
                    "cohort": "rollout-test-cohort",
                    "stage": "replay",
                    "stage_started_at": stage_started_at,
                    "candidate_policy_id": candidate,
                    "baseline_policy_id": baseline_policy,
                    "baseline_artifact_id": baseline,
                    "policy_id": candidate,
                    "incumbent_policy_id": baseline_policy,
                    "served_policy_id": baseline_policy,
                    "selected_candidate_ids": [f"{name}-candidate-{index}"],
                    "incumbent_selected_candidate_ids": [f"{name}-candidate-{index}"],
                    **source_fields,
                    "paired_eligible": True,
                "candidate_pool_refs": pool,
                "candidate_feature_snapshot": features,
                "baseline_pool_refs": pool,
                "baseline_feature_snapshot": features,
                **hashes,
                "runtime_observation": runtime_observation,
                "runtime_observation_sha256": runtime_sha,
                "operational_evidence": evidence,
                "operational_evidence_sha256": evidence_sha,
                    "observed_at": observed_iso,
                },
            schema="chronovisor.recall-distill-shadow-observation.v1",
        )
        shadow_ids.append(shadow_id)
        binding = {
            key: shadow_artifact[key]
            for key in rollout._SHADOW_RECEIPT_BINDING_KEYS
        }
        receipt_rows.append(
            {
                "kind": "shadow-policy-observation",
                "shadow_observation_artifact_id": shadow_id,
                **binding,
                "binding_sha256": distill.canonical_json.canonical_json_sha256_strict(
                    binding
                ),
                "idempotency_sha256": distill.canonical_json.canonical_json_sha256_strict(
                    {
                        key: value
                        for key, value in binding.items()
                        if key not in {"observed_at", "as_of", "row_id"}
                    }
                ),
            }
        )
    store.append_chain_batch(
        store.distillation_dir(root) / "shadow-observation-receipts.jsonl", receipt_rows
    )
    split_artifact = rollout.write_locked_replay_input(
        root,
        shadow_observation_artifact_ids=shadow_ids,
        run_id=run_id,
        stage="replay",
        cohort="rollout-test-cohort",
        candidate_policy_id=candidate,
        baseline_policy_id=baseline_policy,
        baseline_artifact_id=baseline,
    )
    observation = (
        rollout.write_empty_replay_observation(
            root,
            run_id=run_id,
            stage="replay",
            cohort="rollout-test-cohort",
            candidate_policy_id=candidate,
            baseline_policy_id=baseline_policy,
            baseline_artifact_id=baseline,
            split_artifact_id=split_artifact["artifact_id"],
        )
        if count == 0
        else rollout.write_replay_observation(
            root,
            run_id=run_id,
            stage="replay",
            cohort="rollout-test-cohort",
            candidate_policy_id=candidate,
            baseline_policy_id=baseline_policy,
            baseline_artifact_id=baseline,
            split_artifact_id=split_artifact["artifact_id"],
            shadow_observation_artifact_ids=shadow_ids[:count],
        )
    )
    return split_artifact["artifact_id"], observation["artifact_id"]


def _reseal(path: Path, artifact: dict[str, object]) -> None:
    unsigned = {key: value for key, value in artifact.items() if key != "seal_sha256"}
    path.write_bytes(
        distill.canonical_json.canonical_json_bytes_strict(store._sealed(unsigned))
        + b"\n"
    )


_KEEP_IDENTITY = object()


def _clone_shadow_sources(
    root: Path,
    source_ids: list[str],
    *,
    baseline_artifact_id: str | None,
    candidate_policy_id: str | None | object = _KEEP_IDENTITY,
    baseline_policy_id: str | None | object = _KEEP_IDENTITY,
) -> list[str]:
    """Create content-addressed source/receipt pairs with explicit bindings."""

    receipt_rows = store.read_chain(
        store.distillation_dir(root) / "shadow-observation-receipts.jsonl"
    )
    receipts = {
        str(row["shadow_observation_artifact_id"]): row
        for row in receipt_rows
        if isinstance(row, dict)
        and isinstance(row.get("shadow_observation_artifact_id"), str)
    }
    cloned_ids: list[str] = []
    cloned_receipts: list[dict[str, object]] = []
    for source_id in source_ids:
        source = store.read_sealed(
            store.distillation_dir(root)
            / "shadow-observations"
            / f"{source_id}.json",
            schema="chronovisor.recall-distill-shadow-observation.v1",
        )
        source_payload = {
            key: value
            for key, value in source.items()
            if key not in {"schema", "namespace", "artifact_id", "seal_sha256"}
        }
        clone_namespace = str(baseline_artifact_id or "missing")
        source_payload["decision_id"] = hashlib.sha256(
            f"clone-decision:{clone_namespace}:{source_id}".encode()
        ).hexdigest()
        source_payload["session_id_sha256"] = hashlib.sha256(
            f"clone-session:{clone_namespace}:{source_id}".encode()
        ).hexdigest()
        split_prefix = {
            "train": "00",
            "validation": "b5",
            "test": "e0",
        }[str(source_payload["split"])]
        source_payload["query_semantic_sha256"] = split_prefix + hashlib.sha256(
            f"clone-query:{clone_namespace}:{source_id}".encode()
        ).hexdigest()[2:]
        for key, value in (
            ("baseline_artifact_id", baseline_artifact_id),
            ("candidate_policy_id", candidate_policy_id),
            ("baseline_policy_id", baseline_policy_id),
        ):
            if value is _KEEP_IDENTITY:
                continue
            if value is None:
                source_payload.pop(key, None)
            else:
                source_payload[key] = value
        source_fields = distill._shadow_replay_source_fields(
            decision_id=str(source_payload["decision_id"]),
            query_semantic_sha256=str(source_payload["query_semantic_sha256"]),
            observed_at=str(source_payload["observed_at"]),
            pool_rows=source_payload["candidate_pool_refs"],
            selected_candidate_ids=source_payload["selected_candidate_ids"],
            baseline_pool_rows=source_payload["baseline_pool_refs"],
            baseline_selected_candidate_ids=source_payload[
                "incumbent_selected_candidate_ids"
            ],
            paired_eligible=bool(source_payload["paired_eligible"]),
        )
        source_payload.update(source_fields)
        cloned_id, _, _ = store.write_immutable(
            store.distillation_dir(root) / "shadow-observations",
            source_payload,
            schema="chronovisor.recall-distill-shadow-observation.v1",
        )
        cloned_ids.append(cloned_id)

        receipt = receipts[source_id]
        receipt_payload = {
            key: value
            for key, value in receipt.items()
            if key
            not in {"schema", "namespace", "previous_sha256", "record_sha256"}
        }
        receipt_payload["shadow_observation_artifact_id"] = cloned_id
        for key in rollout._SHADOW_RECEIPT_BINDING_KEYS:
            if key in source_payload:
                receipt_payload[key] = source_payload[key]
            else:
                receipt_payload.pop(key, None)
        for key, value in (
            ("baseline_artifact_id", baseline_artifact_id),
            ("candidate_policy_id", candidate_policy_id),
            ("baseline_policy_id", baseline_policy_id),
        ):
            if value is _KEEP_IDENTITY:
                continue
            if value is None:
                receipt_payload.pop(key, None)
            else:
                receipt_payload[key] = value
        if rollout._SHADOW_RECEIPT_BINDING_KEYS.issubset(receipt_payload):
            binding = {
                key: receipt_payload[key]
                for key in rollout._SHADOW_RECEIPT_BINDING_KEYS
            }
            receipt_payload["binding_sha256"] = (
                distill.canonical_json.canonical_json_sha256_strict(binding)
            )
            receipt_payload["idempotency_sha256"] = (
                distill.canonical_json.canonical_json_sha256_strict(
                    {
                        key: value
                        for key, value in binding.items()
                        if key not in {"observed_at", "as_of", "row_id"}
                    }
                )
            )
        cloned_receipts.append(receipt_payload)
    store.append_chain_batch(
        store.distillation_dir(root) / "shadow-observation-receipts.jsonl",
        cloned_receipts,
    )
    return cloned_ids


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
    observation_count: int | None = None,
    observation_first: str | None = "2026-08-01T00:00:00Z",
    observation_last: str | None = "2026-08-08T00:00:00Z",
    replay_min_days: int = 7,
) -> dict[str, object]:
    run_id = hashlib.sha256(name.encode()).hexdigest()
    baseline_artifact = store.read_sealed(
        store.distillation_dir(root) / "baselines" / f"{baseline}.json",
        schema=distill.BASELINE_SCHEMA,
    )
    replay_metrics = _metrics(denominator=denominator, ci_lower=ci_lower)
    if omit_metric:
        replay_metrics.pop(omit_metric)
    for gate in replay_metrics.values():
        gate["min_days"] = replay_min_days
    split_sha256, observation_id = _replay_observation(
        root,
        name=name,
        run_id=run_id,
        candidate=candidate,
        baseline=baseline,
        baseline_policy=incumbent,
        count=denominator if observation_count is None else observation_count,
        first=observation_first,
        last=observation_last,
    )
    operational_metrics = distill._operational_rollout_metrics(
        root,
        candidate,
        incumbent,
        baseline_artifact_id=baseline,
        cohort="rollout-test-cohort",
        qualified_run_id=run_id,
        stage_name="replay",
    )
    operational_source_ids = distill._operational_rollout_source_ids(
        root,
        candidate_id=candidate,
        incumbent_id=incumbent,
        baseline_artifact_id=baseline,
        cohort="rollout-test-cohort",
        qualified_run_id=run_id,
        stage_name="replay",
    )
    artifact_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "evaluations",
        {
            "kind": "automatic-closed-metrics",
            "run_id": run_id,
            "candidate_policy_id": candidate,
            "baseline_artifact_id": baseline,
            "raw_watermark": "b" * 64,
            "baseline_policy_id": incumbent,
            "split_sha256": split_sha256,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "feature_parity_sha256": "d" * 64,
            "offline_gate_sha256": distill.canonical_json.canonical_json_sha256_strict(
                baseline_artifact["offline_training_gate"]
            ),
            "observation_mode": observation_mode,
            "replay_observation_artifact_id": observation_id,
            "replay_metrics": replay_metrics,
            "shadow_metrics": operational_metrics,
            "canary_metrics": operational_metrics,
            "operational_metrics_sha256": distill.canonical_json.canonical_json_sha256_strict(
                operational_metrics
            ),
            "operational_source_sha256": distill.canonical_json.canonical_json_sha256_strict(
                operational_source_ids
            ),
        },
        schema=rollout.EVALUATION_SCHEMA,
    )
    return {"run_id": run_id, "evaluation_artifact_id": artifact_id}


def test_pass_path_is_replay_then_shadow_then_nested_canaries(tmp_path: Path) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    assert rollout.select_policy_id(tmp_path, "any-session") == lkg
    assert (
        _clocked_advance(
            tmp_path,
            "2026-08-08T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "replay"),
        )["status"]
        == "shadow"
    )
    assert (
        _clocked_advance(
            tmp_path,
            "2026-08-15T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "shadow"),
        )["rollout_percent"]
        == 5
    )
    assert (
        _clocked_advance(
            tmp_path,
            "2026-08-22T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "five"),
        )["rollout_percent"]
        == 25
    )
    assert (
        _clocked_advance(
            tmp_path,
            "2026-08-29T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, baseline, "twenty-five"),
        )["rollout_percent"]
        == 100
    )
    assert (
        _clocked_advance(
            tmp_path,
            "2026-09-05T00:00:00Z",
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
    candidate = _policy(tmp_path, "legacy-candidate")
    store.write_pointer(tmp_path, "candidate", candidate)
    store.write_sealed_state(
        store.distillation_dir(tmp_path) / store.STATE_FILE,
        {"kind": "worker-state", "status": "replay", "rollout_percent": 0},
    )
    for now, name, expected in (
        ("2026-08-08T00:00:00Z", "legacy-replay", 0),
        ("2026-08-15T00:00:00Z", "legacy-shadow", 5),
        ("2026-08-22T00:00:00Z", "legacy-five", 25),
        ("2026-08-29T00:00:00Z", "legacy-twenty-five", 100),
    ):
        result = _clocked_advance(
            tmp_path,
            now,
            _evaluation(tmp_path, candidate, legacy["artifact_id"], baseline, name),
        )
        assert result["rollout_percent"] == expected
    result = _clocked_advance(
        tmp_path,
        "2026-09-05T00:00:00Z",
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
        _clocked_advance(
            tmp_path,
            "2026-08-08T00:00:00Z",
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
    first = _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", evaluation)
    second = _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", evaluation)
    assert first["status"] == "replay"
    assert first["changed"] is True
    assert second["changed"] is False
    assert rollout.select_policy_id(tmp_path, "session") == lkg
    evaluation["unexpected"] = True
    with pytest.raises(rollout.RolloutError, match="closed"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", evaluation)


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
    result = _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", evaluation)
    assert result["status"] == "rolled_back"
    assert result["learning_halted"] is True
    assert rollout.select_policy_id(tmp_path, "safe") == lkg


def test_observation_days_come_from_sealed_stage_start_not_evaluation(
    tmp_path: Path,
) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    _clocked_advance(
        tmp_path,
        "2026-08-08T00:00:00Z",
        _evaluation(tmp_path, candidate, lkg, baseline, "replay"),
    )
    hold = _clocked_advance(
        tmp_path,
        "2026-08-14T23:59:59Z",
        _evaluation(tmp_path, candidate, lkg, baseline, "too-early"),
    )
    assert hold["rollout_percent"] == 0
    advanced = _clocked_advance(
        tmp_path,
        "2026-08-15T00:00:00Z",
        _evaluation(tmp_path, candidate, lkg, baseline, "seven-days"),
    )
    assert advanced["rollout_percent"] == 5


@pytest.mark.parametrize(
    ("last", "now"),
    (
        ("2026-08-01T23:59:59Z", "2026-08-01T23:59:59Z"),
        ("2026-08-07T23:59:59Z", "2026-08-07T23:59:59Z"),
    ),
)
def test_replay_500_rows_without_seven_day_span_holds(
    tmp_path: Path, last: str, now: str
) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    result = _clocked_advance(
        tmp_path,
        now,
        _evaluation(
            tmp_path,
            candidate,
            lkg,
            baseline,
            f"short-{last}",
            observation_first="2026-08-01T00:00:00Z",
            observation_last=last,
        ),
    )
    assert result["status"] == "replay"
    assert store.read_sealed(store.distillation_dir(tmp_path) / store.STATE_FILE)[
        "hold_reason"
    ] == "replay_insufficient"


def test_empty_replay_observation_is_a_truthful_hold(tmp_path: Path) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    result = _clocked_advance(
        tmp_path,
        "2026-08-08T00:00:00Z",
        _evaluation(
            tmp_path,
            candidate,
            lkg,
            baseline,
            "automatic-empty",
            denominator=0,
            observation_count=0,
            observation_first=None,
            observation_last=None,
        ),
    )
    assert result["status"] == "replay"
    assert result["rollout_percent"] == 0


def test_replay_rejects_duplicate_pair_ids(tmp_path: Path) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(
        tmp_path, candidate, lkg, baseline, "duplicate-pair-id"
    )
    evaluation_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{evaluation['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    observation_path = (
        store.distillation_dir(tmp_path)
        / "rollout-observations"
        / f"{evaluation_artifact['replay_observation_artifact_id']}.json"
    )
    observation = store.read_sealed(
        observation_path, schema=rollout.REPLAY_OBSERVATION_SCHEMA
    )
    rows = observation["pairs"]
    rows[1]["pair_id"] = rows[0]["pair_id"]
    observation["pairs_sha256"] = distill.canonical_json.canonical_json_sha256_strict(
        rows
    )
    _reseal(observation_path, observation)
    with pytest.raises(rollout.RolloutError, match="content identity"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", evaluation)


def test_replay_rejects_future_invalid_extra_and_resealed_observations(
    tmp_path: Path,
) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    future = _evaluation(
        tmp_path,
        candidate,
        lkg,
        baseline,
        "future-observation",
        observation_last="2026-08-09T00:00:00Z",
    )
    with pytest.raises(rollout.RolloutError, match="future"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", future)

    invalid = _evaluation(
        tmp_path, candidate, lkg, baseline, "invalid-observation"
    )
    invalid_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{invalid['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    invalid_path = (
        store.distillation_dir(tmp_path)
        / "rollout-observations"
        / f"{invalid_artifact['replay_observation_artifact_id']}.json"
    )
    invalid_observation = store.read_sealed(
        invalid_path, schema=rollout.REPLAY_OBSERVATION_SCHEMA
    )
    invalid_observation["pairs"][0]["observed_at"] = "not-a-timestamp"
    _reseal(invalid_path, invalid_observation)
    with pytest.raises(rollout.RolloutError, match="content identity"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", invalid)

    extra = _evaluation(tmp_path, candidate, lkg, baseline, "extra-observation")
    extra_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{extra['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    extra_path = (
        store.distillation_dir(tmp_path)
        / "rollout-observations"
        / f"{extra_artifact['replay_observation_artifact_id']}.json"
    )
    extra_observation = store.read_sealed(
        extra_path, schema=rollout.REPLAY_OBSERVATION_SCHEMA
    )
    extra_observation["unexpected"] = True
    _reseal(extra_path, extra_observation)
    with pytest.raises(rollout.RolloutError, match="content identity"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", extra)


def test_replay_min_days_and_missing_observation_are_fail_closed(
    tmp_path: Path,
) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    short_gate = _evaluation(
        tmp_path,
        candidate,
        lkg,
        baseline,
        "short-min-days",
        replay_min_days=6,
    )
    with pytest.raises(rollout.RolloutError, match="at least seven"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", short_gate)

    missing = _evaluation(tmp_path, candidate, lkg, baseline, "missing-observation")
    artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{missing['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    observation_path = (
        store.distillation_dir(tmp_path)
        / "rollout-observations"
        / f"{artifact['replay_observation_artifact_id']}.json"
    )
    observation_path.unlink()
    with pytest.raises(rollout.RolloutError, match="observation artifact"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", missing)


def test_replay_evaluation_and_split_paths_reject_symlink_or_missing(
    tmp_path: Path,
) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(tmp_path, candidate, lkg, baseline, "path-check")
    evaluation_path = (
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{evaluation['evaluation_artifact_id']}.json"
    )
    evaluation_backup = evaluation_path.with_suffix(".bak")
    evaluation_path.rename(evaluation_backup)
    evaluation_path.symlink_to(evaluation_backup.name)
    with pytest.raises(rollout.RolloutError, match="evaluation artifact"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", evaluation)
    evaluation_path.unlink()
    evaluation_backup.rename(evaluation_path)

    evaluation_artifact = store.read_sealed(
        evaluation_path, schema=rollout.EVALUATION_SCHEMA
    )
    split_path = (
        store.distillation_dir(tmp_path)
        / "locked-replays"
        / f"{evaluation_artifact['split_sha256']}.json"
    )
    split_path.unlink()
    with pytest.raises(rollout.RolloutError, match="split artifact"):
        _clocked_advance(tmp_path, "2026-08-08T00:00:00Z", evaluation)


def test_pair_registry_rejects_cross_artifact_reuse(tmp_path: Path) -> None:
    _lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(tmp_path, candidate, _lkg, baseline, "registry")
    evaluation_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{evaluation['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    observation_path = (
        store.distillation_dir(tmp_path)
        / "rollout-observations"
        / f"{evaluation_artifact['replay_observation_artifact_id']}.json"
    )
    observation = store.read_sealed(observation_path, schema=rollout.REPLAY_OBSERVATION_SCHEMA)
    rollout._register_replay_pairs(tmp_path, observation)
    conflicting = {**observation, "artifact_id": "0" * 64}
    with pytest.raises(rollout.RolloutError, match="already registered"):
        rollout._register_replay_pairs(tmp_path, conflicting)


def test_operational_metrics_use_only_producer_validated_source_set(
    tmp_path: Path,
) -> None:
    _lkg, candidate, baseline = _setup(tmp_path)
    run_id = hashlib.sha256(b"forged-operational-source").hexdigest()
    _split_id, _observation_id = _replay_observation(
        tmp_path,
        name="forged-operational-source",
        run_id=run_id,
        candidate=candidate,
        baseline=baseline,
        baseline_policy=_lkg,
        count=500,
    )
    receipt_rows = store.read_chain(
        store.distillation_dir(tmp_path) / "shadow-observation-receipts.jsonl"
    )
    source_id = str(receipt_rows[0]["shadow_observation_artifact_id"])
    source = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "shadow-observations"
        / f"{source_id}.json",
        schema=distill.SHADOW_OBSERVATION_SCHEMA,
    )
    source_payload = {
        key: value
        for key, value in source.items()
        if key not in {"schema", "namespace", "artifact_id", "seal_sha256"}
    }
    source_payload["decision_id"] = hashlib.sha256(
        b"forged-decision"
    ).hexdigest()
    source_payload["session_id_sha256"] = hashlib.sha256(
        b"forged-session"
    ).hexdigest()
    source_payload["query_semantic_sha256"] = source["query_semantic_sha256"][:2] + hashlib.sha256(
        b"forged-query"
    ).hexdigest()[2:]
    source_fields = distill._shadow_replay_source_fields(
        decision_id=str(source_payload["decision_id"]),
        query_semantic_sha256=str(source_payload["query_semantic_sha256"]),
        observed_at=str(source_payload["observed_at"]),
        pool_rows=source_payload["candidate_pool_refs"],
        selected_candidate_ids=source_payload["selected_candidate_ids"],
        baseline_pool_rows=source_payload["baseline_pool_refs"],
        baseline_selected_candidate_ids=source_payload[
            "incumbent_selected_candidate_ids"
        ],
        paired_eligible=True,
    )
    source_payload.update(source_fields)
    source_payload["row_id"] = "f" * 64
    forged_id, _, _ = store.write_immutable(
        store.distillation_dir(tmp_path) / "shadow-observations",
        source_payload,
        schema=distill.SHADOW_OBSERVATION_SCHEMA,
    )
    original_receipt = next(
        row for row in receipt_rows if row["shadow_observation_artifact_id"] == source_id
    )
    forged_receipt = {
        key: value
        for key, value in original_receipt.items()
        if key
        not in {
            "schema",
            "namespace",
            "previous_sha256",
            "record_sha256",
            "binding_sha256",
            "idempotency_sha256",
        }
    }
    forged_receipt["shadow_observation_artifact_id"] = forged_id
    for key in rollout._SHADOW_RECEIPT_BINDING_KEYS:
        forged_receipt[key] = source_payload[key]
    binding = {
        key: forged_receipt[key] for key in rollout._SHADOW_RECEIPT_BINDING_KEYS
    }
    forged_receipt["binding_sha256"] = distill.canonical_json.canonical_json_sha256_strict(
        binding
    )
    forged_receipt["idempotency_sha256"] = distill.canonical_json.canonical_json_sha256_strict(
        {
            key: value
            for key, value in binding.items()
            if key not in {"observed_at", "as_of", "row_id"}
        }
    )
    store.append_chain(
        store.distillation_dir(tmp_path) / "shadow-observation-receipts.jsonl",
        forged_receipt,
    )
    source_ids = distill._operational_rollout_source_ids(
        tmp_path,
        candidate_id=candidate,
        incumbent_id=_lkg,
        baseline_artifact_id=baseline,
        cohort="rollout-test-cohort",
        qualified_run_id=run_id,
        stage_name="replay",
    )
    assert len(source_ids) == 500
    metrics = distill._operational_rollout_metrics(
        tmp_path,
        candidate,
        _lkg,
        baseline_artifact_id=baseline,
        cohort="rollout-test-cohort",
        qualified_run_id=run_id,
        stage_name="replay",
        source_ids=source_ids,
    )
    assert metrics["coverage_abstain"]["denominator"] == 500
    assert metrics["latency_timeout"]["denominator"] == 500


def test_replay_registry_source_set_binding_is_fail_closed(tmp_path: Path) -> None:
    _lkg, candidate, baseline = _setup(tmp_path)
    run_id = hashlib.sha256(b"registry-source-set").hexdigest()
    _replay_observation(
        tmp_path,
        name="registry-source-set",
        run_id=run_id,
        candidate=candidate,
        baseline=baseline,
        baseline_policy=_lkg,
        count=3,
    )
    ledger = store.distillation_dir(tmp_path) / "replay-observation-pairs.jsonl"
    original = store.read_chain(ledger)[0]
    forged = {
        key: value
        for key, value in original.items()
        if key not in {"schema", "namespace", "previous_sha256", "record_sha256"}
    }
    forged["source_set_sha256"] = "0" * 64
    store.append_chain(ledger, forged)
    with pytest.raises(distill.DistillationError, match="replay source ledger"):
        distill._shadow_replay_artifact_ids(
            tmp_path,
            run_id=run_id,
            stage="replay",
            cohort="rollout-test-cohort",
            candidate_id=candidate,
            incumbent_id=_lkg,
            baseline_artifact_id=baseline,
        )


def test_public_replay_writers_round_trip_and_empty_is_explicit(
    tmp_path: Path,
) -> None:
    _lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(tmp_path, candidate, _lkg, baseline, "public-writers")
    evaluation_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{evaluation['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    observation = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "rollout-observations"
        / f"{evaluation_artifact['replay_observation_artifact_id']}.json",
        schema=rollout.REPLAY_OBSERVATION_SCHEMA,
    )
    assert observation["pair_count"] == 500
    empty = rollout.write_empty_replay_observation(
        tmp_path,
        run_id=evaluation["run_id"],
        stage="replay",
        cohort="rollout-test-cohort",
        candidate_policy_id=candidate,
        baseline_policy_id=_lkg,
        baseline_artifact_id=baseline,
        split_artifact_id=evaluation_artifact["split_sha256"],
    )
    assert empty["pair_count"] == 0
    assert empty["first_observed_at"] is None


def test_invalid_target_and_disabled_config_never_select_candidate(
    tmp_path: Path,
) -> None:
    lkg, candidate, baseline = _setup(tmp_path, status="canary", percent=25)
    evaluation = _evaluation(tmp_path, candidate, lkg, baseline, "disabled")
    candidate_path = store.distillation_dir(tmp_path) / "policies" / f"{candidate}.json"
    candidate_path.write_text("{}\n", encoding="utf-8")
    assert rollout.select_policy_id(tmp_path, "safe") == lkg
    (tmp_path / "config.toml").write_text("[recall.distillation]\nenabled = false\n")
    assert rollout.select_policy_id(tmp_path, "safe") == ""
    result = _clocked_advance(
        tmp_path,
        "2026-08-08T00:00:00Z",
        evaluation,
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
    result = _clocked_advance(
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
        _clocked_advance(
            tmp_path,
            "2026-08-08T00:00:00Z",
            _evaluation(tmp_path, candidate, lkg, ineligible, "fake-baseline"),
        )
    with pytest.raises(rollout.RolloutError, match="named rollout metrics"):
        _clocked_advance(
            tmp_path,
            "2026-08-08T00:00:00Z",
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


def test_public_writer_rejects_generic_self_report_rows(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        rollout.write_replay_observation(  # type: ignore[call-arg]
            tmp_path,
            rows=[{"x": 1}],
        )


def test_shadow_noncanonical_bytes_and_root_symlink_fail_closed(tmp_path: Path) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(
        tmp_path,
        candidate,
        lkg,
        baseline,
        "canonical-bytes",
        denominator=3,
        observation_count=3,
    )
    evaluation_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{evaluation['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    split = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "locked-replays"
        / f"{evaluation_artifact['split_sha256']}.json",
        schema="chronovisor.recall-distill-locked-replay.v1",
    )
    source_id = split["training_rows"][0]["shadow_observation_artifact_id"]
    source_path = (
        store.distillation_dir(tmp_path)
        / "shadow-observations"
        / f"{source_id}.json"
    )
    raw = source_path.read_bytes()
    source_path.write_bytes(raw[:-1] + b"  \n")
    with pytest.raises(rollout.RolloutError, match="shadow observation artifact"):
        rollout.write_locked_replay_input(
            tmp_path,
            shadow_observation_artifact_ids=[
                row["shadow_observation_artifact_id"] for row in split["training_rows"]
            ],
            run_id=evaluation["run_id"],
            stage="replay",
            cohort="rollout-test-cohort",
            candidate_policy_id=candidate,
            baseline_policy_id=lkg,
            baseline_artifact_id=baseline,
        )
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(rollout.RolloutError):
        rollout.write_locked_replay_input(
            linked,
            shadow_observation_artifact_ids=[],
            run_id=evaluation["run_id"],
            stage="replay",
            cohort="rollout-test-cohort",
            candidate_policy_id=candidate,
            baseline_policy_id=lkg,
            baseline_artifact_id=baseline,
        )


def test_shadow_source_identities_require_exact_baseline_rebind(tmp_path: Path) -> None:
    lkg, candidate, baseline_a = _setup(tmp_path)
    run_id = hashlib.sha256(b"baseline-rebind").hexdigest()
    split_a, _ = _replay_observation(
        tmp_path,
        name="baseline-rebind",
        run_id=run_id,
        candidate=candidate,
        baseline=baseline_a,
        baseline_policy=lkg,
        count=3,
        first="2026-08-01T00:00:00Z",
        last="2026-08-08T00:00:00Z",
    )
    split_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "locked-replays"
        / f"{split_a}.json",
        schema="chronovisor.recall-distill-locked-replay.v1",
    )
    source_ids = [
        row["shadow_observation_artifact_id"]
        for row in split_artifact["training_rows"]
    ]
    baseline_b = _baseline(tmp_path, marker="c")

    # Missing baseline identity in all three sealed artifacts and receipts must
    # not be rebound merely because the caller supplies a new context.
    receipt_ledger = (
        store.distillation_dir(tmp_path) / "shadow-observation-receipts.jsonl"
    )
    receipt_checkpoint = receipt_ledger.with_suffix(receipt_ledger.suffix + ".head.json")
    receipt_snapshot = {
        path: path.read_bytes() if path.exists() else None
        for path in (receipt_ledger, receipt_checkpoint)
    }
    missing_ids = _clone_shadow_sources(
        tmp_path, source_ids, baseline_artifact_id=None
    )
    with pytest.raises(
        rollout.RolloutError, match="receipt schema|baseline artifact id"
    ):
        rollout.write_locked_replay_input(
            tmp_path,
            shadow_observation_artifact_ids=missing_ids,
            run_id=run_id,
            stage="replay",
            cohort="rollout-test-cohort",
            candidate_policy_id=candidate,
            baseline_policy_id=lkg,
            baseline_artifact_id=baseline_b,
        )
    for path, data in receipt_snapshot.items():
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(data)

    # Re-sealing each source and receipt with B is the only valid rebind path.
    rebound_ids = _clone_shadow_sources(
        tmp_path, source_ids, baseline_artifact_id=baseline_b
    )
    rebound = rollout.write_locked_replay_input(
        tmp_path,
        shadow_observation_artifact_ids=rebound_ids,
        run_id=run_id,
        stage="replay",
        cohort="rollout-test-cohort",
        candidate_policy_id=candidate,
        baseline_policy_id=lkg,
        baseline_artifact_id=baseline_b,
    )
    assert rebound["baseline_artifact_id"] == baseline_b
    assert {
        row["baseline_artifact_id"] for row in rebound["training_rows"]
    } == {baseline_b}


def test_pair_registry_append_rolls_back_partial_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(
        tmp_path,
        candidate,
        lkg,
        baseline,
        "registry-rollback",
        denominator=3,
        observation_count=3,
    )
    evaluation_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{evaluation['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    observation_path = (
        store.distillation_dir(tmp_path)
        / "rollout-observations"
        / f"{evaluation_artifact['replay_observation_artifact_id']}.json"
    )
    observation = store.read_sealed(observation_path, schema=rollout.REPLAY_OBSERVATION_SCHEMA)
    pair = dict(observation["pairs"][0])
    pair["pair_id"] = "e" * 64
    conflicting = {"artifact_id": "f" * 64, "pairs": [pair]}
    ledger = store.distillation_dir(tmp_path) / "replay-observation-pairs.jsonl"
    before = ledger.read_bytes()
    original = store.append_chain_unique_locked

    def fail_after_append(*args: object, **kwargs: object) -> dict[str, object]:
        original(*args, **kwargs)
        raise OSError("simulated append failure")

    monkeypatch.setattr(store, "append_chain_unique_locked", fail_after_append)
    with pytest.raises(rollout.RolloutError, match="registry"):
        rollout._register_replay_pairs(tmp_path, conflicting)
    assert ledger.read_bytes() == before


def test_shadow_synthetic_or_missing_identity_and_hash_fallback_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lkg, candidate, baseline = _setup(tmp_path)
    evaluation = _evaluation(
        tmp_path,
        candidate,
        lkg,
        baseline,
        "identity-attack",
        denominator=3,
        observation_count=3,
    )
    evaluation_artifact = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "evaluations"
        / f"{evaluation['evaluation_artifact_id']}.json",
        schema=rollout.EVALUATION_SCHEMA,
    )
    split = store.read_sealed(
        store.distillation_dir(tmp_path)
        / "locked-replays"
        / f"{evaluation_artifact['split_sha256']}.json",
        schema="chronovisor.recall-distill-locked-replay.v1",
    )
    source_ids = [
        row["shadow_observation_artifact_id"] for row in split["training_rows"]
    ]
    source_path = (
        store.distillation_dir(tmp_path)
        / "shadow-observations"
        / f"{source_ids[0]}.json"
    )
    source = store.read_sealed(
        source_path, schema="chronovisor.recall-distill-shadow-observation.v1"
    )
    receipt = rollout._shadow_receipt(tmp_path, source_ids[0])
    context = rollout._replay_context(
        run_id=evaluation["run_id"],
        stage="replay",
        cohort="rollout-test-cohort",
        candidate_policy_id=candidate,
        baseline_policy_id=lkg,
        baseline_artifact_id=baseline,
    )
    synthetic = deepcopy(source)
    synthetic["operational_evidence"]["producer"]["synthetic_fixture"] = True
    monkeypatch.setattr(
        rollout,
        "_shadow_observation",
        lambda _root, _artifact_id: synthetic,
    )
    with pytest.raises(rollout.RolloutError, match="operational evidence"):
        rollout._shadow_row(
            tmp_path, source_ids[0], context, receipt=receipt
        )

    for missing_key in (
        "row_id",
        "rally_id",
        "candidate_id",
        "as_of",
        "split",
        "split_role",
        "operational_evidence",
    ):
        missing = deepcopy(source)
        missing.pop(missing_key, None)
        monkeypatch.setattr(
            rollout,
            "_shadow_observation",
            lambda _root, _artifact_id, value=missing: value,
        )
        with pytest.raises(
            rollout.RolloutError, match="typed source|binding fields"
        ):
            rollout._shadow_row(
                tmp_path, source_ids[0], context, receipt=receipt
            )

    for key, value in (
        ("row_id", "a" * 64),
        ("rally_id", "b" * 64),
        ("candidate_id", "forged-candidate"),
        ("as_of", "2026-08-02T00:00:00Z"),
        ("split", "test"),
        ("split_role", "test"),
    ):
        forged_source = deepcopy(source)
        forged_source[key] = value
        monkeypatch.setattr(
            rollout,
            "_shadow_observation",
            lambda _root, _artifact_id, value=forged_source: value,
        )
        with pytest.raises(
            rollout.RolloutError, match="source identity|binding fields"
        ):
            rollout._shadow_row(tmp_path, source_ids[0], context, receipt=receipt)

    forged_receipt = deepcopy(receipt)
    forged_receipt["row_id"] = "c" * 64
    monkeypatch.setattr(
        rollout,
        "_shadow_observation",
        lambda _root, _artifact_id: source,
    )
    with pytest.raises(rollout.RolloutError, match="source identity|binding hash"):
        rollout._shadow_row(tmp_path, source_ids[0], context, receipt=forged_receipt)

    generic_only = deepcopy(source)
    generic_only.pop("candidate_policy_id")
    generic_only.pop("baseline_policy_id")
    monkeypatch.setattr(
        rollout,
        "_shadow_observation",
        lambda _root, _artifact_id: generic_only,
    )
    with pytest.raises(rollout.RolloutError, match="candidate policy id"):
        rollout._shadow_row(tmp_path, source_ids[0], context, receipt=receipt)
    generic_receipt = deepcopy(receipt)
    generic_receipt.pop("candidate_policy_id")
    generic_receipt.pop("baseline_policy_id")
    monkeypatch.setattr(
        rollout,
        "_shadow_observation",
        lambda _root, _artifact_id: source,
    )
    with pytest.raises(rollout.RolloutError, match="receipt schema"):
        rollout._shadow_row(
            tmp_path, source_ids[0], context, receipt=generic_receipt
        )

    monkeypatch.undo()
    expected_hash_keys = {
        "candidate_decision_sha256",
        "baseline_decision_sha256",
        "candidate_pool_sha256",
        "baseline_pool_sha256",
        "candidate_feature_snapshot_sha256",
        "baseline_feature_snapshot_sha256",
        "candidate_feature_bytes_sha256",
        "baseline_feature_bytes_sha256",
        "feature_snapshot_sha256",
        "pair_id",
        "feature_parity",
    }

    def forged_hashes(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            key: False if key == "feature_parity" else "0" * 64
            for key in expected_hash_keys
        }

    monkeypatch.setattr(rollout, "_PRODUCER_SHADOW_HASHES", forged_hashes)
    with pytest.raises(rollout.RolloutError, match="shadow .* hash mismatch"):
        rollout.write_locked_replay_input(
            tmp_path,
            shadow_observation_artifact_ids=source_ids,
            run_id=evaluation["run_id"],
            stage="replay",
            cohort="rollout-test-cohort",
            candidate_policy_id=candidate,
            baseline_policy_id=lkg,
            baseline_artifact_id=baseline,
        )
