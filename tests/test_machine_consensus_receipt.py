from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.core.durable_state import (
    canonical_sha256,
    seal_object,
    write_sealed_json,
)
from chronovisor.core.runtime_config import DecisionRouterConfig
from chronovisor.decision.decision_authority import AUTHORITY_VERSION
from chronovisor.decision.decision_lane_contract_cases import (
    decision_lane_contract_case_manifest_sha256,
)
from chronovisor.decision.decision_lane_contracts import (
    lane_contract_manifest_sha256,
    lane_contract_sha256,
)
from chronovisor.decision.decision_router import (
    QUORUM_SAFETY_POLICY_VERSION,
    DecisionRouter,
    RouterPolicyResolution,
)
from chronovisor.decision.graph_decisions import (
    RECALL_ANSWER_ADJUDICATION_SCHEMA,
    build_recall_answer_adjudication_prompt,
)
from chronovisor.decision.local_structured import ChatRequest
from chronovisor.decision.machine_consensus_receipt import (
    BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256,
    GOLD_ENTRY_PRODUCER_POLICY_SHA256,
    SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256,
    append_machine_consensus_receipt,
    validate_machine_consensus_receipt,
)

LANE = "recall_answer_adjudication"


class ModelTransport:
    def __init__(self, responses: dict[str, list[str | Exception]]) -> None:
        self.responses: dict[str, deque[str | Exception]] = defaultdict(deque)
        for model, values in responses.items():
            self.responses[model].extend(values)
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest) -> str:
        self.requests.append(request)
        value = self.responses[request.model].popleft()
        if isinstance(value, Exception):
            raise value
        return value


def _config(**overrides: object) -> DecisionRouterConfig:
    values = {
        "authority_kind": "quorum_v1",
        "primary_model": "ornith:test",
        "challenger_model": "gpt-oss:test",
        "tie_break_model": "gemma:test",
        "primary_keep_alive": "20m",
        "challenger_keep_alive": "20m",
        "tie_break_keep_alive": "2m",
        "num_ctx": 16_384,
        "num_predict": 256,
        "read_timeout_ms": 5_000,
        "max_input_chars": 20_000,
        "max_output_chars": 2_000,
        "max_feedback_chars": 2_000,
        "quorum": 2,
    }
    values.update(overrides)
    return DecisionRouterConfig(**values)


def _authority(config: DecisionRouterConfig, *, artifact: str = "a" * 64) -> dict:
    routes = [
        {
            "role": f"classification.{role}",
            "provider": "custom_transport",
            "model": model,
            "location": "local",
            "protocol": "custom-transport",
            "endpoint_sha256": None,
            "revision": None,
            "ollama": None,
        }
        for role, model in zip(
            ("primary", "challenger", "tie_break"),
            (config.primary_model, config.challenger_model, config.tie_break_model),
            strict=True,
        )
    ]
    if artifact != "a" * 64:
        routes[0]["revision"] = artifact
    return {
        "source": "configured_runtime_consensus",
        "authority_version": AUTHORITY_VERSION,
        "lane": LANE,
        "lane_contract_sha256": lane_contract_sha256(LANE),
        "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
        "lane_contract_case_manifest_sha256": (
            decision_lane_contract_case_manifest_sha256()
        ),
        "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
        "policy": {
            "kind": "consensus",
            "schema_name": LANE,
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "runtime_role_mapping",
            "error": None,
            "routes": routes,
        },
    }


def _subject() -> dict:
    return {
        "schema_version": 1,
        "subject_kind": "gold_entry",
        "episode_id": "episode-1",
        "split": "train",
        "split_epoch_id": "b" * 64,
        "rubric_sha256": "c" * 64,
        "source_packet_sha256": "d" * 64,
        "evidence_sha256": "e" * 64,
        "producer_kind": "deterministic_evidence_projection",
        "producer_model": None,
        "producer_policy_sha256": GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        "production_answer_used": False,
    }


def _search_packet(
    *,
    content: bytes = b"frozen machine search evidence",
    preregistered_at: str = "2026-07-31T00:00:00Z",
) -> dict:
    content_sha = hashlib.sha256(content).hexdigest()
    candidate_identity = {
        "query": "What is the frozen fact?",
        "expected_pages": ["frozen-page"],
        "source": "recall_questions",
        "page_uid": "uid-frozen-page",
        "content_sha256": content_sha,
        "content_byte_length": len(content),
        "projection_policy_sha256": BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256,
        "search_eval_split": "train",
    }
    candidate_sha = hashlib.sha256(
        json.dumps(
            candidate_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    excerpt = content.decode()
    reference = f"[PAGE frozen-page]\n{excerpt}"
    return {
        "schema_version": 1,
        "packet_kind": "preregistered_rq_page_evidence",
        "candidate_preregistration_sha256": candidate_sha,
        "candidate": {
            **candidate_identity,
            "negative_pages": [],
            "stale_pages": [],
            "source_page": "frozen-page",
            "split_role": "search_eval_only_not_answer_benchmark",
            "language": "en",
            "kind": "question",
            "preregistered_at": preregistered_at,
            "candidate_preregistration_sha256": candidate_sha,
        },
        "page_binding": {
            "page_id": "frozen-page",
            "page_uid": "uid-frozen-page",
            "content_sha256": content_sha,
            "content_byte_length": len(content),
        },
        "evidence_chunk": {
            "page_id": "frozen-page",
            "content_sha256": content_sha,
            "byte_start": 0,
            "byte_end": len(content),
            "excerpt": excerpt,
            "excerpt_sha256": content_sha,
            "truncated": False,
        },
        "reference_evidence_sha256": hashlib.sha256(
            reference.encode("utf-8")
        ).hexdigest(),
        "projection_policy_sha256": BOUNDED_EVIDENCE_PROJECTION_POLICY_SHA256,
    }


def _search_subject(packet: dict) -> dict:
    return {
        "schema_version": 1,
        "subject_kind": "search_label_candidate",
        "candidate_preregistration_sha256": packet["candidate_preregistration_sha256"],
        "source_packet_sha256": canonical_sha256(packet),
        "source_packet": packet,
        "evidence_sha256": packet["reference_evidence_sha256"],
        "producer_kind": "deterministic_evidence_projection",
        "producer_model": None,
        "producer_policy_sha256": SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256,
        "production_answer_used": False,
    }


def _decision(subject: dict, **overrides: object) -> dict:
    value = {
        "decision": "approved",
        "subject_kind": subject["subject_kind"],
        "subject_sha256": canonical_sha256(subject),
        "evidence_complete": True,
        "reference_independent": True,
        "preregistered_before_evaluation": True,
        "split_safe": True,
        "confidence": 0.99,
        "summary": "The frozen deterministic reference is fully supported.",
    }
    value.update(overrides)
    return value


def _router(
    tmp_path: Path,
    config: DecisionRouterConfig,
    transport: ModelTransport,
) -> DecisionRouter:
    router = DecisionRouter(
        config=config,
        transport=transport,
        audit_root=tmp_path / "runtime" / "machine-consensus-audit",
        resolve_adoption=False,
        require_adopted=True,
        artifact_replay=True,
        live_resource_control=False,
    )
    router.policy = RouterPolicyResolution(
        config=config,
        source="runtime_role_mapping",
    )
    router.config = config
    router._adoption_artifact_nominated = True
    return router


@pytest.fixture
def installed_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DecisionRouterConfig, dict]:
    from chronovisor.core import store
    from chronovisor.decision import decision_authority

    config = _config()
    authority = _authority(config)
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(
        decision_authority,
        "current_semantic_authority",
        lambda lane, **_kwargs: (
            (authority, None) if lane == LANE else (None, "wrong_lane")
        ),
    )
    return config, authority


def _append(
    tmp_path: Path,
    config: DecisionRouterConfig,
    authority: dict,
    subject: dict,
    responses: dict[str, list[str | Exception]],
    *,
    kind: str = "gold_entry_review",
    producer_policy_sha256: str = GOLD_ENTRY_PRODUCER_POLICY_SHA256,
    created_at: str = "2026-08-01T00:00:00Z",
) -> tuple[dict, ModelTransport, Path, str]:
    transport = ModelTransport(responses)
    router = _router(tmp_path, config, transport)
    prompt = build_recall_answer_adjudication_prompt(
        {"subject": subject, "subject_sha256": canonical_sha256(subject)}
    )
    ledger = tmp_path / "recall" / "answer-consensus-receipts.jsonl"
    result = append_machine_consensus_receipt(
        kind=kind,
        subject=subject,
        producer_policy_sha256=producer_policy_sha256,
        prompt=prompt,
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=LANE,
        ledger_file=ledger,
        chronovisor_root=tmp_path,
        router_factory=lambda _lane: router,
        authority_provider=lambda _lane: (authority, None),
        created_at=created_at,
    )
    return result, transport, ledger, prompt


def test_real_router_factory_publishes_subject_bound_machine_receipt(
    tmp_path: Path, installed_authority: tuple[DecisionRouterConfig, dict]
) -> None:
    config, authority = installed_authority
    subject = _subject()
    approved = json.dumps(_decision(subject))

    result, transport, ledger, prompt = _append(
        tmp_path,
        config,
        authority,
        subject,
        {
            config.primary_model: [approved],
            config.challenger_model: [approved],
        },
    )

    assert result["status"] == "accepted"
    assert [request.model for request in transport.requests] == [
        config.primary_model,
        config.challenger_model,
    ]
    receipt = result["receipt"]
    check = validate_machine_consensus_receipt(
        receipt["receipt_sha256"],
        expected_kind="gold_entry_review",
        expected_subject=subject,
        expected_producer_policy_sha256=GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        prompt=prompt,
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=LANE,
        ledger_file=ledger,
        chronovisor_root=tmp_path,
        current_authority=authority,
    )
    assert check["passed"] is True
    assert check["artifact"]["provenance"]["vote_manifest_sha256"]
    trace = [
        json.loads(line)
        for line in (
            tmp_path / "runtime" / "machine-consensus-audit" / "trace-events.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    receipt_event = trace[-1]
    assert receipt_event["kind"] == "machine_consensus_receipt"
    assert receipt_event["status"] == "done"
    assert receipt_event["receipt_status"] == "accepted"
    assert receipt_event["receipt_sha256"] == receipt["receipt_sha256"]
    assert (
        receipt_event["decision_artifact_seal_sha256"]
        == receipt["decision_artifact_seal_sha256"]
    )


def test_false_safety_vote_is_held_before_receipt_append(
    tmp_path: Path, installed_authority: tuple[DecisionRouterConfig, dict]
) -> None:
    config, authority = installed_authority
    subject = _subject()
    unsafe = json.dumps(_decision(subject, evidence_complete=False))

    result, _transport, ledger, _prompt = _append(
        tmp_path,
        config,
        authority,
        subject,
        {
            config.primary_model: [unsafe],
            config.challenger_model: [unsafe],
        },
    )

    assert result == {
        "status": "held",
        "reason": "machine_consensus_subject_not_approved",
    }
    assert not ledger.exists()


def test_subject_mutation_authority_drift_and_receipt_replay_are_rejected(
    tmp_path: Path, installed_authority: tuple[DecisionRouterConfig, dict]
) -> None:
    config, authority = installed_authority
    subject = _subject()
    approved = json.dumps(_decision(subject))
    result, _transport, ledger, prompt = _append(
        tmp_path,
        config,
        authority,
        subject,
        {
            config.primary_model: [approved],
            config.challenger_model: [approved],
        },
    )
    receipt_sha = result["receipt"]["receipt_sha256"]
    mutated = {**subject, "split": "locked-test"}
    replay = validate_machine_consensus_receipt(
        receipt_sha,
        expected_kind="gold_entry_review",
        expected_subject=mutated,
        expected_producer_policy_sha256=GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        prompt=build_recall_answer_adjudication_prompt(
            {"subject": mutated, "subject_sha256": canonical_sha256(mutated)}
        ),
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=LANE,
        ledger_file=ledger,
        chronovisor_root=tmp_path,
        current_authority=authority,
    )
    drifted = replace(
        RouterPolicyResolution(config=config, source="adopted_artifact"),
        artifact_sha256="9" * 64,
    )
    changed_authority = _authority(config, artifact=str(drifted.artifact_sha256))
    drift = validate_machine_consensus_receipt(
        receipt_sha,
        expected_kind="gold_entry_review",
        expected_subject=subject,
        expected_producer_policy_sha256=GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        prompt=prompt,
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=LANE,
        ledger_file=ledger,
        chronovisor_root=tmp_path,
        current_authority=changed_authority,
    )

    assert replay["passed"] is False
    assert replay["reason"] == "machine_consensus_subject_binding_invalid"
    assert drift["passed"] is False
    assert drift["reason"] == "decision authority changed before effect"


def test_no_quorum_and_duplicate_models_never_publish_receipt(
    tmp_path: Path, installed_authority: tuple[DecisionRouterConfig, dict]
) -> None:
    config, authority = installed_authority
    subject = _subject()
    approved = json.dumps(_decision(subject))
    abstained = json.dumps(_decision(subject, decision="abstained"))
    rejected = json.dumps(_decision(subject, decision="rejected"))
    no_quorum, transport, ledger, _prompt = _append(
        tmp_path,
        config,
        authority,
        subject,
        {
            config.primary_model: [approved],
            config.challenger_model: [rejected],
            config.tie_break_model: [abstained],
        },
    )
    duplicate = _config(
        challenger_model=config.primary_model,
        tie_break_model=config.tie_break_model,
    )
    duplicate_authority = _authority(duplicate)
    duplicate_result, _dup_transport, duplicate_ledger, _ = _append(
        tmp_path / "duplicate",
        duplicate,
        duplicate_authority,
        subject,
        {duplicate.primary_model: [approved, approved]},
    )

    assert [request.model for request in transport.requests] == [
        config.primary_model,
        config.challenger_model,
        config.tie_break_model,
    ]
    assert no_quorum["status"] == "held"
    assert not ledger.exists()
    assert duplicate_result["status"] == "held"
    assert not duplicate_ledger.exists()


def test_invalid_vote_is_excluded_from_receipt_quorum(
    tmp_path: Path, installed_authority: tuple[DecisionRouterConfig, dict]
) -> None:
    config, authority = installed_authority
    subject = _subject()
    approved = json.dumps(_decision(subject))

    accepted, transport, ledger, _prompt = _append(
        tmp_path,
        config,
        authority,
        subject,
        {
            config.primary_model: [TimeoutError("not durable")],
            config.challenger_model: [approved],
            config.tie_break_model: [approved],
        },
    )

    assert accepted["status"] == "accepted"
    assert len(transport.requests) == 3
    assert ledger.exists()

    held_subject = {**subject, "episode_id": "episode-no-quorum"}
    held_approved = json.dumps(_decision(held_subject))
    held, held_transport, held_ledger, _ = _append(
        tmp_path / "no-quorum",
        config,
        authority,
        held_subject,
        {
            config.primary_model: [TimeoutError("not durable")],
            config.challenger_model: [held_approved],
            config.tie_break_model: [TimeoutError("not durable")],
        },
    )

    assert held["status"] == "held"
    assert len(held_transport.requests) == 3
    assert not held_ledger.exists()


def test_receipt_rechecks_fresh_runtime_authority_after_inference(
    tmp_path: Path,
    installed_authority: tuple[DecisionRouterConfig, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision import machine_consensus_receipt

    config, authority = installed_authority
    changed_authority = _authority(config, artifact="9" * 64)
    subject = _subject()
    approved = json.dumps(_decision(subject))
    transport = ModelTransport(
        {
            config.primary_model: [approved],
            config.challenger_model: [approved],
        }
    )
    router = _router(tmp_path, config, transport)
    observed_routers: list[object | None] = []

    def dynamic_authority(lane: str, **kwargs: object) -> tuple[dict, None]:
        assert lane == LANE
        observed_router = kwargs.get("router")
        observed_routers.append(observed_router)
        return (authority if observed_router is router else changed_authority), None

    monkeypatch.setattr(
        machine_consensus_receipt,
        "current_semantic_authority",
        dynamic_authority,
    )
    prompt = build_recall_answer_adjudication_prompt(
        {"subject": subject, "subject_sha256": canonical_sha256(subject)}
    )
    ledger = tmp_path / "recall" / "drifted-authority-receipts.jsonl"

    result = append_machine_consensus_receipt(
        kind="gold_entry_review",
        subject=subject,
        producer_policy_sha256=GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        prompt=prompt,
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=LANE,
        ledger_file=ledger,
        chronovisor_root=tmp_path,
        router_factory=lambda _lane: router,
        authority_provider=dynamic_authority,
    )

    assert result == {
        "status": "waiting",
        "reason": "decision authority changed before effect",
    }
    assert observed_routers == [router, None]
    assert len(transport.requests) == 2
    assert not ledger.exists()


def test_search_candidate_receipt_builds_offline_exact_source_ledger(
    tmp_path: Path,
    installed_authority: tuple[DecisionRouterConfig, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.recall import recall_answer_eval

    config, authority = installed_authority
    packet = _search_packet()
    subject = _search_subject(packet)
    approved = json.dumps(_decision(subject))
    result, _transport, ledger, _prompt = _append(
        tmp_path,
        config,
        authority,
        subject,
        {
            config.primary_model: [approved],
            config.challenger_model: [approved],
        },
        kind="search_label_candidate_review",
        producer_policy_sha256=SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256,
    )
    receipt_sha = result["receipt"]["receipt_sha256"]
    identity = {
        "candidate_preregistration_sha256": packet["candidate_preregistration_sha256"],
        "source_packet_sha256": canonical_sha256(packet),
        "consensus_receipt_sha256": receipt_sha,
    }
    source_sha = recall_answer_eval._canonical_sha(identity)
    entry = {
        "case_id": f"search-machine-{source_sha[:32]}",
        "source_entry_sha256": source_sha,
        **identity,
        "source_packet": packet,
    }
    source = seal_object(
        {
            "schema_version": 2,
            "artifact_kind": "machine-search-label-answer-source-ledger",
            "frozen_at": "2026-08-02T00:00:00Z",
            "entries": [entry],
            "entries_sha256": recall_answer_eval._canonical_sha([entry]),
            "retirements": [],
            "retirements_sha256": recall_answer_eval._canonical_sha([]),
            "consensus_ledger_path": str(ledger),
            "consensus_ledger_head_sha256": receipt_sha,
        }
    )
    checked = recall_answer_eval.validate_machine_answer_source_ledger(
        source,
        consensus_ledger_file=ledger,
        chronovisor_root=tmp_path,
    )
    monkeypatch.setattr(
        recall_answer_eval,
        "find_page",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("benchmark projection reread live Wiki")
        ),
    )
    projected = recall_answer_eval._packet_from_machine_source_entry(
        entry,
        source_authority_sha256=source["seal_sha256"],
        frozen_at=source["frozen_at"],
    )

    assert checked["passed"] is True
    assert projected["prompt"] == packet["candidate"]["query"]
    assert projected["page_bindings"] == [packet["page_binding"]]
    assert projected["evidence_chunks"] == [packet["evidence_chunk"]]
    benchmark_pointer = tmp_path / "runtime" / "recall-answer-eval" / "benchmark.json"
    benchmark = recall_answer_eval.build_machine_answer_benchmark_epoch(
        output_file=benchmark_pointer,
        source_ledger_dir=tmp_path
        / "runtime"
        / "recall-answer-eval"
        / "benchmark-source-ledgers",
        consensus_ledger_file=ledger,
        chronovisor_root=tmp_path,
    )
    benchmark_check = recall_answer_eval.validate_independent_answer_benchmark(
        benchmark_pointer, chronovisor_root=tmp_path
    )
    assert benchmark["status"] == "complete"
    assert benchmark_check["passed"] is True
    assert list(benchmark_check["entries"].values())[0]["evidence_chunks"] == [
        packet["evidence_chunk"]
    ]
    benchmark_packet = list(benchmark_check["entries"].values())[0]
    rubric_sha = recall_answer_eval._canonical_sha(
        {
            "version": 1,
            "dimensions": list(recall_answer_eval.ANSWER_DIMENSIONS),
            "reference": "deterministic_source_evidence_projection",
        }
    )
    gold_family_id = "machine-gold-" + benchmark_check["split_epoch_id"][:24]
    gold_answer = recall_answer_eval._packet_reference_evidence(benchmark_packet)
    gold_evidence = {
        "source_packet": benchmark_packet,
        "source_packet_sha256": canonical_sha256(benchmark_packet),
        "source_frozen_at": benchmark_check["payload"]["frozen_at"],
        "reference_policy_sha256": (
            recall_answer_eval.DETERMINISTIC_GOLD_PROJECTION_POLICY_SHA256
        ),
    }
    gold_entry = {
        "episode_id": benchmark_packet["case_id"],
        "gold_answer": gold_answer,
        "evidence": gold_evidence,
        "evidence_sha256": recall_answer_eval._canonical_sha(
            {
                "episode_id": benchmark_packet["case_id"],
                "gold_answer": gold_answer,
                "evidence": gold_evidence,
                "rubric_sha256": rubric_sha,
            }
        ),
    }
    gold_subject = recall_answer_eval._gold_machine_subject(
        gold_entry,
        rubric_sha256=rubric_sha,
        gold_family_id=gold_family_id,
        expected_split=benchmark_packet["split"],
        split_epoch_id=benchmark_check["split_epoch_id"],
    )
    source_time = datetime.fromisoformat(
        benchmark_check["payload"]["frozen_at"].replace("Z", "+00:00")
    )
    reviewed_at = (source_time + timedelta(seconds=1)).astimezone(UTC)
    gold_result, _gold_transport, _gold_ledger, _ = _append(
        tmp_path,
        config,
        authority,
        gold_subject,
        {
            config.primary_model: [json.dumps(_decision(gold_subject))],
            config.challenger_model: [json.dumps(_decision(gold_subject))],
        },
        kind="gold_entry_review",
        producer_policy_sha256=GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        created_at=reviewed_at.isoformat().replace("+00:00", "Z"),
    )
    gold_receipt = gold_result["receipt"]
    gold_entry["review_provenance"] = {
        "source_kind": "adjudicated_benchmark",
        "authority_kind": "adopted_local_consensus",
        "consensus_receipt_sha256": gold_receipt["receipt_sha256"],
        "subject_sha256": canonical_sha256(gold_subject),
        "reviewed_at": gold_receipt["created_at"],
    }
    gold_payload = {
        "schema_version": 1,
        "artifact_kind": "immutable-answer-gold-manifest",
        "frozen_at": (source_time + timedelta(seconds=2))
        .astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "gold_id": f"{gold_family_id}-{benchmark_packet['split']}",
        "gold_family_id": gold_family_id,
        "version": "machine-consensus-v1",
        "review_protocol_sha256": GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        "rubric_sha256": rubric_sha,
        "split": benchmark_packet["split"],
        "split_epoch_id": benchmark_check["split_epoch_id"],
        "benchmark_manifest_path": benchmark_check["manifest_path"],
        "benchmark_manifest_sha256": benchmark_check["manifest_sha256"],
        "entries": [gold_entry],
    }
    gold_path = (
        tmp_path
        / "runtime"
        / "recall-answer-eval"
        / "gold"
        / benchmark_packet["split"]
        / f"{benchmark_check['split_epoch_id']}.json"
    )
    write_sealed_json(gold_path, gold_payload, backup=False)
    gold_pointer = (
        tmp_path
        / "runtime"
        / "recall-answer-eval"
        / f"{benchmark_packet['split']}-gold-manifest.json"
    )
    gold_manifest = json.loads(gold_path.read_text(encoding="utf-8"))
    write_sealed_json(
        gold_pointer,
        {
            "schema_version": 1,
            "artifact_kind": "immutable-answer-gold-active-pointer",
            "split": benchmark_packet["split"],
            "split_epoch_id": benchmark_check["split_epoch_id"],
            "manifest_path": str(gold_path),
            "manifest_sha256": gold_manifest["seal_sha256"],
            "updated_at": gold_payload["frozen_at"],
        },
        backup=False,
    )
    # Advancing or corrupting the active benchmark pointer cannot change the
    # immutable epoch pinned inside this gold artifact.
    write_sealed_json(
        benchmark_pointer,
        {
            "schema_version": 1,
            "artifact_kind": "independent-answer-benchmark-active-pointer",
            "epoch_sha256": "f" * 64,
            "manifest_sha256": "e" * 64,
            "manifest_path": str(
                benchmark_pointer.parent / "benchmarks" / f"{'f' * 64}.json"
            ),
            "updated_at": gold_payload["frozen_at"],
        },
        backup=True,
    )
    gold_check = recall_answer_eval.validate_gold_manifest(
        gold_pointer,
        required_episode_ids=[benchmark_packet["case_id"]],
        consensus_ledger_file=ledger,
        chronovisor_root=tmp_path,
        expected_split=benchmark_packet["split"],
        split_epoch_id=benchmark_check["split_epoch_id"],
        benchmark_manifest=benchmark_pointer,
    )
    assert gold_check["passed"] is True

    mixed_payload = json.loads(json.dumps(gold_payload))
    mixed_payload["entries"][0]["review_provenance"]["source_kind"] = "human_review"
    mixed_check = recall_answer_eval.validate_gold_manifest(
        seal_object(mixed_payload),
        required_episode_ids=[benchmark_packet["case_id"]],
        consensus_ledger_file=ledger,
        chronovisor_root=tmp_path,
        expected_split=benchmark_packet["split"],
        split_epoch_id=benchmark_check["split_epoch_id"],
        benchmark_manifest=benchmark_pointer,
    )
    assert mixed_check == {
        "passed": False,
        "reason": "machine_gold_entry_shape_invalid",
    }

    tampered_packet = json.loads(json.dumps(packet))
    tampered_packet["production_answer"] = "forbidden"
    tampered_identity = {
        **identity,
        "source_packet_sha256": canonical_sha256(tampered_packet),
    }
    tampered_sha = recall_answer_eval._canonical_sha(tampered_identity)
    tampered_entry = {
        "case_id": f"search-machine-{tampered_sha[:32]}",
        "source_entry_sha256": tampered_sha,
        **tampered_identity,
        "source_packet": tampered_packet,
    }
    tampered = seal_object(
        {
            **{
                key: value
                for key, value in source.items()
                if key not in {"seal_sha256", "entries", "entries_sha256"}
            },
            "entries": [tampered_entry],
            "entries_sha256": recall_answer_eval._canonical_sha([tampered_entry]),
        }
    )
    rejected = recall_answer_eval.validate_machine_answer_source_ledger(
        tampered,
        consensus_ledger_file=ledger,
        chronovisor_root=tmp_path,
    )
    assert rejected == {
        "passed": False,
        "reason": "machine_answer_source_epoch_invalid",
    }


def test_receipt_ledger_rejects_bad_tail_future_time_and_wrong_policy(
    tmp_path: Path, installed_authority: tuple[DecisionRouterConfig, dict]
) -> None:
    config, authority = installed_authority
    subject = _subject()
    approved = json.dumps(_decision(subject))
    result, _transport, ledger, prompt = _append(
        tmp_path,
        config,
        authority,
        subject,
        {
            config.primary_model: [approved],
            config.challenger_model: [approved],
        },
    )
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write("{malformed-tail\n")
    invalid = validate_machine_consensus_receipt(
        result["receipt"]["receipt_sha256"],
        expected_kind="gold_entry_review",
        expected_subject=subject,
        expected_producer_policy_sha256=GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        prompt=prompt,
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=LANE,
        ledger_file=ledger,
        chronovisor_root=tmp_path,
        current_authority=authority,
    )
    assert invalid == {
        "passed": False,
        "reason": "machine_consensus_ledger_json_invalid",
    }

    future, future_transport, _future_ledger, _ = _append(
        tmp_path / "future",
        config,
        authority,
        subject,
        {
            config.primary_model: [approved],
            config.challenger_model: [approved],
        },
        created_at="2999-01-01T00:00:00Z",
    )
    assert future == {
        "status": "held",
        "reason": "machine_consensus_created_at_invalid",
    }
    assert future_transport.requests == []

    wrong, wrong_transport, _wrong_ledger, _ = _append(
        tmp_path / "wrong-policy",
        config,
        authority,
        {**subject, "producer_policy_sha256": "f" * 64},
        {
            config.primary_model: [approved],
            config.challenger_model: [approved],
        },
        producer_policy_sha256="f" * 64,
    )
    assert wrong["status"] == "held"
    assert wrong["reason"] == "machine_consensus_producer_invalid"
    assert wrong_transport.requests == []


def test_receipt_ledger_rejects_rehashed_extra_canary_field(
    tmp_path: Path, installed_authority: tuple[DecisionRouterConfig, dict]
) -> None:
    config, authority = installed_authority
    subject = _subject()
    approved = json.dumps(_decision(subject))
    result, _transport, ledger, prompt = _append(
        tmp_path,
        config,
        authority,
        subject,
        {
            config.primary_model: [approved],
            config.challenger_model: [approved],
        },
    )
    row = json.loads(ledger.read_text(encoding="utf-8"))
    canary = "CANARY private prompt must not be durable"
    row["prompt"] = canary
    row["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "receipt_sha256"}
    )
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")

    checked = validate_machine_consensus_receipt(
        row["receipt_sha256"],
        expected_kind="gold_entry_review",
        expected_subject=subject,
        expected_producer_policy_sha256=GOLD_ENTRY_PRODUCER_POLICY_SHA256,
        prompt=prompt,
        schema=RECALL_ANSWER_ADJUDICATION_SCHEMA,
        system=None,
        lane=LANE,
        ledger_file=ledger,
        chronovisor_root=tmp_path,
        current_authority=authority,
    )

    assert checked == {
        "passed": False,
        "reason": "machine_consensus_ledger_chain_invalid",
    }
    assert canary not in repr(checked)
    assert result["status"] == "accepted"


def test_newer_packet_supersedes_same_logical_candidate_without_live_read(
    tmp_path: Path, installed_authority: tuple[DecisionRouterConfig, dict]
) -> None:
    from chronovisor.recall import recall_answer_eval

    config, authority = installed_authority
    old_packet = _search_packet(
        content=b"old frozen evidence",
        preregistered_at="2026-07-31T00:00:00Z",
    )
    new_packet = _search_packet(
        content=b"new frozen evidence",
        preregistered_at="2026-07-31T00:00:00.500000Z",
    )
    old_subject = _search_subject(old_packet)
    new_subject = _search_subject(new_packet)
    old_result, _transport, ledger, _ = _append(
        tmp_path,
        config,
        authority,
        old_subject,
        {
            config.primary_model: [json.dumps(_decision(old_subject))],
            config.challenger_model: [json.dumps(_decision(old_subject))],
        },
        kind="search_label_candidate_review",
        producer_policy_sha256=SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256,
        created_at="2026-07-31T00:00:00.100000Z",
    )
    new_result, _transport, _ledger, _ = _append(
        tmp_path,
        config,
        authority,
        new_subject,
        {
            config.primary_model: [json.dumps(_decision(new_subject))],
            config.challenger_model: [json.dumps(_decision(new_subject))],
        },
        kind="search_label_candidate_review",
        producer_policy_sha256=SEARCH_LABEL_CANDIDATE_PRODUCER_POLICY_SHA256,
        created_at="2026-07-31T00:00:00.600000Z",
    )
    chain = recall_answer_eval.list_machine_consensus_receipts(ledger_file=ledger)
    entries, retirements = recall_answer_eval._machine_source_entries_from_receipts(
        chain["receipts"],
        consensus_ledger_file=ledger,
        chronovisor_root=tmp_path,
    )

    assert len(entries) == 1
    assert entries[0]["source_packet"] == new_packet
    assert len(retirements) == 1
    assert (
        retirements[0]["superseded_consensus_receipt_sha256"]
        == old_result["receipt"]["receipt_sha256"]
    )
    assert (
        retirements[0]["superseded_by_consensus_receipt_sha256"]
        == new_result["receipt"]["receipt_sha256"]
    )


def test_future_candidate_does_not_starve_latest_valid_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.recall import recall_answer_eval

    candidate_file = tmp_path / "recall" / "search-label-queue.jsonl"
    candidate_file.parent.mkdir(parents=True)
    rows = [
        {
            "source": "recall_questions",
            "candidate_sha256": "a" * 64,
            "query": "same logical query",
            "page_uid": "same-page-uid",
            "preregistered_at": "2026-08-01T00:00:00Z",
        },
        {
            "source": "recall_questions",
            "candidate_sha256": "b" * 64,
            "query": "same logical query",
            "page_uid": "same-page-uid",
            "preregistered_at": "2999-01-01T00:00:00Z",
        },
    ]
    candidate_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    frozen: list[str] = []

    def freeze(row: dict) -> dict:
        frozen.append(row["candidate_sha256"])
        return {}

    monkeypatch.setattr(
        recall_answer_eval, "_freeze_search_label_candidate_packet", freeze
    )
    result = recall_answer_eval.adjudicate_machine_search_label_candidates(
        candidate_file=candidate_file,
        consensus_ledger_file=tmp_path / "recall" / "answer-consensus-receipts.jsonl",
        chronovisor_root=tmp_path,
        dry_run=True,
    )

    assert frozen == ["a" * 64]
    assert result["stale"] == 1
    assert result["superseded"] == 0
    assert result["status"] == "waiting"
