from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronovisor.core import llm_config, ollama
from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    RouteLocation,
)
from chronovisor.core.reranker import RerankOutcome
from chronovisor.core.runtime_config import RerankerConfig, RerankerServiceConfig
from chronovisor.core.search_types import ScoredPage
from chronovisor.recall import recall_processor
from chronovisor.recall.evidence_certificate import EvidenceCertificate


def page(page_id: str) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-07-30",
        score=1.0,
    )


def shadow_config() -> RerankerConfig:
    return RerankerConfig(
        enabled=True,
        top_n=2,
        service=RerankerServiceConfig(
            enabled=True,
            mode="shadow",
            timeout_ms=500,
        ),
    )


def active_config() -> RerankerConfig:
    return RerankerConfig(
        enabled=True,
        top_n=2,
        service=RerankerServiceConfig(
            enabled=True,
            mode="on",
            timeout_ms=500,
        ),
    )


def test_shadow_reranker_records_before_after_without_mutating(monkeypatch) -> None:
    candidates = [page("a"), page("b")]
    monkeypatch.setattr(
        recall_processor, "load_reranker_config", shadow_config
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "selected_for_rollout",
        lambda _query, _config: True,
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "rerank",
        lambda _query, values, **_kwargs: RerankOutcome(
            [values[1], values[0]],
            {
                "status": "applied",
                "execution": "service",
                "latency_ms": 7,
                "scores": [],
            },
        ),
    )

    payload = recall_processor.shadow_rerank_candidates(
        "query", candidates, timeout_ms=600
    )

    assert [item.page_id for item in candidates] == ["a", "b"]
    assert payload["before_page_ids"] == ["a", "b"]
    assert payload["after_page_ids"] == ["b", "a"]
    assert payload["changed"] is True


def test_shadow_reranker_fails_open(monkeypatch) -> None:
    monkeypatch.setattr(
        recall_processor, "load_reranker_config", shadow_config
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "selected_for_rollout",
        lambda _query, _config: True,
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "rerank",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("service stopped")
        ),
    )

    payload = recall_processor.shadow_rerank_candidates(
        "query", [page("a")], timeout_ms=500
    )

    assert payload["status"] == "unavailable"
    assert payload["reason"] == "reranker_unavailable"
    assert payload["degraded"] is True


def test_active_reranker_changes_candidate_order(monkeypatch) -> None:
    candidates = [page("a"), page("b")]
    monkeypatch.setattr(recall_processor, "load_reranker_config", active_config)
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "selected_for_rollout",
        lambda _query, _config: True,
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "rerank",
        lambda _query, values, **_kwargs: RerankOutcome(
            [values[1], values[0]],
            {"status": "applied", "scores": [], "model": "bge"},
        ),
    )

    ranked, metadata = recall_processor.rank_recall_candidates(
        "query",
        candidates,
        timeout_ms=500,
    )

    assert [item.page_id for item in ranked] == ["b", "a"]
    assert metadata["status"] == "applied"


def test_active_reranker_failure_returns_original_candidates(monkeypatch) -> None:
    candidates = [page("a"), page("b")]
    monkeypatch.setattr(recall_processor, "load_reranker_config", active_config)
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "selected_for_rollout",
        lambda _query, _config: True,
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "rerank",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("service stopped")
        ),
    )

    ranked, metadata = recall_processor.rank_recall_candidates(
        "query",
        candidates,
        timeout_ms=500,
    )

    assert ranked is candidates
    assert metadata["fail_open"] is True
    assert metadata["reason"] == "reranker_unavailable"
    assert metadata["degraded"] is True


def certificate(page_id: str, *, outcome: str = "pass") -> EvidenceCertificate:
    return EvidenceCertificate(
        certificate_id=f"cert-{page_id}",
        page_id=page_id,
        outcome=outcome,
        confidence=0.9 if outcome == "pass" else 0.1,
        label_quality="strong" if outcome == "pass" else "silver",
        supporting_span=f"evidence unique {page_id}",
        source_line=1,
        query_sha256="query",
        content_sha256=f"content-{page_id}",
        policy_sha256="policy",
        model_revision="bge",
        features={},
        reasons=("test",),
        created_at="2026-07-30T22:00:00",
    )


def judge_route(role: str, *, location: str = "remote") -> dict[str, str | None]:
    stage = (
        "primary"
        if role == recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE
        else "escalation"
    )
    return {
        "role": role,
        "provider": "ollama" if location == "local" else "remote-test",
        "model": (
            "judge-local"
            if location == "local" and stage == "primary"
            else f"judge-{stage}-{location}"
        ),
        "location": location,
        "model_digest": "digest-local" if location == "local" else None,
    }


def distinct_judge_routes() -> tuple[ollama.RuntimeGenerationRoute, ...]:
    return tuple(
        ollama.RuntimeGenerationRoute(
            role,
            "remote-test",
            str(judge_route(role)["model"]),
            "remote",
            True,
        )
        for role in (
            recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
            recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE,
        )
    )


def test_selection_is_dynamic_and_caps_rich_and_pointer_counts(monkeypatch) -> None:
    candidates = [page(f"page-{index}") for index in range(8)]
    monkeypatch.setattr(
        recall_processor,
        "certify_candidate",
        lambda _query, value, **_kwargs: certificate(value.page_id),
    )
    monkeypatch.setattr(
        recall_processor,
        "append_certificates",
        lambda values: len(values),
    )

    selected, metadata = recall_processor.select_certified_candidates(
        "multi intent query",
        candidates,
        reranker_metadata={},
        max_candidates=8,
        max_pointer_cards=6,
        max_rich_evidence=2,
        injection_token_budget=1200,
        certificate_required=True,
    )

    assert len(selected) == 6
    assert [item.evidence_kind for item in selected].count("rich") == 2
    assert [item.evidence_kind for item in selected].count("pointer") == 4
    assert metadata["selected_count"] == 6
    assert metadata["ledger_written"] == 8


def test_selection_uses_resolved_rerank_route_model(monkeypatch) -> None:
    seen: list[dict[str, Any] | None] = []

    def certify(_query, value, **kwargs):
        seen.append(kwargs.get("reranker_score"))
        return certificate(value.page_id)

    monkeypatch.setattr(recall_processor, "certify_candidate", certify)
    monkeypatch.setattr(recall_processor, "append_certificates", lambda _values: 1)

    recall_processor.select_certified_candidates(
        "query",
        [page("page-a")],
        reranker_metadata={
            "route": {
                "role": "search.rerank",
                "provider": "provider",
                "model": "route-model",
                "location": "remote",
            },
            "candidate_count": 1,
            "scores": [{"page_id": "page-a", "raw_score": 1.0}],
        },
        max_candidates=1,
        max_pointer_cards=1,
        max_rich_evidence=1,
        injection_token_budget=1200,
        certificate_required=True,
    )

    assert seen[0] is not None
    assert seen[0]["model_revision"] == "route-model"


def test_selection_abstains_when_every_certificate_rejects(monkeypatch) -> None:
    candidates = [page("noise-a"), page("noise-b")]
    monkeypatch.setattr(
        recall_processor,
        "certify_candidate",
        lambda _query, value, **_kwargs: certificate(
            value.page_id,
            outcome="reject",
        ),
    )
    monkeypatch.setattr(recall_processor, "append_certificates", lambda values: 2)

    selected, metadata = recall_processor.select_certified_candidates(
        "unrelated",
        candidates,
        reranker_metadata={},
        max_candidates=10,
        max_pointer_cards=6,
        max_rich_evidence=2,
        injection_token_budget=1200,
        certificate_required=True,
    )

    assert selected == []
    assert metadata["status"] == "abstained"
    assert metadata["certificate_pass_count"] == 0


def test_certificate_judge_resolves_only_two_ambiguous_pages(monkeypatch) -> None:
    certificates = [certificate(f"page-{index}") for index in range(3)]
    certificates = [
        item
        if index == 2
        else EvidenceCertificate(
            **{
                **item.__dict__,
                "confidence": 0.5,
            }
        )
        for index, item in enumerate(certificates)
    ]
    monkeypatch.setattr(
        recall_processor,
        "_run_certificate_judge",
        lambda _query, values, **_kwargs: (
            {
                values[0].page_id: {
                    "page_id": values[0].page_id,
                    "decision": "pass",
                    "confidence": 0.9,
                },
                values[1].page_id: {
                    "page_id": values[1].page_id,
                    "decision": "reject",
                    "confidence": 0.8,
                },
            },
            "ok",
            judge_route(recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE),
        ),
    )
    monkeypatch.setattr(
        ollama, "runtime_generation_routes", lambda _roles: distinct_judge_routes()
    )
    policy = SimpleNamespace(
        processor_judge_timeout_ms=500,
        judge_keep_alive="24h",
    )

    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "first and second",
        certificates,
        policy=policy,
        timeout_ms=500,
    )

    assert resolved[0].outcome == "pass"
    assert resolved[0].label_quality == "strong"
    assert resolved[1].outcome == "reject"
    assert resolved[2].outcome == "reject"
    assert "unjudged_precision_gate" in resolved[2].reasons
    assert metadata["candidate_count"] == 2
    assert metadata["primary_route_identity"]["role"] == (
        recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE
    )
    assert resolved[0].features["certificate_judge"] == {
        "primary_route_identity": metadata["primary_route_identity"],
        "escalation_route_identity": None,
    }


def test_certificate_judge_escalates_uncertain_result(monkeypatch) -> None:
    item = EvidenceCertificate(
        **{
            **certificate("page-a").__dict__,
            "confidence": 0.5,
        }
    )
    calls: list[str] = []

    def fake_judge(_query, values, *, runtime_role, **_kwargs):
        calls.append(runtime_role)
        primary = runtime_role == recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE
        confidence = 0.6 if primary else 0.95
        return (
            {
                values[0].page_id: {
                    "page_id": values[0].page_id,
                    "decision": "uncertain"
                    if primary
                    else "pass",
                    "confidence": confidence,
                    "reason": "fixture",
                }
            },
            "ok",
            judge_route(runtime_role),
        )

    monkeypatch.setattr(recall_processor, "_run_certificate_judge", fake_judge)
    monkeypatch.setattr(
        ollama, "runtime_generation_routes", lambda _roles: distinct_judge_routes()
    )
    policy = SimpleNamespace(
        processor_judge_timeout_ms=300,
        judge_keep_alive="24h",
        processor_escalation_timeout_ms=500,
    )

    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "query",
        [item],
        policy=policy,
        timeout_ms=1000,
    )

    assert calls == [
        recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE,
    ]
    assert resolved[0].outcome == "pass"
    assert "escalation_judge_pass" in resolved[0].reasons
    assert metadata["escalation_status"] == "ok"


def test_local_certificate_judge_binds_digest_and_exact_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = ollama.RuntimeGenerationRoute(
        recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        "ollama",
        "judge-local",
        "local",
        True,
    )
    digest_calls: list[list[str]] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))

    def model_digests(models: list[str]) -> dict[str, str]:
        digest_calls.append(models)
        return {"judge-local": "digest-local"}

    def structured_chat(_messages: object, **kwargs: Any) -> ollama.ChatResponse:
        captured.update(kwargs)
        return ollama.ChatResponse(
            content=json.dumps(
                {
                    "verdicts": [
                        {
                            "page_id": "page-a",
                            "decision": "pass",
                            "confidence": 0.9,
                            "reason": "supported",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(ollama, "model_digests", model_digests)
    monkeypatch.setattr(ollama, "runtime_structured_chat", structured_chat)

    verdicts, status, identity = recall_processor._run_certificate_judge(
        "query",
        [certificate("page-a")],
        runtime_role=recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        timeout_ms=500,
        keep_alive="0",
    )

    assert status == "ok"
    assert verdicts["page-a"]["decision"] == "pass"
    assert identity == judge_route(
        recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE, location="local"
    )
    assert digest_calls == [["judge-local"]]
    assert captured["runtime_role"] == recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE
    assert captured["source_data_class"] == "raw"
    assert captured["source_sensitivity"] == "high"
    assert captured["format"]["properties"]["verdicts"]["minItems"] == 1


def test_missing_local_digest_fails_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = ollama.RuntimeGenerationRoute(
        recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        "ollama",
        "judge-local",
        "local",
        True,
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))
    monkeypatch.setattr(ollama, "model_digests", lambda _models: {})
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: pytest.fail("missing digest reached backend"),
    )

    verdicts, status, identity = recall_processor._run_certificate_judge(
        "query",
        [certificate("page-a")],
        runtime_role=recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        timeout_ms=500,
        keep_alive="0",
    )

    assert verdicts == {}
    assert status == "runtime_route_unavailable"
    assert identity is None


def _forbid_ollama_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("remote certificate judge touched an Ollama control")

    for name in (
        "chat",
        "model_digests",
        "model_resource_lease",
        "plan_model_residency",
        "resident_model_rows",
        "unload_named_model",
        "unload_model",
    ):
        monkeypatch.setattr(ollama, name, forbidden)


def test_decisive_primary_does_not_touch_escalation_digest_or_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = (
        ollama.RuntimeGenerationRoute(
            recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
            "ollama",
            "primary-local",
            "local",
            True,
        ),
        ollama.RuntimeGenerationRoute(
            recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE,
            "ollama",
            "escalation-local",
            "local",
            True,
        ),
    )
    digest_calls: list[list[str]] = []
    backend_roles: list[str] = []
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)

    def model_digests(models: list[str]) -> dict[str, str]:
        digest_calls.append(models)
        return {"primary-local": "digest-primary"}

    def structured_chat(_messages: object, **kwargs: Any) -> ollama.ChatResponse:
        backend_roles.append(str(kwargs["runtime_role"]))
        return ollama.ChatResponse(
            content=json.dumps(
                {
                    "verdicts": [
                        {
                            "page_id": "page-a",
                            "decision": "pass",
                            "confidence": 0.9,
                            "reason": "supported",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(ollama, "model_digests", model_digests)
    monkeypatch.setattr(ollama, "runtime_structured_chat", structured_chat)
    for name in (
        "chat",
        "model_resource_lease",
        "plan_model_residency",
        "resident_model_rows",
        "unload_named_model",
        "unload_model",
    ):
        monkeypatch.setattr(
            ollama,
            name,
            lambda *_args, **_kwargs: pytest.fail("touched Ollama control"),
        )

    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "query",
        [replace_certificate_confidence(certificate("page-a"), 0.5)],
        policy=SimpleNamespace(
            processor_judge_timeout_ms=500,
            processor_escalation_timeout_ms=500,
            judge_keep_alive="0",
        ),
        timeout_ms=1000,
    )

    assert resolved[0].outcome == "pass"
    assert metadata["escalation_status"] == "not_needed"
    assert digest_calls == [["primary-local"]]
    assert backend_roles == [recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE]


def test_duplicate_judge_routes_fail_before_digest_backend_or_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = tuple(
        ollama.RuntimeGenerationRoute(
            role,
            "ollama",
            "same-local-model",
            "local",
            True,
        )
        for role in (
            recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
            recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE,
        )
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: pytest.fail("duplicate routes reached backend"),
    )
    _forbid_ollama_controls(monkeypatch)

    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "query",
        [replace_certificate_confidence(certificate("page-a"), 0.5)],
        policy=SimpleNamespace(
            processor_judge_timeout_ms=500,
            processor_escalation_timeout_ms=500,
            judge_keep_alive="0",
        ),
        timeout_ms=1000,
    )

    assert resolved[0].outcome == "reject"
    assert "primary_judge_fail_closed" in resolved[0].reasons
    assert metadata["primary_status"] == "runtime_route_invalid"
    assert metadata["escalation_status"] == "blocked_by_primary"


def test_unstructured_judge_route_fails_before_backend_or_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = list(distinct_judge_routes())
    routes[0] = ollama.RuntimeGenerationRoute(
        recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        "ollama",
        "primary-local",
        "local",
        False,
    )
    monkeypatch.setattr(
        ollama, "runtime_generation_routes", lambda _roles: tuple(routes)
    )
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: pytest.fail("invalid route reached backend"),
    )
    _forbid_ollama_controls(monkeypatch)

    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "query",
        [replace_certificate_confidence(certificate("page-a"), 0.5)],
        policy=SimpleNamespace(
            processor_judge_timeout_ms=500,
            processor_escalation_timeout_ms=500,
            judge_keep_alive="0",
        ),
        timeout_ms=1000,
    )

    assert resolved[0].outcome == "reject"
    assert metadata["primary_status"] == "runtime_route_invalid"
    assert metadata["escalation_status"] == "blocked_by_primary"


def test_remote_certificate_judge_uses_runtime_without_ollama_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = ollama.RuntimeGenerationRoute(
        recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        "remote-test",
        "judge-remote",
        "remote",
        True,
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))

    def structured_chat(_messages: object, **kwargs: Any) -> ollama.ChatResponse:
        captured.update(kwargs)
        return ollama.ChatResponse(
            content=json.dumps(
                {
                    "verdicts": [
                        {
                            "page_id": "page-a",
                            "decision": "reject",
                            "confidence": 0.8,
                            "reason": "weak",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(ollama, "runtime_structured_chat", structured_chat)
    _forbid_ollama_controls(monkeypatch)

    _verdicts, status, identity = recall_processor._run_certificate_judge(
        "query",
        [certificate("page-a")],
        runtime_role=recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        timeout_ms=500,
        keep_alive="0",
    )

    assert status == "ok"
    assert identity == {
        "role": recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        "provider": "remote-test",
        "model": "judge-remote",
        "location": "remote",
        "model_digest": None,
    }
    assert captured["source_data_class"] == "raw"
    assert captured["source_sensitivity"] == "high"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "verdicts": [
                {
                    "page_id": "page-a",
                    "decision": "pass",
                    "confidence": 0.9,
                    "reason": "missing page",
                }
            ]
        },
        {
            "verdicts": [
                {
                    "page_id": "page-a",
                    "decision": "pass",
                    "confidence": 0.9,
                    "reason": "first",
                },
                {
                    "page_id": "page-a",
                    "decision": "reject",
                    "confidence": 0.8,
                    "reason": "duplicate",
                },
            ]
        },
        {
            "verdicts": [
                {
                    "page_id": "page-a",
                    "decision": "pass",
                    "confidence": 0.9,
                    "reason": "first",
                },
                {
                    "page_id": "unknown",
                    "decision": "reject",
                    "confidence": 0.8,
                    "reason": "unknown",
                },
            ]
        },
        {
            "verdicts": [
                {
                    "page_id": "page-a",
                    "decision": "pass",
                    "confidence": True,
                    "reason": "wrong type",
                },
                {
                    "page_id": "page-b",
                    "decision": "reject",
                    "confidence": 0.8,
                    "reason": "second",
                },
            ]
        },
        {
            "verdicts": [
                {
                    "page_id": "page-a",
                    "decision": "pass",
                    "confidence": 0.9,
                    "reason": "first",
                    "extra": "forbidden",
                },
                {
                    "page_id": "page-b",
                    "decision": "reject",
                    "confidence": 0.8,
                    "reason": "second",
                },
            ]
        },
    ],
    ids=("missing", "duplicate", "unknown", "type", "field"),
)
def test_invalid_primary_verdicts_reject_without_escalation(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    roles: list[tuple[str, ...]] = []

    def resolve(
        requested: tuple[str, ...],
    ) -> tuple[ollama.RuntimeGenerationRoute, ...]:
        roles.append(requested)
        return distinct_judge_routes()

    monkeypatch.setattr(ollama, "runtime_generation_routes", resolve)
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: ollama.ChatResponse(content=json.dumps(payload)),
    )
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: pytest.fail("remote route queried digests"),
    )
    policy = SimpleNamespace(
        processor_judge_timeout_ms=500,
        processor_escalation_timeout_ms=500,
        judge_keep_alive="0",
    )

    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "first and second",
        [
            replace_certificate_confidence(certificate("page-a"), 0.5),
            replace_certificate_confidence(certificate("page-b"), 0.5),
        ],
        policy=policy,
        timeout_ms=1000,
    )

    assert [item.outcome for item in resolved] == ["reject", "reject"]
    assert all("primary_judge_fail_closed" in item.reasons for item in resolved)
    assert metadata["status"] == "primary_judge_fail_closed"
    assert metadata["escalation_status"] == "blocked_by_primary"
    assert roles == [
        (
            recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
            recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE,
        )
    ]


def replace_certificate_confidence(
    value: EvidenceCertificate, confidence: float
) -> EvidenceCertificate:
    return EvidenceCertificate(**{**value.__dict__, "confidence": confidence})


def test_primary_backend_failure_rejects_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles: list[tuple[str, ...]] = []

    def resolve(
        requested: tuple[str, ...],
    ) -> tuple[ollama.RuntimeGenerationRoute, ...]:
        roles.append(requested)
        return distinct_judge_routes()

    monkeypatch.setattr(ollama, "runtime_generation_routes", resolve)
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend")),
    )
    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "query",
        [replace_certificate_confidence(certificate("page-a"), 0.5)],
        policy=SimpleNamespace(
            processor_judge_timeout_ms=500,
            processor_escalation_timeout_ms=500,
            judge_keep_alive="0",
        ),
        timeout_ms=1000,
    )

    assert resolved[0].outcome == "reject"
    assert "primary_judge_fail_closed" in resolved[0].reasons
    assert metadata["escalation_status"] == "blocked_by_primary"
    assert roles == [
        (
            recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
            recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE,
        )
    ]


@pytest.mark.parametrize(
    ("escalation_outcome", "expected_reason"),
    [
        (
            (
                {
                    "page-a": {
                        "page_id": "page-a",
                        "decision": "pass",
                        "confidence": 0.89,
                        "reason": "below threshold",
                    }
                },
                "ok",
                judge_route(recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE),
            ),
            "escalation_judge_reject",
        ),
        (
            (
                {},
                "runtime_backend_unavailable",
                judge_route(recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE),
            ),
            "escalation_judge_fail_closed",
        ),
    ],
    ids=("subthreshold", "backend"),
)
def test_escalation_failure_and_subthreshold_pass_reject(
    monkeypatch: pytest.MonkeyPatch,
    escalation_outcome: tuple[
        dict[str, dict[str, object]], str, dict[str, str | None]
    ],
    expected_reason: str,
) -> None:
    outcomes = iter(
        [
            (
                {
                    "page-a": {
                        "page_id": "page-a",
                        "decision": "uncertain",
                        "confidence": 0.6,
                        "reason": "uncertain",
                    }
                },
                "ok",
                judge_route(recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE),
            ),
            escalation_outcome,
        ]
    )
    monkeypatch.setattr(
        recall_processor,
        "_run_certificate_judge",
        lambda *_args, **_kwargs: next(outcomes),
    )
    monkeypatch.setattr(
        ollama, "runtime_generation_routes", lambda _roles: distinct_judge_routes()
    )

    resolved, _metadata = recall_processor.judge_ambiguous_certificates(
        "query",
        [replace_certificate_confidence(certificate("page-a"), 0.5)],
        policy=SimpleNamespace(
            processor_judge_timeout_ms=500,
            processor_escalation_timeout_ms=500,
            judge_keep_alive="0",
        ),
        timeout_ms=1000,
    )

    assert resolved[0].outcome == "reject"
    assert expected_reason in resolved[0].reasons


def test_certificate_receipt_and_metadata_bind_judge_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.recall.evidence_certificate import append_certificates

    route = judge_route(recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE)
    monkeypatch.setattr(
        recall_processor,
        "certify_candidate",
        lambda _query, value, **_kwargs: replace_certificate_confidence(
            certificate(value.page_id), 0.5
        ),
    )
    monkeypatch.setattr(
        ollama, "runtime_generation_routes", lambda _roles: distinct_judge_routes()
    )
    monkeypatch.setattr(
        recall_processor,
        "_run_certificate_judge",
        lambda _query, values, **_kwargs: (
            {
                values[0].page_id: {
                    "page_id": values[0].page_id,
                    "decision": "pass",
                    "confidence": 0.9,
                    "reason": "supported",
                }
            },
            "ok",
            route,
        ),
    )
    ledger = tmp_path / "certificates.jsonl"
    monkeypatch.setattr(
        recall_processor,
        "append_certificates",
        lambda values: append_certificates(values, path=ledger),
    )

    _selected, metadata = recall_processor.select_certified_candidates(
        "query",
        [page("page-a")],
        reranker_metadata={},
        max_candidates=1,
        max_pointer_cards=1,
        max_rich_evidence=1,
        injection_token_budget=1200,
        certificate_required=True,
        judge_policy=SimpleNamespace(
            processor_judge_timeout_ms=500,
            processor_escalation_timeout_ms=500,
            judge_keep_alive="0",
        ),
        judge_timeout_ms=1000,
    )

    receipt = json.loads(ledger.read_text(encoding="utf-8"))
    assert receipt["features"]["certificate_judge"][
        "primary_route_identity"
    ] == route
    assert metadata["judge"]["primary_route_identity"] == route


def test_remote_raw_high_egress_denial_runs_no_backend_or_ollama_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_calls: list[object] = []

    class Backend:
        provider = "remote-test"
        location = RouteLocation.REMOTE

        def generate(self, request: object, *, model: str) -> GenerationResult:
            backend_calls.append((request, model))
            return GenerationResult(content="{}", provider=self.provider, model=model)

    backend = Backend()
    runtime = LLMRuntime(
        generation={
            recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE: GenerationRoute(
                backend,
                "primary-remote",
                BackendCapabilities(True, False, structured_output=True),
            ),
            recall_processor.ESCALATION_JUDGE_RUNTIME_ROLE: GenerationRoute(
                backend,
                "escalation-remote",
                BackendCapabilities(True, False, structured_output=True),
            ),
        }
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    _forbid_ollama_controls(monkeypatch)

    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "query",
        [replace_certificate_confidence(certificate("page-a"), 0.5)],
        policy=SimpleNamespace(
            processor_judge_timeout_ms=500,
            processor_escalation_timeout_ms=500,
            judge_keep_alive="0",
        ),
        timeout_ms=1000,
    )

    assert resolved[0].outcome == "reject"
    assert "primary_judge_fail_closed" in resolved[0].reasons
    assert metadata["primary_status"] == "runtime_backend_unavailable"
    assert metadata["primary_route_identity"] == {
        "role": recall_processor.PRIMARY_JUDGE_RUNTIME_ROLE,
        "provider": "remote-test",
        "model": "primary-remote",
        "location": "remote",
        "model_digest": None,
    }
    assert metadata["escalation_status"] == "blocked_by_primary"
    assert backend_calls == []
