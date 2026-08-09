from __future__ import annotations

from types import SimpleNamespace

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
    assert payload["reason"] == "RuntimeError"


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
        ),
    )
    policy = SimpleNamespace(
        processor_judge_model="judge-9b",
        judge_model="fallback",
        processor_judge_timeout_ms=500,
        judge_keep_alive="24h",
        processor_escalation_model="",
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


def test_certificate_judge_escalates_uncertain_result(monkeypatch) -> None:
    item = EvidenceCertificate(
        **{
            **certificate("page-a").__dict__,
            "confidence": 0.5,
        }
    )
    calls: list[str] = []

    def fake_judge(_query, values, *, model, **_kwargs):
        calls.append(model)
        confidence = 0.6 if model == "judge-9b" else 0.95
        return (
            {
                values[0].page_id: {
                    "page_id": values[0].page_id,
                    "decision": "uncertain"
                    if model == "judge-9b"
                    else "pass",
                    "confidence": confidence,
                }
            },
            "ok",
        )

    monkeypatch.setattr(recall_processor, "_run_certificate_judge", fake_judge)
    policy = SimpleNamespace(
        processor_judge_model="judge-9b",
        judge_model="fallback",
        processor_judge_timeout_ms=300,
        judge_keep_alive="24h",
        processor_escalation_model="judge-35b",
        processor_escalation_timeout_ms=500,
    )

    resolved, metadata = recall_processor.judge_ambiguous_certificates(
        "query",
        [item],
        policy=policy,
        timeout_ms=1000,
    )

    assert calls == ["judge-9b", "judge-35b"]
    assert resolved[0].outcome == "pass"
    assert "35b_judge_pass" in resolved[0].reasons
    assert metadata["escalation_status"] == "ok"
