from __future__ import annotations

import contextlib
import json
import math
import os
import threading
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from chronovisor.core import llm_config, reranker_client
from chronovisor.core.decision_mlx import (
    ANSWER_AXES,
    JUDGE_ANSWER_AXES,
    JUDGE_CERTIFICATE,
    JUDGE_ROLE_BY_KIND,
    JUDGE_ROLES,
    DecisionRoute,
    decision_fingerprint,
)
from chronovisor.core.llm_runtime import (
    EgressDeniedError,
    LLMRuntime,
    RerankItem,
    RerankRequest,
    RerankResult,
    RerankRoute,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.reranker import RERANK_RUNTIME_ROLE
from chronovisor.core.runtime_config import RerankerConfig, RerankerServiceConfig
from chronovisor.core.search_types import ScoredPage, SemanticEvidence
from chronovisor.search import reranker_service

EVIDENCE_UID = "019fea6e-8a33-7401-b37b-c7afcf6711e1"


def page(page_id: str, score: float = 1.0) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-07-30",
        score=score,
    )


def config(socket_path, *, mode: str = "on") -> RerankerConfig:
    return RerankerConfig(
        enabled=True,
        top_n=2,
        service=RerankerServiceConfig(
            enabled=True,
            socket=str(socket_path),
            timeout_ms=500,
            mode=mode,
            queue_size=2,
        ),
    )


def prepare_state(
    tmp_path,
    monkeypatch,
    cfg: RerankerConfig,
    *,
    location: RouteLocation = RouteLocation.LOCAL,
    egress: set[tuple[str, SourceDataClass]] | None = None,
    provider: str | None = None,
):
    controls = {"lease": 0, "activity": 0}

    class Store:
        def refresh(self) -> None:
            raise AssertionError("per-request full refresh walks the corpus")

        def refresh_if_stale(self) -> None:
            pass

    class Backend:
        def __init__(self) -> None:
            self.provider = provider or (
                "local-reranker"
                if location is RouteLocation.LOCAL
                else "remote-reranker"
            )
            self.location = location
            self.requests: list[RerankRequest] = []

        def rerank(self, request: RerankRequest, *, model: str) -> RerankResult:
            self.requests.append(request)
            return RerankResult(
                tuple(
                    RerankItem(
                        index,
                        0.1 if passage.startswith("a") else 0.9,
                    )
                    for index, passage in enumerate(request.candidates)
                ),
                self.provider,
                model,
            )

        def readout(self, question: str, states: tuple[str, ...]) -> tuple[float, ...]:
            self.readout_calls.append((question, states))
            return tuple(0.0 for _state in states)

    backend = Backend()
    backend.readout_calls = []
    revision = "f" * 64 if backend.provider == "local-decision" else None
    rerank_routes = {RERANK_RUNTIME_ROLE: RerankRoute(backend, "route-model", revision)}
    if backend.provider == "local-decision":
        rerank_routes.update(
            {
                role: RerankRoute(backend, "route-model", revision)
                for role in JUDGE_ROLES
            }
        )
    runtime = LLMRuntime(
        rerank=rerank_routes,
        remote_egress_opt_ins=egress or set(),
    )
    status_file = tmp_path / "reranker-status.json"
    monkeypatch.setattr(reranker_service, "SERVICE_STATUS_FILE", status_file)
    monkeypatch.setattr(
        reranker_service, "runtime_identity", lambda: {"commit_id": "test"}
    )
    monkeypatch.setattr(reranker_service.index_store, "get_store", lambda: Store())
    monkeypatch.setattr(
        reranker_service,
        "resolve_rerank_candidate",
        lambda page_id, **_kwargs: (
            page_id,
            SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL),
            ("pages", page_id, 1, 1, page_id),
        ),
    )

    def lease(**_kwargs):
        controls["lease"] += 1
        return contextlib.nullcontext()

    def activity(**_kwargs):
        controls["activity"] += 1
        return contextlib.nullcontext()

    monkeypatch.setattr(reranker_service, "accelerator_lease", lease)
    monkeypatch.setattr(reranker_service, "model_activity", activity)
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    state = reranker_service.RerankerServiceState(cfg, llm_runtime=runtime)
    backend.requests.clear()
    controls.update(lease=0, activity=0)
    return state, status_file, backend, runtime, controls


def test_service_state_returns_page_keyed_raw_scores(tmp_path, monkeypatch) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, status_file, _backend, _runtime, controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )

    payload = state.handle(
        {"method": "rerank", "query": "query text", "page_ids": ["a", "b"]}
    )

    assert payload["status"] == "ok"
    assert [row["page_id"] for row in payload["scores"]] == ["a", "b"]
    assert [row["raw_score"] for row in payload["scores"]] == [0.1, 0.9]
    assert payload["route"] == {
        "role": RERANK_RUNTIME_ROLE,
        "provider": "local-reranker",
        "model": "route-model",
        "location": "local",
    }
    status_text = status_file.read_text(encoding="utf-8")
    assert "query text" not in status_text
    assert json.loads(status_text)["requests"]["total"] == 1
    assert controls == {"lease": 1, "activity": 1}


def test_idle_service_readiness_does_not_depend_on_status_age(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, status_file, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    status = json.loads(status_file.read_text(encoding="utf-8"))
    status["observed_at_epoch"] = 0
    status_file.write_text(json.dumps(status), encoding="utf-8")

    assert state.ready is True
    assert state._status_payload()["ready"] is True

    installer = Path(__file__).parents[1] / "scripts" / "install-reranker-service"
    source = installer.read_text(encoding="utf-8")
    assert "not 0 <= age <= 30" not in source
    assert "client.connect(str(socket_path))" in source


def test_service_and_client_round_trip_preserves_raw_scores(
    tmp_path, monkeypatch
) -> None:
    socket_path = Path("/tmp") / f"chronovisor-reranker-test-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    cfg = config(socket_path)
    state, _status_file, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    server = reranker_service._Server(str(socket_path), reranker_service._Handler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(reranker_client, "_BREAKER_FAILURES", 0)
    monkeypatch.setattr(reranker_client, "_BREAKER_OPEN_UNTIL", 0.0)
    try:
        outcome = reranker_client.rerank("query", [page("a"), page("b")], config=cfg)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
        socket_path.unlink(missing_ok=True)

    assert [item.page_id for item in outcome.results] == ["b", "a"]
    assert outcome.metadata["status"] == "applied"
    assert outcome.metadata["execution"] == "service"
    assert [detail.raw_score for detail in outcome.scores] == [0.9, 0.1]


def test_client_sends_optional_candidate_evidence(monkeypatch, tmp_path) -> None:
    cfg = config(tmp_path / "reranker.sock")
    _state, _status, _backend, runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    candidate = page("a")
    evidence = SemanticEvidence(
        "a",
        f"{EVIDENCE_UID}#c0",
        "chunk",
        0,
        "a" * 64,
        EVIDENCE_UID,
        "generation-1",
        0.9,
    )
    candidate.evidence = (evidence,)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        reranker_client,
        "request",
        lambda payload, *_a, **_k: (
            seen.update(payload)
            or {
                "status": "ok",
                "route": {
                    "role": RERANK_RUNTIME_ROLE,
                    "provider": "local-reranker",
                    "model": "route-model",
                    "location": "local",
                },
                "scores": [{"page_id": "a", "raw_score": 0.5}],
            }
        ),
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)

    outcome = reranker_client.rerank("query", [candidate], config=cfg)

    assert outcome.metadata["status"] == "applied"
    assert seen["page_ids"] == ["a"]
    assert seen["candidate_evidence"] == [
        {"page_id": "a", "evidence": [asdict(evidence)]}
    ]


def test_service_forwards_optional_candidate_evidence_to_resolver(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    evidence = SemanticEvidence(
        "a",
        f"{EVIDENCE_UID}#c0",
        "chunk",
        0,
        "a" * 64,
        EVIDENCE_UID,
        "generation-1",
        0.9,
    )
    seen: list[tuple[SemanticEvidence, ...]] = []

    def resolve(page_id, *, evidence=(), **_kwargs):
        seen.append(evidence)
        return (
            page_id,
            SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL),
            ("pages", page_id, 1, 1, "a" * 64),
        )

    monkeypatch.setattr(reranker_service, "resolve_rerank_candidate", resolve)
    payload = state.handle(
        {
            "method": "rerank",
            "query": "query",
            "page_ids": ["a"],
            "candidate_evidence": [{"page_id": "a", "evidence": [asdict(evidence)]}],
        }
    )

    assert payload["status"] == "ok"
    assert seen == [(evidence,)]
    assert len(backend.requests) == 1


def test_service_rejects_malformed_candidate_evidence_before_backend(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    bad = {
        "page_id": "a",
        "evidence": [
            {
                "page_id": "a",
                "doc_id": f"{EVIDENCE_UID}#c0",
                "kind": "chunk",
                "ordinal": 0,
                "source_sha256": "bad",
                "page_uid": EVIDENCE_UID,
                "generation_id": "generation-1",
                "score": 0.9,
            }
        ],
    }

    with pytest.raises(ValueError, match="candidate evidence"):
        state.handle(
            {
                "method": "rerank",
                "query": "query",
                "page_ids": ["a"],
                "candidate_evidence": [bad],
            }
        )
    assert backend.requests == []


def test_service_rejects_empty_candidate_evidence_before_backend(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )

    with pytest.raises(ValueError, match="candidate evidence"):
        state.handle(
            {
                "method": "rerank",
                "query": "query",
                "page_ids": ["a"],
                "candidate_evidence": [{"page_id": "a", "evidence": []}],
            }
        )
    assert backend.requests == []


def test_service_rejects_legacy_evidence_alias_before_backend(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )

    with pytest.raises(ValueError, match="candidate evidence"):
        state.handle(
            {
                "method": "rerank",
                "query": "query",
                "page_ids": ["a"],
                "evidence": [],
            }
        )
    assert backend.requests == []


def test_service_socket_and_status_never_expose_backend_details(
    tmp_path, monkeypatch
) -> None:
    socket_path = Path("/tmp") / f"chronovisor-reranker-safe-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    cfg = config(socket_path)
    state, status_file, backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    canary = "CANARY_QUERY_PATH_CREDENTIAL"

    def fail(_request, *, model):
        raise RuntimeError(f"{canary}:{model}")

    monkeypatch.setattr(backend, "rerank", fail)
    server = reranker_service._Server(str(socket_path), reranker_service._Handler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(reranker_client.RerankerServiceUnavailable) as exc:
            reranker_client.request(
                {"method": "rerank", "query": canary, "page_ids": ["a"]},
                cfg,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
        socket_path.unlink(missing_ok=True)

    assert exc.value.category == "backend_error"
    status_text = status_file.read_text(encoding="utf-8")
    assert canary not in status_text
    assert json.loads(status_text)["last_error"] == "backend_error"


def test_existing_state_does_not_publish_into_a_later_states_status_file(
    tmp_path, monkeypatch
) -> None:
    first, first_status, *_ = prepare_state(
        tmp_path / "first", monkeypatch, config(tmp_path / "first.sock")
    )
    _second, second_status, *_ = prepare_state(
        tmp_path / "second", monkeypatch, config(tmp_path / "second.sock")
    )

    first.note_error(RuntimeError("synthetic-backend-detail"))

    assert json.loads(first_status.read_text())["last_error"] == "reranker_unavailable"
    assert json.loads(second_status.read_text())["last_error"] == ""


def test_service_rejects_duplicate_page_ids(tmp_path, monkeypatch) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status_file, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )

    with pytest.raises(ValueError, match="unique"):
        state.handle({"method": "rerank", "query": "query", "page_ids": ["a", "a"]})


def test_decision_route_leases_the_accelerator_per_candidate(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, controls = prepare_state(
        tmp_path, monkeypatch, cfg, provider="local-decision"
    )

    payload = state.handle({"method": "rerank", "query": "q", "page_ids": ["b", "a"]})

    assert [row["raw_score"] for row in payload["scores"]] == [0.9, 0.1]
    assert [request.candidates for request in backend.requests] == [("b",), ("a",)]
    assert controls == {"lease": 2, "activity": 2}


def test_recall_filter_takes_the_next_candidate_turn_ahead_of_search(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    cfg = replace(cfg, service=replace(cfg.service, queue_size=4))
    state, _status, backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg, provider="local-decision"
    )
    entered = threading.Event()
    release = threading.Event()
    calls: list[tuple[str, tuple[str, ...]]] = []
    calls_lock = threading.Lock()

    def rerank(request: RerankRequest, *, model: str) -> RerankResult:
        with calls_lock:
            calls.append((request.purpose, tuple(request.candidates)))
            first = len(calls) == 1
        if first:
            entered.set()
            assert release.wait(timeout=2)
        return RerankResult(
            tuple(
                RerankItem(index, 0.1 if candidate.startswith("a") else 0.9)
                for index, candidate in enumerate(request.candidates)
            ),
            backend.provider,
            model,
        )

    backend.rerank = rerank  # type: ignore[method-assign]
    failures: list[BaseException] = []

    def run(payload: dict[str, object]) -> None:
        try:
            state.handle(payload)
        except BaseException as exc:
            failures.append(exc)

    def await_waiting(priority: int) -> None:
        condition = state._turn._condition
        with condition:
            assert condition.wait_for(
                lambda: state._turn._waiting[priority] == 1, timeout=2
            )

    first_search = threading.Thread(
        target=run,
        args=({"method": "rerank", "query": "first", "page_ids": ["a1", "a2"]},),
        daemon=True,
    )
    second_search = threading.Thread(
        target=run,
        args=({"method": "rerank", "query": "second", "page_ids": ["b1", "b2"]},),
        daemon=True,
    )
    recall = threading.Thread(
        target=run,
        args=(
            {
                "method": "recall_filter",
                "query": "now",
                "context": "before",
                "texts": ["recall"],
            },
        ),
        daemon=True,
    )
    first_search.start()
    try:
        assert entered.wait(timeout=2)
        second_search.start()
        await_waiting(reranker_service.PRIORITY_SEARCH)
        recall.start()
        await_waiting(reranker_service.PRIORITY_RECALL)
    finally:
        release.set()
    threads = (first_search, second_search, recall)
    for thread in threads:
        thread.join(timeout=2)
    assert not any(thread.is_alive() for thread in threads)
    assert failures == []
    assert calls[0] == ("search", ("a1",))
    assert calls[1] == ("recall_filter", ("recall",))
    assert sorted(call[1] for call in calls[2:]) == [("a2",), ("b1",), ("b2",)]


def test_judge_answer_axes_maps_probabilities_and_uses_background_priority(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, controls = prepare_state(
        tmp_path, monkeypatch, cfg, provider="local-decision"
    )
    priorities: list[int] = []
    original_slot = state._turn.slot

    @contextlib.contextmanager
    def observed_slot(priority):
        priorities.append(priority)
        with original_slot(priority):
            yield

    monkeypatch.setattr(state._turn, "slot", observed_slot)
    logits = iter((2.0, -2.0, 0.0))
    seen: list[tuple[str, tuple[str, ...]]] = []

    def readout(question: str, states: tuple[str, ...]) -> tuple[float, ...]:
        seen.append((question, states))
        return (next(logits),)

    backend.readout = readout
    payload = state.handle(
        {
            "method": "judge",
            "kind": JUDGE_ANSWER_AXES,
            "role": JUDGE_ROLE_BY_KIND[JUDGE_ANSWER_AXES],
            "query": "question",
            "answer": "candidate",
            "reference": "[PAGE page-1]\nreference fact",
        }
    )

    assert payload["status"] == "ok"
    assert payload["kind"] == JUDGE_ANSWER_AXES
    assert payload["scores"] == {
        "correctness": round(1 / (1 + math.exp(-2.0)), 9),
        "grounding": round(1 / (1 + math.exp(2.0)), 9),
        "citation": 0.5,
    }
    assert priorities == [reranker_service.PRIORITY_SEARCH] * 3
    assert controls == {"lease": 3, "activity": 3}
    assert [question for question, _states in seen] == [
        reranker_service.ANSWER_AXIS_QUESTIONS[axis] for axis in ANSWER_AXES
    ]
    assert all("Current question:\nquestion" in states[0] for _q, states in seen)
    assert all("[PAGE page-1]" in states[0] for _q, states in seen)
    expected_route = {
        "role": JUDGE_ROLE_BY_KIND[JUDGE_ANSWER_AXES],
        "provider": "local-decision",
        "model": "route-model",
        "location": "local",
    }
    assert payload["route"] == expected_route
    assert payload["decision_fingerprint"] == decision_fingerprint(
        DecisionRoute(
            role=JUDGE_ROLE_BY_KIND[JUDGE_ANSWER_AXES],
            provider="local-decision",
            model="route-model",
            location="local",
            revision="f" * 64,
        )
    )


def test_judge_certificate_uses_recall_priority_and_maps_verdicts(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, controls = prepare_state(
        tmp_path, monkeypatch, cfg, provider="local-decision"
    )
    priorities: list[int] = []
    original_slot = state._turn.slot

    @contextlib.contextmanager
    def observed_slot(priority):
        priorities.append(priority)
        with original_slot(priority):
            yield

    monkeypatch.setattr(state._turn, "slot", observed_slot)
    backend.readout = lambda _question, _states: (5.0,)
    payload = state.handle(
        {
            "method": "judge",
            "kind": JUDGE_CERTIFICATE,
            "role": JUDGE_ROLE_BY_KIND[JUDGE_CERTIFICATE],
            "query": "question",
            "rubric": "Pass only answer-bearing evidence.",
            "rubric_id": "rubric-1",
            "evidence": [{"page_id": "page-1", "span": "supporting span"}],
        }
    )

    assert payload["verdicts"][0]["page_id"] == "page-1"
    assert payload["verdicts"][0]["decision"] == "pass"
    assert payload["verdicts"][0]["confidence"] > 0.99
    assert priorities == [reranker_service.PRIORITY_RECALL]
    assert controls == {"lease": 1, "activity": 1}


def test_judge_rejects_non_decision_routes_and_bad_fields(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )

    with pytest.raises(Exception, match="rerank is not configured"):
        state.handle(
            {
                "method": "judge",
                "kind": JUDGE_ANSWER_AXES,
                "role": JUDGE_ROLE_BY_KIND[JUDGE_ANSWER_AXES],
                "query": "question",
                "answer": "candidate",
                "reference": "reference",
            }
        )
    with pytest.raises(ValueError):
        state.handle(
            {
                "method": "judge",
                "kind": JUDGE_ANSWER_AXES,
                "role": JUDGE_ROLE_BY_KIND[JUDGE_ANSWER_AXES],
                "query": "question",
                "answer": "candidate",
            }
        )


def test_decision_judge_client_checks_route_and_fingerprint(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    route = DecisionRoute(
        role=JUDGE_ROLE_BY_KIND[JUDGE_ANSWER_AXES],
        provider="local-decision",
        model="startlux-model",
        location="local",
        revision="f" * 64,
    )
    fingerprint = decision_fingerprint(route)
    monkeypatch.setattr(reranker_client, "decision_route", lambda _role: route)
    captured: dict[str, object] = {}

    def fake_request(payload, config, *, timeout_ms=None):
        captured.update(payload=payload, config=config, timeout_ms=timeout_ms)
        return {
            "status": "ok",
            "route": {
                "role": route.role,
                "provider": route.provider,
                "model": route.model,
                "location": route.location,
            },
            "decision_fingerprint": fingerprint,
            "scores": {
                "correctness": 0.9,
                "grounding": 0.8,
                "citation": 0.7,
            },
        }

    monkeypatch.setattr(reranker_client, "request", fake_request)
    result = reranker_client.decision_judge(
        JUDGE_ANSWER_AXES,
        {"query": "q", "answer": "a", "reference": "r"},
        role=route.role,
        config=cfg,
        timeout_ms=250,
    )

    assert result["scores"]["correctness"] == 0.9
    assert captured["timeout_ms"] == 250
    assert captured["payload"] == {
        "method": "judge",
        "kind": JUDGE_ANSWER_AXES,
        "role": route.role,
        "query": "q",
        "answer": "a",
        "reference": "r",
    }

    def drifted_request(payload, config, *, timeout_ms=None):
        response = fake_request(payload, config, timeout_ms=timeout_ms)
        response["decision_fingerprint"] = "0" * 64
        return response

    monkeypatch.setattr(reranker_client, "request", drifted_request)
    with pytest.raises(reranker_client.RerankerServiceUnavailable):
        reranker_client.decision_judge(
            JUDGE_ANSWER_AXES,
            {"query": "q", "answer": "a", "reference": "r"},
            role=route.role,
            config=cfg,
        )


def test_recall_filter_scores_exact_texts_with_the_recall_question(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, controls = prepare_state(
        tmp_path, monkeypatch, cfg, provider="local-decision"
    )

    payload = state.handle(
        {
            "method": "recall_filter",
            "query": "now",
            "context": "before",
            "texts": ["a1", "b2"],
        }
    )

    assert payload["status"] == "ok" and payload["scores"] == [0.1, 0.9]
    assert [request.candidates for request in backend.requests] == [("a1",), ("b2",)]
    assert {
        (request.query, request.context, request.purpose)
        for request in backend.requests
    } == {("now", "before", "recall_filter")}
    assert controls == {"lease": 2, "activity": 2}


@pytest.mark.parametrize(
    "payload",
    [
        {"method": "recall_filter", "query": "q", "texts": []},
        {"method": "recall_filter", "query": " ", "texts": ["a"]},
        {"method": "recall_filter", "query": "q", "texts": [""]},
        {"method": "recall_filter", "query": "q", "context": 1, "texts": ["a"]},
        {"method": "recall_filter", "query": "q", "texts": ["a"] * 17},
    ],
)
def test_recall_filter_rejects_malformed_requests_before_backend(
    tmp_path, monkeypatch, payload
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg, provider="local-decision"
    )

    with pytest.raises(ValueError):
        state.handle(payload)
    assert backend.requests == []


def test_recall_filter_refuses_a_relevance_reranker_route(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )

    with pytest.raises(ValueError):
        state.handle({"method": "recall_filter", "query": "q", "texts": ["a"]})
    assert backend.requests == []


def test_recall_filter_client_round_trip_and_contract(tmp_path, monkeypatch) -> None:
    socket_path = Path("/tmp") / f"chronovisor-filter-test-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    cfg = config(socket_path)
    state, _status, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg, provider="local-decision"
    )
    server = reranker_service._Server(str(socket_path), reranker_service._Handler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(reranker_client, "_BREAKER_FAILURES", 0)
    monkeypatch.setattr(reranker_client, "_BREAKER_OPEN_UNTIL", 0.0)
    try:
        scores = reranker_client.recall_filter("q", "", ["b", "a"], config=cfg)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
        socket_path.unlink(missing_ok=True)

    assert scores == [0.9, 0.1]

    route = state._route_payload()
    for response in (
        {"status": "ok", "route": {**route, "model": "drifted"}, "scores": [1.0]},
        {"status": "ok", "route": route, "scores": [1.0, 2.0]},
        {"status": "ok", "route": route, "scores": [float("nan")]},
        {"status": "ok", "route": route, "scores": [True]},
    ):
        monkeypatch.setattr(reranker_client, "request", lambda *_a, r=response, **_k: r)
        with pytest.raises(reranker_client.RerankerServiceUnavailable):
            reranker_client.recall_filter("q", "", ["a"], config=cfg)


def test_remote_default_denial_has_no_backend_or_local_controls(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, controls = prepare_state(
        tmp_path,
        monkeypatch,
        cfg,
        location=RouteLocation.REMOTE,
    )

    with pytest.raises(EgressDeniedError):
        state.handle({"method": "rerank", "query": "private query", "page_ids": ["a"]})

    assert backend.requests == []
    assert controls == {"lease": 0, "activity": 0}


def test_remote_explicit_opt_in_succeeds_without_local_controls(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, controls = prepare_state(
        tmp_path,
        monkeypatch,
        cfg,
        location=RouteLocation.REMOTE,
        egress={(RERANK_RUNTIME_ROLE, SourceDataClass.RAW)},
    )

    payload = state.handle(
        {"method": "rerank", "query": "query", "page_ids": ["a", "b"]}
    )

    assert payload["status"] == "ok"
    assert payload["route"]["location"] == "remote"
    assert len(backend.requests) == 1
    assert controls == {"lease": 0, "activity": 0}


@pytest.mark.parametrize(
    "source",
    [
        SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.HIGH),
        SourceDataClassification(SourceDataClass.SYSTEM, SourceSensitivity.HIGH),
    ],
)
def test_remote_candidate_denial_runs_no_backend_or_local_controls(
    tmp_path, monkeypatch, source
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, controls = prepare_state(
        tmp_path,
        monkeypatch,
        cfg,
        location=RouteLocation.REMOTE,
        egress={(RERANK_RUNTIME_ROLE, SourceDataClass.RAW)},
    )
    monkeypatch.setattr(
        reranker_service,
        "resolve_rerank_candidate",
        lambda page_id, **_kwargs: (
            page_id,
            source,
            ("system", page_id, 1, 1, page_id),
        ),
    )

    with pytest.raises(EgressDeniedError):
        state.handle({"method": "rerank", "query": "query", "page_ids": ["a"]})

    assert backend.requests == []
    assert controls == {"lease": 0, "activity": 0}


def test_warm_failure_is_not_reported_as_success(tmp_path, monkeypatch) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, backend, _runtime, controls = prepare_state(
        tmp_path,
        monkeypatch,
        cfg,
        location=RouteLocation.REMOTE,
    )

    payload = state.handle({"method": "warm"})

    assert payload["status"] == "unavailable"
    assert payload["error"] == "egress_denied"
    assert backend.requests == []
    assert controls == {"lease": 0, "activity": 0}


def test_client_rejects_service_route_drift(tmp_path, monkeypatch) -> None:
    cfg = config(tmp_path / "reranker.sock")
    _state, _status, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    monkeypatch.setattr(
        reranker_client,
        "request",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "route": {
                "role": RERANK_RUNTIME_ROLE,
                "provider": "local-reranker",
                "model": "drifted-model",
                "location": "local",
            },
            "scores": [{"page_id": "a", "raw_score": 1.0}],
        },
    )

    with pytest.raises(reranker_client.RerankerServiceUnavailable) as exc:
        reranker_client.rerank("query", [page("a")], config=cfg)

    assert exc.value.category == "backend_contract_error"


@pytest.mark.parametrize("changed_field", range(5))
def test_service_passage_cache_requires_exact_candidate_identity(
    tmp_path, monkeypatch, changed_field
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    base = ["pages", "/pages/a.md", 1, 2, "digest-a"]
    changed = list(base)
    changed[changed_field] = (
        "system"
        if changed_field == 0
        else "/system/a.md"
        if changed_field == 1
        else 3
        if changed_field in {2, 3}
        else "digest-b"
    )
    responses = iter(
        (
            (
                "old passage",
                SourceDataClassification(
                    SourceDataClass.PAGE, SourceSensitivity.NORMAL
                ),
                tuple(base),
            ),
            (
                "new passage",
                SourceDataClassification(
                    SourceDataClass.SYSTEM, SourceSensitivity.HIGH
                ),
                tuple(changed),
            ),
        )
    )
    monkeypatch.setattr(
        reranker_service,
        "resolve_rerank_candidate",
        lambda *_args, **_kwargs: next(responses),
    )

    first = state._candidate_passage("a", store=None)
    second = state._candidate_passage("a", store=None)

    assert first[0] == "old passage"
    assert second[:2] == (
        "new passage",
        SourceDataClassification(SourceDataClass.SYSTEM, SourceSensitivity.HIGH),
    )


def test_service_passage_cache_includes_selected_evidence_identity(
    tmp_path, monkeypatch
) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )
    identity = ("pages", "/pages/a.md", 1, 2, "same-source-digest")
    responses = iter(
        (
            (
                "passage selected first",
                SourceDataClassification(
                    SourceDataClass.PAGE, SourceSensitivity.NORMAL
                ),
                identity,
            ),
            (
                "passage selected second",
                SourceDataClassification(
                    SourceDataClass.PAGE, SourceSensitivity.NORMAL
                ),
                identity,
            ),
        )
    )
    monkeypatch.setattr(
        reranker_service,
        "resolve_rerank_candidate",
        lambda *_args, **_kwargs: next(responses),
    )
    first_evidence = SemanticEvidence(
        "a",
        f"{EVIDENCE_UID}#c0",
        "chunk",
        0,
        "a" * 64,
        EVIDENCE_UID,
        "generation-1",
        0.9,
    )
    second_evidence = SemanticEvidence(
        "a",
        f"{EVIDENCE_UID}#c1",
        "chunk",
        1,
        "a" * 64,
        EVIDENCE_UID,
        "generation-1",
        0.8,
    )

    first = state._candidate_passage("a", store=None, evidence=(first_evidence,))
    second = state._candidate_passage("a", store=None, evidence=(second_evidence,))

    assert first[0] == "passage selected first"
    assert second[0] == "passage selected second"


def test_reranker_rollout_selection_is_stable(tmp_path) -> None:
    off = config(tmp_path / "off.sock", mode="off")
    on = config(tmp_path / "on.sock", mode="shadow")
    canary = RerankerConfig(
        enabled=True,
        service=RerankerServiceConfig(
            enabled=True,
            socket=str(tmp_path / "canary.sock"),
            mode="canary",
            canary_percent=37,
        ),
    )

    assert reranker_client.selected_for_rollout("query", off) is False
    assert reranker_client.selected_for_rollout("query", on) is True
    assert reranker_client.selected_for_rollout(
        "stable query", canary
    ) == reranker_client.selected_for_rollout("stable query", canary)


def test_idle_server_publishes_bounded_heartbeat(monkeypatch):
    from types import SimpleNamespace

    from chronovisor.search import reranker_service

    server = object.__new__(reranker_service._Server)
    observed = []
    server.state = SimpleNamespace(_publish_status=lambda: observed.append(True))
    clock = [10.0]
    monkeypatch.setattr(reranker_service.time, "monotonic", lambda: clock[0])
    server.service_actions()
    server.service_actions()
    assert len(observed) == 1
    clock[0] += 5
    server.service_actions()
    assert len(observed) == 2

    def unavailable_disk():
        raise OSError("status disk unavailable")

    server.state._publish_status = unavailable_disk
    clock[0] += 5
    server.service_actions()  # Telemetry failure must not stop inference.
