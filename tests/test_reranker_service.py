from __future__ import annotations

import contextlib
import json
import os
import threading
from pathlib import Path

import pytest

from chronovisor.core import llm_config, reranker_client
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
from chronovisor.core.search_types import ScoredPage
from chronovisor.search import reranker_service


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
):
    controls = {"lease": 0, "activity": 0}

    class Store:
        def refresh(self) -> None:
            pass

    class Backend:
        def __init__(self) -> None:
            self.provider = (
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

    backend = Backend()
    runtime = LLMRuntime(
        rerank={RERANK_RUNTIME_ROLE: RerankRoute(backend, "route-model")},
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
            SourceDataClassification(
                SourceDataClass.PAGE, SourceSensitivity.NORMAL
            ),
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


def test_idle_service_readiness_does_not_depend_on_status_age(tmp_path, monkeypatch) -> None:
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
        outcome = reranker_client.rerank(
            "query", [page("a"), page("b")], config=cfg
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
        socket_path.unlink(missing_ok=True)

    assert [item.page_id for item in outcome.results] == ["b", "a"]
    assert outcome.metadata["status"] == "applied"
    assert outcome.metadata["execution"] == "service"
    assert [detail.raw_score for detail in outcome.scores] == [0.9, 0.1]


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


def test_service_rejects_duplicate_page_ids(tmp_path, monkeypatch) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status_file, _backend, _runtime, _controls = prepare_state(
        tmp_path, monkeypatch, cfg
    )

    with pytest.raises(ValueError, match="unique"):
        state.handle(
            {"method": "rerank", "query": "query", "page_ids": ["a", "a"]}
        )


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
        state.handle(
            {"method": "rerank", "query": "private query", "page_ids": ["a"]}
        )

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
