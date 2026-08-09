from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from chronovisor.core import reranker_client
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


def prepare_state(tmp_path, monkeypatch, cfg: RerankerConfig):
    status_file = tmp_path / "reranker-status.json"
    monkeypatch.setattr(reranker_service, "SERVICE_STATUS_FILE", status_file)
    monkeypatch.setattr(
        reranker_service, "runtime_identity", lambda: {"commit_id": "test"}
    )
    monkeypatch.setattr(
        reranker_service,
        "warm_reranker",
        lambda _config: {"status": "ready", "latency_ms": 1},
    )
    monkeypatch.setattr(reranker_service, "find_page", lambda _page_id: None)
    monkeypatch.setattr(
        reranker_service,
        "_score_fn",
        lambda _config: (
            lambda _query, passages, _cfg: [
                0.1 if passage.startswith("a") else 0.9 for passage in passages
            ]
        ),
    )
    return reranker_service.RerankerServiceState(cfg), status_file


def test_service_state_returns_page_keyed_raw_scores(tmp_path, monkeypatch) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, status_file = prepare_state(tmp_path, monkeypatch, cfg)

    payload = state.handle(
        {"method": "rerank", "query": "query text", "page_ids": ["a", "b"]}
    )

    assert payload["status"] == "ok"
    assert [row["page_id"] for row in payload["scores"]] == ["a", "b"]
    assert [row["raw_score"] for row in payload["scores"]] == [0.1, 0.9]
    status_text = status_file.read_text(encoding="utf-8")
    assert "query text" not in status_text
    assert json.loads(status_text)["requests"]["total"] == 1


def test_service_and_client_round_trip_preserves_raw_scores(
    tmp_path, monkeypatch
) -> None:
    socket_path = Path("/tmp") / f"chronovisor-reranker-test-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    cfg = config(socket_path)
    state, _status_file = prepare_state(tmp_path, monkeypatch, cfg)
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


def test_service_rejects_duplicate_page_ids(tmp_path, monkeypatch) -> None:
    cfg = config(tmp_path / "reranker.sock")
    state, _status_file = prepare_state(tmp_path, monkeypatch, cfg)

    with pytest.raises(ValueError, match="unique"):
        state.handle(
            {"method": "rerank", "query": "query", "page_ids": ["a", "a"]}
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
