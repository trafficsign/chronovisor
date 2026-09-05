import pytest

from chronovisor.core import semantic_client
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.core.semantic_client import (
    SemanticServiceUnavailable,
    selected_for_rollout,
)


def test_rollout_selection_is_stable_and_respects_modes() -> None:
    base = SearchEmbeddingConfig()
    assert not selected_for_rollout("query", base)
    assert selected_for_rollout(
        "query",
        SearchEmbeddingConfig(enabled=True, rollout_mode="on"),
    )
    assert not selected_for_rollout(
        "query",
        SearchEmbeddingConfig(
            enabled=True,
            rollout_mode="canary",
            canary_percent=0,
        ),
    )
    assert selected_for_rollout(
        "query",
        SearchEmbeddingConfig(
            enabled=True,
            rollout_mode="canary",
            canary_percent=100,
        ),
    )


def test_search_sends_one_absolute_deadline_and_rejects_zero_budget(
    monkeypatch,
) -> None:
    config = SearchEmbeddingConfig(enabled=True, rollout_mode="on")
    captured: dict[str, object] = {}

    def fake_request(payload, _config, *, timeout_ms=None, deadline_at=None):
        captured.update(
            {
                "payload": payload,
                "timeout_ms": timeout_ms,
                "deadline_at": deadline_at,
            }
        )
        return {"status": "ok", "results": []}

    class Store:
        loaded = False

        def refresh_if_stale(self) -> None:
            self.loaded = True

    monkeypatch.setattr(semantic_client, "request", fake_request)
    store = Store()
    monkeypatch.setattr("chronovisor.core.index_store.get_store", lambda: store)

    assert semantic_client.search(
        "query",
        1,
        include_reference=False,
        config=config,
        timeout_ms=100,
    ) == []
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["timeout_ms"] == 100
    assert isinstance(payload["deadline_at"], float)
    assert captured["deadline_at"] == payload["deadline_at"]
    assert store.loaded is False

    with pytest.raises(SemanticServiceUnavailable, match="deadline"):
        semantic_client.search(
            "query",
            1,
            include_reference=False,
            config=config,
            timeout_ms=0,
        )


def test_verify_sends_deadline_payload_and_rejects_negative_budget(monkeypatch) -> None:
    config = SearchEmbeddingConfig(enabled=True, rollout_mode="on")
    captured: dict[str, object] = {}

    def fake_request(payload, _config, *, timeout_ms=None, deadline_at=None):
        captured.update(
            {
                "payload": payload,
                "timeout_ms": timeout_ms,
                "deadline_at": deadline_at,
            }
        )
        return {"status": "ok", "results": []}

    class Store:
        loaded = False

        def refresh_if_stale(self) -> None:
            self.loaded = True

    monkeypatch.setattr(semantic_client, "request", fake_request)
    store = Store()
    monkeypatch.setattr("chronovisor.core.index_store.get_store", lambda: store)

    assert semantic_client.verify(
        "query",
        ["page"],
        config=config,
        timeout_ms=100,
    ) == []
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["timeout_ms"] == 100
    assert isinstance(payload["deadline_at"], float)
    assert captured["deadline_at"] == payload["deadline_at"]
    assert store.loaded is False

    with pytest.raises(SemanticServiceUnavailable, match="deadline"):
        semantic_client.verify(
            "query",
            ["page"],
            config=config,
            timeout_ms=-1,
        )


def test_request_invalid_utf8_response_is_typed_unavailable(
    monkeypatch, tmp_path
) -> None:
    socket_path = tmp_path / "semantic.sock"
    socket_path.touch()

    class BrokenSocket:
        def settimeout(self, _timeout) -> None:
            pass

        def connect(self, _path) -> None:
            pass

        def sendall(self, _payload) -> None:
            pass

        def recv(self, _size) -> bytes:
            return b"\xff\n"

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        semantic_client.socket, "socket", lambda *_args, **_kwargs: BrokenSocket()
    )
    config = SearchEmbeddingConfig(socket=str(socket_path))

    with pytest.raises(
        SemanticServiceUnavailable, match="invalid semantic service response"
    ):
        semantic_client.request({"method": "health"}, config, timeout_ms=100)
