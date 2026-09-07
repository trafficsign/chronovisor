"""Focused model-fleet coverage for the MTPLX generation runtime."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from chronovisor.ops import dashboard


def _response(url: str, status: int, payload: dict[str, Any] | None = None):
    return dashboard.httpx.Response(
        status,
        request=dashboard.httpx.Request("GET", url),
        json=payload,
    )


def test_mtplx_snapshot_marks_served_model_loaded_after_health(
    monkeypatch,
) -> None:
    endpoint = "http://127.0.0.1:18145/v1"
    model = "Ornith-1.5-9B-MTPLX-4bit"
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_llm_config",
        lambda: SimpleNamespace(
            providers={"mtplx": SimpleNamespace(kind="mtplx", endpoint=endpoint)}
        ),
    )
    observed: list[str] = []

    def get(url: str, **_kwargs: Any):
        observed.append(url)
        if url.endswith("/models/status"):
            return _response(url, 404)
        if url.endswith("/models"):
            return _response(
                url,
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model,
                            "owned_by": "mtplx",
                            "processor": "mlx",
                            "details": {"format": "MLX"},
                        }
                    ],
                },
            )
        assert url == "http://127.0.0.1:18145/health"
        return _response(
            url,
            200,
            {
                "ok": True,
                "model": model,
                "model_path": "/Users/trafficsign/.local/share/mtplx/models/" + model,
                "startup": {
                    "model_id": model,
                    "warmup": {"enabled": True, "ran": True, "error": None},
                },
            },
        )

    monkeypatch.setattr(dashboard.httpx, "get", get)

    runtime = dashboard._mtplx_snapshot()

    assert observed == [
        f"{endpoint}/models/status",
        f"{endpoint}/models",
        "http://127.0.0.1:18145/health",
    ]
    assert runtime["available"] is True
    assert runtime["provider"] == "mtplx"
    row = runtime["models"][0]
    assert row["name"] == model
    assert row["loaded"] is True
    assert row["processor"] == "MTPLX"
    assert row["details"]["format"] == "MTPLX"
    assert row["details"]["engine"] == "MTPLX"


def test_mtplx_snapshot_reports_unavailable_server(monkeypatch) -> None:
    endpoint = "http://127.0.0.1:18145/v1"
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_llm_config",
        lambda: SimpleNamespace(
            providers={"mtplx": SimpleNamespace(kind="mtplx", endpoint=endpoint)}
        ),
    )
    observed: list[str] = []

    def get(url: str, **_kwargs: Any):
        observed.append(url)
        raise RuntimeError("mtplx unavailable")

    monkeypatch.setattr(dashboard.httpx, "get", get)

    runtime = dashboard._mtplx_snapshot()

    assert observed == [f"{endpoint}/models/status"]
    assert runtime["available"] is False
    assert runtime["provider"] == "mtplx"
    assert runtime["models"] == []
    assert "mtplx unavailable" in runtime["error"]


def test_mtplx_snapshot_rejects_health_identity_mismatch(monkeypatch) -> None:
    endpoint = "http://127.0.0.1:18145/v1"
    model = "Ornith-1.5-9B-MTPLX-4bit"
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_llm_config",
        lambda: SimpleNamespace(
            providers={"mtplx": SimpleNamespace(kind="mtplx", endpoint=endpoint)}
        ),
    )

    def get(url: str, **_kwargs: Any):
        if url.endswith("/models/status"):
            return _response(url, 404)
        if url.endswith("/models"):
            return _response(url, 200, {"data": [{"id": model, "owned_by": "mtplx"}]})
        return _response(
            url,
            200,
            {
                "ok": True,
                "model": "different-model",
                "model_path": "/tmp/different-model",
                "startup": {"model_id": "different-model", "warmup": {"error": None}},
            },
        )

    monkeypatch.setattr(dashboard.httpx, "get", get)

    runtime = dashboard._mtplx_snapshot()

    assert runtime["models"][0]["loaded"] is False


def test_mtplx_registered_model_waits_for_health_readiness(monkeypatch) -> None:
    model = "Ornith-1.5-9B-MTPLX-4bit"
    monkeypatch.setattr(
        dashboard,
        "_configured_model_roles",
        lambda: {model: {"decision-primary"}},
    )

    runtime = {
        "available": True,
        "provider": "mtplx",
        "models": [
            {
                "name": model,
                "model": model,
                "provider": "mtplx",
                "loaded": False,
                "processor": "MTPLX",
                "details": {"format": "MTPLX", "engine": "MTPLX"},
            }
        ],
    }

    snapshot = dashboard._model_status_snapshot(runtime)

    row = snapshot["models"][0]
    assert row["status"] == "ready"
    assert row["installed"] is True
    assert row["running"] is False
    assert row["processor"] == "MTPLX"
    assert snapshot["summary"]["installed"] == 1
    assert snapshot["summary"]["loaded"] == 0
    assert snapshot["summary"]["missing"] == 0


def test_local_model_snapshot_keeps_ds4_and_mtplx_fleets_distinct(
    monkeypatch,
) -> None:
    ds4 = {
        "available": True,
        "provider": "omlx",
        "models": [
            {"name": "qwen3.8-flash-next", "provider": "omlx", "loaded": True}
        ],
    }
    mtplx = {
        "available": True,
        "provider": "mtplx",
        "models": [
            {
                "name": "Ornith-1.5-9B-MTPLX-4bit",
                "provider": "mtplx",
                "loaded": True,
                "processor": "MTPLX",
            }
        ],
    }
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_llm_config",
        lambda: SimpleNamespace(
            providers={
                "ds4": SimpleNamespace(kind="omlx", endpoint="http://127.0.0.1:18136/v1"),
                "mtplx": SimpleNamespace(kind="mtplx", endpoint="http://127.0.0.1:18145/v1"),
            },
            roles={
                "classification.primary": SimpleNamespace(provider_id="mtplx"),
                "classification.challenger": SimpleNamespace(provider_id="ds4"),
            },
        ),
    )
    monkeypatch.setattr(dashboard, "_omlx_snapshot", lambda: ds4)
    monkeypatch.setattr(dashboard, "_mtplx_snapshot", lambda: mtplx)
    monkeypatch.setattr(dashboard, "_service_model_rows", lambda *_args, **_kwargs: [])

    runtime = dashboard._local_model_snapshot()

    assert runtime["provider"] == "local-openai"
    assert {row["provider"] for row in runtime["models"]} == {"mtplx", "omlx"}
    assert {row["name"] for row in runtime["models"]} == {
        "qwen3.8-flash-next",
        "Ornith-1.5-9B-MTPLX-4bit",
    }
