"""Tests for the Ollama client wrapper."""

from __future__ import annotations

import json

from llm_wiki_mcp import ollama
from llm_wiki_mcp.runtime_config import IngestConfig


class _StreamResponse:
    def __init__(self, lines: list[dict]) -> None:
        self.lines = lines

    def __enter__(self) -> "_StreamResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        for line in self.lines:
            yield json.dumps(line)


class _StreamClient:
    def __init__(self) -> None:
        self.payload = None

    def stream(self, _method: str, _path: str, *, json: dict, timeout: object) -> _StreamResponse:
        self.payload = json
        return _StreamResponse([
            {"response": "hel", "done": False},
            {
                "response": "lo",
                "done": True,
                "eval_count": 2,
                "eval_duration": 1_000_000_000,
            },
        ])


class _PostResponse:
    status_code = 200

    def json(self) -> dict:
        return {"embeddings": [[1.0, 2.0]]}

    def raise_for_status(self) -> None:
        return None


class _PostClient:
    def __init__(self) -> None:
        self.payload = None

    def post(self, _path: str, *, json: dict, timeout: object) -> _PostResponse:
        self.payload = json
        return _PostResponse()


def test_generate_streams_progress_and_returns_text(monkeypatch) -> None:
    client = _StreamClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)
    monkeypatch.setattr(
        ollama,
        "load_ingest_config",
        lambda: IngestConfig(
            model="qwen3.6:35b-a3b-mxfp8",
            keep_alive="10m",
            temperature=0.1,
            num_ctx=2048,
            max_num_ctx=8192,
            num_predict=128,
            read_timeout_ms=120000,
        ),
    )
    updates: list[dict] = []

    result = ollama.generate("prompt", progress_callback=updates.append)

    assert result == "hello"
    assert client.payload["stream"] is True
    assert client.payload["model"] == "qwen3.6:35b-a3b-mxfp8"
    assert client.payload["keep_alive"] == "10m"
    assert client.payload["options"]["temperature"] == 0.1
    assert client.payload["options"]["num_predict"] == 128
    assert client.payload["options"]["num_ctx"] == 2048
    assert updates[-1]["event"] == "done"
    assert updates[-1]["active"] is False
    assert updates[-1]["generated_chars"] == 5
    assert updates[-1]["eval_count"] == 2


def test_num_ctx_grows_for_long_prompts_without_crossing_cap() -> None:
    config = IngestConfig(num_ctx=2048, max_num_ctx=4096, num_predict=128)

    assert ollama._num_ctx_for_prompt("short", None, config) == 2048
    assert ollama._num_ctx_for_prompt("x" * 10_000, None, config) == 4096


def test_embed_uses_explicit_model(monkeypatch) -> None:
    client = _PostClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)

    assert ollama.embed(["hello"], model="bge-m3") == [[1.0, 2.0]]
    assert client.payload["model"] == "bge-m3"
