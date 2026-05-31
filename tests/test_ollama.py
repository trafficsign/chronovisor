"""Tests for the Ollama client wrapper."""

from __future__ import annotations

import json

from llm_wiki_mcp import ollama


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


def test_generate_streams_progress_and_returns_text(monkeypatch) -> None:
    client = _StreamClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)
    updates: list[dict] = []

    result = ollama.generate("prompt", progress_callback=updates.append)

    assert result == "hello"
    assert client.payload["stream"] is True
    assert updates[-1]["event"] == "done"
    assert updates[-1]["active"] is False
    assert updates[-1]["generated_chars"] == 5
    assert updates[-1]["eval_count"] == 2
