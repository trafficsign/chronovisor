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


class _ChatResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def json(self) -> dict:
        return {
            "message": {
                "role": "assistant",
                "content": self.content,
                "thinking": "this must not be returned",
            },
            "prompt_eval_count": 42,
            "eval_count": 7,
        }

    def raise_for_status(self) -> None:
        return None


class _ChatClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.path = None
        self.payload = None

    def post(self, path: str, *, json: dict, timeout: object) -> _ChatResponse:
        self.path = path
        self.payload = json
        return _ChatResponse(self.content)


class _TagsResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "models": [
                {
                    "name": "ornith:test",
                    "model": "ornith:test",
                    "digest": "ornith-digest",
                },
                {
                    "name": "gpt-oss:test",
                    "model": "gpt-oss:test",
                    "digest": "gpt-oss-digest",
                },
            ]
        }


class _TagsClient:
    def get(self, path: str, *, timeout: object) -> _TagsResponse:
        assert path == "/api/tags"
        assert timeout == 3
        return _TagsResponse()


def test_triage_prompt_requires_filename_for_updates() -> None:
    assert 'MUST use "filename"' in ollama.TRIAGE_SYSTEM_PROMPT
    assert 'Never emit a "page_id" field' in ollama.TRIAGE_SYSTEM_PROMPT


def test_generation_prompts_forbid_invented_dates() -> None:
    assert "exact current date" in ollama.GENERATE_SYSTEM_PROMPT
    assert "Never invent or infer dates" in ollama.GENERATE_SYSTEM_PROMPT
    assert "Never invent or infer dates" in ollama.UPDATE_SYSTEM_PROMPT


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


def test_chat_uses_fixed_structured_options_and_returns_final_content(monkeypatch) -> None:
    client = _ChatClient('{"decision":"apply"}')
    monkeypatch.setattr(ollama, "_client", lambda: client)
    schema = {
        "type": "object",
        "properties": {"decision": {"type": "string"}},
        "required": ["decision"],
    }

    result = ollama.chat(
        [{"role": "user", "content": "decide"}],
        model="ornith:test",
        format=schema,
        num_ctx=32768,
        num_predict=1024,
        keep_alive="20m",
        read_timeout_ms=120000,
        max_output_chars=1000,
    )

    assert result == '{"decision":"apply"}'
    assert client.path == "/api/chat"
    assert client.payload["model"] == "ornith:test"
    assert client.payload["messages"] == [{"role": "user", "content": "decide"}]
    assert client.payload["stream"] is False
    assert client.payload["think"] is False
    assert client.payload["format"] == schema
    assert client.payload["keep_alive"] == "20m"
    assert client.payload["options"] == {
        "temperature": 0,
        "num_predict": 1024,
        "num_ctx": 32768,
    }


def test_chat_enforces_output_char_cap(monkeypatch) -> None:
    client = _ChatClient("x" * 11)
    monkeypatch.setattr(ollama, "_client", lambda: client)

    try:
        ollama.chat(
            [{"role": "user", "content": "decide"}],
            model="ornith:test",
            format={"type": "string"},
            num_ctx=4096,
            num_predict=128,
            keep_alive="20m",
            read_timeout_ms=120000,
            max_output_chars=10,
        )
    except ollama.OutputTooLargeError:
        pass
    else:
        raise AssertionError("expected OutputTooLargeError")


def test_model_digests_returns_exact_installed_identities(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "_client", lambda: _TagsClient())

    assert ollama.model_digests(
        ["ornith:test", "gpt-oss:test", "missing:test"]
    ) == {
        "ornith:test": "ornith-digest",
        "gpt-oss:test": "gpt-oss-digest",
        "missing:test": "",
    }


def test_chat_can_return_context_accounting(monkeypatch) -> None:
    client = _ChatClient('{"decision":"apply"}')
    monkeypatch.setattr(ollama, "_client", lambda: client)

    result = ollama.chat(
        [{"role": "user", "content": "decide"}],
        model="ornith:test",
        format={"type": "object"},
        num_ctx=4096,
        num_predict=128,
        keep_alive="20m",
        read_timeout_ms=120000,
        max_output_chars=1000,
        return_metadata=True,
    )

    assert isinstance(result, ollama.ChatResponse)
    assert result.content == '{"decision":"apply"}'
    assert result.prompt_eval_count == 42
    assert result.eval_count == 7


def test_embed_uses_explicit_model(monkeypatch) -> None:
    client = _PostClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)

    assert ollama.embed(["hello"], model="bge-m3") == [[1.0, 2.0]]
    assert client.payload["model"] == "bge-m3"
