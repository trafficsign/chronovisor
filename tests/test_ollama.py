"""Tests for the Ollama client wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import httpx
import pytest

from chronovisor.core import ollama
from chronovisor.core.runtime_config import IngestConfig


class _StreamResponse:
    def __init__(self, lines: list[dict]) -> None:
        self.lines = lines

    def __enter__(self) -> _StreamResponse:
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

    def stream(
        self, _method: str, _path: str, *, json: dict, timeout: object
    ) -> _StreamResponse:
        self.payload = json
        return _StreamResponse(
            [
                {"response": "hel", "done": False},
                {
                    "response": "lo",
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 11,
                    "eval_count": 2,
                    "eval_duration": 1_000_000_000,
                },
            ]
        )


class _IncompleteStreamClient(_StreamClient):
    def stream(
        self, _method: str, _path: str, *, json: dict, timeout: object
    ) -> _StreamResponse:
        self.payload = json
        return _StreamResponse([{"response": "partial", "done": False}])


class _PostResponse:
    status_code = 200

    def json(self) -> dict:
        return {"embeddings": [[1.0, 2.0]]}

    def raise_for_status(self) -> None:
        return None


class _PostClient:
    def __init__(self) -> None:
        self.payload = None
        self.timeout = None

    def post(self, _path: str, *, json: dict, timeout: object) -> _PostResponse:
        self.payload = json
        self.timeout = timeout
        return _PostResponse()


class _ChatResponse:
    def __init__(
        self,
        content: str,
        *,
        done: bool | None = True,
        done_reason: str | None = "stop",
    ) -> None:
        self.content = content
        self.done = done
        self.done_reason = done_reason

    def json(self) -> dict:
        body = {
            "message": {
                "role": "assistant",
                "content": self.content,
                "thinking": "this must not be returned",
            },
            "prompt_eval_count": 42,
            "eval_count": 7,
        }
        if self.done is not None:
            body["done"] = self.done
        if self.done_reason is not None:
            body["done_reason"] = self.done_reason
        return body

    def raise_for_status(self) -> None:
        return None


class _ChatClient:
    def __init__(
        self,
        content: str,
        *,
        done: bool | None = True,
        done_reason: str | None = "stop",
    ) -> None:
        self.content = content
        self.done = done
        self.done_reason = done_reason
        self.path = None
        self.payload = None

    def post(self, path: str, *, json: dict, timeout: object) -> _ChatResponse:
        self.path = path
        self.payload = json
        return _ChatResponse(
            self.content,
            done=self.done,
            done_reason=self.done_reason,
        )


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


class _JsonResponse:
    status_code = 200

    def __init__(self, body: dict) -> None:
        self.body = body

    def json(self) -> dict:
        return self.body

    def raise_for_status(self) -> None:
        return None


class _ResourceClient:
    def __init__(self, *, resident_size_vram: int, resident_size: int) -> None:
        self.resident_size_vram = resident_size_vram
        self.resident_size = resident_size

    def get(self, path: str, *, timeout: object) -> _JsonResponse:
        assert timeout == 3
        if path == "/api/tags":
            return _JsonResponse(
                {
                    "models": [
                        {
                            "name": "ornith:test",
                            "model": "ornith:test",
                            "size": 10,
                        }
                    ]
                }
            )
        assert path == "/api/ps"
        return _JsonResponse(
            {
                "models": [
                    {
                        "name": "ornith:test",
                        "model": "ornith:test",
                        "size_vram": self.resident_size_vram,
                        "size": self.resident_size,
                        "context_length": 4096,
                    }
                ]
            }
        )


def test_triage_prompt_requires_filename_for_updates() -> None:
    assert 'MUST use "filename"' in ollama.TRIAGE_SYSTEM_PROMPT
    assert 'Never emit a "page_id" field' in ollama.TRIAGE_SYSTEM_PROMPT
    assert "a bare filename is forbidden" in ollama.TRIAGE_SYSTEM_PROMPT
    assert "when no existing folder fits" in ollama.TRIAGE_SYSTEM_PROMPT


def test_http_error_preserves_ollama_response_detail() -> None:
    request = httpx.Request("POST", "http://localhost:11434/api/generate")
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": (
                "Failed to initialize samplers: failed to parse grammar\n"
                "repetition exceeds sane defaults"
            )
        },
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Ollama HTTP 400: Failed to initialize samplers: "
            "failed to parse grammar repetition exceeds sane defaults"
        ),
    ):
        ollama._raise_for_status_with_detail(response)


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
    assert client.payload["shift"] is False
    assert client.payload["truncate"] is False
    assert client.payload["options"]["temperature"] == 0.1
    assert client.payload["options"]["num_predict"] == 128
    assert client.payload["options"]["num_ctx"] == 2048
    assert updates[-1]["event"] == "done"
    assert updates[-1]["active"] is False
    assert updates[-1]["generated_chars"] == 5
    assert updates[-1]["eval_count"] == 2


def test_generate_forwards_per_request_runtime_overrides(monkeypatch) -> None:
    client = _StreamClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)

    result = ollama.generate(
        "prompt",
        progress_callback=lambda _event: None,
        model="ornith:override",
        num_ctx=65536,
        num_predict=2048,
        keep_alive="90s",
        read_timeout_ms=180000,
        temperature=0,
        seed=0,
    )

    assert result == "hello"
    assert client.payload["model"] == "ornith:override"
    assert client.payload["keep_alive"] == "90s"
    assert client.payload["shift"] is False
    assert client.payload["truncate"] is False
    assert client.payload["options"] == {
        "temperature": 0,
        "num_predict": 2048,
        "num_ctx": 65536,
        "seed": 0,
    }


def test_generate_publishes_redacted_model_activity(tmp_path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    observed: dict = {}
    marker_bytes = b""
    marker_mode = 0

    def fake_generate(*_args: object, **_kwargs: object) -> str:
        nonlocal marker_bytes, marker_mode
        markers = list(
            (chronovisor_root / "runtime" / "model-activity" / "active").glob(
                "*.json"
            )
        )
        assert len(markers) == 1
        marker_bytes = markers[0].read_bytes()
        marker_mode = markers[0].stat().st_mode & 0o777
        observed.update(json.loads(marker_bytes))
        return "ok"

    monkeypatch.setattr(ollama, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(ollama, "_generate_unlocked", fake_generate)

    assert ollama.generate("secret prompt", model="ornith:test") == "ok"
    assert observed["schema_version"] == 1
    assert observed["model"] == "ornith:test"
    assert observed["operation"] == "generate"
    assert observed["component"] == __name__
    assert observed["caller"] == "test_generate_publishes_redacted_model_activity"
    assert observed["pipeline"] == "audit"
    assert marker_bytes == (
        json.dumps(observed, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    assert marker_mode == 0o600
    assert "prompt" not in observed
    assert not list(
        (chronovisor_root / "runtime" / "model-activity" / "active").glob(
            "*.json"
        )
    )
    recent = json.loads(
        (
            chronovisor_root
            / "runtime"
            / "model-activity"
            / "recent"
            / "audit.json"
        ).read_text(encoding="utf-8")
    )
    assert recent["activity_id"] == observed["activity_id"]
    assert recent["finished_at"]


def test_model_activity_uses_live_root_when_entered(tmp_path, monkeypatch) -> None:
    initial_root = tmp_path / "initial"
    live_root = tmp_path / "live"
    monkeypatch.setattr(ollama, "CHRONOVISOR_ROOT", initial_root)
    activity = ollama.model_activity(model="ornith:test", operation="generate")

    monkeypatch.setattr(ollama, "CHRONOVISOR_ROOT", live_root)
    with activity:
        assert len(
            list(
                (live_root / "runtime" / "model-activity" / "active").glob("*.json")
            )
        ) == 1

    assert not (initial_root / "runtime" / "model-activity").exists()
    assert (live_root / "runtime" / "model-activity" / "recent" / "audit.json").is_file()


def test_generate_can_return_explicit_completion_metadata(monkeypatch) -> None:
    client = _StreamClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)

    result = ollama.generate(
        "prompt",
        progress_callback=lambda _event: None,
        return_metadata=True,
    )

    assert result == ollama.GenerateResponse(
        content="hello",
        done=True,
        done_reason="stop",
        prompt_eval_count=11,
        eval_count=2,
        streamed=True,
    )


def test_generate_metadata_preserves_incomplete_stream_for_fail_closed_caller(
    monkeypatch,
) -> None:
    client = _IncompleteStreamClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)

    result = ollama.generate(
        "prompt",
        progress_callback=lambda _event: None,
        return_metadata=True,
    )

    assert result == ollama.GenerateResponse(
        content="partial",
        done=False,
        done_reason=None,
        streamed=True,
    )


def test_num_ctx_grows_for_long_prompts_without_crossing_cap() -> None:
    config = IngestConfig(num_ctx=2048, max_num_ctx=4096, num_predict=128)

    assert ollama._num_ctx_for_prompt("short", None, config) == 2048
    assert ollama._num_ctx_for_prompt("x" * 10_000, None, config) == 4096


def test_chat_uses_fixed_structured_options_and_returns_final_content(
    monkeypatch,
) -> None:
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
    assert client.payload["shift"] is False
    assert client.payload["truncate"] is False
    assert client.payload["format"] == schema
    assert client.payload["keep_alive"] == "20m"
    assert client.payload["options"] == {
        "temperature": 0,
        "seed": 0,
        "num_predict": 1024,
        "num_ctx": 32768,
    }


def test_chat_forwards_explicit_reasoning_level(monkeypatch) -> None:
    client = _ChatClient('{"decision":"apply"}')
    monkeypatch.setattr(ollama, "_client", lambda: client)

    result = ollama.chat(
        [{"role": "user", "content": "decide"}],
        model="gpt-oss:20b",
        format={"type": "object"},
        num_ctx=65_536,
        num_predict=1_024,
        keep_alive="20m",
        read_timeout_ms=120_000,
        max_output_chars=1_000,
        think="low",
    )

    assert result == '{"decision":"apply"}'
    assert client.payload["think"] == "low"


def test_chat_enforces_output_char_cap(monkeypatch) -> None:
    client = _ChatClient("x" * 11)
    monkeypatch.setattr(ollama, "_client", lambda: client)

    with pytest.raises(ollama.OutputTooLargeError):
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


@pytest.mark.parametrize("seed", [-1, True])
def test_chat_rejects_invalid_deterministic_seed(monkeypatch, seed: object) -> None:
    client = _ChatClient('{"decision":"apply"}')
    monkeypatch.setattr(ollama, "_client", lambda: client)

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        ollama.chat(
            [{"role": "user", "content": "decide"}],
            model="ornith:test",
            format={"type": "object"},
            num_ctx=32768,
            num_predict=1024,
            keep_alive="20m",
            read_timeout_ms=120000,
            max_output_chars=1000,
            seed=seed,  # type: ignore[arg-type]
        )

    assert client.payload is None


def test_model_digests_returns_exact_installed_identities(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "_client", lambda: _TagsClient())

    assert ollama.model_digests(["ornith:test", "gpt-oss:test", "missing:test"]) == {
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
    assert result.done is True
    assert result.done_reason == "stop"


def test_chat_metadata_treats_missing_done_as_incomplete(monkeypatch) -> None:
    client = _ChatClient(
        '{"decision":"apply"}',
        done=None,
        done_reason=None,
    )
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
    assert result.done is False
    assert result.done_reason is None


def test_chat_metadata_returns_oversize_content_for_bounded_session_repair(
    monkeypatch,
) -> None:
    client = _ChatClient("x" * 11)
    monkeypatch.setattr(ollama, "_client", lambda: client)

    result = ollama.chat(
        [{"role": "user", "content": "decide"}],
        model="ornith:test",
        format={"type": "object"},
        num_ctx=4096,
        num_predict=128,
        keep_alive="20m",
        read_timeout_ms=120000,
        max_output_chars=10,
        return_metadata=True,
    )

    assert isinstance(result, ollama.ChatResponse)
    assert result.content == "x" * 11


def test_embed_uses_explicit_model(monkeypatch) -> None:
    client = _PostClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)

    assert ollama.embed(["hello"], model="bge-m3") == [[1.0, 2.0]]
    assert client.payload["model"] == "bge-m3"


def test_embed_uses_remaining_recall_timeout(monkeypatch) -> None:
    client = _PostClient()
    monkeypatch.setattr(ollama, "_client", lambda: client)

    assert ollama.embed(["hello"], model="bge-m3", read_timeout_ms=750) == [[1.0, 2.0]]

    assert isinstance(client.timeout, httpx.Timeout)
    assert client.timeout.read == 0.75
    assert client.timeout.connect == 0.75


def test_resource_lease_blocks_exclusive_across_threads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "resource.lock"))
    shared_entered = threading.Event()
    release_shared = threading.Event()
    exclusive_attempted = threading.Event()
    exclusive_entered = threading.Event()
    failures: list[BaseException] = []

    def hold_shared() -> None:
        try:
            with ollama.model_resource_lease(exclusive=False):
                shared_entered.set()
                assert release_shared.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    def take_exclusive() -> None:
        try:
            exclusive_attempted.set()
            with ollama.model_resource_lease(exclusive=True):
                exclusive_entered.set()
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    reader = threading.Thread(target=hold_shared)
    writer = threading.Thread(target=take_exclusive)
    reader.start()
    assert shared_entered.wait(timeout=5)
    writer.start()
    assert exclusive_attempted.wait(timeout=5)
    assert not exclusive_entered.wait(timeout=0.1)
    release_shared.set()
    reader.join(timeout=5)
    writer.join(timeout=5)

    assert not reader.is_alive()
    assert not writer.is_alive()
    assert exclusive_entered.is_set()
    assert failures == []


def test_resource_lease_reentry_and_upgrade_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "resource.lock"))

    assert ollama.model_resource_lease_mode() is None
    with ollama.model_resource_lease(exclusive=True):
        assert ollama.model_resource_lease_mode() == "exclusive"
        with ollama.model_resource_lease(exclusive=False):
            assert ollama.model_resource_lease_mode() == "exclusive"
            with ollama.model_resource_lease(exclusive=True):
                assert ollama.model_resource_lease_mode() == "exclusive"
                pass
    assert ollama.model_resource_lease_mode() is None

    with ollama.model_resource_lease(exclusive=False):
        assert ollama.model_resource_lease_mode() == "shared"
        with pytest.raises(RuntimeError, match="cannot upgrade"):
            with ollama.model_resource_lease(exclusive=True):
                pass
    assert ollama.model_resource_lease_mode() is None


def test_resource_lease_facade_uses_current_chronovisor_root(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "wiki"
    monkeypatch.delenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", raising=False)
    monkeypatch.setattr(ollama, "CHRONOVISOR_ROOT", root)

    with ollama.model_resource_lease(exclusive=False):
        pass

    assert (root / "runtime" / "ollama-resource.lock").is_file()


def test_resource_lease_blocks_another_process(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "resource.lock"
    ready_path = tmp_path / "ready"
    acquired_path = tmp_path / "acquired"
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(lock_path))
    env = os.environ.copy()
    env.update(
        {
            "LEASE_READY_PATH": str(ready_path),
            "LEASE_ACQUIRED_PATH": str(acquired_path),
        }
    )
    script = """
import os
from pathlib import Path
from chronovisor.core import ollama

Path(os.environ["LEASE_READY_PATH"]).write_text("ready", encoding="utf-8")
with ollama.model_resource_lease(exclusive=False):
    Path(os.environ["LEASE_ACQUIRED_PATH"]).write_text("acquired", encoding="utf-8")
"""

    with ollama.model_resource_lease(exclusive=True):
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 5
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_path.exists()
        assert not acquired_path.exists()

    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    assert acquired_path.exists()


def test_resource_lease_timeout_does_not_wait_for_another_process(
    tmp_path, monkeypatch
) -> None:
    lock_path = tmp_path / "resource.lock"
    ready_path = tmp_path / "ready"
    timed_out_path = tmp_path / "timed-out"
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(lock_path))
    env = os.environ.copy()
    env.update(
        {
            "LEASE_READY_PATH": str(ready_path),
            "LEASE_TIMED_OUT_PATH": str(timed_out_path),
        }
    )
    script = """
import os
from pathlib import Path
from chronovisor.core import ollama

Path(os.environ["LEASE_READY_PATH"]).write_text("ready", encoding="utf-8")
try:
    with ollama.model_resource_lease(exclusive=True, timeout_ms=25):
        raise AssertionError("contended lease must not be acquired")
except TimeoutError:
    Path(os.environ["LEASE_TIMED_OUT_PATH"]).write_text("timeout", encoding="utf-8")
"""

    with ollama.model_resource_lease(exclusive=True):
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 5
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_path.exists()
        stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 0, (stdout, stderr)
    assert timed_out_path.exists()


def test_public_heavy_operations_take_expected_resource_lease(monkeypatch) -> None:
    modes: list[bool] = []

    @contextmanager
    def record_lease(*, exclusive: bool):
        modes.append(exclusive)
        yield

    monkeypatch.setattr(ollama, "model_resource_lease", record_lease)
    monkeypatch.setattr(ollama, "_generate_unlocked", lambda *_args, **_kwargs: "ok")
    monkeypatch.setattr(ollama, "_chat_unlocked", lambda *_args, **_kwargs: "ok")
    monkeypatch.setattr(ollama, "_client", lambda: _PostClient())
    monkeypatch.setattr(ollama, "_ollama_resource_rows", lambda: ({}, {}))

    assert ollama.generate("prompt") == "ok"
    assert (
        ollama.chat(
            [{"role": "user", "content": "prompt"}],
            model="ornith:test",
            format={"type": "object"},
            num_ctx=4096,
            num_predict=128,
            keep_alive="1m",
            read_timeout_ms=1000,
            max_output_chars=1000,
        )
        == "ok"
    )
    assert ollama.embed(["hello"], model="bge-m3") == [[1.0, 2.0]]
    assert ollama.unload_named_model("ornith:test") is True

    assert modes == [False, False, False, True]


@pytest.mark.parametrize("size_vram", [0, 4])
def test_resource_rows_use_total_size_for_cpu_or_partial_offload(
    monkeypatch,
    size_vram: int,
) -> None:
    monkeypatch.setattr(
        ollama,
        "_client",
        lambda: _ResourceClient(resident_size_vram=size_vram, resident_size=12),
    )

    installed, resident = ollama._ollama_resource_rows()

    assert installed == {"ornith:test": 10}
    assert resident == {"ornith:test": (12, 4096)}


def test_macos_memory_snapshot_uses_kernel_pressure_availability(monkeypatch) -> None:
    class _Darwin:
        sysname = "Darwin"

    calls: list[tuple[str, ...]] = []

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[0] == "sysctl":
            stdout = str(128 * ollama.GIB)
        elif args[0] == "memory_pressure":
            stdout = (
                "The system has 137438953472 (8388608 pages with a page size of 16384).\n"
                "System-wide memory free percentage: 82%\n"
            )
        else:  # pragma: no cover - proves vm_stat is not consulted
            pytest.fail(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(ollama.os, "uname", lambda: _Darwin())
    monkeypatch.setattr(ollama.subprocess, "run", fake_run)

    snapshot = ollama.memory_snapshot()

    assert snapshot == ollama.MemorySnapshot(
        total_bytes=128 * ollama.GIB,
        available_bytes=(128 * ollama.GIB * 82) // 100,
        source="macos_memory_pressure",
    )
    assert calls == [
        ("sysctl", "-n", "hw.memsize"),
        ("memory_pressure", "-Q"),
    ]


def test_macos_memory_snapshot_falls_back_to_vm_stat_on_invalid_pressure_probe(
    monkeypatch,
) -> None:
    class _Darwin:
        sysname = "Darwin"

    calls: list[str] = []

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args[0])
        if args[0] == "sysctl":
            stdout = str(128 * ollama.GIB)
        elif args[0] == "memory_pressure":
            stdout = "unexpected output\n"
        elif args[0] == "vm_stat":
            stdout = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages inactive: 200.
Pages speculative: 25.
Pages purgeable: 50.
"""
        else:  # pragma: no cover - all commands are enumerated above
            pytest.fail(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(ollama.os, "uname", lambda: _Darwin())
    monkeypatch.setattr(ollama.subprocess, "run", fake_run)

    snapshot = ollama.memory_snapshot()

    assert snapshot == ollama.MemorySnapshot(
        total_bytes=128 * ollama.GIB,
        available_bytes=375 * 16_384,
        source="macos_vm_stat",
    )
    assert calls == ["sysctl", "memory_pressure", "sysctl", "vm_stat"]


def test_macos_memory_snapshot_rejects_pressure_total_from_another_host(
    monkeypatch,
) -> None:
    class _Darwin:
        sysname = "Darwin"

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if args[0] == "sysctl":
            stdout = str(128 * ollama.GIB)
        elif args[0] == "memory_pressure":
            stdout = (
                "The system has 68719476736 (4194304 pages with a page size of 16384).\n"
                "System-wide memory free percentage: 82%\n"
            )
        elif args[0] == "vm_stat":
            stdout = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages inactive: 200.
"""
        else:  # pragma: no cover - all commands are enumerated above
            pytest.fail(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(ollama.os, "uname", lambda: _Darwin())
    monkeypatch.setattr(ollama.subprocess, "run", fake_run)

    snapshot = ollama.memory_snapshot()

    assert snapshot.source == "macos_vm_stat"
    assert snapshot.available_bytes == 300 * 16_384


def test_pressure_aware_snapshot_bootstraps_exactly_one_uncalibrated_runner() -> None:
    memory = ollama.MemorySnapshot(
        total_bytes=128 * ollama.GIB,
        available_bytes=(128 * ollama.GIB * 82) // 100,
        source="macos_memory_pressure",
    )
    plan = ollama.build_model_residency_plan(
        ["ornith:35b", "gpt-oss:20b", "gemma4:26b"],
        num_ctx=16_384,
        max_num_ctx=114_688,
        memory=memory,
        installed_sizes={
            "ornith:35b": 24_729_131_264,
            "gpt-oss:20b": 13_793_441_244,
            "gemma4:26b": 17_987_581_215,
        },
        resident={},
        calibrated_sizes={},
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
    )

    assert plan.estimate("ornith:35b") == 49_458_262_528
    assert plan.max_resident_models == 1
    assert plan.forced_single is False
    assert plan.calibrated_models == ()


def test_macos_pressure_snapshot_reads_compressor_and_swap(monkeypatch) -> None:
    class _Darwin:
        sysname = "Darwin"

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if args == ["vm_stat"]:
            stdout = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages occupied by compressor: 2523793.
"""
        elif args == ["sysctl", "-n", "vm.swapusage"]:
            stdout = "vm.swapusage: total = 9216.00M  used = 8155.19M  free = 1060.81M"
        else:  # pragma: no cover - all commands are enumerated above
            pytest.fail(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(ollama.os, "uname", lambda: _Darwin())
    monkeypatch.setattr(ollama.subprocess, "run", fake_run)

    snapshot = ollama.macos_pressure_snapshot()

    assert snapshot == ollama.MacOSPressureSnapshot(
        compressed_bytes=2_523_793 * 16_384,
        swap_used_bytes=int(8_155.19 * 1024**2),
        source="vm_stat+swapusage",
    )


def test_compressed_memory_forces_single_resident_even_when_three_fit() -> None:
    memory = ollama.MemorySnapshot(
        total_bytes=128 * ollama.GIB,
        available_bytes=96 * ollama.GIB,
        source="macos_memory_pressure",
    )
    plan = ollama.build_model_residency_plan(
        ["ornith:35b", "gpt-oss:20b", "gemma4:26b"],
        num_ctx=16_384,
        max_num_ctx=114_688,
        memory=memory,
        installed_sizes={
            "ornith:35b": 8 * ollama.GIB,
            "gpt-oss:20b": 8 * ollama.GIB,
            "gemma4:26b": 8 * ollama.GIB,
        },
        resident={},
        calibrated_sizes={
            ("ornith:35b", 16_384): 10 * ollama.GIB,
            ("gpt-oss:20b", 16_384): 10 * ollama.GIB,
            ("gemma4:26b", 16_384): 10 * ollama.GIB,
        },
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
        pressure=ollama.MacOSPressureSnapshot(
            compressed_bytes=40 * ollama.GIB,
            swap_used_bytes=8 * ollama.GIB,
            source="test",
        ),
    )

    assert plan.max_resident_models == 1
    assert plan.pressure_forced_single is True
    assert plan.compressed_bytes == 40 * ollama.GIB
    assert plan.swap_used_bytes == 8 * ollama.GIB
    assert plan.audit_record()["pressure_forced_single"] is True


def test_daemon_identity_ignores_runner_and_changes_on_restart(monkeypatch) -> None:
    process_table = {
        "stdout": """
120 Sun Jul 13 06:00:00 2026 /opt/homebrew/bin/ollama serve
121 Sun Jul 13 06:00:01 2026 /opt/homebrew/bin/llama-server -c 32768
"""
    }

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=process_table["stdout"], stderr=""
        )

    monkeypatch.setattr(ollama.subprocess, "run", fake_run)

    first = ollama._ollama_daemon_process_identity()
    process_table["stdout"] = """
121 Sun Jul 13 06:00:01 2026 /opt/homebrew/bin/llama-server -c 32768
220 Sun Jul 13 07:00:00 2026 /opt/homebrew/bin/ollama serve
"""
    second = ollama._ollama_daemon_process_identity()

    assert first != second


def test_engine_identity_ignores_caller_env_for_same_daemon(monkeypatch) -> None:
    class VersionClient:
        def get(self, path: str, *, timeout: object) -> _JsonResponse:
            assert path == "/api/version"
            assert timeout == 3
            return _JsonResponse({"version": "0.11.0"})

    monkeypatch.setattr(ollama, "_client", lambda: VersionClient())
    monkeypatch.setattr(
        ollama, "_ollama_daemon_process_identity", lambda: "daemon-epoch"
    )
    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE", raising=False)
    default_identity = ollama._ollama_engine_identity()

    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
    configured_identity = ollama._ollama_engine_identity()

    assert default_identity == configured_identity
    assert default_identity.startswith("ollama-engine-v2:")
    assert len(default_identity) == len("ollama-engine-v2:") + 64


def test_previous_calibration_schema_is_ignored(tmp_path) -> None:
    path = tmp_path / "ollama-footprints.json"
    path.write_text(
        json.dumps({"schema_version": 1, "entries": [{"size_bytes": 1}]}),
        encoding="utf-8",
    )

    assert ollama._read_calibration_payload(path) == {}


def test_footprint_calibration_persists_and_invalidates_by_identity(
    tmp_path, monkeypatch
) -> None:
    calibration_file = tmp_path / "ollama-footprints.json"
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_CALIBRATION_FILE", str(calibration_file))
    previous = dict(ollama._MODEL_FOOTPRINT_CALIBRATION)
    ollama._MODEL_FOOTPRINT_CALIBRATION.clear()
    resident: dict[str, tuple[int, int]] = {"ornith:test": (20 * ollama.GIB, 32_768)}
    monkeypatch.setattr(
        ollama,
        "_ollama_resource_rows",
        lambda: ({"ornith:test": 10 * ollama.GIB}, dict(resident)),
    )
    monkeypatch.setattr(
        ollama, "model_digests", lambda _models: {"ornith:test": "digest-a"}
    )
    monkeypatch.setattr(ollama, "_ollama_engine_identity", lambda: "engine-a")
    monkeypatch.setattr(
        ollama,
        "memory_snapshot",
        lambda: ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=96 * ollama.GIB,
            source="test",
        ),
    )
    try:
        assert ollama.observe_model_runtime("ornith:test") == (
            20 * ollama.GIB,
            32_768,
        )
        assert calibration_file.exists()
        assert calibration_file.stat().st_mode & 0o777 == 0o600

        ollama._MODEL_FOOTPRINT_CALIBRATION.clear()
        resident.clear()
        plan = ollama.plan_model_residency(
            ["ornith:test"],
            num_ctx=32_768,
            max_num_ctx=131_072,
            reserve_bytes=16 * ollama.GIB,
            configured_max_resident=1,
        )
        assert plan.estimate("ornith:test") == 20 * ollama.GIB
        assert plan.calibrated_models == ("ornith:test",)

        assert (
            ollama._matching_persisted_calibrations(
                installed={"ornith:test": 10 * ollama.GIB},
                digests={"ornith:test": "digest-b"},
                engine="engine-a",
            )
            == {}
        )
        assert (
            ollama._matching_persisted_calibrations(
                installed={"ornith:test": 10 * ollama.GIB},
                digests={"ornith:test": "digest-a"},
                engine="engine-b",
            )
            == {}
        )
        assert (
            ollama._matching_persisted_calibrations(
                installed={"other:test": 10 * ollama.GIB},
                digests={"other:test": "digest-a"},
                engine="engine-a",
            )
            == {}
        )

        different_context = ollama.plan_model_residency(
            ["ornith:test"],
            num_ctx=65_536,
            max_num_ctx=131_072,
            reserve_bytes=16 * ollama.GIB,
            configured_max_resident=1,
        )
        assert different_context.calibrated_models == ()
    finally:
        ollama._MODEL_FOOTPRINT_CALIBRATION.clear()
        ollama._MODEL_FOOTPRINT_CALIBRATION.update(previous)


def test_persisted_footprint_is_readable_from_a_fresh_process(
    tmp_path, monkeypatch
) -> None:
    calibration_file = tmp_path / "ollama-footprints.json"
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_CALIBRATION_FILE", str(calibration_file))
    ollama._persist_model_calibration(
        model="ornith:test",
        context=32_768,
        installed_size=10,
        digest="digest-a",
        engine="engine-a",
        size_bytes=20,
    )
    script = """
import json
from chronovisor.core import ollama

rows = ollama._matching_persisted_calibrations(
    installed={"ornith:test": 10},
    digests={"ornith:test": "digest-a"},
    engine="engine-a",
)
print(json.dumps({f"{model}:{context}": size for (model, context), size in rows.items()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=5,
    )

    assert json.loads(result.stdout) == {"ornith:test:32768": 20}


def test_unload_fails_closed_on_probe_failure_or_timeout(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "_client", lambda: _PostClient())

    def failed_probe():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(ollama, "_ollama_resource_rows", failed_probe)
    assert ollama.unload_named_model("ornith:test", verify_timeout=0) is False

    monkeypatch.setattr(
        ollama,
        "_ollama_resource_rows",
        lambda: ({"ornith:test": 10}, {"ornith:test": (12, 4096)}),
    )
    assert ollama.unload_named_model("ornith:test", verify_timeout=0) is False


def test_residency_plan_returns_zero_when_no_runner_fits() -> None:
    plan = ollama.build_model_residency_plan(
        ["ornith:test"],
        num_ctx=32_768,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=17 * ollama.GIB,
            source="test",
        ),
        installed_sizes={"ornith:test": 20 * ollama.GIB},
        resident={},
        calibrated_sizes={
            ("ornith:test", 32_768): 24 * ollama.GIB,
        },
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=1,
    )

    assert plan.capacity_bytes == ollama.GIB
    assert plan.max_resident_models == 0
    assert plan.forced_single is True


def test_unobserved_context_uses_known_footprint_as_a_lower_bound() -> None:
    plan = ollama.build_model_residency_plan(
        ["ornith:test"],
        num_ctx=65_536,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=64 * ollama.GIB,
            available_bytes=38 * ollama.GIB,
            source="test",
        ),
        installed_sizes={"ornith:test": 20 * ollama.GIB},
        resident={},
        calibrated_sizes={
            ("ornith:test", 32_768): 40 * ollama.GIB,
        },
        reserve_bytes=8 * ollama.GIB,
        configured_max_resident=1,
    )

    assert plan.capacity_bytes == 30 * ollama.GIB
    assert plan.estimate("ornith:test") >= 40 * ollama.GIB
    assert plan.max_resident_models == 0


def test_uncalibrated_single_runner_does_not_ignore_current_pressure() -> None:
    plan = ollama.build_model_residency_plan(
        ["ornith:test", "challenger:test"],
        num_ctx=32_768,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=32 * ollama.GIB,
            source="test",
        ),
        installed_sizes={
            "ornith:test": 24 * ollama.GIB,
            "challenger:test": 14 * ollama.GIB,
        },
        resident={"embedding:test": (60 * ollama.GIB, 8_192)},
        calibrated_sizes={},
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
    )

    assert plan.capacity_bytes == 16 * ollama.GIB
    assert plan.estimate("ornith:test") == 48 * ollama.GIB
    assert plan.max_resident_models == 0
    assert plan.calibrated_models == ()


def test_uncalibrated_single_runner_bootstraps_with_current_headroom() -> None:
    plan = ollama.build_model_residency_plan(
        ["ornith:test", "challenger:test"],
        num_ctx=32_768,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=80 * ollama.GIB,
            source="test",
        ),
        installed_sizes={
            "ornith:test": 24 * ollama.GIB,
            "challenger:test": 14 * ollama.GIB,
        },
        resident={},
        calibrated_sizes={},
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
    )

    assert plan.capacity_bytes == 64 * ollama.GIB
    assert plan.estimate("ornith:test") == 48 * ollama.GIB
    assert plan.max_resident_models == 1
    assert plan.calibrated_models == ()


def test_live_residency_probe_failure_returns_zero_runner_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        ollama,
        "memory_snapshot",
        lambda: ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=96 * ollama.GIB,
            source="test",
        ),
    )
    monkeypatch.setattr(
        ollama,
        "_ollama_resource_rows",
        lambda: (_ for _ in ()).throw(RuntimeError("probe unavailable")),
    )

    plan = ollama.plan_model_residency(
        ["ornith:test"],
        num_ctx=32_768,
        max_num_ctx=131_072,
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=1,
    )

    assert plan.source == "test+ollama_unavailable"
    assert plan.max_resident_models == 0
    assert plan.forced_single is True
