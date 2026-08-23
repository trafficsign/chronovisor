from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from urllib.request import Request

import httpx
import pytest

from chronovisor.core.llm_security import CredentialRef, CredentialResolver
from chronovisor.core.openai_compatible_adapter import compose_openai_compatible_adapter
from chronovisor.core.provider_profiles import generic_openai_profile
from chronovisor.recall.recall_distillation_remote_teacher import (
    OX_ALPHA_ENDPOINT,
    OX_ALPHA_ROUTE_MODEL,
    OpenCodeOxAlphaTeacher,
)

CANARY = "sk-CANARY-REMOTE-TEACHER"
CREDENTIAL_REF = CredentialRef.parse("env:REMOTE_TEACHER_API_KEY")
ENDPOINT = OX_ALPHA_ENDPOINT


class FakeSender:
    def __init__(
        self, response: httpx.Response, *, error: Exception | None = None
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[Request] = []

    def __call__(
        self, request: Request, *, follow_redirects: bool, timeout_seconds: float
    ) -> object:
        assert follow_redirects is False
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.response


def _resolver(tmp_path: Path) -> CredentialResolver:
    return CredentialResolver(
        environ={CREDENTIAL_REF.target: CANARY},
        repo_root=tmp_path / "repo",
        home_root=tmp_path / "home",
    )


def _response(content: str, *, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "model": "ox-alpha-free",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
        headers={"x-request-id": "ox_req_1"},
    )


def _payload(*, query: str = "which note answers the query") -> dict[str, object]:
    return {
        "schema": "chronovisor.recall-distill-teacher-batch.v1",
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "rally_id": "rally-1",
                "query": query,
                "context": ["A bounded historical context."],
                "evidence": "The candidate note contains the answer.",
            }
        ],
    }


def _label_response() -> str:
    return json.dumps(
        {
            "labels": [
                {
                    "candidate_id": "candidate-1",
                    "verdict": "relevant",
                    "confidence": 0.9,
                    "rationale": "The evidence directly answers the query.",
                    "minimal_atom_ids": ["atom-1"],
                    "missing_slots": [],
                    "changing_claim": "",
                }
            ]
        }
    )


def _teacher(
    tmp_path: Path,
    sender: FakeSender,
    *,
    endpoint: str = ENDPOINT,
    provider: str = "opencode-go",
    configured_model: str = OX_ALPHA_ROUTE_MODEL,
) -> OpenCodeOxAlphaTeacher:
    profile = generic_openai_profile(
        provider,
        endpoint,
        CREDENTIAL_REF,
        structured_output_models={"ox-alpha-free"},
    )
    backend = compose_openai_compatible_adapter(
        profile,
        _resolver(tmp_path),
        sender=sender,
    )
    return OpenCodeOxAlphaTeacher(backend, configured_model=configured_model)


def test_success_uses_shared_adapter_and_records_safe_digests(
    tmp_path: Path,
) -> None:
    sender = FakeSender(_response(_label_response()))
    teacher = _teacher(tmp_path, sender)

    result = teacher.evaluate(_payload())

    assert result["labels"][0]["verdict"] == "relevant"
    assert result["_route_identity"] == {
        "provider": "opencode-go",
        "model": OX_ALPHA_ROUTE_MODEL,
        "location": "remote",
    }
    for key in ("_model_digest", "_route_digest", "_prompt_digest", "_schema_digest"):
        assert len(cast(str, result[key])) == 64
    body = json.loads(cast(bytes, sender.calls[0].data))
    assert body["model"] == "ox-alpha-free"
    assert sender.calls[0].full_url == f"{ENDPOINT}/chat/completions"
    assert body["response_format"]["json_schema"]["schema"]["properties"][
        "labels"
    ]["items"]["properties"]["candidate_id"]["enum"] == ["candidate-1"]
    assert CANARY not in cast(bytes, sender.calls[0].data).decode("utf-8")


@pytest.mark.parametrize(
    "payload",
    [
        {**_payload(), "unexpected": "no"},
        _payload(query="send password=do-not-send"),
        _payload(query="token=do-not-send"),
        _payload(query="Authorization: Bearer sk-CANARY-REMOTE-TEACHER"),
        _payload(query="contact user@example.com"),
        _payload(query="internal customer case"),
        _payload(query="read file:///Users/trafficsign/private/note.txt"),
    ],
)
def test_egress_allowlist_is_fail_closed_and_call_free(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    sender = FakeSender(_response(_label_response()))
    teacher = _teacher(tmp_path, sender)

    result = teacher.evaluate(payload)

    assert result["_failure"]["class"] == "remote_payload_rejected"
    assert result["_failure"]["labelable"] is False
    assert sender.calls == []


@pytest.mark.parametrize(
    "status, failure",
    [
        (301, "redirect_rejected"),
        (401, "http_401"),
        (402, "http_error"),
        (429, "http_429"),
        (503, "http_5xx"),
    ],
)
def test_provider_failure_is_safe_and_has_no_label(
    tmp_path: Path, status: int, failure: str
) -> None:
    sender = FakeSender(_response(CANARY, status=status))
    teacher = _teacher(tmp_path, sender)

    result = teacher.evaluate(_payload())

    assert result["_failure"]["class"] == failure
    assert result["_failure"]["labelable"] is False
    assert "labels" not in result
    assert CANARY not in repr(result)


def test_timeout_is_safe_and_has_no_label(tmp_path: Path) -> None:
    sender = FakeSender(
        _response(_label_response()), error=httpx.ReadTimeout("provider timeout")
    )
    teacher = _teacher(tmp_path, sender)

    result = teacher.evaluate(_payload())

    assert result["_failure"]["class"] == "timeout"
    assert result["_failure"]["labelable"] is False
    assert "labels" not in result


def test_invalid_provider_content_is_not_reflected(tmp_path: Path) -> None:
    sender = FakeSender(_response(CANARY))
    teacher = _teacher(tmp_path, sender)

    result = teacher.evaluate(_payload())

    assert result["_failure"]["class"] == "invalid_response"
    assert result["_failure"]["labelable"] is False
    assert CANARY not in repr(result)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "other-provider"},
        {"endpoint": "https://opencode.ai/zen/v1"},
        {"configured_model": "opencode-go/ox-alpha"},
    ],
)
def test_route_contract_is_exact_and_rejects_paid_alternatives(
    tmp_path: Path, kwargs: dict[str, str]
) -> None:
    sender = FakeSender(_response(_label_response()))

    with pytest.raises(ValueError):
        _teacher(tmp_path, sender, **kwargs)
    assert sender.calls == []


def test_invalid_provider_output_is_not_converted_to_a_label(tmp_path: Path) -> None:
    sender = FakeSender(_response(json.dumps({"labels": []})))
    teacher = _teacher(tmp_path, sender)

    result = teacher.evaluate(_payload())

    assert result["_failure"]["class"] == "invalid_response"
    assert result["_failure"]["labelable"] is False


def test_kill_switch_and_paid_fallback_guard(tmp_path: Path) -> None:
    sender = FakeSender(_response(_label_response()))
    teacher = _teacher(tmp_path, sender)
    teacher.disable()

    result = teacher.evaluate(_payload())

    assert result["_failure"]["class"] == "remote_teacher_disabled"
    assert sender.calls == []
    with pytest.raises(ValueError):
        OpenCodeOxAlphaTeacher(
            teacher._backend,
            free_only=False,
        )
