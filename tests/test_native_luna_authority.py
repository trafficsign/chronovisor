from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.decision import native_luna_authority as native


def _runtime_identity() -> dict[str, object]:
    route = {
        "role": native.NATIVE_LUNA_ROLE,
        "provider": native.NATIVE_LUNA_PROVIDER,
        "model": native.NATIVE_LUNA_MODEL,
        "location": "native",
        "protocol": native.NATIVE_LUNA_PROTOCOL,
        "endpoint_sha256": "a" * 64,
        "revision": f"{native.NATIVE_LUNA_REVISION_PREFIX}{'b' * 16}",
        "ollama": None,
    }
    return {
        **route,
        "_role_config_sha256": "c" * 64,
        "_codex_version_sha256": "d" * 64,
        "_codex_path": "/private/codex",
    }


def _review() -> dict[str, object]:
    route = native.native_luna_route_identity()
    return {
        "decision": "approved",
        "confidence": 0.9,
        "expected_pages": ["page-a"],
        "negative_pages": [],
        "stale_pages": [],
        "summary": "approved",
        "notes": None,
        "reviewer": "native_luna",
        "decision_policy": {
            "router_policy": {
                "source": native.NATIVE_LUNA_SOURCE,
                "error": None,
                "routes": [route],
            }
        },
    }


def _receipt(review: dict[str, object]) -> dict[str, object]:
    identity = _runtime_identity()
    receipt = {
        "schema": native.NATIVE_LUNA_RECEIPT_SCHEMA,
        "run_id": f"native-search-{'a' * 32}",
        "status": "completed",
        "exit_code": 0,
        "model": native.NATIVE_LUNA_MODEL,
        "provider": native.NATIVE_LUNA_PROVIDER,
        "route": native.NATIVE_LUNA_ROUTE,
        "role": native.NATIVE_LUNA_ROLE,
        "agent_type": native.NATIVE_LUNA_AGENT_TYPE,
        "revision": identity["revision"],
        "request_sha256": "e" * 64,
        "input_sha256": "e" * 64,
        "prompt_sha256": "f" * 64,
        "schema_sha256": "1" * 64,
        "output_sha256": canonical_json_sha256_strict(review),
        "role_config_sha256": identity["_role_config_sha256"],
        "codex_binary_sha256": identity["endpoint_sha256"],
        "codex_version_sha256": identity["_codex_version_sha256"],
        "session_model": native.NATIVE_LUNA_MODEL,
        "session_provider": native.NATIVE_LUNA_PROVIDER,
        "session_reasoning_effort": "max",
        "session_id": "01a14a46-be40-75a2-9391-c464d8d97cd6",
        "session_provenance_sha256": canonical_json_sha256_strict(
            {
                "session_id": "01a14a46-be40-75a2-9391-c464d8d97cd6",
                "model": native.NATIVE_LUNA_MODEL,
                "provider": native.NATIVE_LUNA_PROVIDER,
                "reasoning_effort": "max",
            }
        ),
    }
    receipt["receipt_sha256"] = canonical_json_sha256_strict(receipt)
    return receipt


def _persist_receipt(root: Path, receipt: dict[str, object]) -> None:
    destination = root / "runtime/native-luna-authority/receipts"
    destination.mkdir(parents=True)
    (destination / f"{receipt['run_id']}.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _authority() -> dict[str, object]:
    route = native.native_luna_route_identity()
    return {
        "source": native.NATIVE_LUNA_SOURCE,
        "router": {
            "source": native.NATIVE_LUNA_SOURCE,
            "error": None,
            "routes": [route],
        },
    }


def test_native_config_is_shared_when_environment_override_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.delenv(native.NATIVE_LUNA_AUTHORITY_ENV, raising=False)
    (tmp_path / "config.toml").write_text(
        "[search_label_review]\nauthority = 'native_luna'\n",
        encoding="utf-8",
    )

    # Resolve twice to model independent processes reading the same canonical
    # root config rather than inheriting a per-service environment override.
    assert native.native_luna_authority_for_lane("search_label") == "native_luna"
    assert native.native_luna_enabled_for_lane("search_label") is True
    assert native.native_luna_enabled_for_lane("search_label") is True
    assert native.native_luna_enabled_for_lane("frontier") is False


def test_config_native_authority_is_current_for_decision_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.decision import decision_authority

    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.delenv(native.NATIVE_LUNA_AUTHORITY_ENV, raising=False)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_SEARCH_LABEL", "enabled")
    (tmp_path / "config.toml").write_text(
        "[decision_policies.search_label]\nmode = 'enabled'\n"
        "[search_label_review]\nauthority = 'native_luna'\n",
        encoding="utf-8",
    )
    route = {
        key: value
        for key, value in _runtime_identity().items()
        if not key.startswith("_")
    }
    monkeypatch.setattr(native, "native_luna_route_identity", lambda: route)

    authority, error = decision_authority.base_semantic_authority("search_label")

    assert error is None
    assert authority is not None
    assert authority["source"] == native.NATIVE_LUNA_SOURCE
    assert authority["router"]["routes"] == [route]


@pytest.mark.parametrize(
    "config",
    (
        "[search_label_review\n",
        "[search_label_review]\nauthority = 'unsupported'\n",
    ),
)
def test_native_config_errors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: str,
) -> None:
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.delenv(native.NATIVE_LUNA_AUTHORITY_ENV, raising=False)
    (tmp_path / "config.toml").write_text(config, encoding="utf-8")

    with pytest.raises(native.NativeLunaIdentityError):
        native.native_luna_enabled_for_lane("search_label")
    # A malformed search-label setting does not alter unrelated lanes.
    assert native.native_luna_enabled_for_lane("frontier") is False


def test_native_receipt_binds_result_and_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(native, "_runtime_identity", lambda: _runtime_identity())
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.delenv(native.NATIVE_LUNA_AUTHORITY_ENV, raising=False)
    (tmp_path / "config.toml").write_text(
        "[search_label_review]\nauthority = 'native_luna'\n",
        encoding="utf-8",
    )
    assert native.native_luna_enabled_for_lane("search_label") is True
    review = _review()
    receipt = _receipt(review)
    _persist_receipt(tmp_path, receipt)
    review["native_luna_receipt"] = receipt
    assert native.validate_native_luna_review(review, _authority(), lane="search_label") is None

    review["summary"] = "tampered"
    assert native.validate_native_luna_review(review, _authority(), lane="search_label")


def test_native_receipt_rejects_wrong_model_and_schema_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native, "_runtime_identity", lambda: _runtime_identity())
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    review = _review()
    receipt = _receipt(review)
    receipt["model"] = "wrong-model"
    receipt["receipt_sha256"] = canonical_json_sha256_strict(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    _persist_receipt(tmp_path, receipt)
    review["native_luna_receipt"] = receipt
    assert native.validate_native_luna_review(review, _authority(), lane="search_label")


def test_native_request_binding_rejects_stale_prompt_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native,
        "decision_request_fingerprint_sha256",
        lambda **kwargs: hashlib.sha256(
            json.dumps(
                {"prompt": kwargs["prompt"], "schema": kwargs["schema"]},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    )
    prompt = "synthetic prompt"
    schema = {"type": "object"}
    receipt = {
        "request_sha256": native.decision_request_fingerprint_sha256(
            prompt=prompt, schema=schema, system=None, decision_lane="search_label"
        ),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": native.schema_sha256(schema),
    }
    assert native.validate_native_luna_request(receipt, prompt, schema) is None
    assert native.validate_native_luna_request(receipt, "stale prompt", schema)
    assert native.validate_native_luna_request(receipt, prompt, {"type": "array"})


def test_native_authority_rejects_stale_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _runtime_identity()
    monkeypatch.setattr(native, "_runtime_identity", lambda: identity)
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    review = _review()
    receipt = _receipt(review)
    _persist_receipt(tmp_path, receipt)
    review["native_luna_receipt"] = receipt
    changed = dict(identity)
    changed["endpoint_sha256"] = "9" * 64
    monkeypatch.setattr(native, "_runtime_identity", lambda: changed)
    assert native.validate_native_luna_review(review, _authority(), lane="search_label")


def test_native_session_provenance_is_fail_closed() -> None:
    assert native._session_provenance("") is None
    event = "\n".join(
        (
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"model_provider": native.NATIVE_LUNA_PROVIDER},
                }
            ),
            json.dumps(
                {
                    "type": "turn_context",
                    "payload": {
                        "model": native.NATIVE_LUNA_MODEL,
                        "effort": "max",
                        "collaboration_mode": {
                            "settings": {
                                "model": native.NATIVE_LUNA_MODEL,
                                "reasoning_effort": "max",
                            }
                        },
                    },
                }
            ),
        )
    )
    assert native._session_provenance(event) == {
        "model": native.NATIVE_LUNA_MODEL,
        "provider": native.NATIVE_LUNA_PROVIDER,
        "reasoning_effort": "max",
    }
    assert native._session_provenance(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "model": native.NATIVE_LUNA_MODEL,
                    "provider": native.NATIVE_LUNA_PROVIDER,
                    "reasoning_effort": "max",
                },
            }
        )
    ) is None


def test_native_session_metadata_binds_stdout_thread_to_private_transcript(
    tmp_path: Path,
) -> None:
    session = tmp_path / "sessions/2026/10/17/rollout.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "session-1",
                            "model_provider": native.NATIVE_LUNA_PROVIDER,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn_context",
                        "payload": {
                            "model": native.NATIVE_LUNA_MODEL,
                            "effort": "max",
                            "collaboration_mode": {
                                "settings": {
                                    "model": native.NATIVE_LUNA_MODEL,
                                    "reasoning_effort": "max",
                                }
                            },
                        },
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert native._thread_started_id(
        json.dumps({"type": "thread.started", "thread_id": "session-1"})
    ) == "session-1"
    assert native._isolated_session_metadata(tmp_path, "session-1") == (
        "session-1",
        {
            "model": native.NATIVE_LUNA_MODEL,
            "provider": native.NATIVE_LUNA_PROVIDER,
            "reasoning_effort": "max",
        },
    )
    assert native._isolated_session_metadata(tmp_path, "wrong-session") is None


def test_native_command_pins_provider_and_keeps_session_metadata(tmp_path: Path) -> None:
    command = native._build_command(
        Path("/bin/codex"),
        repo_root=tmp_path,
        schema_path=tmp_path / "schema.json",
        output_path=tmp_path / "output.json",
    )
    assert 'model_provider="openai"' in command
    assert "--ephemeral" not in command


def test_native_receipt_requires_canonical_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(native, "_runtime_identity", lambda: _runtime_identity())
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    review = _review()
    receipt = _receipt(review)
    review["native_luna_receipt"] = receipt
    assert native.validate_native_luna_review(review, _authority(), lane="search_label")
