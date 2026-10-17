from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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
    source = root / "session-source" / "sessions/2026/10/17/rollout.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": receipt["session_id"],
                            "model_provider": receipt["session_provider"],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "turn_context",
                        "payload": {
                            "model": receipt["session_model"],
                            "effort": receipt["session_reasoning_effort"],
                            "collaboration_mode": {
                                "settings": {
                                    "model": receipt["session_model"],
                                    "reasoning_effort": receipt[
                                        "session_reasoning_effort"
                                    ],
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
    receipt.pop("rollout_sha256", None)
    receipt.pop("receipt_sha256", None)
    receipt["rollout_sha256"] = native._persist_rollout_evidence(
        root,
        str(receipt["run_id"]),
        stdout=json.dumps(
            {"type": "thread.started", "thread_id": receipt["session_id"]}
        )
        + "\n",
        stderr="",
        output="{}\n",
        codex_home=root / "session-source",
        session_evidence=(
            str(receipt["session_id"]),
            {
                "model": str(receipt["session_model"]),
                "provider": str(receipt["session_provider"]),
                "reasoning_effort": str(receipt["session_reasoning_effort"]),
            },
            source,
        ),
    )
    receipt["receipt_sha256"] = canonical_json_sha256_strict(receipt)
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


def _codex_executable(root: Path, name: str) -> Path:
    path = root / name
    path.write_text(f"#!/bin/sh\n# {name}\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def test_configured_executable_keeps_authority_stable_across_service_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _codex_executable(tmp_path, "wrapper")
    direct = _codex_executable(tmp_path, "direct")
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.delenv("CHRONOVISOR_CODEX_EXECUTABLE", raising=False)
    (tmp_path / "config.toml").write_text(
        f"[search_label_review]\ncodex_executable = {json.dumps(str(wrapper))}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(native, "_read_role", lambda _path: ({}, "a" * 64))
    monkeypatch.setattr(native, "_codex_version", lambda _path: "codex test")
    try:
        routes = []
        for executable in (wrapper, direct):
            native.clear_native_luna_runtime_identity_cache()
            monkeypatch.setattr(
                native.shutil, "which", lambda _name, path=executable: str(path)
            )
            routes.append(native.native_luna_route_identity())
            assert native._runtime_identity()["_codex_path"] == str(wrapper)
        assert routes[0] == routes[1]
        assert routes[0]["endpoint_sha256"] == hashlib.sha256(wrapper.read_bytes()).hexdigest()
    finally:
        native.clear_native_luna_runtime_identity_cache()


@pytest.mark.parametrize("value", ('""', '"codex"', "42", "[]"))
def test_invalid_executable_pin_does_not_fall_back_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    fallback = _codex_executable(tmp_path, "fallback")
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.delenv("CHRONOVISOR_CODEX_EXECUTABLE", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda _name: str(fallback))
    (tmp_path / "config.toml").write_text(
        f"[search_label_review]\ncodex_executable = {value}\n", encoding="utf-8"
    )
    with pytest.raises(native.NativeLunaIdentityError, match="config_executable_invalid"):
        native._resolve_codex()


@pytest.mark.parametrize("kind", ("missing", "directory", "not_executable", "symlink_loop"))
def test_unavailable_executable_pin_does_not_fall_back_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fallback = _codex_executable(tmp_path, "fallback")
    configured = tmp_path / "configured"
    if kind == "directory":
        configured.mkdir()
    elif kind == "not_executable":
        configured.write_text("not executable", encoding="utf-8")
        configured.chmod(0o600)
    elif kind == "symlink_loop":
        configured.symlink_to(configured)
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.delenv("CHRONOVISOR_CODEX_EXECUTABLE", raising=False)
    if kind == "symlink_loop":
        monkeypatch.setenv("CHRONOVISOR_CODEX_EXECUTABLE", str(configured))
    monkeypatch.setattr(native.shutil, "which", lambda _name: str(fallback))
    (tmp_path / "config.toml").write_text(
        f"[search_label_review]\ncodex_executable = {json.dumps(str(configured))}\n",
        encoding="utf-8",
    )
    with pytest.raises(native.NativeLunaIdentityError, match="codex_unavailable"):
        native._resolve_codex()


def test_executable_environment_cannot_override_canonical_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _codex_executable(tmp_path, "configured")
    other = _codex_executable(tmp_path, "other")
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.setenv("CHRONOVISOR_CODEX_EXECUTABLE", str(other))
    assert native._resolve_codex() == other
    (tmp_path / "config.toml").write_text(
        f"[search_label_review]\ncodex_executable = {json.dumps(str(configured))}\n",
        encoding="utf-8",
    )
    with pytest.raises(native.NativeLunaIdentityError, match="override_conflict"):
        native._resolve_codex()
    monkeypatch.setenv("CHRONOVISOR_CODEX_EXECUTABLE", str(configured))
    assert native._resolve_codex() == configured


def test_changed_executable_pin_invalidates_cached_path_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _codex_executable(tmp_path, "first")
    second = _codex_executable(tmp_path, "second")
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.delenv("CHRONOVISOR_CODEX_EXECUTABLE", raising=False)
    monkeypatch.setattr(native, "_read_role", lambda _path: ({}, "a" * 64))
    monkeypatch.setattr(native, "_codex_version", lambda _path: "codex test")
    config = tmp_path / "config.toml"
    try:
        native.clear_native_luna_runtime_identity_cache()
        routes = []
        for path in (first, second):
            config.write_text(
                f"[search_label_review]\ncodex_executable = {json.dumps(str(path))}\n",
                encoding="utf-8",
            )
            routes.append(native.native_luna_route_identity())
            assert native._runtime_identity()["_codex_path"] == str(path)
        assert routes[0] != routes[1]
        config.write_text("[search_label_review]\ncodex_executable = 42\n", encoding="utf-8")
        with pytest.raises(native.NativeLunaIdentityError, match="config_executable_invalid"):
            native.native_luna_route_identity()
    finally:
        native.clear_native_luna_runtime_identity_cache()


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


def test_native_rollout_history_change_and_deletion_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(native, "_runtime_identity", lambda: _runtime_identity())
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    review = _review()
    receipt = _receipt(review)
    _persist_receipt(tmp_path, receipt)
    review["native_luna_receipt"] = receipt
    assert native.validate_native_luna_review(review, _authority(), lane="search_label") is None

    session_file = next(
        (tmp_path / "runtime/native-luna-authority/runs" / receipt["run_id"])
        .glob("sessions/**/*.jsonl")
    )
    original = session_file.read_text(encoding="utf-8")
    assert native._validate_rollout_evidence(
        {**receipt, "session_id": "another-session"}
    ) == "native Luna rollout session does not match receipt"
    session_file.write_text(original + "tampered\n", encoding="utf-8")
    assert native.validate_native_luna_review(review, _authority(), lane="search_label")
    session_file.unlink()
    assert native.validate_native_luna_review(review, _authority(), lane="search_label")


def test_native_timeout_keeps_private_rollout_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.decision import frontier_review

    codex_home = tmp_path / "isolated-home"
    session = codex_home / "sessions/2026/10/17/timeout.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": "timeout-session", "model_provider": "openai"},
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

    @contextmanager
    def isolated_environment():
        yield {"CODEX_HOME": str(codex_home)}

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd="codex",
            timeout=1,
            output=json.dumps(
                {"type": "thread.started", "thread_id": "timeout-session"}
            )
            + "\n",
            stderr="timed out",
        )

    monkeypatch.setattr(frontier_review, "_isolated_codex_environment", isolated_environment)
    monkeypatch.setattr(native.subprocess, "run", timeout)
    record = native._execute_native_luna_process(
        codex=Path("/bin/codex"),
        prompt="private prompt",
        repo_root=tmp_path,
        schema={"type": "object"},
        timeout=1,
        root=tmp_path,
        run_id="native-search-" + "c" * 32,
        capacity=SimpleNamespace(pass_fds=()),
    )

    assert record["error"] == "timeout"
    run_dir = tmp_path / "runtime/native-luna-authority/runs" / record["run_id"]
    assert (run_dir / "stdout.jsonl").is_file()
    assert (run_dir / "stderr.log").read_text(encoding="utf-8") == "timed out"
    assert next(run_dir.glob("sessions/**/*.jsonl")).is_file()
    assert (run_dir / "manifest.json").is_file()


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
