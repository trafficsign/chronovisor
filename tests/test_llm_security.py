from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from urllib.request import Request

import keyring.errors
import pytest

from chronovisor.core.llm_security import (
    MAX_SECRET_FILE_BYTES,
    AuthenticatedTransport,
    AuthScheme,
    CredentialBackend,
    CredentialBinding,
    CredentialFailureCategory,
    CredentialFailureTelemetry,
    CredentialRef,
    CredentialResolver,
    CredentialSecurityError,
    OSKeyringCredentialStore,
    SecretValue,
    canonical_endpoint,
)

CANARY = "sk-CANARY-DO-NOT-LEAK"


class MemoryKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.set_calls: list[tuple[str, str]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.failure: Exception | None = None

    def get_password(self, service: str, username: str) -> str | None:
        self.get_calls.append((service, username))
        if self.failure is not None:
            raise self.failure
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.set_calls.append((service, username))
        if self.failure is not None:
            raise self.failure
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.delete_calls.append((service, username))
        if self.failure is not None:
            raise self.failure
        try:
            del self.values[(service, username)]
        except KeyError:
            raise keyring.errors.PasswordDeleteError("missing") from None


def _keyring_backend(
    module: str = "keyring.backends.macOS", class_name: str = "Keyring"
) -> MemoryKeyring:
    backend_type = type(class_name, (MemoryKeyring,), {"__module__": module})
    return backend_type()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "env",
        " env:API_KEY",
        "env:API KEY",
        "env:1API_KEY",
        "mounted-file:relative/key",
        "oskeyring:missing-account",
        "unknown:key",
    ],
)
def test_credential_ref_parse_is_strict(value: str) -> None:
    with pytest.raises(CredentialSecurityError) as exc:
        CredentialRef.parse(value)
    assert exc.value.category is CredentialFailureCategory.INVALID_REF

    assert CredentialRef.parse("env:API_KEY") == CredentialRef(
        CredentialBackend.ENV, "API_KEY"
    )
    assert str(CredentialRef.parse("oskeyring:openai/default")) == (
        "oskeyring:openai/default"
    )


def test_secret_is_opaque_redacted_and_not_json_serializable() -> None:
    secret = SecretValue(CANARY)

    assert repr(secret) == str(secret) == "<redacted>"
    assert CANARY not in repr(secret)
    with pytest.raises(TypeError) as exc:
        json.dumps(secret)
    assert CANARY not in str(exc.value)
    with pytest.raises(TypeError) as pickle_exc:
        pickle.dumps(secret)
    assert CANARY not in str(pickle_exc.value)
    assert not any(
        name in {"reveal", "get", "value", "to_json"} for name in dir(secret)
    )


def test_env_resolution_is_exact_and_removes_child_inheritance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "CHRONOVISOR_LLM_SECURITY_CANARY"
    monkeypatch.setenv(variable, CANARY)
    resolver = CredentialResolver()

    secret = resolver.resolve(CredentialRef.parse(f"env:{variable}"))

    assert repr(secret) == "<redacted>"
    assert variable not in os.environ
    child = subprocess.run(
        [sys.executable, "-c", f"import os; print(os.getenv({variable!r}))"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.stdout.strip() == "None"
    with pytest.raises(CredentialSecurityError) as exc:
        resolver.resolve(CredentialRef.parse(f"env:{variable}"))
    assert exc.value.category is CredentialFailureCategory.MISSING


def test_resolver_has_no_backend_fallback_and_emits_safe_categories(
    tmp_path: Path,
) -> None:
    events: list[CredentialFailureTelemetry] = []
    resolver = CredentialResolver(
        environ={"FALLBACK": CANARY},
        repo_root=tmp_path / "repo",
        home_root=tmp_path / "home",
        keyring_backend=MemoryKeyring(),
        telemetry=events.append,
    )

    with pytest.raises(CredentialSecurityError) as missing:
        resolver.resolve(CredentialRef.parse(f"mounted-file:{tmp_path / 'missing'}"))
    with pytest.raises(CredentialSecurityError) as rejected:
        resolver.resolve(CredentialRef.parse("oskeyring:openai/default"))

    assert missing.value.category is CredentialFailureCategory.MISSING
    assert rejected.value.category is CredentialFailureCategory.BACKEND_REJECTED
    assert events == [
        CredentialFailureTelemetry("credential_missing"),
        CredentialFailureTelemetry("backend_rejected"),
    ]
    assert CANARY not in repr(events)
    assert CANARY not in repr(missing.value)


@pytest.mark.parametrize(
    "module,class_name",
    [
        ("keyring.backends.macOS", "Keyring"),
        ("keyring.backends.Windows", "WinVaultKeyring"),
        ("keyring.backends.SecretService", "Keyring"),
        ("keyring.backends.kwallet", "DBusKeyring"),
        ("keyring.backends.kwallet", "DBusKeyringKWallet4"),
    ],
)
def test_os_keyring_resolver_accepts_only_known_native_backend_identities(
    module: str,
    class_name: str,
) -> None:
    backend = _keyring_backend(module, class_name)
    backend.values[("chronovisor/openai", "default")] = CANARY
    resolver = CredentialResolver(keyring_backend=backend)

    secret = resolver.resolve(CredentialRef.parse("oskeyring:openai/default"))

    assert repr(secret) == "<redacted>"
    assert backend.get_calls == [("chronovisor/openai", "default")]


@pytest.mark.parametrize(
    "module,class_name",
    [
        ("keyring.backends.null", "Keyring"),
        ("keyring.backends.chainer", "ChainerBackend"),
        ("keyrings.alt.file", "PlaintextKeyring"),
        ("project.keyring", "CustomKeyring"),
    ],
)
def test_os_keyring_resolver_rejects_null_chainer_alt_and_unknown_backends(
    module: str,
    class_name: str,
) -> None:
    resolver = CredentialResolver(keyring_backend=_keyring_backend(module, class_name))

    with pytest.raises(CredentialSecurityError) as exc:
        resolver.resolve(CredentialRef.parse("oskeyring:openai/default"))

    assert exc.value.category is CredentialFailureCategory.BACKEND_REJECTED
    assert CANARY not in repr(exc.value)


def test_os_keyring_store_namespaces_status_and_delete_without_preread() -> None:
    backend = _keyring_backend()
    store = OSKeyringCredentialStore(backend=backend)
    ref = CredentialRef.parse("oskeyring:openai/default")

    assert store.status(ref).category == "missing"
    assert store.set(ref, CANARY).category == "present"
    status = store.status(ref)
    assert status.present and status.category == "present"
    assert store.delete(ref).category == "deleted"
    assert store.delete(ref).category == "missing"

    assert backend.set_calls == [("chronovisor/openai", "default")]
    assert backend.delete_calls == [
        ("chronovisor/openai", "default"),
        ("chronovisor/openai", "default"),
    ]
    assert backend.get_calls == [
        ("chronovisor/openai", "default"),
        ("chronovisor/openai", "default"),
    ]
    assert CANARY not in repr(status)


@pytest.mark.parametrize(
    "failure,category",
    [
        (
            keyring.errors.KeyringLocked("locked"),
            CredentialFailureCategory.STORE_LOCKED,
        ),
        (OSError(CANARY), CredentialFailureCategory.STORE_UNAVAILABLE),
    ],
)
def test_os_keyring_failures_are_safe(
    failure: Exception,
    category: CredentialFailureCategory,
) -> None:
    backend = _keyring_backend()
    backend.failure = failure
    resolver = CredentialResolver(keyring_backend=backend)

    with pytest.raises(CredentialSecurityError) as exc:
        resolver.resolve(CredentialRef.parse("oskeyring:openai/default"))

    assert exc.value.category is category
    assert exc.value.__cause__ is None
    assert CANARY not in repr(exc.value)


def _mounted_resolver(tmp_path: Path) -> CredentialResolver:
    return CredentialResolver(
        environ={},
        repo_root=tmp_path / "repo",
        home_root=tmp_path / "home",
    )


def _write_secret(path: Path, payload: str = CANARY, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    path.chmod(mode)
    return path


def test_mounted_file_accepts_only_bounded_external_owner_only_regular_file(
    tmp_path: Path,
) -> None:
    resolver = _mounted_resolver(tmp_path)
    mounted = _write_secret(tmp_path / "external" / "api-key", f"{CANARY}\n")

    assert (
        repr(resolver.resolve(CredentialRef.parse(f"mounted-file:{mounted}")))
        == "<redacted>"
    )

    unsafe_mode = _write_secret(tmp_path / "external" / "unsafe-mode", mode=0o640)
    oversized = _write_secret(
        tmp_path / "external" / "oversized", "x" * (MAX_SECRET_FILE_BYTES + 1)
    )
    directory = tmp_path / "external" / "directory"
    directory.mkdir()
    symlink = tmp_path / "external" / "symlink"
    symlink.symlink_to(mounted)
    stored_in_repo = _write_secret(tmp_path / "repo" / "api-key")
    stored_in_home = _write_secret(tmp_path / "home" / "api-key")

    for path in (
        unsafe_mode,
        oversized,
        directory,
        symlink,
        stored_in_repo,
        stored_in_home,
    ):
        with pytest.raises(CredentialSecurityError) as exc:
            resolver.resolve(CredentialRef.parse(f"mounted-file:{path}"))
        assert exc.value.category is CredentialFailureCategory.MOUNT_REJECTED
        assert CANARY not in repr(exc.value)


@pytest.mark.parametrize(
    "endpoint, cloud_secret",
    [
        ("http://api.example.com/v1", False),
        ("http://127.0.0.1:8080/v1", True),
        (f"https://user:{CANARY}@api.example.com/v1", True),
        (f"https://api.example.com/v1?api_key={CANARY}", True),
        ("https://api.example.com/v1#fragment", True),
    ],
)
def test_endpoint_policy_rejects_insecure_or_credentialed_urls(
    endpoint: str, cloud_secret: bool
) -> None:
    with pytest.raises(CredentialSecurityError) as exc:
        canonical_endpoint(endpoint, cloud_secret=cloud_secret)
    assert exc.value.category is CredentialFailureCategory.ENDPOINT_REJECTED
    assert CANARY not in str(exc.value)


def test_endpoint_policy_canonicalizes_origin_and_limits_plain_http() -> None:
    remote = canonical_endpoint(
        "HTTPS://API.Example.COM:443/v1?api-version=1", cloud_secret=True
    )
    local = canonical_endpoint("http://[::1]:80/v1", cloud_secret=False)

    assert remote.origin == "https://api.example.com"
    assert remote.url == "https://api.example.com/v1?api-version=1"
    assert not remote.is_loopback
    assert local.origin == "http://[::1]"
    assert local.is_loopback


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://localhost/v1",
        "https://api.localhost/v1",
        "https://0.0.0.0/v1",
        "https://127.0.0.1/v1",
        "https://10.0.0.1/v1",
        "https://169.254.169.254/v1",
        "https://224.0.0.1/v1",
        "https://240.0.0.1/v1",
        "https://[::]/v1",
        "https://[::1]/v1",
        "https://[fc00::1]/v1",
        "https://[fe80::1]/v1",
        "https://[ff00::1]/v1",
    ],
)
def test_cloud_endpoint_policy_rejects_local_and_special_ip_literals(
    endpoint: str,
) -> None:
    with pytest.raises(CredentialSecurityError) as exc:
        canonical_endpoint(endpoint, cloud_secret=True)

    assert exc.value.category is CredentialFailureCategory.ENDPOINT_REJECTED


def test_local_loopback_endpoint_remains_available_without_cloud_secret() -> None:
    endpoint = canonical_endpoint("http://127.0.0.1:11434/api", cloud_secret=False)

    assert endpoint.origin == "http://127.0.0.1:11434"


class RecordingSender:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.calls: list[tuple[Request, bool, float]] = []
        self.failure = failure

    def __call__(
        self, request: Request, *, follow_redirects: bool, timeout_seconds: float
    ) -> object:
        self.calls.append((request, follow_redirects, timeout_seconds))
        if self.failure is not None:
            raise self.failure
        return "ok"


def _transport(
    sender: RecordingSender,
    *,
    profile_id: str = "openai-primary",
    binding_profile_id: str = "openai-primary",
    endpoint: str = "https://api.example.com/v1",
    binding_endpoint: str = "https://api.example.com",
    binding_scheme: AuthScheme = AuthScheme.BEARER,
    requested_scheme: AuthScheme = AuthScheme.BEARER,
    events: list[CredentialFailureTelemetry] | None = None,
) -> AuthenticatedTransport:
    return AuthenticatedTransport(
        profile_id=profile_id,
        endpoint=endpoint,
        secret=SecretValue(CANARY),
        binding=CredentialBinding.bind(
            binding_profile_id, binding_endpoint, binding_scheme
        ),
        auth_scheme=requested_scheme,
        sender=sender,
        telemetry=None if events is None else events.append,
    )


def test_transport_injects_auth_only_at_wire_and_never_follows_redirects() -> None:
    sender = RecordingSender()
    transport = _transport(sender)

    assert transport.send("https://API.example.com:443/v1/chat", data=b"{}") == "ok"

    request, follow_redirects, timeout_seconds = sender.calls[0]
    assert request.get_header("Authorization") == f"Bearer {CANARY}"
    assert follow_redirects is False
    assert timeout_seconds == 60.0
    assert CANARY not in repr(request)
    with pytest.raises(TypeError):
        json.dumps(request)


@pytest.mark.parametrize(
    "profile_id, binding_profile_id, binding_endpoint, binding_scheme, requested_scheme",
    [
        (
            "openai-primary",
            "other-profile",
            "https://api.example.com",
            AuthScheme.BEARER,
            AuthScheme.BEARER,
        ),
        (
            "openai-primary",
            "openai-primary",
            "https://other.example.com",
            AuthScheme.BEARER,
            AuthScheme.BEARER,
        ),
        (
            "openai-primary",
            "openai-primary",
            "https://api.example.com",
            AuthScheme.X_API_KEY,
            AuthScheme.BEARER,
        ),
    ],
)
def test_transport_rejects_binding_mismatch_before_sender_call(
    profile_id: str,
    binding_profile_id: str,
    binding_endpoint: str,
    binding_scheme: AuthScheme,
    requested_scheme: AuthScheme,
) -> None:
    sender = RecordingSender()

    with pytest.raises(CredentialSecurityError) as exc:
        _transport(
            sender,
            profile_id=profile_id,
            binding_profile_id=binding_profile_id,
            binding_endpoint=binding_endpoint,
            binding_scheme=binding_scheme,
            requested_scheme=requested_scheme,
        )

    assert exc.value.category is CredentialFailureCategory.ORIGIN_MISMATCH
    assert sender.calls == []


@pytest.mark.parametrize(
    "profile_id",
    ["", "OpenAI", " openai", "openai/default", "openai primary", "1openai"],
)
def test_credential_binding_rejects_noncanonical_profile_id(
    profile_id: str,
) -> None:
    with pytest.raises(CredentialSecurityError) as exc:
        CredentialBinding.bind(profile_id, "https://api.example.com", AuthScheme.BEARER)

    assert exc.value.category is CredentialFailureCategory.ORIGIN_MISMATCH


@pytest.mark.parametrize(
    "header",
    [
        "Host",
        "Content-Length",
        "Transfer-Encoding",
        "Connection",
        "Proxy-Connection",
        "Upgrade",
        "TE",
        "Trailer",
        "Authorization",
        "Proxy-Authorization",
        "X-API-Key",
    ],
)
def test_transport_rejects_caller_authority_framing_and_auth_headers(
    header: str,
) -> None:
    sender = RecordingSender()
    transport = _transport(sender)

    with pytest.raises(CredentialSecurityError) as exc:
        transport.send(
            "https://api.example.com/v1/chat", headers={header.swapcase(): CANARY}
        )

    assert exc.value.category is CredentialFailureCategory.ENDPOINT_REJECTED
    assert sender.calls == []


def test_transport_rejects_cross_origin_and_caller_auth_before_sender_call() -> None:
    sender = RecordingSender()
    transport = _transport(sender)

    with pytest.raises(CredentialSecurityError) as origin:
        transport.send("https://other.example.com/v1/chat")
    with pytest.raises(CredentialSecurityError) as header:
        transport.send(
            "https://api.example.com/v1/chat", headers={"authorization": CANARY}
        )

    assert origin.value.category is CredentialFailureCategory.ORIGIN_MISMATCH
    assert header.value.category is CredentialFailureCategory.ENDPOINT_REJECTED
    assert sender.calls == []


def test_transport_rejects_invalid_timeout_before_sender_call() -> None:
    sender = RecordingSender()
    transport = _transport(sender)

    with pytest.raises(CredentialSecurityError) as exc:
        transport.send("https://api.example.com/v1/chat", timeout_ms=0)

    assert exc.value.category is CredentialFailureCategory.ENDPOINT_REJECTED
    assert sender.calls == []


def test_transport_preserves_safe_credential_category_from_sender() -> None:
    sender = RecordingSender(
        failure=CredentialSecurityError(CredentialFailureCategory.ORIGIN_MISMATCH)
    )
    transport = _transport(sender)

    with pytest.raises(CredentialSecurityError) as exc:
        transport.send("https://api.example.com/v1/chat")

    assert exc.value.category is CredentialFailureCategory.ORIGIN_MISMATCH
    assert exc.value.__context__ is None


def test_transport_sanitizes_sender_errors_and_telemetry() -> None:
    events: list[CredentialFailureTelemetry] = []
    sender = RecordingSender(failure=OSError(CANARY))
    transport = _transport(sender, events=events)

    with pytest.raises(CredentialSecurityError) as exc:
        transport.send("https://api.example.com/v1/chat")

    assert exc.value.category is CredentialFailureCategory.TRANSPORT_ERROR
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert events == [CredentialFailureTelemetry("transport_error")]
    assert CANARY not in str(exc.value)
    assert CANARY not in repr(exc.value)
    assert CANARY not in repr(events)
