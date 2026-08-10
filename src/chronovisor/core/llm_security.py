"""Fail-closed credential and authenticated transport boundary for LLM APIs."""

from __future__ import annotations

import ipaddress
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Never, Protocol, SupportsIndex, cast
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from urllib.request import Request

import keyring
import keyring.errors

MAX_SECRET_FILE_BYTES = 16_384
DEFAULT_REQUEST_TIMEOUT_MS = 60_000
MAX_REQUEST_TIMEOUT_MS = 900_000
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_KEYRING_TARGET = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z"
)
_PROFILE_ID = re.compile(r"[a-z][a-z0-9._-]{0,63}\Z")
_HTTP_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_CREDENTIAL_QUERY_NAMES = frozenset(
    {
        "apikey",
        "authorization",
        "auth",
        "accesstoken",
        "clientsecret",
        "credential",
        "key",
        "password",
        "secret",
        "token",
    }
)
_CALLER_FORBIDDEN_HEADERS = frozenset(
    {
        "api-key",
        "authorization",
        "connection",
        "content-length",
        "host",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-api-key",
        "x-goog-api-key",
    }
)
_SECURE_KEYRING_BACKENDS = frozenset(
    {
        ("keyring.backends.macOS", "Keyring"),
        ("keyring.backends.Windows", "WinVaultKeyring"),
        ("keyring.backends.SecretService", "Keyring"),
        ("keyring.backends.kwallet", "DBusKeyring"),
        ("keyring.backends.kwallet", "DBusKeyringKWallet4"),
    }
)
_CHILD_ENV_NAMES = frozenset(
    {
        "COMSPEC",
        "CURL_CA_BUNDLE",
        "DBUS_SESSION_BUS_ADDRESS",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ADDRESS",
        "LC_ALL",
        "LC_COLLATE",
        "LC_CTYPE",
        "LC_IDENTIFICATION",
        "LC_MEASUREMENT",
        "LC_MESSAGES",
        "LC_MONETARY",
        "LC_NAME",
        "LC_NUMERIC",
        "LC_PAPER",
        "LC_TELEPHONE",
        "LC_TIME",
        "PATH",
        "PATHEXT",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "TZDIR",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "XDG_RUNTIME_DIR",
    }
)


def build_child_process_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy only non-secret runtime settings into an isolated child."""

    source = os.environ if environ is None else environ
    return {name: value for name, value in source.items() if name in _CHILD_ENV_NAMES}


class CredentialFailureCategory(StrEnum):
    INVALID_REF = "credential_ref_invalid"
    MISSING = "credential_missing"
    BACKEND_REJECTED = "backend_rejected"
    STORE_LOCKED = "store_locked"
    STORE_UNAVAILABLE = "store_unavailable"
    MOUNT_REJECTED = "mounted_file_rejected"
    ENDPOINT_REJECTED = "endpoint_rejected"
    ORIGIN_MISMATCH = "origin_mismatch"
    TRANSPORT_ERROR = "transport_error"


class CredentialSecurityError(RuntimeError):
    """A safe credential failure whose text never includes credential material."""

    category: CredentialFailureCategory

    def __init__(self, category: CredentialFailureCategory) -> None:
        self.category = category
        super().__init__(category.value)


@dataclass(frozen=True)
class CredentialFailureTelemetry:
    category: str


FailureSink = Callable[[CredentialFailureTelemetry], None]


def _error(
    category: CredentialFailureCategory,
    telemetry: FailureSink | None = None,
) -> CredentialSecurityError:
    if telemetry is not None:
        try:
            telemetry(CredentialFailureTelemetry(category.value))
        except Exception:
            pass
    return CredentialSecurityError(category)


class CredentialBackend(StrEnum):
    ENV = "env"
    MOUNTED_FILE = "mounted-file"
    OS_KEYRING = "oskeyring"


@dataclass(frozen=True)
class CredentialRef:
    backend: CredentialBackend
    target: str

    def __post_init__(self) -> None:
        if not isinstance(self.backend, CredentialBackend) or not isinstance(
            self.target, str
        ):
            raise _error(CredentialFailureCategory.INVALID_REF)
        if self.backend is CredentialBackend.ENV:
            valid = _ENV_NAME.fullmatch(self.target) is not None
        elif self.backend is CredentialBackend.MOUNTED_FILE:
            valid = (
                bool(self.target)
                and "\x00" not in self.target
                and not any(ord(character) < 32 for character in self.target)
                and Path(self.target).is_absolute()
            )
        else:
            valid = _KEYRING_TARGET.fullmatch(self.target) is not None
        if not valid:
            raise _error(CredentialFailureCategory.INVALID_REF)

    @classmethod
    def parse(cls, value: str) -> CredentialRef:
        if not isinstance(value, str) or value != value.strip():
            raise _error(CredentialFailureCategory.INVALID_REF)
        backend_text, separator, target = value.partition(":")
        if not separator:
            raise _error(CredentialFailureCategory.INVALID_REF)
        try:
            backend = CredentialBackend(backend_text)
        except ValueError:
            raise _error(CredentialFailureCategory.INVALID_REF) from None
        return cls(backend, target)

    def __str__(self) -> str:
        return f"{self.backend.value}:{self.target}"


class SecretValue:
    """Opaque process-local secret; Python memory zeroization is not promised."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        if (
            not isinstance(value, str)
            or not value
            or any(character in value for character in ("\x00", "\r", "\n"))
        ):
            raise _error(CredentialFailureCategory.MISSING)
        self.__value = value

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__

    def __reduce_ex__(self, _protocol: SupportsIndex) -> Never:
        raise TypeError("SecretValue is not serializable")


class KeyringBackend(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...

    def set_password(self, service: str, username: str, password: str) -> None: ...

    def delete_password(self, service: str, username: str) -> None: ...


def _secure_keyring_backend(
    backend: KeyringBackend | None,
    telemetry: FailureSink | None,
) -> KeyringBackend:
    if backend is None:
        try:
            loaded = keyring.get_keyring()
        except Exception:
            loaded = None
        if loaded is None:
            raise _error(CredentialFailureCategory.STORE_UNAVAILABLE, telemetry)
        backend = cast(KeyringBackend, loaded)
    identity = (type(backend).__module__, type(backend).__qualname__)
    if identity not in _SECURE_KEYRING_BACKENDS:
        raise _error(CredentialFailureCategory.BACKEND_REJECTED, telemetry)
    return backend


def _keyring_target(ref: CredentialRef) -> tuple[str, str]:
    if (
        not isinstance(ref, CredentialRef)
        or ref.backend is not CredentialBackend.OS_KEYRING
    ):
        raise _error(CredentialFailureCategory.INVALID_REF)
    profile, account = ref.target.split("/", 1)
    return f"chronovisor/{profile}", account


def _keyring_read(
    backend: KeyringBackend,
    service: str,
    account: str,
    telemetry: FailureSink | None,
) -> str | None:
    try:
        value = backend.get_password(service, account)
    except keyring.errors.KeyringLocked:
        locked = True
    except Exception:
        locked = False
    else:
        if value is None or isinstance(value, str):
            return value
        locked = False
    raise _error(
        CredentialFailureCategory.STORE_LOCKED
        if locked
        else CredentialFailureCategory.STORE_UNAVAILABLE,
        telemetry,
    )


@dataclass(frozen=True)
class CredentialStoreStatus:
    present: bool
    category: str


class OSKeyringCredentialStore:
    """Value-free CLI boundary for an allowlisted OS-native credential store."""

    def __init__(
        self,
        *,
        backend: KeyringBackend | None = None,
        telemetry: FailureSink | None = None,
    ) -> None:
        self._backend = _secure_keyring_backend(backend, telemetry)
        self._telemetry = telemetry

    def set(self, ref: CredentialRef, secret: str) -> CredentialStoreStatus:
        service, account = _keyring_target(ref)
        SecretValue(secret)
        try:
            self._backend.set_password(service, account, secret)
        except keyring.errors.KeyringLocked:
            locked = True
        except Exception:
            locked = False
        else:
            return CredentialStoreStatus(True, "present")
        raise _error(
            CredentialFailureCategory.STORE_LOCKED
            if locked
            else CredentialFailureCategory.STORE_UNAVAILABLE,
            self._telemetry,
        )

    def status(self, ref: CredentialRef) -> CredentialStoreStatus:
        service, account = _keyring_target(ref)
        value = _keyring_read(self._backend, service, account, self._telemetry)
        present = value is not None
        del value
        return CredentialStoreStatus(present, "present" if present else "missing")

    def delete(self, ref: CredentialRef) -> CredentialStoreStatus:
        service, account = _keyring_target(ref)
        try:
            self._backend.delete_password(service, account)
        except keyring.errors.PasswordDeleteError:
            return CredentialStoreStatus(False, "missing")
        except keyring.errors.KeyringLocked:
            locked = True
        except Exception:
            locked = False
        else:
            return CredentialStoreStatus(False, "deleted")
        raise _error(
            CredentialFailureCategory.STORE_LOCKED
            if locked
            else CredentialFailureCategory.STORE_UNAVAILABLE,
            self._telemetry,
        )


def _default_repo_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    return None


def _within(path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class CredentialResolver:
    """Resolve exactly the backend named by a credential reference."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        repo_root: Path | None = None,
        home_root: Path | None = None,
        max_file_bytes: int = MAX_SECRET_FILE_BYTES,
        keyring_backend: KeyringBackend | None = None,
        telemetry: FailureSink | None = None,
    ) -> None:
        if (
            isinstance(max_file_bytes, bool)
            or not isinstance(max_file_bytes, int)
            or max_file_bytes <= 0
        ):
            raise ValueError("max_file_bytes must be a positive integer")
        self._environ = os.environ if environ is None else environ
        self._consume_process_env = environ is None or environ is os.environ
        self._repo_root = (
            repo_root.resolve() if repo_root is not None else _default_repo_root()
        )
        self._home_root = (
            home_root.resolve() if home_root is not None else Path.home().resolve()
        )
        self._max_file_bytes = max_file_bytes
        self._keyring_backend = keyring_backend
        self._telemetry = telemetry

    def resolve(self, ref: CredentialRef) -> SecretValue:
        if not isinstance(ref, CredentialRef):
            raise _error(CredentialFailureCategory.INVALID_REF, self._telemetry)
        if ref.backend is CredentialBackend.OS_KEYRING:
            service, account = _keyring_target(ref)
            backend = _secure_keyring_backend(self._keyring_backend, self._telemetry)
            value = _keyring_read(backend, service, account, self._telemetry)
            if value is None:
                raise _error(CredentialFailureCategory.MISSING, self._telemetry)
            try:
                return SecretValue(value)
            except CredentialSecurityError:
                pass
            raise _error(CredentialFailureCategory.MISSING, self._telemetry)
        if ref.backend is CredentialBackend.ENV:
            value = (
                os.environ.pop(ref.target, None)
                if self._consume_process_env
                else self._environ.get(ref.target)
            )
            if value is None:
                raise _error(CredentialFailureCategory.MISSING, self._telemetry)
            try:
                return SecretValue(value)
            except CredentialSecurityError:
                pass
            raise _error(CredentialFailureCategory.MISSING, self._telemetry)
        return self._resolve_mounted_file(Path(ref.target))

    def _resolve_mounted_file(self, path: Path) -> SecretValue:
        try:
            canonical = path.resolve(strict=False)
            if any(
                _within(candidate, root)
                for candidate in (path, canonical)
                for root in (self._repo_root, self._home_root)
            ):
                raise _error(CredentialFailureCategory.MOUNT_REJECTED, self._telemetry)
            path_stat = path.lstat()
            if not stat.S_ISREG(path_stat.st_mode):
                raise _error(CredentialFailureCategory.MOUNT_REJECTED, self._telemetry)
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
            try:
                opened_stat = os.fstat(descriptor)
                getuid = getattr(os, "getuid", None)
                valid = (
                    stat.S_ISREG(opened_stat.st_mode)
                    and getuid is not None
                    and opened_stat.st_uid == getuid()
                    and stat.S_IMODE(opened_stat.st_mode) & 0o077 == 0
                    and opened_stat.st_nlink == 1
                    and opened_stat.st_size <= self._max_file_bytes
                    and (opened_stat.st_dev, opened_stat.st_ino)
                    == (path_stat.st_dev, path_stat.st_ino)
                )
                if not valid:
                    raise _error(
                        CredentialFailureCategory.MOUNT_REJECTED, self._telemetry
                    )
                chunks: list[bytes] = []
                remaining = self._max_file_bytes + 1
                while remaining:
                    chunk = os.read(descriptor, remaining)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
            finally:
                os.close(descriptor)
            payload = b"".join(chunks)
            if len(payload) > self._max_file_bytes:
                raise _error(CredentialFailureCategory.MOUNT_REJECTED, self._telemetry)
            text = payload.rstrip(b"\r\n").decode("utf-8")
            if not text:
                raise _error(CredentialFailureCategory.MISSING, self._telemetry)
            return SecretValue(text)
        except CredentialSecurityError:
            raise
        except UnicodeError:
            pass
        except OSError:
            raise _error(CredentialFailureCategory.MISSING, self._telemetry) from None
        raise _error(CredentialFailureCategory.MOUNT_REJECTED, self._telemetry)


def _validated_headers(
    headers: Mapping[str, str] | None,
    telemetry: FailureSink | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for name, value in (headers or {}).items():
            if (
                not isinstance(name, str)
                or not isinstance(value, str)
                or _HTTP_TOKEN.fullmatch(name) is None
                or name.lower() in _CALLER_FORBIDDEN_HEADERS
                or any(
                    ord(character) < 32 or ord(character) == 127 for character in value
                )
            ):
                raise _error(CredentialFailureCategory.ENDPOINT_REJECTED, telemetry)
            result[name] = value
    except CredentialSecurityError:
        raise
    except Exception:
        pass
    else:
        return result
    raise _error(CredentialFailureCategory.ENDPOINT_REJECTED, telemetry)


@dataclass(frozen=True)
class CanonicalEndpoint:
    url: str
    origin: str
    is_loopback: bool


def _canonical_host(host: str) -> tuple[str, bool]:
    host = host.rstrip(".")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is None:
        try:
            canonical = host.encode("idna").decode("ascii").lower()
        except UnicodeError:
            raise _error(CredentialFailureCategory.ENDPOINT_REJECTED) from None
        labels = canonical.split(".")
        if not canonical or any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or not re.fullmatch(r"[a-z0-9-]+", label)
            for label in labels
        ):
            raise _error(CredentialFailureCategory.ENDPOINT_REJECTED)
        return canonical, canonical == "localhost"
    canonical = address.compressed.lower()
    return canonical, address.is_loopback


def _forbidden_cloud_host(host: str) -> bool:
    """Reject local and special-use literal targets for credentialed endpoints."""

    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(
        (
            address.is_unspecified,
            address.is_loopback,
            address.is_private,
            address.is_link_local,
            address.is_multicast,
            address.is_reserved,
        )
    )


def _has_credential_query(query: str) -> bool:
    for key, _value in parse_qsl(query, keep_blank_values=True):
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        if (
            normalized in _CREDENTIAL_QUERY_NAMES
            or "credential" in normalized
            or normalized.endswith(("apikey", "password", "secret", "token"))
        ):
            return True
    return False


def canonical_endpoint(value: str, *, cloud_secret: bool) -> CanonicalEndpoint:
    """Validate and canonicalize an endpoint without weakening TLS policy."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise _error(CredentialFailureCategory.ENDPOINT_REJECTED)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise _error(CredentialFailureCategory.ENDPOINT_REJECTED) from None
    scheme = parsed.scheme.lower()
    if (
        scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or _has_credential_query(parsed.query)
        or port == 0
    ):
        raise _error(CredentialFailureCategory.ENDPOINT_REJECTED)
    host, is_loopback = _canonical_host(parsed.hostname)
    if cloud_secret and _forbidden_cloud_host(host):
        raise _error(CredentialFailureCategory.ENDPOINT_REJECTED)
    if scheme == "http" and (not is_loopback or cloud_secret):
        raise _error(CredentialFailureCategory.ENDPOINT_REJECTED)
    default_port = 443 if scheme == "https" else 80
    authority_host = f"[{host}]" if ":" in host else host
    authority = (
        authority_host if port in {None, default_port} else f"{authority_host}:{port}"
    )
    origin = f"{scheme}://{authority}"
    return CanonicalEndpoint(
        url=urlunsplit((scheme, authority, parsed.path, parsed.query, "")),
        origin=origin,
        is_loopback=is_loopback,
    )


class AuthScheme(StrEnum):
    BEARER = "bearer"
    X_API_KEY = "x-api-key"

    @classmethod
    def parse(cls, value: AuthScheme | str) -> AuthScheme:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or value != value.strip():
            raise _error(CredentialFailureCategory.ORIGIN_MISMATCH)
        try:
            return cls(value.lower())
        except ValueError:
            raise _error(CredentialFailureCategory.ORIGIN_MISMATCH) from None


@dataclass(frozen=True)
class CredentialBinding:
    profile_id: str
    origin: str
    auth_scheme: AuthScheme

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profile_id, str)
            or _PROFILE_ID.fullmatch(self.profile_id) is None
            or not isinstance(self.auth_scheme, AuthScheme)
        ):
            raise _error(CredentialFailureCategory.ORIGIN_MISMATCH)
        canonical = canonical_endpoint(self.origin, cloud_secret=True)
        if canonical.origin != self.origin:
            raise _error(CredentialFailureCategory.ORIGIN_MISMATCH)

    @classmethod
    def bind(
        cls,
        profile_id: str,
        endpoint: str,
        auth_scheme: AuthScheme | str,
    ) -> CredentialBinding:
        canonical = canonical_endpoint(endpoint, cloud_secret=True)
        return cls(profile_id, canonical.origin, AuthScheme.parse(auth_scheme))


class RequestSender(Protocol):
    def __call__(
        self,
        request: Request,
        *,
        follow_redirects: Literal[False],
        timeout_seconds: float,
    ) -> object: ...


class AuthenticatedTransport:
    """The sole boundary that turns a SecretValue into an outbound auth header."""

    def __init__(
        self,
        *,
        profile_id: str,
        endpoint: str,
        secret: SecretValue,
        binding: CredentialBinding,
        auth_scheme: AuthScheme | str,
        sender: RequestSender,
        telemetry: FailureSink | None = None,
    ) -> None:
        if not isinstance(secret, SecretValue) or not isinstance(
            binding, CredentialBinding
        ):
            raise _error(CredentialFailureCategory.ORIGIN_MISMATCH, telemetry)
        canonical = canonical_endpoint(endpoint, cloud_secret=True)
        bound = canonical_endpoint(binding.origin, cloud_secret=True)
        requested_scheme = AuthScheme.parse(auth_scheme)
        if (
            profile_id != binding.profile_id
            or _PROFILE_ID.fullmatch(profile_id) is None
            or canonical.origin != bound.origin
            or requested_scheme is not binding.auth_scheme
        ):
            raise _error(CredentialFailureCategory.ORIGIN_MISMATCH, telemetry)
        self._endpoint = canonical
        self._secret = secret
        self._auth_scheme = requested_scheme
        self._sender = sender
        self._telemetry = telemetry

    def send(
        self,
        url: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        method: str = "POST",
        timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    ) -> object:
        target = canonical_endpoint(url, cloud_secret=True)
        if target.origin != self._endpoint.origin:
            raise _error(CredentialFailureCategory.ORIGIN_MISMATCH, self._telemetry)
        if not isinstance(method, str) or _HTTP_TOKEN.fullmatch(method) is None:
            raise _error(CredentialFailureCategory.ENDPOINT_REJECTED, self._telemetry)
        if data is not None and not isinstance(data, bytes):
            raise _error(CredentialFailureCategory.ENDPOINT_REJECTED, self._telemetry)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 0 < timeout_ms <= MAX_REQUEST_TIMEOUT_MS
        ):
            raise _error(CredentialFailureCategory.ENDPOINT_REJECTED, self._telemetry)
        request_headers = _validated_headers(headers, self._telemetry)
        raw_secret = cast(
            str,
            object.__getattribute__(self._secret, "_SecretValue__value"),
        )
        if self._auth_scheme is AuthScheme.BEARER:
            request_headers["Authorization"] = f"Bearer {raw_secret}"
        else:
            request_headers["X-API-Key"] = raw_secret
        try:
            request = Request(
                target.url,
                data=data,
                headers=request_headers,
                method=method.upper(),
            )
            return self._sender(
                request,
                follow_redirects=False,
                timeout_seconds=timeout_ms / 1000,
            )
        except CredentialSecurityError:
            raise
        except Exception:
            pass
        raise _error(CredentialFailureCategory.TRANSPORT_ERROR, self._telemetry)
