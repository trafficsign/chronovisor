"""Network-egress and untrusted-content policy for research tools."""

from __future__ import annotations

import ipaddress
import socket
import ssl
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, cast
from urllib.parse import urlsplit

import httpcore
import httpx

from chronovisor.core.egress_policy import (
    PolicyDecision,
    guard_egress_query,
    invisible_unicode,
)

__all__ = [
    "EgressPolicyError",
    "PolicyDecision",
    "Resolver",
    "external_content_metadata",
    "guard_egress_query",
    "guard_url",
    "guarded_http_client",
    "invisible_unicode",
    "resolve_host",
]

_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "developer message",
    "tool call",
    "命令を無視",
    "指示を無視",
)


def _public_ip(address: str, *, allow_private_network: bool) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if allow_private_network:
        return not (ip.is_unspecified or ip.is_multicast)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


Resolver = Callable[[str, int], list[str]]


def resolve_host(host: str, port: int) -> list[str]:
    return sorted(
        {
            str(row[4][0])
            for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
    )


def guard_url(
    url: str,
    *,
    allow_private_network: bool = False,
    resolver: Resolver = resolve_host,
) -> tuple[PolicyDecision, tuple[str, ...]]:
    try:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return PolicyDecision(False, "invalid_url"), ()
    if parsed.scheme not in {"http", "https"}:
        return PolicyDecision(False, "unsupported_scheme"), ()
    if not parsed.hostname:
        return PolicyDecision(False, "missing_hostname"), ()
    if parsed.username or parsed.password:
        return PolicyDecision(False, "url_credentials_forbidden"), ()
    host = parsed.hostname.casefold().rstrip(".")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return PolicyDecision(False, "local_hostname"), ()
    try:
        addresses = tuple(resolver(host, port))
    except OSError:
        return PolicyDecision(False, "dns_failure"), ()
    if not addresses:
        return PolicyDecision(False, "dns_empty"), ()
    if not all(
        _public_ip(address, allow_private_network=allow_private_network)
        for address in addresses
    ):
        return PolicyDecision(False, "private_or_special_address"), addresses
    return PolicyDecision(True, "allowed", parsed.geturl()), addresses


class EgressPolicyError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _PinnedBackend(httpcore.SyncBackend):
    def __init__(self, host: str, port: int, address: str) -> None:
        self.host = host.casefold().rstrip(".")
        self.port = port
        self.address = address

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        if host.casefold().rstrip(".") != self.host or port != self.port:
            raise httpcore.ConnectError("connection origin escaped validated URL")
        return super().connect_tcp(
            self.address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


class _CoreStream(httpx.SyncByteStream):
    def __init__(self, stream: Iterable[bytes]) -> None:
        self.stream = stream

    def __iter__(self) -> Iterator[bytes]:
        yield from self.stream

    def close(self) -> None:
        close = getattr(self.stream, "close", None)
        if close is not None:
            close()


class _PinnedTransport(httpx.BaseTransport):
    def __init__(self, host: str, port: int, address: str) -> None:
        self.pool = httpcore.ConnectionPool(
            ssl_context=ssl.create_default_context(),
            max_connections=1,
            max_keepalive_connections=0,
            network_backend=_PinnedBackend(host, port, address),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            request.method,
            httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        try:
            response = self.pool.handle_request(core_request)
        except (
            httpcore.NetworkError,
            httpcore.ProtocolError,
            httpcore.ProxyError,
            httpcore.TimeoutException,
            httpcore.UnsupportedProtocol,
        ) as exc:
            raise httpx.TransportError(str(exc)) from exc
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_CoreStream(cast(Iterable[bytes], response.stream)),
            extensions=response.extensions,
        )

    def close(self) -> None:
        self.pool.close()


@contextmanager
def guarded_http_client(
    url: str,
    *,
    resolver: Resolver = resolve_host,
    allow_private_network: bool = False,
    client: httpx.Client | None = None,
) -> Iterator[tuple[httpx.Client, tuple[str, ...]]]:
    policy, addresses = guard_url(
        url,
        resolver=resolver,
        allow_private_network=allow_private_network,
    )
    if not policy.allowed:
        raise EgressPolicyError(policy.reason)
    if client is not None:
        yield client, addresses
        return
    parsed = urlsplit(url)
    assert parsed.hostname is not None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host = parsed.hostname.encode("idna").decode("ascii")
    with httpx.Client(
        transport=_PinnedTransport(host, port, addresses[0]),
        follow_redirects=False,
        trust_env=False,
    ) as pinned:
        yield pinned, addresses


def external_content_metadata(text: str) -> dict[str, object]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    markers = [marker for marker in _INJECTION_MARKERS if marker in folded]
    return {
        "trust": "untrusted",
        "instruction_boundary": "data_not_instructions",
        "injection_markers": markers,
        "invisible_unicode": invisible_unicode(text),
    }
