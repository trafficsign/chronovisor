"""Network-egress and untrusted-content policy for research tools."""

from __future__ import annotations

import ipaddress
import re
import socket
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])-[-A-Za-z0-9_]{16,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*\S{8,}"),
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?81[- ]?)?0\d{1,4}[- ]?\d{1,4}[- ]?\d{3,4}(?!\d)")
_PRIVATE_PATH = re.compile(r"(?:^|\s)(?:/Users/[^/\s]+|/home/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)")
_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "developer message",
    "tool call",
    "命令を無視",
    "指示を無視",
)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    normalized: str = ""


def invisible_unicode(text: str) -> list[str]:
    allowed = {"\n", "\r", "\t"}
    return sorted(
        {
            f"U+{ord(char):04X}"
            for char in text
            if char not in allowed
            and unicodedata.category(char) in {"Cf", "Cc", "Cs", "Co"}
        }
    )


def guard_egress_query(query: str, *, max_chars: int = 500) -> PolicyDecision:
    normalized = unicodedata.normalize("NFKC", query).strip()
    if not normalized:
        return PolicyDecision(False, "empty_query")
    if len(normalized) > max_chars:
        return PolicyDecision(False, "query_too_long")
    if invisible_unicode(query):
        return PolicyDecision(False, "invisible_unicode")
    if any(pattern.search(normalized) for pattern in _SECRET_PATTERNS):
        return PolicyDecision(False, "secret_detected")
    if _EMAIL.search(normalized) or _PHONE.search(normalized):
        return PolicyDecision(False, "pii_detected")
    if _PRIVATE_PATH.search(normalized):
        return PolicyDecision(False, "private_path_detected")
    return PolicyDecision(True, "allowed", normalized)


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
            row[4][0]
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
    if not all(_public_ip(address, allow_private_network=allow_private_network) for address in addresses):
        return PolicyDecision(False, "private_or_special_address"), addresses
    return PolicyDecision(True, "allowed", parsed.geturl()), addresses


def external_content_metadata(text: str) -> dict[str, object]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    markers = [marker for marker in _INJECTION_MARKERS if marker in folded]
    return {
        "trust": "untrusted",
        "instruction_boundary": "data_not_instructions",
        "injection_markers": markers,
        "invisible_unicode": invisible_unicode(text),
    }
