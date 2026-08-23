"""Shared fail-closed policy for text sent to network egress."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["PolicyDecision", "guard_egress_query", "invisible_unicode"]

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])-[-A-Za-z0-9_]{16,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*\S{8,}"),
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?81[- ]?)?0\d{1,4}[- ]?\d{1,4}[- ]?\d{3,4}(?!\d)")
_PRIVATE_PATH = re.compile(
    r"(?:^|\s)(?:/Users/[^/\s]+|/home/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)"
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
