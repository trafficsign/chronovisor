"""Named canonical JSON contracts used by durable identities.

The variants are deliberately separate. Callers must choose whether unknown
values are rejected or stringified and whether non-finite floats are rejected.
Those choices are part of every durable artifact identity.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_strict(value: Any) -> str:
    """Serialize deterministically and reject invalid JSON numbers."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json_sha256_strict(value: Any) -> str:
    """Hash :func:`canonical_json_strict` as UTF-8."""

    return hashlib.sha256(canonical_json_strict(value).encode("utf-8")).hexdigest()


def canonical_json_bytes_strict(value: Any) -> bytes:
    """Return strict canonical UTF-8 bytes without a trailing newline."""

    return canonical_json_strict(value).encode("utf-8")


def canonical_json_line_bytes_strict(value: Any) -> bytes:
    """Return strict canonical UTF-8 bytes with one trailing newline."""

    return (canonical_json_strict(value) + "\n").encode("utf-8")


def canonical_json_stringifying(value: Any) -> str:
    """Serialize deterministically while preserving legacy ``default=str``."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def canonical_json_permissive(value: Any) -> str:
    """Preserve canonical ordering while allowing non-finite floats."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_sha256_stringifying(value: Any) -> str:
    """Hash :func:`canonical_json_stringifying` as UTF-8."""

    return hashlib.sha256(
        canonical_json_stringifying(value).encode("utf-8")
    ).hexdigest()


def canonical_json_bytes_stringifying(value: Any) -> bytes:
    """Return stringifying canonical UTF-8 bytes without a newline."""

    return canonical_json_stringifying(value).encode("utf-8")


def canonical_json_stringifying_strict(value: Any) -> str:
    """Stringify unknown values but reject non-finite JSON numbers."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def canonical_json_sha256_stringifying_strict(value: Any) -> str:
    """Hash :func:`canonical_json_stringifying_strict` as UTF-8."""

    return hashlib.sha256(
        canonical_json_stringifying_strict(value).encode("utf-8")
    ).hexdigest()
