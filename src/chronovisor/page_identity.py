"""Stable, semantically opaque page identities.

Chronovisor page slugs remain human-facing mutable names.  UUIDv7 values are
generated once and persisted by :mod:`chronovisor.page_registry`; the embedded
millisecond timestamp is technical creation metadata and is never used for
classification, retention, or page semantics.
"""

from __future__ import annotations

import secrets
import time
import uuid


def new_page_uid(
    *,
    timestamp_ms: int | None = None,
    random_bits: int | None = None,
) -> str:
    """Return an RFC 9562 UUIDv7 string.

    ``timestamp_ms`` and ``random_bits`` are injectable for deterministic
    tests.  Production callers must persist the returned value immediately;
    regenerating an identity for an existing page is not supported.
    """

    ts = int(time.time_ns() // 1_000_000 if timestamp_ms is None else timestamp_ms)
    if not 0 <= ts < (1 << 48):
        raise ValueError("timestamp_ms must fit in 48 bits")
    randomness = secrets.randbits(74) if random_bits is None else int(random_bits)
    if not 0 <= randomness < (1 << 74):
        raise ValueError("random_bits must fit in 74 bits")

    rand_a = randomness >> 62
    rand_b = randomness & ((1 << 62) - 1)
    value = (ts << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(uuid.UUID(int=value))


def normalize_page_uid(value: object) -> str:
    """Validate and normalize a UUIDv7 page identity."""

    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("invalid page UID") from exc
    if parsed.version != 7 or parsed.variant != uuid.RFC_4122:
        raise ValueError("page UID must be RFC 9562 UUIDv7")
    return str(parsed)


def page_uid_timestamp_ms(value: object) -> int:
    """Return the technical creation timestamp encoded by UUIDv7."""

    parsed = uuid.UUID(normalize_page_uid(value))
    return parsed.int >> 80
