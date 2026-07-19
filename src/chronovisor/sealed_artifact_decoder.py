"""Canonical schema validation for durable Chronovisor artifacts."""

from __future__ import annotations

CURRENT_PREFIX = "chronovisor."


def schema_matches(observed: object, current: str) -> bool:
    """Accept only the exact canonical schema id."""

    return current.startswith(CURRENT_PREFIX) and observed == current


def canonical_schema(observed: str) -> str:
    """Validate and return a canonical schema id without rewriting it."""

    if not observed.startswith(CURRENT_PREFIX):
        raise ValueError(f"not a canonical Chronovisor schema id: {observed}")
    return observed
