"""Explicit UTC timestamp helpers with stable precision contracts."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return an aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime | None = None) -> datetime:
    """Return an aware UTC datetime, normalizing an optional input."""
    current = value or utc_now()
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def utc_iso_milliseconds() -> str:
    """Return current UTC in ISO 8601 with millisecond precision."""
    return utc_now().isoformat(timespec="milliseconds")


def utc_iso_seconds() -> str:
    """Return current UTC in ISO 8601 with second precision."""
    return utc_now().isoformat(timespec="seconds")


def iso_seconds(value: datetime) -> str:
    """Format a datetime without changing its timezone at second precision."""
    return value.isoformat(timespec="seconds")


def iso_milliseconds(value: datetime) -> str:
    """Format a datetime without changing its timezone at millisecond precision."""
    return value.isoformat(timespec="milliseconds")
