"""Compatibility helpers for immutable pre-Chronovisor artifacts."""

from __future__ import annotations

CURRENT_PREFIX = "chronovisor."
LEGACY_PREFIX = "llm-wiki."

MIGRATED_SCHEMA_SUFFIXES = (
    "canonical-decision-artifact.v1",
    "dashboard-component.v1",
    "deadman-heartbeat.v1",
    "deadman-incident.v1",
    "deadman-threshold-state.v1",
    "quality-anchor.v1",
    "quality-behavior-pointer.v1",
    "quality-behavior-snapshot.v1",
    "quality-metamorphic.v1",
    "quality-probe.v1",
    "raw-capture-fragment.v1",
    "raw-completion-ack.v1",
    "raw-legacy-archive.v1",
    "raw-reference.v1",
    "raw-relocation-ledger.v1",
    "raw-replay-semantic-bundle.v1",
    "raw-segment-commit.v1",
    "raw-segment-manifest.v1",
    "raw-semantic-projection-bundle-receipt.v1",
    "raw-semantic-projection-child.v1",
    "raw-semantic-projection-manifest.v1",
    "raw-semantic-projection-noop.v1",
)


def legacy_schema(current: str) -> str:
    """Return the read-only legacy spelling for a canonical schema id."""

    if not current.startswith(CURRENT_PREFIX):
        raise ValueError(f"not a Chronovisor schema id: {current}")
    return LEGACY_PREFIX + current.removeprefix(CURRENT_PREFIX)


def schema_matches(observed: object, current: str) -> bool:
    """Accept current and immutable legacy schema ids on read."""

    return observed in {current, legacy_schema(current)}


def canonical_schema(observed: str) -> str:
    """Normalize a supported schema id without mutating its source artifact."""

    if observed.startswith(LEGACY_PREFIX):
        return CURRENT_PREFIX + observed.removeprefix(LEGACY_PREFIX)
    return observed
