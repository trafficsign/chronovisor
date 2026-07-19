"""Read-only decoder for immutable artifacts written before Chronovisor v2."""

from __future__ import annotations

CURRENT_PREFIX = "chronovisor."
SEALED_PREVIOUS_PREFIX = "llm-wiki."

SEALED_SCHEMA_SUFFIXES = (
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


def previous_schema(current: str) -> str:
    """Return the read-only previous spelling for a canonical schema id."""

    if not current.startswith(CURRENT_PREFIX):
        raise ValueError(f"not a Chronovisor schema id: {current}")
    suffix = current.removeprefix(CURRENT_PREFIX)
    if suffix not in SEALED_SCHEMA_SUFFIXES:
        raise ValueError(f"schema is not in the sealed-artifact allowlist: {current}")
    return SEALED_PREVIOUS_PREFIX + suffix


def schema_matches(observed: object, current: str) -> bool:
    """Accept current and sealed previous schema ids on read."""

    if observed == current:
        return True
    try:
        return observed == previous_schema(current)
    except ValueError:
        return False


def canonical_schema(observed: str) -> str:
    """Normalize a supported schema id without mutating its source artifact."""

    if observed.startswith(SEALED_PREVIOUS_PREFIX):
        suffix = observed.removeprefix(SEALED_PREVIOUS_PREFIX)
        if suffix not in SEALED_SCHEMA_SUFFIXES:
            raise ValueError(
                f"schema is not in the sealed-artifact allowlist: {observed}"
            )
        return CURRENT_PREFIX + suffix
    return observed
