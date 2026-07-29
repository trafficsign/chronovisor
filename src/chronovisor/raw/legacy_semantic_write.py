"""Fail-closed guard for obsolete semantic page mutation scripts.

The scripts that import this module predate the durable frontier-review lanes.
They may still be useful for read-only diagnostics, but they must never mutate
wiki knowledge from a heuristic or local-model decision.
"""

from __future__ import annotations


class LegacySemanticMutationDisabled(RuntimeError):
    """Raised before an obsolete script can mutate wiki knowledge."""


def block_legacy_semantic_mutation(*, tool: str, replacement: str) -> None:
    """Refuse a semantic write that is outside the frontier-managed pipeline."""

    raise LegacySemanticMutationDisabled(
        f"{tool} semantic writes are disabled: local or heuristic decisions "
        f"cannot mutate Chronovisor knowledge. Use {replacement}; it persists an "
        "exact proposal, requires a frontier-model final decision, and applies "
        "through the shared CAS writer."
    )
