"""Pure target-identity checks for local ingest triage plans."""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence


def canonical_triage_path(filename: str) -> str:
    """Return the case/Unicode-insensitive path used for exact dedupe."""

    normalized = filename.strip()
    if not normalized.endswith(".md"):
        normalized += ".md"
    return unicodedata.normalize("NFC", normalized).casefold()


def canonical_triage_target(filename: str) -> str:
    """Return the page-id collision key used by the ingest apply boundary."""

    return PurePosixPath(canonical_triage_path(filename)).stem


def _exact_operation_signature(operation: Mapping[str, Any]) -> str:
    """Bind an exact duplicate to both its path and every semantic field."""

    payload = dict(operation)
    filename = payload.get("filename")
    if isinstance(filename, str):
        payload["filename"] = canonical_triage_path(filename)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _collapse_exact_with_indices(
    operations: Sequence[Mapping[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    """Return first occurrences while retaining original plan positions."""

    collapsed: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, operation in enumerate(operations):
        signature = _exact_operation_signature(operation)
        if signature in seen:
            continue
        seen.add(signature)
        collapsed.append((index, dict(operation)))
    return collapsed


def collapse_exact_duplicate_operations(
    operations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse only byte-equivalent plan entries for the same exact path.

    Distinct summaries, keywords, titles, operation types, or folder choices
    are never merged here.  Those differences can encode separate facts or an
    unresolved target choice and therefore require a model repair.
    """

    return [operation for _index, operation in _collapse_exact_with_indices(operations)]


def duplicate_target_groups(
    operations: Sequence[Mapping[str, Any]],
    *,
    effective_filename: Callable[[Mapping[str, Any]], str | None] | None = None,
    operation_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Describe collisions compactly without copying model-owned semantics.

    The complete invalid plan is already retained as the prior assistant turn
    by :class:`LocalStructuredSession`.  Repeating titles, summaries, and
    keywords in validator feedback can push an otherwise repairable 8 KiB plan
    over the 4 KiB feedback cap.  Stable indices are enough to point the model
    back to every conflicting operation without losing any source meaning.
    """

    grouped: dict[str, list[int]] = defaultdict(list)
    if operation_indices is not None and len(operation_indices) != len(operations):
        raise ValueError("operation_indices must align with operations")
    for position, operation in enumerate(operations):
        index = (
            operation_indices[position] if operation_indices is not None else position
        )
        filename = (
            effective_filename(operation)
            if effective_filename is not None
            else operation.get("filename")
        )
        if not isinstance(filename, str) or not filename.strip():
            continue
        target = canonical_triage_target(filename)
        grouped[target].append(index)

    return [
        {
            "target_page_id": target,
            "operation_indices": indices,
            "operation_count": len(indices),
        }
        for target, indices in sorted(grouped.items())
        if len(indices) > 1
    ]


def distinct_target_collisions(
    operations: Sequence[Mapping[str, Any]],
    *,
    effective_filename: Callable[[Mapping[str, Any]], str | None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exact-deduped operations and remaining target collisions."""

    indexed = _collapse_exact_with_indices(operations)
    collapsed = [operation for _index, operation in indexed]
    return collapsed, duplicate_target_groups(
        collapsed,
        effective_filename=effective_filename,
        operation_indices=[index for index, _operation in indexed],
    )
