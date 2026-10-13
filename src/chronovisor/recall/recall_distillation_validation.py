"""Validated, bounded source views for Recall materialization bindings.

The materialization path is called once per candidate row.  This module keeps
the validated source joins per Chronovisor root so the binding check does not
decode the complete ledgers for every row.  The cache is only an optimization:
every read is bracketed by file identity signatures and any source drift fails
closed without publishing a cache entry.
"""

from __future__ import annotations

import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.recall import recall_distillation_store as store


class SourceBindingError(RuntimeError):
    """The source view was absent, invalid, or changed during validation."""


_FileSignature = tuple[int, int, int, int, int]
_Signature = tuple[tuple[str, _FileSignature | None], ...]
_BindingInputs = tuple[
    dict[str, dict[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, str], dict[str, float]],
    dict[str, dict[str, Any]],
]


@dataclass(frozen=True)
class _CacheEntry:
    signature: _Signature
    values: _BindingInputs


_CACHE: dict[Path, _CacheEntry] = {}
_CACHE_LOCK = threading.RLock()


def _runtime_paths(root: Path) -> tuple[Path, ...]:
    """Return every mutable source and index file used by the join."""

    from chronovisor.recall import recall_distillation_catalog as catalog

    distillation = store.distillation_dir(root)
    candidate = distillation / "candidate-ledger.jsonl"
    label = distillation / store.LABEL_LEDGER_FILE
    historical_catalog = catalog.catalog_path(root)
    return (
        label,
        distillation / "rally-manifest.jsonl",
        historical_catalog,
        historical_catalog.with_suffix(historical_catalog.suffix + ".checkpoint.json"),
        historical_catalog.with_suffix(".sqlite.candidate-state.json"),
        candidate,
    )


def _file_signature(path: Path) -> _FileSignature | None:
    try:
        state = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SourceBindingError(f"source signature unavailable: {path.name}") from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
        raise SourceBindingError(f"source is not a regular file: {path.name}")
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
    )


def _signature(root: Path) -> _Signature:
    return tuple(
        (str(path), _file_signature(path)) for path in _runtime_paths(root)
    )


def _read_labels(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], tuple[dict[str, str], ...]]:
    path = store.distillation_dir(root) / store.LABEL_LEDGER_FILE
    try:
        rows = store.read_chain(path)
    except Exception as exc:
        raise SourceBindingError("label ledger is not a validated chain") from exc
    labels_by_record: dict[str, dict[str, Any]] = {}
    rally_ids: set[str] = set()
    claims: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise SourceBindingError("label ledger row is not an object")
        record = row.get("record_sha256")
        if (
            not isinstance(record, str)
            or len(record) != 64
            or any(character not in "0123456789abcdef" for character in record)
        ):
            raise SourceBindingError("label ledger record identity is missing")
        if record in labels_by_record:
            raise SourceBindingError("label ledger record identity is duplicated")
        rally_id = row.get("rally_id")
        if not isinstance(rally_id, str) or not rally_id:
            raise SourceBindingError("label rally identity is missing")
        candidate_id = row.get("candidate_id")
        if candidate_id is not None and (
            not isinstance(candidate_id, str) or not candidate_id
        ):
            raise SourceBindingError("label candidate identity is invalid")
        labels_by_record[record] = dict(row)
        rally_ids.add(rally_id)
        claims.add((rally_id, "" if candidate_id is None else candidate_id))
    return (
        labels_by_record,
        tuple(sorted(rally_ids)),
        tuple(
            {"rally_id": rally_id, **({"candidate_id": candidate} if candidate else {})}
            for rally_id, candidate in sorted(claims)
        ),
    )


def _legacy_sources(
    root: Path, rally_ids: Sequence[str]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    # Importing distill at module import time would create a cycle when the
    # parent binding function imports this helper.  Keep the legacy fallback
    # lazy and use the same source readers as materialization itself.
    from chronovisor.recall import recall_distillation as distill

    try:
        rally_rows = distill._materialization_rallies(root, None)
        snapshot_rows = distill._materialization_snapshots(root, None)
    except Exception as exc:
        raise SourceBindingError("legacy materialization sources are invalid") from exc
    wanted = set(rally_ids)
    return (
        {
            rally_id: row
            for rally_id, row in rally_rows.items()
            if rally_id in wanted and isinstance(row, Mapping)
        },
        {
            rally_id: row
            for rally_id, row in snapshot_rows.items()
            if rally_id in wanted and isinstance(row, Mapping)
        },
    )


def _catalog_sources(
    root: Path,
    rally_ids: Sequence[str],
    claims: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]] | None:
    from chronovisor.recall import recall_distillation_catalog as catalog

    try:
        checkpoint = catalog._read_catalog_checkpoint(root)
    except Exception as exc:
        raise SourceBindingError("catalog checkpoint validation failed") from exc
    if checkpoint is None:
        return None
    try:
        rally_rows = catalog.rallies(root, rally_ids)
        rallies = {
            str(row.get("rally_id") or ""): row
            for row in rally_rows
            if isinstance(row, Mapping) and row.get("rally_id")
        }
        if set(rallies) != set(rally_ids):
            raise SourceBindingError("catalog is missing a label rally")
        snapshots = catalog.read_candidate_snapshots(
            root,
            store.distillation_dir(root) / "candidate-ledger.jsonl",
            claims,
        )
        if set(snapshots) != set(rally_ids):
            raise SourceBindingError("candidate index is missing a label rally")
        return rallies, snapshots
    except SourceBindingError:
        raise
    except Exception as exc:
        raise SourceBindingError("catalog source validation failed") from exc


def _load(root: Path) -> _BindingInputs:
    from chronovisor.recall import recall_distillation as distill

    labels_by_record, rally_ids, claims = _read_labels(root)
    catalog_sources = _catalog_sources(root, rally_ids, claims)
    if catalog_sources is None:
        rallies, snapshots = _legacy_sources(root, rally_ids)
    else:
        rallies, snapshots = catalog_sources
    try:
        features_by_pair, snapshot_contracts = distill._materialization_feature_pairs(
            root, rallies, snapshots
        )
    except Exception as exc:
        raise SourceBindingError("materialization feature source is invalid") from exc
    return labels_by_record, dict(rallies), features_by_pair, snapshot_contracts


def binding_inputs(root: Path) -> _BindingInputs:
    """Return one validated, cached source view for ``root``.

    The caller must treat the returned mappings as read-only.  A cache hit is
    accepted only when every source signature is unchanged.  If any source is
    replaced, appended, tampered with, or changes while loading, no stale entry
    is returned and the call raises :class:`SourceBindingError`.
    """

    canonical_root = root.expanduser().resolve(strict=False)
    with _CACHE_LOCK:
        before = _signature(canonical_root)
        cached = _CACHE.get(canonical_root)
        if cached is not None and cached.signature == before:
            return cached.values
        _CACHE.pop(canonical_root, None)
        try:
            values = _load(canonical_root)
            after = _signature(canonical_root)
        except SourceBindingError:
            _CACHE.pop(canonical_root, None)
            raise
        except Exception as exc:
            _CACHE.pop(canonical_root, None)
            raise SourceBindingError("validated source view is unavailable") from exc
        if before != after:
            _CACHE.pop(canonical_root, None)
            raise SourceBindingError("source changed during binding validation")
        _CACHE[canonical_root] = _CacheEntry(before, values)
        return values


def clear_cache(root: Path | None = None) -> None:
    """Clear validation cache entries; intended for tests and process shutdown."""

    with _CACHE_LOCK:
        if root is None:
            _CACHE.clear()
        else:
            _CACHE.pop(root.expanduser().resolve(strict=False), None)
