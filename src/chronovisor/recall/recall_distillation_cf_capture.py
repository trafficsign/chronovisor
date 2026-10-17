"""Build source-bound CF-only rows for a fresh candidate snapshot.

This module is intentionally a pure adapter.  It never appends to a ledger and
never changes an existing snapshot.  A caller may use it exactly once, after a
new historical snapshot has been produced and before its candidate row is
appended.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import canonical_json
from chronovisor.ingest.page_registry import PageRegistry, PageRegistryError
from chronovisor.recall import recall_distillation_store as store

_PAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CUTOFF_UTC = datetime(2026, 10, 8, 15, 0, tzinfo=UTC)
_SOURCE_KIND = "exact-exposure-page-v1"
_COMPOSITE_RETRIEVER = "historical-fts-v1+exact-exposure-page-v1"


def _copy_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(snapshot))


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_after_cutoff(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        return False
    return parsed.astimezone(UTC) >= _CUTOFF_UTC


def _recompute_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: value for key, value in snapshot.items() if key != "snapshot_sha256"
    }
    snapshot["snapshot_sha256"] = canonical_json.canonical_json_sha256_strict(unsigned)
    return snapshot


def _unavailable(snapshot: Mapping[str, Any], reason: str) -> dict[str, Any]:
    result = _copy_snapshot(snapshot)
    result["counterfactual_capture"] = {
        "status": "unavailable",
        "reason": reason,
    }
    return _recompute_snapshot(result)


def _canonical_exposure(
    root: Path, rally: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return one rally-declared receipt and its verified canonical receipt."""

    host = str(rally.get("host") or "")
    session_sha = str(rally.get("session_id_sha256") or "")
    query_sha = str(rally.get("query_sha256") or "")
    if not host or not _SHA256.fullmatch(session_sha) or not _SHA256.fullmatch(query_sha):
        return None
    declared = rally.get("exposure_receipts")
    if not isinstance(declared, list) or not declared:
        return None
    try:
        # Keep this import local: recall_distillation integrates this adapter
        # into its candidate-snapshot producer.
        from chronovisor.recall import recall_distillation

        entries = recall_distillation._exposure_map(root).get(
            (host, session_sha, query_sha), []
        )
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in declared:
        if not isinstance(item, Mapping):
            continue
        artifact_id = str(item.get("exposure_artifact_id") or "")
        if not _SHA256.fullmatch(artifact_id):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("exposure_artifact_id") or "") != artifact_id:
                continue
            # The map was produced by the existing binding verifier.  Keep the
            # rally's candidate declaration aligned as an extra identity check.
            if item.get("candidate_ids") != entry.get("candidate_ids"):
                continue
            if item.get("candidate_snapshot_sha256") != entry.get(
                "candidate_snapshot_sha256"
            ):
                continue
            candidates.append((dict(item), dict(entry)))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _artifact_hash(artifact: Mapping[str, Any], field: str, value: object) -> bool:
    expected = artifact.get(field)
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        return False
    try:
        return canonical_json.canonical_json_sha256_strict(value) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def _semantic_source_ref(
    rendered_context: str,
    *,
    page_id: str,
    page_sha: str,
    page_uid: str,
    file_size: int,
) -> dict[str, Any] | None:
    """Extract only source identity metadata from the rendered JSON."""

    try:
        payload = json.loads(rendered_context)
    except (TypeError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    found: list[Mapping[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping) and item.get("page_id") == page_id:
            candidate = item.get("source_ref")
            if candidate is not None:
                if not isinstance(candidate, Mapping):
                    raise ValueError("semantic source binding is invalid")
                found.append(candidate)
    if not found:
        return None
    normalized: list[dict[str, Any]] = []
    for source in found:
        uid = str(source.get("uid") or "")
        sha = str(source.get("sha256") or "")
        doc_id = str(source.get("doc_id") or "")
        generation_id = str(source.get("generation_id") or "")
        start = source.get("byte_start")
        end = source.get("byte_end")
        if (
            not uid
            or uid != page_uid
            or sha != page_sha
            or not doc_id
            or not generation_id
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end > file_size
        ):
            raise ValueError("semantic source binding is inconsistent")
        normalized.append(
            {
                "doc_id": doc_id,
                "generation_id": generation_id,
                "sha256": sha,
                "uid": uid,
                "byte_start": start,
                "byte_end": end,
            }
        )
    identities = {
        (item["doc_id"], item["generation_id"], item["sha256"], item["uid"])
        for item in normalized
    }
    if len(identities) != 1:
        raise ValueError("semantic source binding is inconsistent")
    return normalized[0]


def _context_receipt_matches(
    root: Path,
    *,
    decision_id: str,
    selected: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Validate an available recall-log receipt using its existing checker."""

    log_path = root / "recall" / "recall-log.jsonl"
    if not log_path.is_file() or not decision_id:
        return True
    found: Mapping[str, Any] | None = None
    try:
        with log_path.open(encoding="utf-8") as handle:
            for line in deque(handle, maxlen=5_000):
                try:
                    row = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue
                if isinstance(row, Mapping) and row.get("decision_id") == decision_id:
                    found = row
    except OSError:
        return True
    if found is None or "context_receipt" not in found:
        return True
    receipt = found.get("context_receipt")
    if not isinstance(receipt, Mapping):
        return False
    from chronovisor.recall.recall_answer_eval import _context_receipt_error

    if _context_receipt_error(receipt):
        return False
    bindings = receipt.get("page_bindings")
    if not isinstance(bindings, list):
        return False
    actual: dict[str, tuple[str, str]] = {}
    for item in bindings:
        if not isinstance(item, Mapping):
            return False
        page_id = str(item.get("page_id") or "")
        page_sha = str(item.get("content_sha256") or "")
        page_uid = str(item.get("page_uid") or "")
        if not page_id or not _SHA256.fullmatch(page_sha):
            return False
        actual[page_id] = (page_sha, page_uid)
    expected = {
        page_id: (str(item["page_content_sha256"]), str(item["page_uid"]))
        for page_id, item in selected.items()
    }
    if set(actual) != set(expected):
        return False
    return all(
        actual[page_id][0] == expected[page_id][0]
        and (not expected[page_id][1] or actual[page_id][1] == expected[page_id][1])
        for page_id in expected
    )


def _selected_rows(
    root: Path,
    rally: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """Validate source, feature, and receipt bindings; return page rows."""

    from chronovisor.recall import recall_distillation as distill

    candidate_ids = artifact.get("candidate_ids")
    refs = artifact.get("candidate_refs")
    features = artifact.get("candidate_feature_snapshot")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or len(candidate_ids) != len({str(value) for value in candidate_ids})
        or not all(isinstance(value, str) and value for value in candidate_ids)
        or not isinstance(refs, list)
        or not isinstance(features, list)
    ):
        return [], "selected_source_invalid"
    if not _artifact_hash(artifact, "candidate_refs_sha256", refs):
        return [], "artifact_invalid"
    pool = artifact.get("candidate_pool_refs")
    if not isinstance(pool, list) or not _artifact_hash(
        artifact, "candidate_pool_refs_sha256", pool
    ):
        return [], "artifact_invalid"
    if not _artifact_hash(artifact, "candidate_feature_snapshot_sha256", features):
        return [], "artifact_invalid"
    feature_map: dict[str, Mapping[str, Any]] = {}
    for item in features:
        if not isinstance(item, Mapping):
            return [], "selected_feature_invalid"
        candidate_id = str(item.get("candidate_id") or "")
        values = item.get("features")
        if (
            not candidate_id
            or candidate_id in feature_map
            or not isinstance(values, Mapping)
        ):
            return [], "selected_feature_invalid"
        try:
            normalized = distill.build_fast_features(values)
        except distill.DistillationError:
            return [], "selected_feature_invalid"
        if set(values) != set(distill.FAST_FEATURE_KEYS) or dict(values) != normalized:
            return [], "selected_feature_invalid"
        feature_map[candidate_id] = normalized
    if not set(candidate_ids).issubset(feature_map):
        return [], "selected_feature_invalid"
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in refs:
        if not isinstance(item, Mapping):
            return [], "selected_source_invalid"
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id or candidate_id in by_id:
            return [], "selected_source_invalid"
        by_id[candidate_id] = item
    if set(by_id) != set(candidate_ids):
        return [], "selected_source_invalid"
    pool_selected: dict[str, Mapping[str, Any]] = {}
    for item in pool:
        if not isinstance(item, Mapping):
            return [], "artifact_invalid"
        candidate_id = str(item.get("candidate_id") or "")
        if item.get("selected") is True:
            if candidate_id in pool_selected:
                return [], "artifact_invalid"
            pool_selected[candidate_id] = item
    if pool and set(pool_selected) != set(candidate_ids):
        return [], "selected_source_invalid"
    registry = PageRegistry(root)
    normalized: list[dict[str, Any]] = []
    selected_for_context: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        item = by_id[candidate_id]
        page_id = item.get("page_id")
        page_sha = str(item.get("page_content_sha256") or "")
        rendered = item.get("rendered_context")
        rendered_sha = str(item.get("rendered_context_sha256") or "")
        if (
            not isinstance(page_id, str)
            or not _PAGE_ID.fullmatch(page_id)
            or not _SHA256.fullmatch(page_sha)
            or not isinstance(rendered, str)
            or not rendered
            or len(rendered.encode()) > 12_000
            or hashlib.sha256(rendered.encode()).hexdigest() != rendered_sha
        ):
            return [], "selected_source_invalid"
        if pool:
            pool_item = pool_selected[candidate_id]
            for key in ("page_id", "page_content_sha256", "rendered_context_sha256"):
                if pool_item.get(key) != item.get(key):
                    return [], "selected_source_invalid"
        try:
            resolved = registry.resolve(page_id)
            path = registry.path_for(page_id, require_stable=True)
        except (OSError, KeyError, TypeError, ValueError, PageRegistryError):
            return [], "page_unavailable"
        if (
            not isinstance(resolved, Mapping)
            or path is None
            or resolved.get("status") != "stable"
            or resolved.get("canonical_uid")
            or str(resolved.get("page_id") or "") != page_id
            or not path.is_file()
        ):
            return [], "page_unavailable"
        try:
            actual_sha, file_size = _sha256_file(path)
        except OSError:
            return [], "page_unavailable"
        if actual_sha != page_sha or str(resolved.get("content_sha256") or "") != page_sha:
            return [], "page_hash_mismatch"
        page_uid = str(resolved.get("uid") or "")
        if not page_uid:
            return [], "page_identity_mismatch"
        try:
            semantic_ref = _semantic_source_ref(
                rendered,
                page_id=page_id,
                page_sha=page_sha,
                page_uid=page_uid,
                file_size=file_size,
            )
        except ValueError:
            return [], "page_identity_mismatch"
        selected_for_context[page_id] = {
            "page_content_sha256": page_sha,
            "page_uid": page_uid,
        }
        row = {
            "candidate_id": candidate_id,
            "rank": 0,
            # The CF binding validator compares text_sha256 to declared source
            # hashes.  This is deliberately a page digest, not a Raw atom ID.
            "text_sha256": page_sha,
            "page_id": page_id,
            "page_content_sha256": page_sha,
            "page_uid": page_uid,
            "source_kind": _SOURCE_KIND,
            "exposure_artifact_id": str(artifact.get("artifact_id") or ""),
            "source_provenance": {
                "candidate_snapshot_sha256": str(
                    artifact.get("candidate_snapshot_sha256") or ""
                ),
                "candidate_refs_sha256": str(
                    artifact.get("candidate_refs_sha256") or ""
                ),
                "candidate_pool_refs_sha256": str(
                    artifact.get("candidate_pool_refs_sha256") or ""
                ),
                "candidate_feature_snapshot_sha256": str(
                    artifact.get("candidate_feature_snapshot_sha256") or ""
                ),
                "observed_at": str(artifact.get("observed_at") or ""),
            },
        }
        if semantic_ref is not None:
            row["source_provenance"]["semantic_source_ref"] = semantic_ref
        normalized.append(row)
    decision_id = str(artifact.get("decision_id") or "")
    if not _context_receipt_matches(
        root, decision_id=decision_id, selected=selected_for_context
    ):
        return [], "context_receipt_mismatch"
    return normalized, None


def augment_candidate_snapshot(
    root: Path, rally: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach verified page candidates for CF without changing historical rows.

    The function is pure: it returns a new mapping and performs only read-only
    source checks.  A caller must append the returned snapshot as the rally's
    first candidate-ledger row; existing rows are intentionally not repaired.
    """

    original = _copy_snapshot(snapshot)
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("candidates"), list):
        return _unavailable(original, "candidate_snapshot_invalid")
    if not _is_after_cutoff(rally.get("as_of")):
        return original
    if not isinstance(rally.get("exposure_receipts"), list) or not rally.get(
        "exposure_receipts"
    ):
        return original
    exposure = _canonical_exposure(Path(root), rally)
    if exposure is None:
        return _unavailable(original, "receipt_mismatch")
    _declared, canonical = exposure
    artifact_id = str(canonical.get("exposure_artifact_id") or "")
    if not _SHA256.fullmatch(artifact_id):
        return _unavailable(original, "receipt_mismatch")
    try:
        artifact = store.read_sealed(
            store.distillation_dir(Path(root)) / "exposures" / f"{artifact_id}.json",
            schema="chronovisor.recall-exact-exposure.v1",
        )
    except (OSError, KeyError, TypeError, ValueError, store.DistillationStoreError):
        return _unavailable(original, "artifact_invalid")
    if (
        artifact.get("artifact_id") != artifact_id
        or artifact.get("candidate_ids") != canonical.get("candidate_ids")
        or artifact.get("candidate_snapshot_sha256")
        != canonical.get("candidate_snapshot_sha256")
        or artifact.get("decision_id") != canonical.get("decision_id")
    ):
        return _unavailable(original, "artifact_invalid")
    selected = artifact.get("candidate_ids")
    if not isinstance(selected, list) or not selected:
        return original
    try:
        rows, reason = _selected_rows(Path(root), rally, artifact)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        store.DistillationStoreError,
    ):
        return _unavailable(original, "artifact_invalid")
    if reason is not None:
        return _unavailable(original, reason)
    existing = original["candidates"]
    existing_ids = {
        str(item.get("candidate_id") or "")
        for item in existing
        if isinstance(item, Mapping)
    }
    if any(str(row["candidate_id"]) in existing_ids for row in rows):
        return _unavailable(original, "candidate_id_collision")
    used_ranks = {
        int(item.get("rank"))
        for item in existing
        if isinstance(item, Mapping)
        and isinstance(item.get("rank"), int)
        and not isinstance(item.get("rank"), bool)
        and int(item.get("rank")) > 0
    }
    next_rank = max(used_ranks, default=0) + 1
    for row in rows:
        row["rank"] = next_rank
        next_rank += 1
    result = _copy_snapshot(original)
    result["candidates"] = [*existing, *rows]
    result["retriever_revision"] = _COMPOSITE_RETRIEVER
    result["counterfactual_capture"] = {
        "status": "ready",
        "source_kind": _SOURCE_KIND,
        "exposure_artifact_id": artifact_id,
        "candidate_refs_sha256": str(artifact.get("candidate_refs_sha256") or ""),
        "candidate_pool_refs_sha256": str(
            artifact.get("candidate_pool_refs_sha256") or ""
        ),
        "candidate_feature_snapshot_sha256": str(
            artifact.get("candidate_feature_snapshot_sha256") or ""
        ),
        "observed_at": str(artifact.get("observed_at") or ""),
        "selected_count": len(rows),
    }
    return _recompute_snapshot(result)
