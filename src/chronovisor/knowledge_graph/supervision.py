"""Join actual Recall use to saved relation paths and advance lifecycle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import read_sealed_json
from chronovisor.knowledge_graph.store import KnowledgeGraphStore
from chronovisor.recall.recall_label_factory import _read_jsonl


def advance_used_relations(
    *,
    relation_path_file: Path,
    pull_log_file: Path,
    store: KnowledgeGraphStore,
    min_sessions: int = 3,
) -> dict[str, Any]:
    """Promote only paths actually used in distinct sessions; preserve events."""

    used = {
        str(row.get("decision_id") or ""): row
        for row in _read_jsonl(pull_log_file)
        if row.get("type") == "used" and str(row.get("decision_id") or "")
    }
    sessions_by_relation: dict[str, set[str]] = {}
    for row in _read_jsonl(relation_path_file):
        decision_id = str(row.get("decision_id") or "")
        use = used.get(decision_id)
        if use is None:
            continue
        page_ids = {
            str(value)
            for value in use.get("page_ids", [])
            if isinstance(value, str) and value
        }
        if str(row.get("page_id") or "") not in page_ids:
            continue
        session = str(use.get("session_id") or "")
        if not session:
            continue
        for relation_id in row.get("relation_ids", []):
            if isinstance(relation_id, str) and relation_id:
                sessions_by_relation.setdefault(relation_id, set()).add(session)

    latest = {row.relation_id: row for row in store.relations()}
    advanced = 0
    observed = 0
    for relation_id, sessions in sorted(sessions_by_relation.items()):
        record = latest.get(relation_id)
        if record is None or record.status in {
            "stale",
            "retracted",
            "proposed",
            "held",
        }:
            continue
        combined = tuple(sorted(set(record.used_sessions) | sessions))
        status = (
            "repeatedly_used"
            if len(combined) >= min_sessions and record.status == "verified"
            else record.status
        )
        updated = replace(
            record,
            status=status,
            used_count=max(record.used_count, len(combined)),
            used_sessions=combined,
            reason_code="actual_recall_path_used",
        )
        store.append(updated, action="use", reason_code="actual_recall_path_used")
        observed += 1
        advanced += status != record.status
    return {
        "status": "ok",
        "used_relations": observed,
        "advanced_repeatedly_used": advanced,
        "orphan_paths": sum(
            relation_id not in latest for relation_id in sessions_by_relation
        ),
    }


def advance_used_entities(
    *,
    relation_path_file: Path,
    pull_log_file: Path,
    store: KnowledgeGraphStore,
    min_sessions: int = 3,
) -> dict[str, Any]:
    """Advance entity merges only when their retrieval path was actually used."""

    try:
        snapshot = read_sealed_json(store.entity_snapshot_file, recover_backup=True)
    except Exception:
        return {"status": "waiting", "used_entities": 0, "advanced": 0}
    merge_values = snapshot.get("merge_candidates")
    merges = (
        {str(key): dict(value) for key, value in merge_values.items() if isinstance(value, dict)}
        if isinstance(merge_values, dict)
        else {}
    )
    used = {
        str(row.get("decision_id") or ""): row
        for row in _read_jsonl(pull_log_file)
        if row.get("type") == "used" and str(row.get("decision_id") or "")
    }
    sessions_by_merge: dict[str, set[str]] = {}
    for row in _read_jsonl(relation_path_file):
        use = used.get(str(row.get("decision_id") or ""))
        if use is None:
            continue
        page_ids = {
            str(value)
            for value in use.get("page_ids", [])
            if isinstance(value, str) and value
        }
        if str(row.get("page_id") or "") not in page_ids:
            continue
        session = str(use.get("session_id") or "")
        if not session:
            continue
        for merge_id in row.get("entity_merge_ids", []):
            if isinstance(merge_id, str) and merge_id.startswith("merge_"):
                sessions_by_merge.setdefault(merge_id, set()).add(session)
    observed = advanced = 0
    for merge_id, sessions in sessions_by_merge.items():
        merge = merges.get(merge_id)
        if merge is None or merge.get("status") in {"held", "retracted"}:
            continue
        combined = sorted(
            {
                str(value)
                for value in merge.get("used_sessions") or []
                if str(value)
            }
            | sessions
        )
        prior_status = str(merge.get("status") or "verified")
        status = (
            "repeatedly_used"
            if prior_status == "verified" and len(combined) >= min_sessions
            else prior_status
        )
        merges[merge_id] = {
            **merge,
            "status": status,
            "used_count": max(int(merge.get("used_count") or 0), len(combined)),
            "used_sessions": combined,
            "reason_code": "actual_recall_entity_path_used",
        }
        observed += 1
        advanced += status != prior_status
    if observed:
        store.write_derived_snapshot(
            "entities",
            {
                **{
                    key: value
                    for key, value in snapshot.items()
                    if key not in {"merge_candidates", "seal_sha256"}
                },
                "merge_candidates": dict(sorted(merges.items())),
            },
        )
    return {
        "status": "ok",
        "used_entities": observed,
        "advanced_repeatedly_used": advanced,
        "orphan_paths": sum(merge_id not in merges for merge_id in sessions_by_merge),
    }


def promote_authoritative_entities(
    *,
    store: KnowledgeGraphStore,
    enabled: bool,
    min_sessions: int,
) -> dict[str, Any]:
    try:
        snapshot = read_sealed_json(store.entity_snapshot_file, recover_backup=True)
    except Exception:
        return {"status": "waiting", "eligible": 0, "promoted": 0}
    merge_values = snapshot.get("merge_candidates")
    merges = (
        {str(key): dict(value) for key, value in merge_values.items() if isinstance(value, dict)}
        if isinstance(merge_values, dict)
        else {}
    )
    if not enabled:
        return {"status": "held", "eligible": 0, "promoted": 0}
    eligible = promoted = 0
    for merge_id, merge in merges.items():
        sessions = {
            str(value) for value in merge.get("used_sessions") or [] if str(value)
        }
        if merge.get("status") != "repeatedly_used" or len(sessions) < min_sessions:
            continue
        eligible += 1
        merges[merge_id] = {
            **merge,
            "status": "authoritative",
            "reason_code": "sealed_rollout_and_repeated_actual_use",
        }
        promoted += 1
    if promoted:
        store.write_derived_snapshot(
            "entities",
            {
                **{
                    key: value
                    for key, value in snapshot.items()
                    if key not in {"merge_candidates", "seal_sha256"}
                },
                "merge_candidates": dict(sorted(merges.items())),
            },
        )
    return {"status": "ok", "eligible": eligible, "promoted": promoted}


def mark_stale_source(
    *,
    page_id: str,
    current_content_sha256: str,
    store: KnowledgeGraphStore,
) -> dict[str, Any]:
    stale = 0
    for record in store.relations():
        if record.source_page_id != page_id or record.status in {"stale", "retracted"}:
            continue
        if any(row.content_sha256 == current_content_sha256 for row in record.evidence):
            continue
        updated = replace(record, status="stale", reason_code="source_digest_changed")
        store.append(updated, action="stale", reason_code="source_digest_changed")
        stale += 1
    return {"status": "ok", "stale_relations": stale}


def promote_authoritative_relations(
    *,
    store: KnowledgeGraphStore,
    enabled: bool,
    min_sessions: int,
) -> dict[str, Any]:
    """Promote mature used relations only after the sealed global rollout wins."""

    promoted = 0
    eligible = 0
    if not enabled:
        return {"status": "held", "eligible": 0, "promoted": 0}
    for record in store.relations(statuses={"repeatedly_used"}):
        if len(set(record.used_sessions)) < max(1, min_sessions):
            continue
        eligible += 1
        updated = replace(
            record,
            status="authoritative",
            reason_code="sealed_rollout_and_repeated_actual_use",
        )
        event = store.append(
            updated,
            action="promote",
            reason_code="sealed_rollout_and_repeated_actual_use",
        )
        promoted += int(event.get("idempotent") is not True)
    return {"status": "ok", "eligible": eligible, "promoted": promoted}


def retract_relation(
    relation_id: str,
    *,
    reason: str,
    store: KnowledgeGraphStore,
) -> dict[str, Any]:
    record = next(
        (row for row in store.relations() if row.relation_id == relation_id), None
    )
    if record is None:
        return {"status": "missing", "relation_id": relation_id}
    updated = replace(record, status="retracted", reason_code=reason[:160])
    event = store.append(updated, action="retract", reason_code=reason[:160])
    return {
        "status": "retracted",
        "relation_id": relation_id,
        "event_id": event["event_id"],
    }
