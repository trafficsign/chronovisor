"""Derived provenance ledger for Recall supervision."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall.evidence_certificate import CERTIFICATE_LEDGER
from chronovisor.recall.feedback_ledger import active_feedback_rows
from chronovisor.recall.recall_log_schema import join_used_recall_episodes

LABEL_LEDGER = CHRONOVISOR_ROOT / "runtime" / "recall-labels" / "ledger.jsonl"
RELATION_RECEIPT_LEDGER = (
    CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "consensus-receipts.jsonl"
)
RELATION_PATH_LEDGER = (
    CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "candidate-trace.jsonl"
)
ENTITY_DECISION_LEDGER = (
    CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "entity-decisions.jsonl"
)
RUBRIC_OUTCOME_LEDGER = (
    CHRONOVISOR_ROOT / "runtime" / "recall-rubric" / "outcomes.jsonl"
)
RELATION_EVENT_LEDGER = CHRONOVISOR_ROOT / "knowledge-graph" / "relation-events.jsonl"
QUALITY_WEIGHTS = {"silver": 0.25, "strong": 1.0, "gold": 2.0}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _session_hash(value: object) -> str:
    session = str(value or "").strip()
    return _digest(session)[:16] if session else ""


def _label(
    *,
    page_id: str,
    query_sha256: str,
    session_hash: str,
    polarity: str,
    quality: str,
    provenance: dict[str, Any],
    observed_at: str = "",
    subject_kind: str = "page_recall",
    subject_id: str = "",
) -> dict[str, Any]:
    resolved_subject_id = subject_id or page_id
    identity = session_hash or query_sha256 or str(provenance.get("event_id") or "")
    stable = ":".join(
        [
            identity,
            subject_kind,
            resolved_subject_id,
            polarity,
            str(provenance.get("source") or ""),
        ]
    )
    return {
        "schema_version": 2,
        "label_id": _digest(stable)[:24],
        "subject_kind": subject_kind,
        "subject_id": resolved_subject_id,
        "page_id": page_id,
        "query_sha256": query_sha256,
        "session_hash": session_hash,
        "polarity": polarity,
        "quality": quality,
        "weight": QUALITY_WEIGHTS.get(quality, 0.0),
        "split": "unassigned",
        "observed_at": observed_at,
        "provenance": provenance,
    }


def assign_temporal_splits(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign chronological 70/20/10 splits without session/query leakage."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    row_keys: list[list[str]] = []
    for row in labels:
        keys = []
        session = str(row.get("session_hash") or "")
        query = str(row.get("query_sha256") or "")
        if session:
            keys.append(f"session:{session}")
        if query:
            keys.append(f"query:{query}")
        if not keys:
            keys.append(f"label:{row.get('label_id')}")
        for key in keys:
            find(key)
        for key in keys[1:]:
            union(keys[0], key)
        row_keys.append(keys)

    groups: dict[str, list[int]] = {}
    for index, keys in enumerate(row_keys):
        groups.setdefault(find(keys[0]), []).append(index)
    ordered = sorted(
        groups.values(),
        key=lambda indexes: (
            max(str(labels[index].get("observed_at") or "") for index in indexes),
            min(str(labels[index].get("label_id") or "") for index in indexes),
        ),
    )
    count = len(ordered)
    train_end = max(1, math.ceil(count * 0.70)) if count else 0
    holdout_end = max(train_end, math.ceil(count * 0.90))
    assigned: list[dict[str, Any]] = [dict(row) for row in labels]
    for group_index, indexes in enumerate(ordered):
        split = (
            "train"
            if group_index < train_end
            else "holdout"
            if group_index < holdout_end
            else "locked-test"
        )
        for index in indexes:
            assigned[index]["split"] = split
    return assigned


def _certificate_labels(path: Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        page_id = str(row.get("page_id") or "")
        query_sha = str(row.get("query_sha256") or "")
        if not page_id or not query_sha:
            continue
        outcome = str(row.get("outcome") or "")
        labels.append(
            _label(
                page_id=page_id,
                query_sha256=query_sha,
                session_hash="",
                polarity="positive" if outcome == "pass" else "exposure",
                quality="silver",
                provenance={
                    "source": "certificate",
                    "certificate_id": str(row.get("certificate_id") or ""),
                    "outcome": outcome,
                    "certificate_quality": str(row.get("label_quality") or "silver"),
                    "content_sha256": str(row.get("content_sha256") or ""),
                    "policy_sha256": str(row.get("policy_sha256") or ""),
                },
                observed_at=str(row.get("created_at") or ""),
            )
        )
    return labels


def _explicit_feedback_labels(path: Path | None) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for row in active_feedback_rows(path) if path else []:
        if row.get("kind") != "page_ignored":
            continue
        values = row.get("negative_pages")
        if not isinstance(values, list):
            continue
        snapshot_value = row.get("snapshot")
        snapshot = snapshot_value if isinstance(snapshot_value, dict) else {}
        prompt = str(row.get("prompt") or "")
        query_sha = _digest(prompt) if prompt else str(snapshot.get("prompt_hash") or "")
        for page_id in values:
            if not isinstance(page_id, str) or not page_id:
                continue
            labels.append(
                _label(
                    page_id=page_id,
                    query_sha256=query_sha,
                    session_hash=_session_hash(snapshot.get("session_id")),
                    polarity="negative",
                    quality="strong" if row.get("frontier_reviewed") is True else "silver",
                    provenance={
                        "source": "explicit_page_feedback",
                        "ref": str(row.get("ref") or ""),
                        "feedback_sha256": _digest(
                            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
                        ),
                        "reviewed": row.get("frontier_reviewed") is True,
                    },
                    observed_at=str(row.get("ts") or ""),
                )
            )
    return labels


def _opposing_relation_labels(path: Path | None) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for row in _read_jsonl(path) if path else []:
        action = str(row.get("action") or "")
        relation_value = row.get("relation")
        relation = relation_value if isinstance(relation_value, dict) else {}
        relation_id = str(relation.get("relation_id") or "")
        if action not in {"stale", "retract"} or not relation_id:
            continue
        labels.append(
            _label(
                page_id="",
                query_sha256="",
                session_hash="",
                polarity="negative",
                quality="strong" if action == "retract" else "silver",
                subject_kind="relation",
                subject_id=relation_id,
                provenance={
                    "source": "relation_opposing_event",
                    "event_id": str(row.get("event_id") or ""),
                    "action": action,
                    "reason_code": str(row.get("reason_code") or "")[:160],
                    "event_hash": str(row.get("event_hash") or ""),
                },
                observed_at=str(row.get("created_at") or ""),
            )
        )
    return labels


def _summarize_label_ledger(
    labels: list[dict[str, Any]], joined: dict[str, Any]
) -> dict[str, Any]:
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for label in labels:
        identity = str(label["session_hash"] or label["query_sha256"])
        key = (
            identity,
            str(label["subject_kind"]),
            str(label["subject_id"]),
            str(label["polarity"]),
        )
        current = deduped.get(key)
        if current is None or float(label["weight"]) > float(current["weight"]):
            deduped[key] = label
    rows = assign_temporal_splits(list(deduped.values()))
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["split"]),
            str(row["query_sha256"]),
            str(row["subject_kind"]),
            str(row["subject_id"]),
            str(row["polarity"]),
        ),
    )
    trusted_positive = [
        row
        for row in rows
        if row["subject_kind"] == "page_recall"
        and row["polarity"] == "positive"
        and row["quality"] in {"strong", "gold"}
    ]
    strong_positive = [row for row in trusted_positive if row["quality"] == "strong"]
    sessions = {
        str(row["session_hash"]) for row in strong_positive if row["session_hash"]
    }
    subject_counts: dict[str, dict[str, int]] = {}
    subject_sessions: dict[str, set[str]] = {}
    for row in rows:
        kind = str(row["subject_kind"])
        bucket = subject_counts.setdefault(
            kind, {"total": 0, "silver": 0, "strong": 0, "gold": 0}
        )
        bucket["total"] += 1
        bucket[str(row["quality"])] += 1
        if row["quality"] in {"strong", "gold"} and row["session_hash"]:
            subject_sessions.setdefault(kind, set()).add(str(row["session_hash"]))
    for kind, bucket in subject_counts.items():
        bucket["sessions"] = len(subject_sessions.get(kind, set()))
    return {
        "schema_version": 2,
        "labels": rows,
        "counts": {
            "total": len(rows),
            "strong_positive": len(strong_positive),
            "gold_positive": len(trusted_positive) - len(strong_positive),
            "strong_positive_sessions": len(sessions),
            "joined_used": int(joined["accepted"]),
            "rejected_used": int(joined["rejected"]),
            "by_subject_kind": subject_counts,
        },
        "gates": {
            "field_learning_allowed": len(strong_positive) >= 200
            and len(sessions) >= 20,
            "calibration_allowed": len(rows) >= 500,
            "relation_learning_allowed": subject_counts.get("relation", {}).get(
                "strong", 0
            )
            >= 20
            and subject_counts.get("relation", {}).get("sessions", 0) >= 5,
            "entity_learning_allowed": subject_counts.get("entity_merge", {}).get(
                "strong", 0
            )
            >= 20
            and subject_counts.get("entity_merge", {}).get("sessions", 0) >= 5,
            "rubric_learning_allowed": subject_counts.get("rubric", {}).get(
                "gold", 0
            )
            >= 30,
        },
    }


def build_label_ledger(
    *,
    certificate_file: Path,
    recall_log_file: Path,
    pull_log_file: Path,
    golden_file: Path,
    relation_receipt_file: Path | None = None,
    relation_path_file: Path | None = None,
    entity_decision_file: Path | None = None,
    rubric_outcome_file: Path | None = None,
    feedback_file: Path | None = None,
    relation_event_file: Path | None = None,
) -> dict[str, Any]:
    """Join evidence without treating exposure or rejects as negatives."""

    labels = _certificate_labels(certificate_file)

    recall_rows = _read_jsonl(recall_log_file)
    pull_rows = _read_jsonl(pull_log_file)
    joined = join_used_recall_episodes(recall_rows, pull_rows)
    for episode in joined["episodes"]:
        recall = episode["recall"]
        query_sha = str(
            recall.get("prompt_sha256")
            or recall.get("prompt_hash")
            or recall.get("query_sha256")
            or ""
        )
        session = _session_hash(episode.get("session_id"))
        for page_id in episode["page_ids"]:
            labels.append(
                _label(
                    page_id=page_id,
                    query_sha256=query_sha,
                    session_hash=session,
                    polarity="positive",
                    quality="strong",
                    provenance={
                        "source": "recall_used",
                        "event_id": episode["event_id"],
                        "decision_id": episode["decision_id"],
                    },
                    observed_at=str(episode.get("pull", {}).get("ts") or ""),
                )
            )
    for row in pull_rows:
        if row.get("type") != "read":
            continue
        page_id = str(row.get("page_id") or "")
        if not page_id:
            continue
        labels.append(
            _label(
                page_id=page_id,
                query_sha256="",
                session_hash=_session_hash(row.get("session_id")),
                polarity="exposure",
                quality="silver",
                provenance={
                    "source": "read",
                    "decision_id": str(row.get("decision_id") or ""),
                },
                observed_at=str(row.get("ts") or ""),
            )
        )

    for row in _read_jsonl(golden_file):
        if row.get("reviewed") is not True:
            continue
        query = str(row.get("query") or "")
        query_sha = _digest(query) if query else str(row.get("query_sha256") or "")
        for polarity, field in (
            ("positive", "expected_pages"),
            ("negative", "negative_pages"),
        ):
            values = row.get(field)
            if not isinstance(values, list):
                continue
            for page_id in values:
                if not isinstance(page_id, str) or not page_id:
                    continue
                labels.append(
                    _label(
                        page_id=page_id,
                        query_sha256=query_sha,
                        session_hash="",
                        polarity=polarity,
                        quality="gold",
                        provenance={
                            "source": "explicit_eval",
                            "ref": str(row.get("ref") or ""),
                            "reviewed": True,
                        },
                        observed_at=str(row.get("ts") or ""),
                    )
                )

    # Only explicit, still-active page corrections are negative supervision.
    # Passive exposure and generic ``injection_ignored`` events remain non-negative.
    labels.extend(_explicit_feedback_labels(feedback_file))

    # Stale and retract are opposing relation events. They do not erase the
    # earlier positive receipt, preserving replayable supervision history.
    labels.extend(_opposing_relation_labels(relation_event_file))

    relation_receipts = _read_jsonl(relation_receipt_file or RELATION_RECEIPT_LEDGER)
    for row in relation_receipts:
        relation_id = str(row.get("relation_id") or "")
        receipt_id = str(row.get("receipt_id") or "")
        if not relation_id or not receipt_id:
            continue
        outcome = str(row.get("outcome") or "held")
        labels.append(
            _label(
                page_id="",
                query_sha256=str(row.get("query_sha256") or ""),
                session_hash="",
                polarity="positive" if outcome == "verified" else "exposure",
                quality="silver",
                subject_kind="relation",
                subject_id=relation_id,
                provenance={
                    "source": "relation_consensus",
                    "receipt_id": receipt_id,
                    "outcome": outcome,
                    "producer_role": str(row.get("producer_role") or ""),
                    "vote_manifest_sha256": str(row.get("vote_manifest_sha256") or ""),
                },
                observed_at=str(row.get("created_at") or ""),
            )
        )

    path_rows = _read_jsonl(relation_path_file or RELATION_PATH_LEDGER)
    pull_used = {
        str(row.get("decision_id") or ""): row
        for row in pull_rows
        if row.get("type") == "used" and str(row.get("decision_id") or "")
    }
    for row in path_rows:
        decision_id = str(row.get("decision_id") or "")
        used = pull_used.get(decision_id)
        relation_ids = row.get("relation_ids")
        if not isinstance(relation_ids, list):
            continue
        used_pages = {
            str(value)
            for value in (used or {}).get("page_ids", [])
            if isinstance(value, str)
        }
        page_id = str(row.get("page_id") or "")
        actually_used = bool(used is not None and page_id and page_id in used_pages)
        for relation_id in relation_ids:
            if not isinstance(relation_id, str) or not relation_id:
                continue
            labels.append(
                _label(
                    page_id=page_id,
                    query_sha256=str(row.get("query_sha256") or ""),
                    session_hash=(
                        _session_hash(used.get("session_id"))
                        if used is not None
                        else str(row.get("session_hash") or "")
                    ),
                    polarity="positive" if actually_used else "exposure",
                    quality="strong" if actually_used else "silver",
                    subject_kind="relation",
                    subject_id=relation_id,
                    provenance={
                        "source": "used_relation_path"
                        if actually_used
                        else "relation_path_exposure",
                        "decision_id": decision_id,
                        "path_id": str(row.get("path_id") or ""),
                    },
                    observed_at=str((used or row).get("ts") or ""),
                )
            )
        entity_merge_ids = {
            str(value)
            for value in row.get("entity_merge_ids", [])
            if isinstance(value, str) and value.startswith("merge_")
        }
        for merge_id in entity_merge_ids:
            labels.append(
                _label(
                    page_id=page_id,
                    query_sha256=str(row.get("query_sha256") or ""),
                    session_hash=(
                        _session_hash(used.get("session_id"))
                        if used is not None
                        else str(row.get("session_hash") or "")
                    ),
                    polarity="positive" if actually_used else "exposure",
                    quality="strong" if actually_used else "silver",
                    subject_kind="entity_merge",
                    subject_id=merge_id,
                    provenance={
                        "source": "used_entity_merge_path"
                        if actually_used
                        else "entity_merge_path_exposure",
                        "decision_id": decision_id,
                        "path_id": str(row.get("path_id") or ""),
                    },
                    observed_at=str((used or row).get("ts") or ""),
                )
            )

    for kind, path in (
        ("entity_merge", entity_decision_file or ENTITY_DECISION_LEDGER),
        ("rubric", rubric_outcome_file or RUBRIC_OUTCOME_LEDGER),
    ):
        for row in _read_jsonl(path):
            subject_id = str(row.get("subject_id") or row.get(f"{kind}_id") or "")
            if not subject_id:
                continue
            quality = str(row.get("quality") or "silver")
            if quality not in QUALITY_WEIGHTS:
                quality = "silver"
            labels.append(
                _label(
                    page_id="",
                    query_sha256=str(row.get("query_sha256") or ""),
                    session_hash=_session_hash(row.get("session_id")),
                    polarity=str(row.get("polarity") or "exposure"),
                    quality=quality,
                    subject_kind=kind,
                    subject_id=subject_id,
                    provenance={
                        "source": f"{kind}_outcome",
                        "receipt_id": str(row.get("receipt_id") or ""),
                    },
                    observed_at=str(row.get("observed_at") or ""),
                )
            )

    return _summarize_label_ledger(labels, joined)


def materialize_label_ledger(
    *,
    certificate_file: Path,
    recall_log_file: Path,
    pull_log_file: Path,
    golden_file: Path,
    output_file: Path = LABEL_LEDGER,
    relation_receipt_file: Path | None = None,
    relation_path_file: Path | None = None,
    entity_decision_file: Path | None = None,
    rubric_outcome_file: Path | None = None,
    feedback_file: Path | None = None,
    relation_event_file: Path | None = None,
) -> dict[str, Any]:
    payload = build_label_ledger(
        certificate_file=certificate_file,
        recall_log_file=recall_log_file,
        pull_log_file=pull_log_file,
        golden_file=golden_file,
        relation_receipt_file=relation_receipt_file,
        relation_path_file=relation_path_file,
        entity_decision_file=entity_decision_file,
        rubric_outcome_file=rubric_outcome_file,
        feedback_file=feedback_file,
        relation_event_file=relation_event_file,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output_file,
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in payload["labels"]
        ),
    )
    return {**payload, "output_file": str(output_file)}


def default_label_ledger_inputs() -> dict[str, Path]:
    from chronovisor.recall.recall_runtime import (
        RECALL_FEEDBACK_FILE,
        RECALL_LOG_FILE,
        RECALL_PULL_LOG_FILE,
    )
    from chronovisor.search.search_eval import GOLDEN_FILE

    return {
        "certificate_file": CERTIFICATE_LEDGER,
        "recall_log_file": RECALL_LOG_FILE,
        "pull_log_file": RECALL_PULL_LOG_FILE,
        "golden_file": GOLDEN_FILE,
        "feedback_file": RECALL_FEEDBACK_FILE,
        "relation_event_file": RELATION_EVENT_LEDGER,
    }
