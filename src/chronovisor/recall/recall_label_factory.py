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
from chronovisor.recall.recall_log_schema import join_used_recall_episodes

LABEL_LEDGER = CHRONOVISOR_ROOT / "runtime" / "recall-labels" / "ledger.jsonl"
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
) -> dict[str, Any]:
    identity = session_hash or query_sha256 or str(provenance.get("event_id") or "")
    stable = ":".join(
        [
            identity,
            page_id,
            polarity,
            str(provenance.get("source") or ""),
        ]
    )
    return {
        "schema_version": 1,
        "label_id": _digest(stable)[:24],
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


def build_label_ledger(
    *,
    certificate_file: Path,
    recall_log_file: Path,
    pull_log_file: Path,
    golden_file: Path,
) -> dict[str, Any]:
    """Join evidence without treating exposure or rejects as negatives."""

    labels: list[dict[str, Any]] = []
    for row in _read_jsonl(certificate_file):
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

    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for label in labels:
        identity = str(label["session_hash"] or label["query_sha256"])
        key = (identity, str(label["page_id"]), str(label["polarity"]))
        current = deduped.get(key)
        if current is None or float(label["weight"]) > float(current["weight"]):
            deduped[key] = label
    rows = assign_temporal_splits(list(deduped.values()))
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["split"]),
            str(row["query_sha256"]),
            str(row["page_id"]),
            str(row["polarity"]),
        ),
    )
    trusted_positive = [
        row
        for row in rows
        if row["polarity"] == "positive" and row["quality"] in {"strong", "gold"}
    ]
    strong_positive = [row for row in trusted_positive if row["quality"] == "strong"]
    sessions = {
        str(row["session_hash"]) for row in strong_positive if row["session_hash"]
    }
    return {
        "schema_version": 1,
        "labels": rows,
        "counts": {
            "total": len(rows),
            "strong_positive": len(strong_positive),
            "gold_positive": len(trusted_positive) - len(strong_positive),
            "strong_positive_sessions": len(sessions),
            "joined_used": int(joined["accepted"]),
            "rejected_used": int(joined["rejected"]),
        },
        "gates": {
            "field_learning_allowed": (
                len(strong_positive) >= 200 and len(sessions) >= 20
            ),
            "calibration_allowed": len(rows) >= 500,
        },
    }


def materialize_label_ledger(
    *,
    certificate_file: Path,
    recall_log_file: Path,
    pull_log_file: Path,
    golden_file: Path,
    output_file: Path = LABEL_LEDGER,
) -> dict[str, Any]:
    payload = build_label_ledger(
        certificate_file=certificate_file,
        recall_log_file=recall_log_file,
        pull_log_file=pull_log_file,
        golden_file=golden_file,
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
        RECALL_LOG_FILE,
        RECALL_PULL_LOG_FILE,
    )
    from chronovisor.search.search_eval import GOLDEN_FILE

    return {
        "certificate_file": CERTIFICATE_LEDGER,
        "recall_log_file": RECALL_LOG_FILE,
        "pull_log_file": RECALL_PULL_LOG_FILE,
        "golden_file": GOLDEN_FILE,
    }
