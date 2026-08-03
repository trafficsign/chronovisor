from __future__ import annotations

import hashlib
import json

import pytest

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.recall import recall_label_factory
from chronovisor.recall.feedback_ledger import feedback_row_sha256
from chronovisor.recall.recall_label_factory import (
    assign_temporal_splits,
    build_label_ledger,
)


def write_rows(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_duplicate_order_cannot_change_temporal_split() -> None:
    common = {
        "page_id": "page",
        "query_sha256": "a" * 64,
        "session_hash": "session",
        "polarity": "positive",
        "quality": "strong",
        "provenance": {"source": "test", "outcome_grounded": True},
        "content_sha256": "b" * 64,
    }
    early = recall_label_factory._label(
        **common, observed_at="2026-07-01T00:00:00Z"
    )
    late = recall_label_factory._label(
        **common, observed_at="2026-07-10T00:00:00Z"
    )
    joined = {"accepted": 0, "rejected": 0}
    first = recall_label_factory._summarize_label_ledger(
        [early, late],
        joined,
        embargo_seconds=recall_label_factory.DEFAULT_EMBARGO_SECONDS,
        answer_diagnostics={},
    )
    second = recall_label_factory._summarize_label_ledger(
        [late, early],
        joined,
        embargo_seconds=recall_label_factory.DEFAULT_EMBARGO_SECONDS,
        answer_diagnostics={},
    )

    assert first["labels"] == second["labels"]
    assert first["labels"][0]["observed_at"] == "2026-07-01T00:00:00Z"
    with pytest.raises(ValueError, match="fixed at 24 hours"):
        assign_temporal_splits([early], embargo_seconds=-1)


def test_label_factory_keeps_exposure_non_negative_and_deduplicates(tmp_path) -> None:
    certificates = tmp_path / "certificates.jsonl"
    recalls = tmp_path / "recall.jsonl"
    pulls = tmp_path / "pull.jsonl"
    golden = tmp_path / "golden.jsonl"
    write_rows(
        certificates,
        [
            {
                "certificate_id": "cert-a",
                "page_id": "page-a",
                "query_sha256": "q" * 64,
                "outcome": "pass",
                "label_quality": "silver",
            },
            {
                "certificate_id": "cert-b",
                "page_id": "page-b",
                "query_sha256": "q" * 64,
                "outcome": "reject",
                "label_quality": "silver",
            },
        ],
    )
    write_rows(
        recalls,
        [
            {
                "decision_id": "decision-1",
                "session_id": "session-1",
                "prompt_hash": "q" * 64,
            }
        ],
    )
    write_rows(
        pulls,
        [
            {
                "type": "read",
                "session_id": "session-1",
                "decision_id": "decision-1",
                "page_id": "page-b",
            },
            {
                "type": "used",
                "event_id": "used-1",
                "session_id": "session-1",
                "decision_id": "decision-1",
                "page_ids": ["page-a"],
            },
        ],
    )
    write_rows(
        golden,
        [
            {
                "query": "same query",
                "expected_pages": ["page-a"],
                "negative_pages": ["page-c"],
                "reviewed": True,
            },
            {
                "query": "forged auto gold",
                "expected_pages": ["forged-page"],
                "reviewed": True,
                "source": "recall_questions",
                "reviewer": "chronovisor:recall-questions",
            },
        ],
    )

    payload = build_label_ledger(
        certificate_file=certificates,
        recall_log_file=recalls,
        pull_log_file=pulls,
        golden_file=golden,
    )
    labels = payload["labels"]
    assert not any(row["page_id"] == "forged-page" for row in labels)

    assert any(
        row["page_id"] == "page-b" and row["polarity"] == "exposure" for row in labels
    )
    assert not any(
        row["page_id"] == "page-b" and row["polarity"] == "negative" for row in labels
    )
    used = [
        row
        for row in labels
        if row["page_id"] == "page-a" and row["provenance"]["source"] == "recall_used"
    ]
    assert used[0]["quality"] == "strong"
    assert any(
        row["page_id"] == "page-c"
        and row["quality"] == "gold"
        and row["polarity"] == "negative"
        for row in labels
    )


def test_same_session_never_crosses_temporal_split(tmp_path) -> None:
    certificates = tmp_path / "certificates.jsonl"
    recalls = tmp_path / "recall.jsonl"
    pulls = tmp_path / "pull.jsonl"
    golden = tmp_path / "golden.jsonl"
    write_rows(certificates, [])
    write_rows(golden, [])
    write_rows(
        recalls,
        [
            {"decision_id": "d1", "session_id": "s", "prompt_hash": "a" * 64},
            {"decision_id": "d2", "session_id": "s", "prompt_hash": "b" * 64},
        ],
    )
    write_rows(
        pulls,
        [
            {
                "type": "used",
                "event_id": "e1",
                "decision_id": "d1",
                "session_id": "s",
                "page_ids": ["one"],
            },
            {
                "type": "used",
                "event_id": "e2",
                "decision_id": "d2",
                "session_id": "s",
                "page_ids": ["two"],
            },
        ],
    )

    payload = build_label_ledger(
        certificate_file=certificates,
        recall_log_file=recalls,
        pull_log_file=pulls,
        golden_file=golden,
    )
    used_splits = {
        row["split"]
        for row in payload["labels"]
        if row["provenance"]["source"] == "recall_used"
    }

    assert len(used_splits) == 1
    assert payload["gates"]["field_learning_allowed"] is False
    assert payload["gates"]["calibration_allowed"] is False


def test_same_query_across_sessions_never_crosses_temporal_split(tmp_path) -> None:
    from chronovisor.recall.recall_label_factory import assign_temporal_splits

    labels = [
        {
            "label_id": "one",
            "session_hash": "session-a",
            "query_sha256": "shared-query",
            "observed_at": "2026-01-01T00:00:00Z",
        },
        {
            "label_id": "two",
            "session_hash": "session-b",
            "query_sha256": "shared-query",
            "observed_at": "2026-07-01T00:00:00Z",
        },
        {
            "label_id": "three",
            "session_hash": "session-c",
            "query_sha256": "other-query",
            "observed_at": "2026-07-02T00:00:00Z",
        },
    ]

    assigned = assign_temporal_splits(labels)

    shared = {row["split"] for row in assigned if row["query_sha256"] == "shared-query"}
    assert len(shared) == 1


def test_relation_labels_cannot_inflate_page_learning_gate(tmp_path) -> None:
    certificates = tmp_path / "certificates.jsonl"
    recalls = tmp_path / "recall.jsonl"
    pulls = tmp_path / "pull.jsonl"
    golden = tmp_path / "golden.jsonl"
    receipts = tmp_path / "receipts.jsonl"
    paths = tmp_path / "paths.jsonl"
    entities = tmp_path / "entities.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    for path in (certificates, recalls, pulls, golden, paths, entities, rubrics):
        write_rows(path, [])
    write_rows(
        receipts,
        [
            {
                "relation_id": f"relation-{index}",
                "receipt_id": f"receipt-{index}",
                "outcome": "verified",
            }
            for index in range(500)
        ],
    )

    payload = build_label_ledger(
        certificate_file=certificates,
        recall_log_file=recalls,
        pull_log_file=pulls,
        golden_file=golden,
        relation_receipt_file=receipts,
        relation_path_file=paths,
        entity_decision_file=entities,
        rubric_outcome_file=rubrics,
    )

    assert payload["counts"]["strong_positive"] == 0
    assert payload["counts_by_split"]["unassigned"]["total"] == 500
    assert sum(row["subject_kind"] == "relation" for row in payload["labels"]) == 500
    assert payload["gates"]["field_learning_allowed"] is False
    assert all(row["subject_kind"] == "relation" for row in payload["labels"])


def test_used_relation_and_entity_paths_remain_silver_without_unlocking(tmp_path) -> None:
    certificates = tmp_path / "certificates.jsonl"
    recalls = tmp_path / "recall.jsonl"
    pulls = tmp_path / "pull.jsonl"
    golden = tmp_path / "golden.jsonl"
    receipts = tmp_path / "receipts.jsonl"
    paths = tmp_path / "paths.jsonl"
    entities = tmp_path / "entities.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    for path in (certificates, recalls, golden, receipts, entities, rubrics):
        write_rows(path, [])
    write_rows(
        pulls,
        [
            {
                "type": "used",
                "decision_id": "decision-entity",
                "session_id": "session-entity",
                "page_ids": ["target"],
            }
        ],
    )
    write_rows(
        paths,
        [
            {
                "decision_id": "decision-entity",
                "page_id": "target",
                "query_sha256": "q" * 64,
                "relation_ids": ["relation-used"],
                "entity_merge_ids": ["merge_entity"],
            }
        ],
    )

    payload = build_label_ledger(
        certificate_file=certificates,
        recall_log_file=recalls,
        pull_log_file=pulls,
        golden_file=golden,
        relation_receipt_file=receipts,
        relation_path_file=paths,
        entity_decision_file=entities,
        rubric_outcome_file=rubrics,
    )

    entity_labels = [
        row for row in payload["labels"] if row["subject_kind"] == "entity_merge"
    ]
    relation_labels = [
        row for row in payload["labels"] if row["subject_kind"] == "relation"
    ]
    assert entity_labels[0]["quality"] == "silver"
    assert entity_labels[0]["polarity"] == "exposure"
    assert relation_labels[0]["quality"] == "silver"
    assert relation_labels[0]["polarity"] == "exposure"
    assert payload["counts"]["strong_positive"] == 0
    assert payload["gates"]["field_learning_allowed"] is False
    assert payload["gates"]["relation_learning_allowed"] is False
    assert payload["gates"]["entity_learning_allowed"] is False


def test_explicit_feedback_and_relation_retraction_remain_opposing_events(
    tmp_path,
) -> None:
    inputs = {
        name: tmp_path / f"{name}.jsonl"
        for name in (
            "certificate_file",
            "recall_log_file",
            "pull_log_file",
            "golden_file",
            "relation_receipt_file",
            "relation_path_file",
            "entity_decision_file",
            "rubric_outcome_file",
            "feedback_file",
            "relation_event_file",
        )
    }
    for path in inputs.values():
        write_rows(path, [])
    active = {
        "kind": "page_ignored",
        "prompt": "explicit bad recall",
        "negative_pages": ["bad-page"],
        "frontier_reviewed": True,
        "content_correction_key": "active-key",
        "ref": "active-ref",
        "ts": "2026-08-01T00:00:00Z",
    }
    retracted = {
        "kind": "page_ignored",
        "prompt": "withdrawn feedback",
        "negative_pages": ["withdrawn-page"],
        "frontier_reviewed": True,
        "content_correction_key": "withdrawn-key",
        "ref": "withdrawn-ref",
    }
    write_rows(
        inputs["feedback_file"],
        [
            active,
            retracted,
            {
                "kind": "page_ignored_retracted",
                "target_kind": "page_ignored",
                "content_correction_key": "withdrawn-key",
                "target_feedback_sha256": feedback_row_sha256(retracted),
            },
            {"kind": "injection_ignored", "negative_pages": ["not-explicit"]},
        ],
    )
    write_rows(
        inputs["relation_receipt_file"],
        [
            {
                "relation_id": "relation-one",
                "receipt_id": "receipt-one",
                "outcome": "verified",
            }
        ],
    )
    write_rows(
        inputs["relation_event_file"],
        [
            {
                "event_id": "event-retract",
                "event_hash": "a" * 64,
                "action": "retract",
                "reason_code": "source correction",
                "created_at": "2026-08-01T00:00:01Z",
                "relation": {"relation_id": "relation-one"},
            }
        ],
    )

    payload = build_label_ledger(**inputs)
    labels = payload["labels"]

    assert not any(row["page_id"] == "bad-page" for row in labels)
    assert not any(
        row["page_id"] in {"withdrawn-page", "not-explicit"} for row in labels
    )
    relation_labels = [row for row in labels if row["subject_id"] == "relation-one"]
    assert {row["polarity"] for row in relation_labels} == {"positive", "negative"}
    assert any(
        row["provenance"]["source"] == "relation_opposing_event"
        for row in relation_labels
    )


def test_split_assignment_groups_content_and_quarantines_bad_timestamps() -> None:
    labels = [
        {
            "label_id": f"label-{index}",
            "session_hash": f"session-{index}",
            "query_sha256": f"query-{index}",
            "page_id": f"page-{index}",
            "page_uid": f"uid-{index}",
            "content_sha256": f"{index + 1:064x}",
            "observed_at": f"2026-07-{index + 1:02d}T00:00:00Z",
        }
        for index in range(10)
    ]
    labels.extend(
        [
            {
                "label_id": "same-content-alias",
                "session_hash": "different-session",
                "query_sha256": "different-query",
                "page_id": "renamed-page",
                "page_uid": "renamed-uid",
                "content_sha256": f"{1:064x}",
                "observed_at": "2026-07-10T12:00:00Z",
            },
            {
                "label_id": "undated",
                "session_hash": "undated-session",
                "query_sha256": "undated-query",
                "page_id": "undated-page",
                "observed_at": "",
            },
            {
                "label_id": "naive-time",
                "session_hash": "naive-session",
                "query_sha256": "naive-query",
                "page_id": "naive-page",
                "observed_at": "2026-08-01T00:00:00",
            },
        ]
    )

    assigned = assign_temporal_splits(labels)

    same_content = {
        row["split"]
        for row in assigned
        if row.get("content_sha256") == f"{1:064x}"
    }
    assert len(same_content) == 1
    assert next(row for row in assigned if row["label_id"] == "undated")[
        "split_diagnostic"
    ] == "legacy_undated"
    assert next(row for row in assigned if row["label_id"] == "naive-time")[
        "split_diagnostic"
    ] == "invalid_timestamp"
    assert any(row["split"] == "embargo" for row in assigned)


def test_only_sealed_train_answer_outcomes_unlock_page_labels(tmp_path) -> None:
    inputs = {
        name: tmp_path / f"{name}.jsonl"
        for name in (
            "certificate_file",
            "recall_log_file",
            "pull_log_file",
            "golden_file",
        )
    }
    for path in inputs.values():
        write_rows(path, [])

    def artifact(path, split: str, page_id: str, session: str) -> None:
        manifest = {"split": split}
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        write_sealed_json(
            path,
            {
                "schema_version": 1,
                "status": "passed",
                "manifest": manifest,
                "confidence_bound": {
                    "valid": True,
                    "method": "connected-cluster-bootstrap-percentile",
                    "lower": 0.1,
                },
                "gates": {"verified": True},
                "page_rewards": [
                    {
                        "page_id": page_id,
                        "content_sha256": "a" * 64,
                        "reward": 0.2,
                        "producer": "verified_answer_pair_v1",
                        "decision_id": f"decision-{split}",
                        "episode_id": f"episode-{split}",
                        "query_sha256": "b" * 64,
                        "session_hash": session,
                        "observed_at": "2026-07-01T00:00:00Z",
                    }
                ],
            },
        )

    locked = tmp_path / "locked.json"
    artifact(locked, "locked-test", "locked-page", "locked-session")
    locked_payload = build_label_ledger(**inputs, answer_outcome_file=locked)
    assert locked_payload["counts"]["strong_positive"] == 0
    assert locked_payload["counts_by_split"]["locked-test"][
        "outcome_grounded_positive"
    ] == 0

    train = tmp_path / "train.json"
    artifact(train, "train", "train-page", "train-session")
    train_payload = build_label_ledger(**inputs, answer_outcome_file=train)
    assert train_payload["counts"]["scope"] == "train"
    assert train_payload["counts"]["strong_positive"] == 0
