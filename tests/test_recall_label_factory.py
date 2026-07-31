from __future__ import annotations

import json

from chronovisor.recall.recall_label_factory import build_label_ledger


def write_rows(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


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
            }
        ],
    )

    payload = build_label_ledger(
        certificate_file=certificates,
        recall_log_file=recalls,
        pull_log_file=pulls,
        golden_file=golden,
    )
    labels = payload["labels"]

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
