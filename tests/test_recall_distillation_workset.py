from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chronovisor.recall.recall_distillation_workset import (
    DistillationWorkset,
    DistillationWorksetError,
)


def _item(work_id: str, *, priority: int = 0, kind: str = "label") -> dict[str, object]:
    return {
        "work_id": work_id,
        "kind": kind,
        "payload_ref": f"candidate-ledger:{work_id}",
        "payload_digest": "a" * 64,
        "priority": priority,
        "temporal_split": {"partition": "train", "cutoff": "2026-08-20"},
        "provenance": {"teacher_cohort": "ox-alpha-backfill-v1"},
    }


def _completed() -> dict[str, str]:
    return {
        "status": "completed",
        "completion_ref": "label-ledger:row-1",
        "completion_digest": "b" * 64,
    }


def test_advance_is_idempotent_and_rejects_identity_mutation(tmp_path: Path) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)

    assert workset.advance([_item("one")], {"source": 1}) == {
        "inserted": 1,
        "existing": 0,
        "watermark": {"source": 1},
    }
    assert workset.advance([_item("one")], {"source": 2})["existing"] == 1
    changed = _item("one")
    changed["payload_ref"] = "candidate-ledger:changed"

    with pytest.raises(DistillationWorksetError, match="identity conflict"):
        workset.advance([changed], {"source": 3})
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT value_json FROM workset_state WHERE key = 'watermark'"
        ).fetchone() == ('{"source":2}',)


def test_claim_obeys_priority_then_fifo_and_preserves_metadata(tmp_path: Path) -> None:
    now = [100.0]
    workset = DistillationWorkset(tmp_path / "workset.sqlite3", clock=lambda: now[0])
    workset.advance([_item("first"), _item("urgent", priority=1), _item("second")], 3)

    claims = workset.claim("label", 3, "ox-1", 10)

    assert [claim.work_id for claim in claims] == ["urgent", "first", "second"]
    assert claims[1].temporal_split == {"partition": "train", "cutoff": "2026-08-20"}
    assert claims[1].provenance == {"teacher_cohort": "ox-alpha-backfill-v1"}
    assert claims[1].attempt == 1


def test_expired_lease_is_reclaimed_and_attempt_increments(tmp_path: Path) -> None:
    now = [100.0]
    workset = DistillationWorkset(tmp_path / "workset.sqlite3", clock=lambda: now[0])
    workset.advance([_item("one")], 1)
    original = workset.claim("label", 1, "ox-1", 10)[0]
    now[0] = 110.0

    reopened = DistillationWorkset(tmp_path / "workset.sqlite3", clock=lambda: now[0])
    reclaimed = reopened.claim("label", 1, "ox-2", 10)[0]

    assert reclaimed.work_id == original.work_id
    assert reclaimed.owner == "ox-2"
    assert reclaimed.attempt == 2
    with pytest.raises(DistillationWorksetError, match="ownership lost"):
        workset.commit([original], [_completed()])


def test_concurrent_claims_are_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)
    workset.advance([_item(f"work-{index}") for index in range(20)], 1)

    def claim(worker: int) -> tuple[object, ...]:
        return DistillationWorkset(path).claim("label", 3, f"worker-{worker}", 60)

    with ThreadPoolExecutor(max_workers=8) as executor:
        batches = list(executor.map(claim, range(8)))

    claims = [claim for batch in batches for claim in batch]
    assert len(claims) == 20
    assert len({claim.work_id for claim in claims}) == 20
    assert {claim.attempt for claim in claims} == {1}


def test_completed_commit_is_idempotent_and_conflicts_fail(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_item("one")], 1)
    claim = workset.claim("label", 1, "ox-1", 60)[0]
    outcome = _completed()

    assert workset.commit([claim], [outcome]) == {
        "completed": 1,
        "retry": 0,
        "quarantined": 0,
    }
    assert workset.commit([claim], [outcome]) == {
        "completed": 1,
        "retry": 0,
        "quarantined": 0,
    }
    with pytest.raises(DistillationWorksetError, match="completion identity conflict"):
        workset.commit(
            [claim],
            [{**outcome, "completion_ref": "label-ledger:other"}],
        )


def test_status_reports_backlog_by_kind_and_state(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance(
        [
            _item("ready-a"),
            _item("ready-b"),
            _item("other", kind="counterfactual"),
        ],
        1,
    )
    claims = workset.claim("label", 2, "ox-1", 60)

    assert workset.status("label") == {
        "ready": 0,
        "leased": 2,
        "completed": 0,
        "quarantined": 0,
        "backlog": 2,
        "total": 2,
    }
    workset.commit(
        claims,
        [
            _completed(),
            {"status": "quarantined", "error_class": "policy_veto"},
        ],
    )
    assert workset.status() == {
        "ready": 1,
        "leased": 0,
        "completed": 1,
        "quarantined": 1,
        "backlog": 1,
        "total": 3,
    }


def test_commit_distinguishes_retry_quarantine_and_label_invalid_output(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_item("retry"), _item("quarantine"), _item("invalid")], 3)
    claims = workset.claim("label", 3, "ox-1", 60)
    by_id = {claim.work_id: claim for claim in claims}

    totals = workset.commit(
        [by_id["retry"], by_id["quarantine"], by_id["invalid"]],
        [
            {"status": "retry", "error_class": "transport_timeout"},
            {"status": "quarantined", "error_class": "policy_veto"},
            {"status": "retry", "error_class": "invalid_teacher_output"},
        ],
    )

    assert totals == {"completed": 0, "retry": 2, "quarantined": 1}
    retry = workset.claim("label", 10, "ox-2", 60)
    assert [claim.work_id for claim in retry] == ["retry", "invalid"]


def test_payload_bodies_and_invalid_teacher_completion_are_rejected(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    with sqlite3.connect(tmp_path / "workset.sqlite3") as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(work_items)")
        }
    assert "payload" not in columns

    raw = _item("raw")
    raw["payload"] = "private raw text"
    with pytest.raises(DistillationWorksetError, match="unsupported fields"):
        workset.advance([raw], 1)

    workset.advance([_item("invalid")], 2)
    claim = workset.claim("label", 1, "ox-1", 60)[0]
    with pytest.raises(DistillationWorksetError, match="cannot have an error_class"):
        workset.commit(
            [claim],
            [
                {
                    **_completed(),
                    "error_class": "invalid_teacher_output",
                }
            ],
        )
    assert workset.status()["completed"] == 0
    assert workset.status()["leased"] == 1
