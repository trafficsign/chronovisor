from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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


def _split_item(work_id: str, plan_id: str) -> dict[str, object]:
    item = _item(work_id)
    item["temporal_split"] = {
        "as_of": "2026-08-20T00:00:00Z",
        "group_id": "group-1",
        "split": "train",
        "split_plan_id": plan_id,
    }
    return item


def test_advance_is_idempotent_and_rejects_identity_mutation(tmp_path: Path) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)

    assert workset.watermark() is None
    assert workset.advance([_item("one")], {"source": 1}) == {
        "inserted": 1,
        "existing": 0,
        "watermark": {"source": 1},
    }
    assert workset.watermark() == {"source": 1}
    assert workset.advance([_item("one")], {"source": 2})["existing"] == 1
    assert workset.watermark() == {"source": 2}
    changed = _item("one")
    changed["payload_ref"] = "candidate-ledger:changed"

    with pytest.raises(DistillationWorksetError, match="identity conflict"):
        workset.advance([changed], {"source": 3})
    assert workset.watermark() == {"source": 2}
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT value_json FROM workset_state WHERE key = 'watermark'"
        ).fetchone() == ('{"source":2}',)


@pytest.mark.parametrize("quarantined", [False, True])
def test_advance_rebinds_unfinished_identical_split_to_new_plan(
    tmp_path: Path, quarantined: bool
) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)
    workset.advance([_split_item("one", "a" * 64)], {"source": 1})
    if quarantined:
        claim = workset.claim("label", 1, "ox-1", 60)[0]
        workset.commit(
            [claim],
            [{"status": "quarantined", "error_class": "invalid_response"}],
        )

    result = workset.advance([_split_item("one", "b" * 64)], {"source": 2})

    assert result["existing"] == 1
    with sqlite3.connect(path) as connection:
        temporal, state, attempt, error = connection.execute(
            "SELECT temporal_split_json, state, attempt_count, last_error_class "
            "FROM work_items WHERE work_id = 'one'"
        ).fetchone()
    assert '"split_plan_id":"' + "b" * 64 + '"' in temporal
    assert (state, attempt, error) == (
        ("quarantined", 1, "invalid_response") if quarantined else ("ready", 0, "")
    )


def test_advance_refuses_split_plan_rebind_for_leased(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_split_item("one", "a" * 64)], {"source": 1})
    workset.claim("label", 1, "ox-1", 60)

    with pytest.raises(DistillationWorksetError, match="identity conflict"):
        workset.advance([_split_item("one", "b" * 64)], {"source": 2})
    assert workset.watermark() == {"source": 1}


def test_advance_completed_split_plan_rebind_keeps_completed_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workset.sqlite3"
    now = [100.0]
    workset = DistillationWorkset(path, clock=lambda: now[0])
    workset.advance([_split_item("one", "a" * 64)], {"source": 1})
    claim = workset.claim("label", 1, "ox-1", 60)[0]
    workset.commit([claim], [_completed()])
    with sqlite3.connect(path) as connection:
        before = connection.execute(
            "SELECT temporal_split_json, state, attempt_count, completion_ref, "
            "completion_digest, created_at, updated_at "
            "FROM work_items WHERE work_id = 'one'"
        ).fetchone()

    now[0] = 200.0
    assert workset.advance([_split_item("one", "b" * 64)], {"source": 2}) == {
        "inserted": 0,
        "existing": 1,
        "watermark": {"source": 2},
    }

    with sqlite3.connect(path) as connection:
        after = connection.execute(
            "SELECT temporal_split_json, state, attempt_count, completion_ref, "
            "completion_digest, created_at, updated_at "
            "FROM work_items WHERE work_id = 'one'"
        ).fetchone()
    assert after == before


def test_advance_refuses_semantic_split_change(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_split_item("one", "a" * 64)], {"source": 1})
    changed = _split_item("one", "b" * 64)
    assert isinstance(changed["temporal_split"], dict)
    changed["temporal_split"]["split"] = "test"

    with pytest.raises(DistillationWorksetError, match="identity conflict"):
        workset.advance([changed], {"source": 2})
    assert workset.watermark() == {"source": 1}


def test_advance_split_rebind_rejects_tampered_metadata(tmp_path: Path) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)
    workset.advance([_split_item("one", "a" * 64)], {"source": 1})
    tampered = _split_item("one", "a" * 64)["temporal_split"]
    assert isinstance(tampered, dict)
    tampered["note"] = "/tmp/private"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE work_items SET temporal_split_json = ? WHERE work_id = 'one'",
            (json.dumps(tampered, sort_keys=True, separators=(",", ":")),),
        )

    with pytest.raises(DistillationWorksetError, match="identity conflict"):
        workset.advance([_split_item("one", "b" * 64)], {"source": 2})
    assert workset.watermark() == {"source": 1}


def test_advance_split_rebind_rolls_back_with_later_conflict(tmp_path: Path) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)
    workset.advance(
        [
            _split_item("one", "a" * 64),
            _split_item("two", "a" * 64),
            _split_item("three", "a" * 64),
        ],
        {"source": 1},
    )
    claim = workset.claim("label", 1, "ox-1", 60)[0]
    workset.commit([claim], [_completed()])
    conflicting = _split_item("three", "b" * 64)
    conflicting["payload_ref"] = "candidate-ledger:changed"

    with pytest.raises(DistillationWorksetError, match="identity conflict"):
        workset.advance(
            [
                _split_item("one", "b" * 64),
                _split_item("two", "b" * 64),
                conflicting,
            ],
            {"source": 2},
        )

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT work_id, temporal_split_json, state FROM work_items "
            "WHERE work_id IN ('one', 'two') ORDER BY work_id"
        ).fetchall()
    assert [(work_id, state) for work_id, _, state in rows] == [
        ("one", "completed"),
        ("two", "ready"),
    ]
    assert all(
        '"split_plan_id":"' + "a" * 64 + '"' in temporal for _, temporal, _ in rows
    )
    assert workset.watermark() == {"source": 1}


def test_failed_advance_keeps_prior_watermark(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_item("one")], {"source": 1})
    unsafe = _item("unsafe")
    unsafe["provenance"] = {"source": "/tmp/private"}

    with pytest.raises(DistillationWorksetError):
        workset.advance([unsafe], {"source": 2})

    assert workset.watermark() == {"source": 1}
    assert workset.status()["total"] == 1


def test_watermark_is_validated_and_returned_as_a_fresh_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)
    expected = {"source": 1, "nested": {"values": ["safe"]}}
    workset.advance([_item("one")], expected)

    value = workset.watermark()
    assert value == expected
    assert isinstance(value, dict)
    assert isinstance(value["nested"], dict)
    assert isinstance(value["nested"]["values"], list)
    value["nested"]["values"].append("mutated")
    assert workset.watermark() == expected


@pytest.mark.parametrize(
    "tampered",
    [
        "{",
        '{"secret":"private"}',
        '{"source":"/tmp/private"}',
        sqlite3.Binary(b"{}"),
    ],
)
def test_watermark_tampering_fails_closed(tmp_path: Path, tampered: object) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)
    workset.advance([_item("one")], {"source": 1})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE workset_state SET value_json = ? WHERE key = 'watermark'",
            (tampered,),
        )

    with pytest.raises(DistillationWorksetError):
        workset.watermark()


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


def test_release_unattempted_preserves_attempt_budget(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_item("one")], 1)
    original = workset.claim("label", 1, "ox-1", 60)[0]

    with pytest.raises(DistillationWorksetError, match="ownership lost"):
        workset.release_unattempted([replace(original, payload_digest="c" * 64)])
    assert workset.status("label")["leased"] == 1
    assert workset.release_unattempted([original]) == 1
    reclaimed = workset.claim("label", 1, "ox-2", 60)[0]

    assert reclaimed.attempt == 1
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


def test_commit_distinguishes_retry_quarantine_and_label_invalid_output(
    tmp_path: Path,
) -> None:
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


def test_payload_bodies_and_invalid_teacher_completion_are_rejected(
    tmp_path: Path,
) -> None:
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


@pytest.mark.parametrize("lease_seconds", [math.nan, math.inf, -math.inf])
def test_claim_rejects_nonfinite_lease_without_stranding_work(
    tmp_path: Path, lease_seconds: float
) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_item("one")], 1)

    with pytest.raises(DistillationWorksetError, match="finite number"):
        workset.claim("label", 1, "ox-1", lease_seconds)

    assert workset.status("label")["ready"] == 1
    assert workset.status("label")["leased"] == 0


@pytest.mark.parametrize("clock_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_clock_is_rejected_before_mutation(
    tmp_path: Path, clock_value: float
) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path, clock=lambda: clock_value)

    with pytest.raises(DistillationWorksetError, match="finite number"):
        workset.advance([_item("one")], 1)

    assert workset.status()["total"] == 0


def test_lease_overflow_is_rejected_without_stranding_work(tmp_path: Path) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path, clock=lambda: 1e308)
    workset.advance([_item("one")], 1)

    with pytest.raises(DistillationWorksetError, match="lease expiry"):
        workset.claim("label", 1, "ox-1", 1e308)

    assert workset.status("label")["ready"] == 1


def test_sqlite_database_and_sidecars_are_private_and_existing_mode_is_narrowed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workset.sqlite3"
    workset = DistillationWorkset(path)
    workset.advance([_item("one")], 1)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600

    os.chmod(path, 0o644)
    DistillationWorkset(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_permission_failure_rolls_back_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    original = workset._secure_sqlite_files
    calls = 0

    def fail_before_commit() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise DistillationWorksetError("cannot secure SQLite path")
        original()

    monkeypatch.setattr(workset, "_secure_sqlite_files", fail_before_commit)
    with pytest.raises(DistillationWorksetError, match="cannot secure"):
        workset.advance([_item("one")], 1)

    monkeypatch.setattr(workset, "_secure_sqlite_files", original)
    assert workset.status()["total"] == 0


def test_sqlite_symlink_and_nonregular_paths_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"not a workset")
    link = tmp_path / "workset.sqlite3"
    link.symlink_to(target)
    with pytest.raises(DistillationWorksetError, match="symlink"):
        DistillationWorkset(link)

    directory = tmp_path / "directory.sqlite3"
    directory.mkdir()
    with pytest.raises(DistillationWorksetError, match="regular file"):
        DistillationWorkset(directory)


def test_commit_rejects_duplicate_work_and_lease_ids_before_transaction(
    tmp_path: Path,
) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_item("one"), _item("two")], 1)
    first, second = workset.claim("label", 2, "ox-1", 60)

    with pytest.raises(DistillationWorksetError, match="duplicate work_id"):
        workset.commit([first, first], [_completed(), _completed()])
    assert workset.status("label")["leased"] == 2

    duplicate_lease = replace(second, lease_id=first.lease_id)
    with pytest.raises(DistillationWorksetError, match="duplicate lease_id"):
        workset.commit([first, duplicate_lease], [_completed(), _completed()])
    assert workset.status("label")["leased"] == 2


@pytest.mark.parametrize(
    "payload_ref",
    [
        "/tmp/private",
        "../private",
        "C:\\private\\payload",
        "https://example.test/secret",
    ],
)
def test_payload_reference_paths_are_rejected_without_persistence(
    tmp_path: Path, payload_ref: str
) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    item = _item("unsafe")
    item["payload_ref"] = payload_ref

    with pytest.raises(DistillationWorksetError, match="safe ledger reference"):
        workset.advance([item], 1)
    assert workset.status()["total"] == 0


def test_main_candidate_snapshot_reference_is_allowed(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    item = _item("snapshot")
    item["payload_ref"] = "candidate-snapshot:" + "a" * 64 + ":" + "b" * 64
    assert workset.advance([item], 1)["inserted"] == 1


def test_profile_provenance_key_is_allowed(tmp_path: Path) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    item = _item("profile")
    item["provenance"] = {
        "profile": "ox-alpha-single-v1",
        "route": "opencode-go/ox-alpha-free",
    }
    assert workset.advance([item], 1)["inserted"] == 1


@pytest.mark.parametrize(
    "provenance",
    [
        {"api_key": "private"},
        {"note": "private secret value"},
        {"source": "/Users/private/raw.json"},
        {"source": "relative/path.json"},
        {"blob": "x" * 257},
    ],
)
def test_provenance_is_bounded_and_payload_free(
    tmp_path: Path, provenance: dict[str, object]
) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    item = _item("unsafe-provenance")
    item["provenance"] = provenance

    with pytest.raises(DistillationWorksetError):
        workset.advance([item], 1)
    assert workset.status()["total"] == 0


def test_error_class_and_completion_reference_cannot_persist_payloads(
    tmp_path: Path,
) -> None:
    workset = DistillationWorkset(tmp_path / "workset.sqlite3")
    workset.advance([_item("one")], 1)
    claim = workset.claim("label", 1, "ox-1", 60)[0]

    with pytest.raises(DistillationWorksetError):
        workset.commit(
            [claim],
            [{"status": "retry", "error_class": "private secret value"}],
        )
    with pytest.raises(DistillationWorksetError):
        workset.commit(
            [claim],
            [
                {
                    **_completed(),
                    "completion_ref": "/tmp/private-label",
                }
            ],
        )

    assert workset.status()["leased"] == 1
    with sqlite3.connect(tmp_path / "workset.sqlite3") as connection:
        row = connection.execute(
            "SELECT last_error_class, completion_ref FROM work_items WHERE work_id = 'one'"
        ).fetchone()
    assert row == ("", "")
