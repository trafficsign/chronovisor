from __future__ import annotations

import json
from pathlib import Path

from chronovisor.classification_query2doc_v2_unseen import (
    FIXTURE_EPOCH,
    PASS_HITS,
    select_unseen_rows,
    unseen_gate_passed,
)
from chronovisor.durable_state import write_sealed_json


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _candidate(uid: str, group_id: str) -> dict[str, object]:
    return {
        "uid": uid,
        "source_sha256": f"sha-{uid}",
        "fixture_group_id": group_id,
        "fixture_group_basis": "raw-provenance",
        "title": uid,
        "lifecycle": "active",
        "sensitivity": "normal",
    }


def test_selection_excludes_prior_provenance_groups(tmp_path: Path) -> None:
    write_sealed_json(
        tmp_path / "classification" / "query2doc-pilot" / "evaluation.json",
        {"cases": [{"uid": "old-fixed"}]},
    )
    write_sealed_json(
        tmp_path / "classification" / "query2doc-unseen" / "selection.json",
        {"cases": [{"uid": "old-unseen"}]},
    )
    write_sealed_json(
        tmp_path / "classification" / "query2doc-v2-unseen" / "selection.json",
        {"cases": [{"uid": "old-v2-unseen"}]},
    )
    write_sealed_json(
        tmp_path / "classification" / "query2doc-v2-2-unseen" / "selection.json",
        {"cases": [{"uid": "old-v2-2-unseen"}]},
    )
    epoch = (
        tmp_path
        / "classification"
        / "fixtures"
        / "epochs"
        / FIXTURE_EPOCH
    )
    _write_jsonl(epoch / "adjudication.jsonl", [{"uid": "old-adjudicated"}])
    _write_jsonl(
        epoch / "candidates.jsonl",
        [
            _candidate("old-fixed", "group-fixed"),
            _candidate("same-fixed-group", "group-fixed"),
            _candidate("old-unseen", "group-unseen"),
            _candidate("old-v2-unseen", "group-v2-unseen"),
            _candidate("same-v2-unseen-group", "group-v2-unseen"),
            _candidate("old-v2-2-unseen", "group-v2-2-unseen"),
            _candidate("same-v2-2-unseen-group", "group-v2-2-unseen"),
            _candidate("old-adjudicated", "group-adjudicated"),
            _candidate("new-a", "group-a"),
            _candidate("new-a-duplicate", "group-a"),
            _candidate("new-b", "group-b"),
            _candidate("new-c", "group-c"),
        ],
    )

    first = select_unseen_rows(tmp_path, sample_size=2)
    second = select_unseen_rows(tmp_path, sample_size=2)

    assert first == second
    assert len({row["fixture_group_id"] for row in first}) == 2
    assert {
        row["fixture_group_id"] for row in first
    }.isdisjoint(
        {
            "group-fixed",
            "group-unseen",
            "group-v2-unseen",
            "group-v2-2-unseen",
            "group-adjudicated",
        }
    )


def test_unseen_gate_requires_absolute_and_raw_noninferiority() -> None:
    assert unseen_gate_passed(
        {
            "fused": {"hit_count": PASS_HITS},
            "raw_lexical": {"hit_count": PASS_HITS - 1},
            "raw_dense": {"hit_count": 1},
        }
    )
    assert not unseen_gate_passed(
        {
            "fused": {"hit_count": PASS_HITS - 1},
            "raw_lexical": {"hit_count": 1},
            "raw_dense": {"hit_count": 1},
        }
    )
    assert not unseen_gate_passed(
        {
            "fused": {"hit_count": PASS_HITS},
            "raw_lexical": {"hit_count": PASS_HITS + 1},
            "raw_dense": {"hit_count": 1},
        }
    )
