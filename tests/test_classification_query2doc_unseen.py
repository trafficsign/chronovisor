from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.classification import ClassificationError
from chronovisor.classification_query2doc_unseen import (
    MANUAL_GOLD_SCHEMA,
    SAMPLE_SIZE,
    lock_preregistration,
    prepare_selection,
    select_unseen_rows,
    unseen_gate_passed,
)
from chronovisor.durable_state import write_sealed_json


def _write_sources(root: Path) -> None:
    fixture_root = root / "classification" / "fixtures"
    fixture_root.mkdir(parents=True)
    rows = []
    for index in range(35):
        rows.append(
            {
                "uid": f"page-{index:02d}",
                "source_sha256": f"{index + 1:064x}",
                "title": f"Page {index}",
                "summary": "",
                "excerpt": f"Body {index}",
                "tags": [],
                "raw_keywords": [],
                "adjudication_status": "accepted",
                "gold_expected_status": "proposed",
            }
        )
    (fixture_root / "classification-dev-200.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    epoch_root = fixture_root / "epochs" / "epoch-3-library-evidence-v1"
    epoch_root.mkdir(parents=True)
    (epoch_root / "adjudication.jsonl").write_text(
        json.dumps({"uid": "page-00"}) + "\n",
        encoding="utf-8",
    )
    write_sealed_json(
        root / "classification" / "query2doc-pilot" / "evaluation.json",
        {
            "schema": "chronovisor.classification-query2doc-evaluation.v1",
            "cases": [{"uid": "page-01"}],
        },
        backup=False,
    )


def test_unseen_selection_is_deterministic_and_excludes_design_rows(
    tmp_path: Path,
) -> None:
    _write_sources(tmp_path)

    first = select_unseen_rows(tmp_path)
    second = select_unseen_rows(tmp_path)

    assert len(first) == SAMPLE_SIZE
    assert [row["uid"] for row in first] == [row["uid"] for row in second]
    assert "page-00" not in {row["uid"] for row in first}
    assert "page-01" not in {row["uid"] for row in first}


def test_lock_rejects_manual_gold_uid_drift(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    prepare_selection(tmp_path)
    manual = tmp_path / "manual-gold.json"
    manual.write_text(
        json.dumps(
            {
                "schema": MANUAL_GOLD_SCHEMA,
                "reviewer": "codex",
                "cases": [
                    {
                        "uid": "wrong-page",
                        "expected_primary_notations": ["004.8"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ClassificationError,
        match="UIDs do not match sealed selection",
    ):
        lock_preregistration(tmp_path, manual)


@pytest.mark.parametrize(
    ("fused", "raw_lexical", "raw_dense", "expected"),
    [
        (24, 20, 22, True),
        (23, 10, 10, False),
        (24, 25, 22, False),
    ],
)
def test_unseen_gate_is_fixed_and_not_worse_than_raw(
    fused: int,
    raw_lexical: int,
    raw_dense: int,
    expected: bool,
) -> None:
    assert (
        unseen_gate_passed(
            {
                "fused": {"hit_count": fused},
                "raw_lexical": {"hit_count": raw_lexical},
                "raw_dense": {"hit_count": raw_dense},
            }
        )
        is expected
    )
