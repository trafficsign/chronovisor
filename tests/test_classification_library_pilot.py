from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.jsonl_write import write_jsonl_atomic as _write_jsonl
from chronovisor.lab import classification_library_pilot
from chronovisor.lab.classification_library_pilot import (
    FIXTURE_EPOCH,
    _advance,
    _phase_e4_resource,
    _run_latin_square_decisions,
    build_parser,
    load_state,
    run_once,
    save_state,
)
from chronovisor.recall.classification import (
    ClassificationError,
    default_udc_package,
)
from chronovisor.recall.classification_engine import record_from_consensus
from chronovisor.recall.classification_fixture_set import fixture_set_paths


def test_new_state_and_terminal_status_are_idempotent(tmp_path: Path) -> None:
    assert load_state(tmp_path)["stage"] == "e0_baseline"
    state = {
        **load_state(tmp_path),
        "status": "awaiting_user",
        "stage": "awaiting_explicit_adoption",
    }
    save_state(tmp_path, state)

    result = run_once(root=tmp_path, repo_root=tmp_path)
    assert result["status"] == "awaiting_user"
    assert result["stage"] == "awaiting_explicit_adoption"


def test_run_once_records_stage_timing_and_blocks_after_three_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = {**load_state(tmp_path), "stage": "e2_index"}
    save_state(tmp_path, state)
    monkeypatch.setattr(
        classification_library_pilot,
        "_phase_e2_index",
        lambda root, current: _advance(root, current, "e3_candidates"),
    )

    result = run_once(root=tmp_path, repo_root=tmp_path)

    assert result["stage"] == "e3_candidates"
    assert result["stage_timings"]["e2_index"]["invocations"] == 1

    state = {
        **load_state(tmp_path),
        "status": "running",
        "stage": "e3_candidates",
    }
    save_state(tmp_path, state)

    def fail(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        classification_library_pilot,
        "_phase_e3_candidates",
        fail,
    )
    for expected in ("retrying", "retrying", "blocked"):
        with pytest.raises(RuntimeError, match="boom"):
            run_once(root=tmp_path, repo_root=tmp_path)
        assert load_state(tmp_path)["status"] == expected


def test_latin_square_executes_every_arm_in_every_position(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = [
        {"uid": f"uid-{index}", "source_sha256": f"sha256:{index}"}
        for index in range(6)
    ]
    paths = fixture_set_paths(tmp_path, FIXTURE_EPOCH)
    _write_jsonl(paths.dev, fixture)
    orders = (
        ["J1", "J2", "J3"],
        ["J2", "J3", "J1"],
        ["J3", "J1", "J2"],
    )
    rows_by_arm = {
        arm: [
            {
                **row,
                "latin_square_order": orders[index % len(orders)],
                "candidates": [{"notation": "004.8"}],
            }
            for index, row in enumerate(fixture)
        ]
        for arm in ("J1", "J2", "J3")
    }
    monkeypatch.setattr(
        classification_library_pilot,
        "_dev_paired_rows",
        lambda _root: rows_by_arm,
    )
    calls = []

    def consensus(rows, **kwargs):
        calls.append((kwargs["run_namespace"], [row["uid"] for row in rows]))
        return [
            {
                "uid": row["uid"],
                "status": "proposed",
                "primary_notation": "004.8",
            }
            for row in rows
        ]

    monkeypatch.setattr(
        classification_library_pilot,
        "run_consensus_batches",
        consensus,
    )

    result = _run_latin_square_decisions(tmp_path, split="dev")

    assert len(calls) == 9
    assert all(len(rows) == len(fixture) for rows in result.values())
    assert all(len(uids) == 2 for _namespace, uids in calls)


@pytest.mark.parametrize(
    ("resource_status", "expected_status"),
    [("passed", "running"), ("failed", "blocked")],
)
def test_resource_phase_is_fail_closed(
    tmp_path: Path,
    monkeypatch,
    resource_status: str,
    expected_status: str,
) -> None:
    state = {**load_state(tmp_path), "stage": "e4_resource"}
    save_state(tmp_path, state)
    monkeypatch.setattr(
        classification_library_pilot,
        "run_resource_burn",
        lambda *_args, **_kwargs: {"status": resource_status},
    )

    result = _phase_e4_resource(tmp_path, state)

    assert result["status"] == expected_status
    if resource_status == "passed":
        assert result["stage"] == "e5_dev"
    else:
        assert result["stage"] == "e4_resource"


def test_parser_requires_explicit_adoption_inputs() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "adopt",
            "--actor",
            "user",
            "--parent-phase4-receipt",
            "/tmp/phase4.json",
        ]
    )

    assert args.command == "adopt"
    assert args.actor == "user"
    assert args.parent_phase4_receipt == Path("/tmp/phase4.json")


def test_vnext_record_requires_and_preserves_authority_digest() -> None:
    page = {
        "source_sha256": "page",
        "page_type": "knowledge",
        "lifecycle": "active",
        "sensitivity": "normal",
    }
    decision = {
        "primary_notation": "004.8",
        "secondary_notations": [],
        "confidence": 0.9,
        "consensus_sha256": "consensus",
    }
    with pytest.raises(ClassificationError, match="authority digest"):
        record_from_consensus(
            page,
            decision,
            package=default_udc_package(),
            authority_epoch=3,
            status="proposed",
        )

    record = record_from_consensus(
        page,
        decision,
        package=default_udc_package(),
        authority_epoch=3,
        status="proposed",
        authority_digest="sha256:authority",
    )

    assert record.to_dict()["classifier_authority_digest"] == "sha256:authority"


def test_complete_state_runs_supervisor_and_rolls_back_on_breach(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = {**load_state(tmp_path), "status": "complete", "stage": "complete"}
    save_state(tmp_path, state)
    adopted = tmp_path / "adopted.json"
    adopted.write_text("{}", encoding="utf-8")
    write_sealed_json(
        tmp_path / "classification" / "library-evidence" / "receipts" / "phase-e8.json",
        {
            "schema": "chronovisor.classification-library-phase-receipt.v1",
            "adopted_manifest": str(adopted),
        },
    )
    monkeypatch.setattr(
        classification_library_pilot,
        "probe_decision_only_authority",
        lambda *_args, **_kwargs: {
            "schema": "chronovisor.classification-authority-probe.v1",
            "status": "critical-breach",
        },
    )
    monkeypatch.setattr(
        classification_library_pilot,
        "rollback_authority",
        lambda _root: {"status": "disabled"},
    )

    result = run_once(root=tmp_path, repo_root=tmp_path)

    assert result["status"] == "blocked"
    assert result["stage"] == "authority_rolled_back"
    rollback = (
        tmp_path
        / "classification"
        / "library-evidence"
        / "supervisor"
        / "rollback-latest.json"
    )
    assert rollback.is_file()
