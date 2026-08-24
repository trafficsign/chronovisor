from __future__ import annotations

import importlib.util
import signal
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r3_harness_test_module", ROOT / "scripts" / "recall_r3_harness.py"
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


def test_r3_contract_constants_are_bounded() -> None:
    assert HARNESS.R3_SCHEMA == "chronovisor.recall-r3.v1"
    assert HARNESS.DEFAULT_SAMPLES == HARNESS.MIN_SAMPLES == 100
    assert HARNESS.CLAIM_P95_LIMIT_NS == 500_000_000
    assert HARNESS.TEACHER_HANDOFF_LIMIT_NS == 10_000_000_000
    assert HARNESS.OX_WORKSET_RELATIVE.as_posix() == (
        "runtime/recall-distillation/ox-workset.sqlite3"
    )
    assert HARNESS.OX_WORKSET_EXPECTED_ROWS == 32_522
    assert HARNESS.OX_WORKSET_EXPECTED_STATES == {
        "ready": 19_400,
        "leased": 0,
        "completed": 152,
        "quarantined": 12_970,
    }


def test_p95_uses_nearest_rank() -> None:
    assert HARNESS._p95(list(range(1, 21))) == 19
    assert HARNESS._p95([10, 1, 5, 2, 9]) == 10


def test_root_matrix_rejects_overlap_and_symlink(tmp_path: Path) -> None:
    production = tmp_path / "production"
    source = tmp_path / "source"
    production.mkdir()
    source.mkdir()
    output = tmp_path / "output"
    HARNESS._assert_root_matrix(production, source, output)
    with pytest.raises(HARNESS.R3Error, match="overlap"):
        HARNESS._assert_root_matrix(production, production / "nested", output)
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(HARNESS.R3Error, match="symlink"):
        HARNESS._assert_root_matrix(link, production, output)


def test_output_tree_rejects_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(HARNESS.R3Error, match="symlink"):
        HARNESS._assert_output_safe(output)


def test_payload_free_guard_rejects_payload_fields() -> None:
    HARNESS._assert_payload_free({"payload_free": True, "count": 1})
    with pytest.raises(HARNESS.R3Error, match="payload"):
        HARNESS._assert_payload_free({"payload": "private raw"})


def test_run_workset_covers_durability_and_recovery(tmp_path: Path) -> None:
    from chronovisor.recall import recall_distillation_workset as workset

    result = HARNESS._run_workset(workset, ROOT, tmp_path, HARNESS.UNIT_MIN_SAMPLES)
    assert result["fairness"]["passed"] is True
    assert result["cross_kind_fairness"] is True
    assert result["claim"]["p95_ns"] <= HARNESS.CLAIM_P95_LIMIT_NS
    assert result["claim"]["successful_count"] == HARNESS.UNIT_MIN_SAMPLES
    assert result["claim"]["observation_calls"] == HARNESS.UNIT_MIN_SAMPLES + 1
    assert result["claim"]["final_empty_excluded"] is True
    assert result["teacher_handoff"]["wall_time_ns"] <= HARNESS.TEACHER_HANDOFF_LIMIT_NS
    assert result["teacher_handoff"]["receiptized"] is True
    assert result["teacher_handoff"]["completed"] == HARNESS.UNIT_MIN_SAMPLES
    assert result["teacher_handoff"]["dispatcher"] == "single-teacher-v1"
    assert result["teacher_handoff"]["lease_observed"] is True
    assert result["teacher_handoff"]["process_returncode"] == 0
    assert set(result["stages"]) == set(HARNESS.SIX_STAGES)
    assert result["stages"]["teacher"]["retry_wait"] == 1
    assert result["stages"]["retry_wait"]["retry_wait"] == 1
    assert result["retry_wait"]["count"] == 1
    assert result["retry_wait"]["next_retry_in_seconds"] == 30
    assert result["sigterm_reopen"]["old_owner_rejected"] is True
    assert result["sigterm_reopen"]["idempotent_commit"] is True
    assert result["sigterm_reopen"]["child_returncode"] == -signal.SIGTERM
    assert result["sigterm_reopen"]["sigterm_process"]["asserted"] is True
    assert result["durability"]["receipt_coverage_pct"] >= 99
    assert result["durability"]["progress_coverage_pct"] >= 99
    assert result["durability"]["coverage"]["denominator"] == result["durability"]["coverage"]["receipts"]
    assert result["durability"]["progress_coverage"]["denominator"] == result["durability"]["progress_coverage"]["receipts"]
    assert result["durability"]["audit_status"] == "verified"
    assert result["duplicates"] == 0
    assert result["payload_free"] is True


def test_clone_inventory_is_bounded_and_payload_free(tmp_path: Path) -> None:
    from chronovisor.recall import recall_distillation_workset as workset

    path = tmp_path / "runtime" / "recall-distillation" / "ox-workset.sqlite3"
    queue = workset.DistillationWorkset(path)
    queue.advance([HARNESS._item("inventory-1", "ox")], {"source": "test"})
    inventory = HARNESS._clone_workset_inventory(path)
    assert inventory["row_count"] == 1
    assert inventory["unique_work_ids"] == 1
    assert inventory["bounded"] is True
    assert inventory["production_path_used"] is False
    HARNESS._assert_payload_free(inventory)


def test_clone_cycles_migrate_legacy_ox_schema_and_preserve_counts(
    tmp_path: Path,
) -> None:
    from chronovisor.recall import recall_distillation_workset as workset

    path = tmp_path / HARNESS.OX_WORKSET_RELATIVE
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE work_items (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                work_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                payload_ref TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                temporal_split_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                priority INTEGER NOT NULL,
                watermark_json TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error_class TEXT NOT NULL DEFAULT '',
                lease_id TEXT,
                lease_owner TEXT,
                lease_expires_at REAL,
                completion_ref TEXT NOT NULL DEFAULT '',
                completion_digest TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE workset_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            """
        )
        rows = []
        for index in range(HARNESS.OX_WORKSET_EXPECTED_ROWS):
            state = (
                "ready"
                if index < 19_400
                else "quarantined"
                if index < 19_400 + 12_970
                else "completed"
            )
            completed = state == "completed"
            rows.append(
                (
                    f"legacy-{index}",
                    "ox",
                    f"candidate-ledger:legacy-{index}",
                    "a" * 64,
                    "{}",
                    "{}",
                    0,
                    "{}",
                    state,
                    0,
                    "",
                    None,
                    None,
                    None,
                    f"label-ledger:legacy-{index}" if completed else "",
                    "b" * 64 if completed else "",
                    1,
                    1,
                )
            )
        connection.executemany(
            "INSERT INTO work_items ("
            "work_id, kind, payload_ref, payload_digest, temporal_split_json, "
            "provenance_json, priority, watermark_json, state, attempt_count, "
            "last_error_class, lease_id, lease_owner, lease_expires_at, "
            "completion_ref, completion_digest, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.execute("INSERT INTO workset_state VALUES ('watermark', '{}')")

    legacy = HARNESS._clone_workset_inventory(
        path, expected_rows=HARNESS.OX_WORKSET_EXPECTED_ROWS
    )
    assert legacy["states"] == {
        "completed": 152,
        "quarantined": 12_970,
        "ready": 19_400,
    }
    assert legacy["schema"]["receipt_table_present"] is False
    result = HARNESS._run_clone_workset_cycles(workset, tmp_path, cycles=100)
    assert result["legacy_status"]["states"] == {
        "ready": 19_400,
        "leased": 0,
        "completed": 152,
        "quarantined": 12_970,
    }
    assert result["migration"]["schema_before"]["receipt_table_present"] is False
    assert result["migration"]["schema_after"]["receipt_table_present"] is True
    assert result["migration"]["status_unchanged"] is True
    assert result["migration"]["status_before_cycles"]["leased"] == 0
    assert result["receipt_chain_verified"] is True
    assert result["audit_status"] == "legacy-unverified"
    assert result["successful_cycles"] == 100
    assert result["observation_calls"] == 100
    assert result["empty_probe"]["kind"] == "r3-empty-probe"
    assert result["empty_probe"]["excluded_from_p95"] is True
    HARNESS._assert_payload_free(result)


def test_source_clean_guard_fails_closed() -> None:
    HARNESS._assert_source_clean({"git_status_count": 0}, when="test")
    with pytest.raises(HARNESS.R3Error, match="dirty"):
        HARNESS._assert_source_clean({"git_status_count": 1}, when="test")


def test_run_once_disables_bytecode_for_guarded_window(monkeypatch) -> None:
    seen: list[bool] = []

    def guarded(**_kwargs):
        seen.append(sys.dont_write_bytecode)
        return {"ok": True}

    monkeypatch.setattr(HARNESS, "_run_once_guarded", guarded)
    previous = sys.dont_write_bytecode
    assert HARNESS._run_once(
        production=Path("/tmp/production"),
        source_root=Path("/tmp/source"),
        source_commit="0" * 40,
        output=Path("/tmp/output"),
        samples=HARNESS.MIN_SAMPLES,
    ) == {"ok": True}
    assert seen == [True]
    assert sys.dont_write_bytecode is previous


def test_run_workset_rejects_under_sampled_formal_run(tmp_path: Path) -> None:
    from chronovisor.recall import recall_distillation_workset as workset

    with pytest.raises(HARNESS.R3Error, match="at least"):
        HARNESS._run_workset(workset, ROOT, tmp_path, HARNESS.UNIT_MIN_SAMPLES - 1)


def test_main_fails_closed_for_isolated_root(tmp_path: Path, capsys) -> None:
    result = HARNESS.main(
        [
            "--production-root",
            str(tmp_path),
            "--source-root",
            str(tmp_path),
            "--source-commit",
            "0" * 40,
            "--output",
            str(tmp_path / "out"),
            "--isolated-root",
            str(tmp_path / "isolated"),
        ]
    )
    assert result == 2
    assert "isolated-root" in capsys.readouterr().err
