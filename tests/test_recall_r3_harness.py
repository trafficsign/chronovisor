from __future__ import annotations

import importlib.util
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
    assert result["claim"]["p95_ns"] <= HARNESS.CLAIM_P95_LIMIT_NS
    assert result["teacher_handoff"]["wall_time_ns"] <= HARNESS.TEACHER_HANDOFF_LIMIT_NS
    assert set(result["stages"]) == set(HARNESS.SIX_STAGES)
    assert result["stages"]["teacher"]["retry_wait"] == 1
    assert result["stages"]["retry_wait"]["retry_wait"] == 1
    assert result["sigterm_reopen"]["old_owner_rejected"] is True
    assert result["sigterm_reopen"]["idempotent_commit"] is True
    assert result["durability"]["receipt_coverage_pct"] >= 99
    assert result["durability"]["progress_coverage_pct"] >= 99
    assert result["durability"]["coverage"]["denominator"] == result["durability"]["coverage"]["receipts"]
    assert result["durability"]["progress_coverage"]["denominator"] == result["durability"]["progress_coverage"]["receipts"]
    assert result["durability"]["audit_status"] == "verified"
    assert result["duplicates"] == 0
    assert result["payload_free"] is True


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
