from __future__ import annotations

from chronovisor.research.research_verification import run_verification


def test_independent_verifier_is_temp_only_and_reports_commands() -> None:
    result = run_verification()
    assert result["status"] == "PASS"
    assert result["failed"] == 0
    assert result["mutation_scope"] == "temporary-directory-only"
    assert all(row["command"] and row["status"] == "PASS" for row in result["checks"])
