from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_wiki_mcp import read_back_repair, recall_hints
from llm_wiki_mcp.convergence import CycleBudget


NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


def _approve(_proposal: dict) -> dict:
    return {
        "decision": "approved",
        "confidence": 0.95,
        "summary": "exact query hint is justified",
    }


def _write_failures(path: Path, failures: list[dict]) -> None:
    rows = [
        {
            "timestamp": f"2026-07-10T12:{index:02d}:00+0000",
            "checked": 1,
            "passed": 0,
            "failed": [failure],
        }
        for index, failure in enumerate(failures)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _allow_pages(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(recall_hints.wiki, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    monkeypatch.setattr(recall_hints.wiki, "SYSTEM_DIR", tmp_path / "system")


def test_failure_key_is_stable_across_case_whitespace_and_diagnostics() -> None:
    left = {
        "page_id": "Page-A",
        "reason": "not_in_top_results",
        "query": "  Where   is Recall? ",
        "top": ["x", "y"],
    }
    right = {
        "page_id": "page-a",
        "reason": "NOT-IN-TOP-RESULTS",
        "query": "where is recall?",
        "top": ["different"],
    }

    assert read_back_repair.failure_key(left) == read_back_repair.failure_key(right)


def test_not_in_top_results_applies_exact_query_hint_once(tmp_path: Path, monkeypatch) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    hints_file = tmp_path / "query-hints.json"
    failure = {
        "page_id": "target-page",
        "reason": "not-in-top-results",
        "query": "Where is the target page?",
        "top": ["other-page"],
    }
    _write_failures(failure_file, [failure, dict(failure)])
    _allow_pages(monkeypatch, tmp_path)

    first = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        now=NOW,
        reviewer=_approve,
    )
    second = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        now=NOW + timedelta(days=1),
        reviewer=_approve,
    )

    assert first["applied"] == 1
    assert first["observed_failures"] == 2
    assert second["processed"] == 0
    hints = recall_hints.load_query_hints(hints_file)
    assert len(hints) == 1
    assert hints[0]["count"] == 1
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    entry = next(iter(ledger["entries"].values()))
    assert entry["status"] == "applied"
    assert entry["occurrences"] == 2

    ledger_file.unlink()
    recovered = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        now=NOW + timedelta(days=2),
        reviewer=_approve,
    )
    assert recovered["already_present"] == 1
    assert recall_hints.load_query_hints(hints_file)[0]["count"] == 1


def test_query_hint_never_applies_without_frontier_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    hints_file = tmp_path / "query-hints.json"
    _write_failures(
        failure_file,
        [
            {
                "page_id": "target-page",
                "reason": "not-in-top-results",
                "query": "specific target query",
            }
        ],
    )
    _allow_pages(monkeypatch, tmp_path)

    result = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        now=NOW,
        reviewer=lambda _proposal: {
            "decision": "rejected",
            "confidence": 0.96,
            "summary": "query is not specific to the page",
        },
    )

    assert result["rejected"] == 1
    assert not hints_file.exists()
    entry = next(
        iter(json.loads(ledger_file.read_text(encoding="utf-8"))["entries"].values())
    )
    assert entry["status"] == "rejected"
    assert entry["frontier_review"]["decision"] == "rejected"


def test_approved_frontier_verdict_survives_crash_before_hint_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    hints_file = tmp_path / "query-hints.json"
    _write_failures(
        failure_file,
        [
            {
                "page_id": "target-page",
                "reason": "not-in-top-results",
                "query": "specific target query",
            }
        ],
    )
    _allow_pages(monkeypatch, tmp_path)
    original_ensure = read_back_repair._ensure_query_hint
    monkeypatch.setattr(
        read_back_repair,
        "_ensure_query_hint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        read_back_repair.run_read_back_repair(
            failure_file=failure_file,
            ledger_file=ledger_file,
            hints_file=hints_file,
            now=NOW,
            reviewer=_approve,
        )

    persisted = next(
        iter(json.loads(ledger_file.read_text(encoding="utf-8"))["entries"].values())
    )
    assert persisted["status"] == "frontier_approved"
    assert persisted["frontier_review"]["decision"] == "approved"
    assert not hints_file.exists()

    monkeypatch.setattr(read_back_repair, "_ensure_query_hint", original_ensure)
    recovered = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        now=NOW + timedelta(seconds=1),
        reviewer=lambda _proposal: (_ for _ in ()).throw(
            AssertionError("durable frontier verdict must be reused")
        ),
    )

    assert recovered["applied"] == 1
    assert recovered["actions"][0]["frontier_review_reused"] is True
    assert len(recall_hints.load_query_hints(hints_file)) == 1


def test_processing_is_bounded_and_leaves_remaining_entry_pending(tmp_path: Path, monkeypatch) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    hints_file = tmp_path / "query-hints.json"
    _write_failures(
        failure_file,
        [
            {"page_id": "a", "reason": "not-in-top-results", "query": "specific query a"},
            {"page_id": "b", "reason": "not-in-top-results", "query": "specific query b"},
        ],
    )
    _allow_pages(monkeypatch, tmp_path)

    result = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        max_items=1,
        now=NOW,
        reviewer=_approve,
    )

    assert result["processed"] == 1
    assert result["deferred_by_limit"] == 1
    assert len(recall_hints.load_query_hints(hints_file)) == 1
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    assert sorted(entry["status"] for entry in ledger["entries"].values()) == ["applied", "pending"]


def test_dry_run_is_fully_read_only(tmp_path: Path, monkeypatch) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "runtime" / "ledger.json"
    hints_file = tmp_path / "recall" / "query-hints.json"
    _write_failures(
        failure_file,
        [{"page_id": "target", "reason": "not-in-top-results", "query": "specific target query"}],
    )
    _allow_pages(monkeypatch, tmp_path)
    before = failure_file.read_bytes()

    result = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        dry_run=True,
        now=NOW,
    )

    assert result["actions"][0]["outcome"] == "would_request_frontier"
    assert failure_file.read_bytes() == before
    assert not ledger_file.exists()
    assert not hints_file.exists()
    assert not ledger_file.parent.exists()
    assert not hints_file.parent.exists()


def test_transient_failure_backs_off_then_quarantines(tmp_path: Path) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    _write_failures(
        failure_file,
        [{"page_id": "target", "reason": "search-error", "error": "model temporarily unavailable"}],
    )

    first = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW,
        max_attempts=2,
        retry_base_seconds=60,
    )
    too_early = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW + timedelta(seconds=30),
        max_attempts=2,
        retry_base_seconds=60,
    )
    exhausted = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW + timedelta(seconds=61),
        max_attempts=2,
        retry_base_seconds=60,
    )

    assert first["retry_scheduled"] == 1
    assert too_early["processed"] == 0
    assert too_early["waiting_for_retry"] == 1
    assert exhausted["quarantined"] == 1
    entry = next(iter(json.loads(ledger_file.read_text(encoding="utf-8"))["entries"].values()))
    assert entry["status"] == "quarantined"
    assert entry["attempts"] == 2
    assert "next_attempt_at" not in entry

    still_cooling = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW + timedelta(seconds=61, hours=6) - timedelta(seconds=1),
        max_attempts=2,
        retry_base_seconds=60,
    )
    resumed = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW + timedelta(seconds=61, hours=6, microseconds=1),
        max_attempts=2,
        retry_base_seconds=60,
    )

    assert still_cooling["processed"] == 0
    assert still_cooling["waiting_in_quarantine"] == 1
    assert resumed["resumed_quarantined"] == 1
    assert resumed["retry_scheduled"] == 1
    resumed_entry = next(
        iter(json.loads(ledger_file.read_text(encoding="utf-8"))["entries"].values())
    )
    assert resumed_entry["status"] == "retry_wait"
    assert resumed_entry["attempts"] == 1
    assert resumed_entry["quarantine_resume_count"] == 1


def test_missing_query_hint_target_retries_instead_of_requiring_human(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    _write_failures(
        failure_file,
        [{"page_id": "gone", "reason": "not-in-top-results", "query": "where is gone"}],
    )
    monkeypatch.setattr(recall_hints.wiki, "find_page", lambda page_id: None)
    monkeypatch.setattr(recall_hints.wiki, "SYSTEM_DIR", tmp_path / "system")

    result = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=tmp_path / "query-hints.json",
        now=NOW,
    )

    assert result["retry_scheduled"] == 1
    assert result["human_required"] == 0


def test_access_or_billing_failure_is_the_only_human_required_class(tmp_path: Path) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    _write_failures(
        failure_file,
        [
            {"page_id": "auth", "reason": "search-error", "error": "401 Unauthorized"},
            {"page_id": "missing", "reason": "missing-meta"},
            {"page_id": "empty", "reason": "empty-query"},
            {"page_id": "search", "reason": "search-error", "error": "temporary timeout"},
        ],
    )

    result = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW,
    )

    assert result["human_required"] == 1
    assert result["retry_scheduled"] == 3
    statuses = sorted(
        entry["status"]
        for entry in json.loads(ledger_file.read_text(encoding="utf-8"))["entries"].values()
    )
    assert statuses == ["human_required", "retry_wait", "retry_wait", "retry_wait"]

    read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW + timedelta(days=2),
        max_attempts=1,
        quarantine_cooldown_seconds=1,
    )
    entries = json.loads(ledger_file.read_text(encoding="utf-8"))["entries"].values()
    auth_entry = next(entry for entry in entries if entry["failure"]["page_id"] == "auth")
    assert auth_entry["status"] == "human_required"
    assert int(auth_entry.get("attempts") or 0) == 0


def test_human_required_detection_uses_narrow_words_and_failure_classes() -> None:
    assert read_back_repair._human_required({"error": "author metadata missing"}) is False
    assert read_back_repair._human_required({"error": "quota exceeded"}) is True
    assert (
        read_back_repair._human_required({"failure_class": "frontier_tool_unavailable"})
        is False
    )
    assert read_back_repair._human_required({"error": "temporary model unavailable"}) is False
    assert read_back_repair._human_required({"error": "page file permission denied"}) is False
    assert read_back_repair._human_required({"error": "403 forbidden object policy"}) is False
    assert read_back_repair._human_required({"error": "forbidden page mutation"}) is False
    assert read_back_repair._human_required({"error": "keychain helper unavailable"}) is False
    assert read_back_repair._human_required({"error": "credential store access denied"}) is True


def test_applied_read_back_failure_reopens_when_observed_again(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    hints_file = tmp_path / "query-hints.json"
    failure = {
        "page_id": "target",
        "reason": "not-in-top-results",
        "query": "specific target query",
    }
    _write_failures(failure_file, [failure])
    _allow_pages(monkeypatch, tmp_path)
    first = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        now=NOW,
        retry_base_seconds=60,
        reviewer=_approve,
    )
    assert first["applied"] == 1

    with failure_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-07-10T13:00:00+00:00",
                    "checked": 1,
                    "passed": 0,
                    "failed": [failure],
                }
            )
            + "\n"
        )
    reopened = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        hints_file=hints_file,
        now=NOW + timedelta(hours=2),
        retry_base_seconds=60,
        reviewer=lambda _proposal: (_ for _ in ()).throw(
            AssertionError("durable frontier review must be reused")
        ),
    )

    assert reopened["retry_scheduled"] == 1
    assert recall_hints.load_query_hints(hints_file)[0]["count"] == 1
    entry = next(iter(json.loads(ledger_file.read_text(encoding="utf-8"))["entries"].values()))
    assert entry["status"] == "retry_wait"
    assert entry["attempts"] == 1
    assert entry["reopen_count"] == 1


def test_mutation_budget_defers_without_persisting_or_burning_attempt(
    tmp_path: Path,
) -> None:
    failure_file = tmp_path / "failures.jsonl"
    ledger_file = tmp_path / "ledger.json"
    _write_failures(
        failure_file,
        [{"page_id": "target", "reason": "search-error", "error": "temporary timeout"}],
    )

    deferred = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW,
        budget=CycleBudget(max_mutations=0),
    )

    assert deferred["status"] == "budget_deferred"
    assert deferred["budget_deferred"] == 1
    assert deferred["processed"] == 0
    assert not ledger_file.exists()

    attempted = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger_file,
        now=NOW,
        budget=CycleBudget(max_mutations=1),
    )
    entry = next(iter(json.loads(ledger_file.read_text(encoding="utf-8"))["entries"].values()))
    assert attempted["retry_scheduled"] == 1
    assert entry["attempts"] == 1
